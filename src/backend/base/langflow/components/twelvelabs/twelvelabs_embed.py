# from langflow.field_typing import Data
from langflow.custom import Component
from langflow.inputs import FileInput, SecretStrInput
from langflow.io import Output
from langflow.schema import Data
from typing import List, Dict, Any
import numpy as np
from loguru import logger
from twelvelabs import TwelveLabs
import tempfile
import time
import os
import mimetypes
import subprocess
import json
import magic  # for better file type detection
import base64

class TwelveLabsEmbed(Component):
    display_name = "Video to Embeddings"
    description = "Converts video content to embeddings using Twelve Labs API."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "video"
    name = "VideoToEmbeddings"

    inputs = [
        FileInput(
            name="file_input",
            display_name="Video Input",
            info="Upload a video file to generate embeddings",
            file_types=["mp4", "video/mp4", "video/*", "application/mp4"],  # More permissive file types
        ),
        SecretStrInput(
            name="api_key",
            display_name="Twelve Labs API Key",
            info="Enter your Twelve Labs API Key."
        )
    ]

    outputs = [
        Output(display_name="Embeddings", name="embeddings", method="generate_embeddings"),
    ]

    def wait_for_task_completion(
        self, 
        client: TwelveLabs, 
        task_id: str, 
        max_retries: int = 60, 
        sleep_time: int = 5
    ) -> Dict[str, Any]:
        """Wait for task completion with timeout.
        
        Args:
            client: TwelveLabs client instance
            task_id: ID of the task to monitor
            max_retries: Maximum number of retry attempts
            sleep_time: Time to wait between retries in seconds
            
        Returns:
            Dict containing the task result
            
        Raises:
            Exception: If task fails or times out
        """
        retries = 0
        while retries < max_retries:
            try:
                self.log("Checking task status (attempt {})".format(retries + 1))
                result = client.embed.task.retrieve(id=task_id)
                
                if result.status == "ready":
                    self.log("Task completed successfully!")
                    return result
                elif result.status == "failed":
                    error_msg = f"Task failed with status: {result.status}"
                    self.log(error_msg, "ERROR")
                    raise Exception(error_msg)
                
                time.sleep(sleep_time)
                retries += 1
                status_msg = f"Processing video... {retries * sleep_time}s elapsed"
                self.status = status_msg
                self.log(status_msg)
                
            except Exception as e:
                error_msg = f"Error checking task status: {str(e)}"
                self.log(error_msg, "ERROR")
                raise Exception(error_msg)
        
        timeout_msg = f"Timeout after {max_retries * sleep_time} seconds"
        self.log(timeout_msg, "ERROR")
        raise Exception(timeout_msg)

    def validate_video_file(self, filepath: str) -> tuple[bool, str]:
        """
        Validate video file using ffprobe.
        Returns (is_valid, error_message)
        """
        try:
            cmd = [
                'ffprobe',
                '-loglevel', 'error',
                '-show_entries', 'stream=codec_type,codec_name',
                '-of', 'default=nw=1',
                '-print_format', 'json',
                '-show_format',
                filepath
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, f"FFprobe error: {result.stderr}"
            
            probe_data = json.loads(result.stdout)
            
            # Check if we have a video stream
            has_video = any(
                stream.get('codec_type') == 'video' 
                for stream in probe_data.get('streams', [])
            )
            
            if not has_video:
                return False, "No video stream found in file"
                
            self.log(f"Video validation successful: {json.dumps(probe_data, indent=2)}")
            return True, ""
            
        except subprocess.SubprocessError as e:
            return False, f"FFprobe process error: {str(e)}"
        except json.JSONDecodeError as e:
            return False, f"FFprobe output parsing error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def generate_embeddings(self) -> Data:
        temp_file = None
        try:
            self.log("Starting video embedding process")
            
            if not self.file_input:
                self.log("No file provided", "ERROR")
                return Data(value="No file provided")

            if not self.api_key:
                self.log("No API key provided", "ERROR")
                return Data(value="No API key provided")

            self.log(f"Initializing client with API key: {self.api_key[:4]}...")
            client = TwelveLabs(api_key=self.api_key)

            # At this point self.file_input should be bytes
            self.log(f"Received file content of size: {len(self.file_input)} bytes")

            self.log("Creating temporary file for video...")
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, mode='wb')
            temp_file.write(self.file_input)
            temp_file.close()
            video_path = temp_file.name

            # Debug the file content and type
            with open(video_path, 'rb') as f:
                first_bytes = f.read(16)
                self.log(f"First 16 bytes of file: {first_bytes.hex()}")

            # Use ffprobe for validation first
            is_valid, error_message = self.validate_video_file(video_path)
            if not is_valid:
                self.log(f"Video validation failed: {error_message}", "ERROR")
                return Data(value={"error": f"Invalid video file: {error_message}"})

            self.log("Video file validated successfully")

            # Create video embedding task
            with open(video_path, 'rb') as video_file:
                task = client.embed.task.create(
                    model_name="Marengo-retrieval-2.7",
                    video_file=video_file
                )
                self.log(f"Task created with ID: {task.id}")

            # Wait for task completion synchronously
            result = self.wait_for_task_completion(client, task.id)
            
            if (result.video_embedding is not None and 
                result.video_embedding.segments is not None):
                
                self.log("Processing embedding results...")
                # Get video-level embedding
                video_segments = [seg for seg in result.video_embedding.segments 
                                if seg.embedding_scope == "video"]
                clip_segments = [seg for seg in result.video_embedding.segments 
                               if seg.embedding_scope == "clip"]
                
                embeddings = {
                    'task_id': task.id,
                    'video_embedding': None,
                    'clip_embeddings': []
                }
                
                if video_segments:
                    embeddings['video_embedding'] = [float(x) for x in video_segments[0].embeddings_float]
                    self.log(f"Generated video embedding (dim={len(embeddings['video_embedding'])})")
                
                if clip_segments:
                    embeddings['clip_embeddings'] = [
                        [float(x) for x in segment.embeddings_float]
                        for segment in clip_segments
                    ]
                    self.log(f"Generated {len(embeddings['clip_embeddings'])} clip embeddings")
                
                status_msg = f"Generated video embedding and {len(embeddings['clip_embeddings'])} clip embeddings"
                self.status = status_msg
                self.log(status_msg)
                return Data(value=embeddings)
            else:
                error_msg = "No embeddings found in response"
                self.status = error_msg
                self.log(error_msg, "ERROR")
                return Data(value={"error": error_msg})
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.status = error_msg
            self.log(error_msg, "ERROR")
            return Data(value={"error": str(e)})
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    self.log("Cleaning up temporary file...")
                    os.unlink(temp_file.name)
                    self.log("Temporary file cleaned up successfully")
                except Exception as e:
                    self.log(f"Error cleaning up temporary file: {str(e)}", "ERROR")
