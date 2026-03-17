
import os
import re
import shutil
import tempfile
import logging
from abc import ABC, abstractmethod
import cv2
from pathlib import Path
import yt_dlp

# -----------------------------
# Constants
# -----------------------------
MAX_SIZE_MB = 400 # Just to not starting to download hour long youtube videos

# -----------------------------
# Base source interface
# -----------------------------
class BaseSource(ABC):
    @abstractmethod
    def get_frame(self):
        """Returns a frame"""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup"""
        pass


# -----------------------------
# Video source
# -----------------------------
class VideoSource(BaseSource):
    def __init__(self, path: str):
        """
        Args:
            path (str): Path to video file
        """
        self.path = path
        self.cap = cv2.VideoCapture(self.path)

        # check that video opened correctly
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

    def get_frame(self):
        """
        Returns a frame from the video.
        """
        ret, frame = self.cap.read()

        # if no frame is returned, video is finished
        if not ret:
            self.cap.release()
            return None

        return frame
    
    def cleanup(self):
        """Release the video."""
        if self.cap.isOpened():
            logging.info("Closing video..")
            self.cap.release()


# -----------------------------
# Youtube source
# -----------------------------
class YoutubeSource(BaseSource):
    def __init__(self, url: str):
        """
        Args:
            path (str): url to youtube
        """

        # Use a secure temporary directory for downloads
        temp_dir = Path(tempfile.mkdtemp())
        ydl_opts = {
            "format": "bestvideo[ext=mp4]",
            "outtmpl": str(temp_dir / "video.%(ext)s"),
            "quiet": True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                filesize_bytes = info.get("filesize") or info.get("filesize_approx")
                if filesize_bytes is None:
                    raise ValueError("Could not determine video file size.")

                filesize_mb = filesize_bytes / (1024 * 1024)
                if filesize_mb > MAX_SIZE_MB:
                    raise ValueError(f"The video exceeds the maximum size of {MAX_SIZE_MB}MB. Video is {int(filesize_mb)}MB.")

                ydl.download([url])
                self.downloaded_path = ydl.prepare_filename(info)
            # store the temp dir so we can clean it up later
            self._temp_dir = temp_dir
        except Exception:
            # remove temp dir on failure
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception:
                logging.warning("Failed to remove temporary download dir")
            raise

        self.cap = cv2.VideoCapture(self.downloaded_path)

        # check that video opened correctly
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.downloaded_path}")

    def get_frame(self):
        """
        Returns a frame from the video.
        """
        ret, frame = self.cap.read()

        # if no frame is returned, video is finished
        if not ret:
            self.cap.release()
            return None

        return frame
    
    def cleanup(self):
        """Release the video."""
        if self.cap.isOpened():
            logging.info("Closing video..")
            self.cap.release()



# -----------------------------
# Source factory
# -----------------------------
class SourceFactory:
    @staticmethod
    def create(source_path):
        # If source is an image or a video
        if os.path.isfile(source_path):
            ext = os.path.splitext(source_path)[1].lower()
            if ext in [".mp4", ".avi", ".mov", ".mkv"]:
                return VideoSource(source_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
    
        elif SourceFactory._is_youtube_url(source_path):
            return YoutubeSource(source_path)
        else:
            raise ValueError(f"Invalid source: {source_path}")
        
    @staticmethod
    def _is_youtube_url(path: str) -> bool:
        return bool(re.match(
            r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/", path
        ))