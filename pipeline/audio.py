import os
import sys
from pathlib import Path

# Ensure workspace root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ingestion.audio import transcribe_video


if __name__ == "__main__":
    video_path = "data/raw/meeting.mp4"
    output_path = "data/processed/transcript.json"

    if not os.path.exists(video_path):
        print(f"Error: {video_path} does not exist. Please place a test video in data/raw/meeting.mp4.")
    else:
        transcribe_video(video_path, output_path)
