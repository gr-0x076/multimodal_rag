import os
import cv2
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path

# Ensure workspace root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence


def extract_frames(
    video_path: str,
    output_dir: str = "data/processed/frames",
    interval_seconds: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Extract video frames at every `interval_seconds` using OpenCV.
    Returns a list of metadata dicts for saved frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        video.release()
        raise ValueError(f"Could not read valid video metadata from {video_path}")

    duration = frame_count / fps

    print(f"Video FPS: {fps:.2f}")
    print(f"Video Duration: {duration:.2f} seconds")

    current_time = 0.0
    extracted_frames = []

    while current_time < duration:
        video.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        success, frame = video.read()

        if success:
            filename = f"frame_{int(current_time):06d}.jpg"
            frame_path = os.path.join(output_dir, filename)
            cv2.imwrite(frame_path, frame)
            print(f"Saved frame at {current_time:.2f}s -> {frame_path}")

            extracted_frames.append({
                "timestamp": round(current_time, 2),
                "filename": filename,
                "path": frame_path
            })

        current_time += interval_seconds

    video.release()
    return extracted_frames


def extract_video_evidence(
    video_path: str,
    output_dir: str = "data/processed/frames",
    interval_seconds: float = 5.0,
    source_name: Optional[str] = None
) -> List[Evidence]:
    """
    Extracts frames from a video file and converts them into Evidence objects.
    """
    source = source_name or os.path.basename(video_path)
    source_clean = Path(source).stem.lower()

    frames = extract_frames(
        video_path,
        output_dir=output_dir,
        interval_seconds=interval_seconds
    )

    evidence_list = []
    for frame_info in frames:
        t = frame_info["timestamp"]
        frame_path = frame_info["path"]
        frame_filename = frame_info["filename"]

        ev_id = f"{source_clean}_frame_{int(t)}"
        content = f"Video frame extracted at {t:.2f}s from {source} ({frame_filename})"

        ev = Evidence(
            id=ev_id,
            content=content,
            modality="video_frame",
            source=source,
            timestamp=t,
            entities=[],
            confidence=1.0,
            relationships=[],
            metadata={
                "frame_path": frame_path,
                "filename": frame_filename,
                "interval_seconds": interval_seconds
            }
        )
        evidence_list.append(ev)

    return evidence_list


if __name__ == "__main__":
    import sys
    test_video = sys.argv[1] if len(sys.argv) > 1 else "data/raw/meeting.mp4"
    if os.path.exists(test_video):
        evidences = extract_video_evidence(test_video, "data/processed/frames", interval_seconds=5)
        print(f"Extracted {len(evidences)} frame evidence items.")
    else:
        print(f"File not found: {test_video}. Please place a test video in data/raw/meeting.mp4.")
