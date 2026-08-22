import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure workspace root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence
from ingestion.audio import extract_audio_evidence, extract_entities
from ingestion.video import extract_video_evidence


def load_transcript(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_temporal_relationships(
    audio_evidences: List[Evidence],
    frame_evidences: List[Evidence],
    window_seconds: float = 5.0
) -> None:
    """
    Connect audio evidence objects with frame evidence objects that occur near the same timestamp.
    """
    for audio_ev in audio_evidences:
        if audio_ev.timestamp is None:
            continue
        audio_time = audio_ev.timestamp
        for frame_ev in frame_evidences:
            if frame_ev.timestamp is None:
                continue
            frame_time = frame_ev.timestamp
            if abs(frame_time - audio_time) <= window_seconds:
                if frame_ev.id not in audio_ev.relationships:
                    audio_ev.relationships.append(frame_ev.id)
                if audio_ev.id not in frame_ev.relationships:
                    frame_ev.relationships.append(audio_ev.id)


def main():
    video_path = "data/raw/meeting.mp4"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    audio_evidences: List[Evidence] = []
    frame_evidences: List[Evidence] = []

    if os.path.exists(video_path):
        print(f"Ingesting video & audio from {video_path}...")
        audio_evidences = extract_audio_evidence(
            video_path,
            output_transcript_path=os.path.join(processed_dir, "transcript.json")
        )
        frame_evidences = extract_video_evidence(
            video_path,
            output_dir=os.path.join(processed_dir, "frames"),
            interval_seconds=5
        )
    elif os.path.exists(os.path.join(processed_dir, "transcript.json")):
        print(f"Loading transcript from {os.path.join(processed_dir, 'transcript.json')}...")
        transcript = load_transcript(os.path.join(processed_dir, "transcript.json"))
        for i, segment in enumerate(transcript):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            audio_evidences.append(Evidence(
                id=f"AUDIO_{i:04d}",
                content=text,
                modality="audio",
                source="meeting.mp4",
                timestamp=start,
                entities=extract_entities(text),
                confidence=0.95,
                relationships=[],
                metadata={"start": start, "end": end}
            ))

    build_temporal_relationships(audio_evidences, frame_evidences)

    all_evidence = [ev.to_dict() for ev in (audio_evidences + frame_evidences)]

    evidence_output_path = os.path.join(processed_dir, "evidence.json")
    with open(evidence_output_path, "w", encoding="utf-8") as f:
        json.dump(all_evidence, f, indent=2, ensure_ascii=False)

    print(f"Created {len(all_evidence)} total evidence objects ({len(audio_evidences)} audio, {len(frame_evidences)} frame).")
    print(f"Saved to {evidence_output_path}")


if __name__ == "__main__":
    main()
