import json
import os
import sys
import glob
from pathlib import Path
from typing import List, Dict, Any

# Ensure workspace root is on sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence
from ingestion.audio import extract_audio_evidence, extract_entities
from ingestion.video import extract_video_evidence
from ingestion.pdf import extract_pdf
from ingestion.image import extract_image


def load_transcript(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_temporal_relationships(
    audio_evidences: List[Evidence],
    frame_evidences: List[Evidence],
    window_seconds: float = 6.0
) -> None:
    """
    Connect audio evidence objects with frame evidence objects that occur within a temporal window
    STRICTLY within the same video source file. Also enriches metadata with synchronized snippets.
    """
    for audio_ev in audio_evidences:
        if audio_ev.timestamp is None:
            continue
        audio_time = audio_ev.timestamp
        best_frame = None
        min_frame_diff = float("inf")

        for frame_ev in frame_evidences:
            if frame_ev.timestamp is None:
                continue
            # Enforce same source file requirement for temporal relationships
            if audio_ev.source != frame_ev.source:
                continue
            frame_time = frame_ev.timestamp
            diff = abs(frame_time - audio_time)
            if diff <= window_seconds:
                if frame_ev.id not in audio_ev.relationships:
                    audio_ev.relationships.append(frame_ev.id)
                if audio_ev.id not in frame_ev.relationships:
                    frame_ev.relationships.append(audio_ev.id)

                if diff < min_frame_diff:
                    min_frame_diff = diff
                    best_frame = frame_ev

        # Attach nearest frame's OCR text to audio metadata
        if best_frame and best_frame.metadata.get("ocr_text"):
            audio_ev.metadata["synchronized_screen_text"] = best_frame.metadata["ocr_text"]
            audio_ev.metadata["synchronized_frame_id"] = best_frame.id

    # Inverse pass: attach nearest audio's transcript to frame metadata
    for frame_ev in frame_evidences:
        if frame_ev.timestamp is None:
            continue
        frame_time = frame_ev.timestamp
        best_audio = None
        min_audio_diff = float("inf")

        for audio_ev in audio_evidences:
            if audio_ev.timestamp is None or audio_ev.source != frame_ev.source:
                continue
            diff = abs(audio_ev.timestamp - frame_time)
            if diff <= window_seconds and diff < min_audio_diff:
                min_audio_diff = diff
                best_audio = audio_ev

        if best_audio:
            frame_ev.metadata["synchronized_audio_transcript"] = best_audio.content
            frame_ev.metadata["synchronized_audio_id"] = best_audio.id


def build_entity_relationships(all_evidences: List[Evidence]) -> None:
    """
    Connect evidence objects that share technical entities (cross-modal linking),
    excluding generic system/boilerplate tokens.
    """
    boilerplate_words = {
        "image", "jpeg", "jpg", "png", "ocr", "path", "readme", "see",
        "dimensions", "format", "unavailable", "detected", "extracted",
        "video", "frame", "file", "text", "information", "page"
    }

    for i, ev_a in enumerate(all_evidences):
        entities_a = set(e.lower() for e in ev_a.entities if e.lower() not in boilerplate_words)
        if not entities_a:
            continue
        for j, ev_b in enumerate(all_evidences):
            if i >= j:
                continue
            entities_b = set(e.lower() for e in ev_b.entities if e.lower() not in boilerplate_words)
            if not entities_b:
                continue
            common = entities_a.intersection(entities_b)
            if common:
                if ev_b.id not in ev_a.relationships:
                    ev_a.relationships.append(ev_b.id)
                if ev_a.id not in ev_b.relationships:
                    ev_b.relationships.append(ev_a.id)


def ingest_all(
    data_dir: str = "data",
    processed_dir: str = "data/processed"
) -> List[Evidence]:
    """
    Runs unified multimodal ingestion over videos, audio, PDFs, and images.
    Outputs unified evidence to data/processed/evidence.json.
    """
    os.makedirs(processed_dir, exist_ok=True)
    all_evidences: List[Evidence] = []
    audio_evidences: List[Evidence] = []
    frame_evidences: List[Evidence] = []
    pdf_evidences: List[Evidence] = []
    image_evidences: List[Evidence] = []

    # 1. Ingest Video & Audio
    all_mp4s = list(dict.fromkeys(
        glob.glob(os.path.join(data_dir, "raw", "*.mp4")) + glob.glob(os.path.join(data_dir, "*.mp4"))
    ))
    if any(os.path.basename(v).lower() == "meeting.mp4" for v in all_mp4s):
        meeting_variants = [v for v in all_mp4s if os.path.basename(v).lower() in ("meeting.mp4", "meeting_audio.mp4")]
        if len(meeting_variants) > 1:
            canonical_meeting = next((v for v in meeting_variants if os.path.basename(v).lower() == "meeting.mp4"), meeting_variants[0])
            all_mp4s = [v for v in all_mp4s if v not in meeting_variants] + [canonical_meeting]

    print(f"\n[1/4] Ingesting {len(all_mp4s)} Video & Audio file(s)...")
    for video_path in all_mp4s:
        v_basename = os.path.basename(video_path)
        v_stem = Path(v_basename).stem.lower()
        print(f"  - Processing Video & Audio: {video_path}")

        custom_transcript_candidates = [
            os.path.join(data_dir, f"{v_stem}_transcript.json"),
            os.path.join(data_dir, "raw", f"{v_stem}_transcript.json"),
            os.path.join(processed_dir, f"{v_stem}_transcript.json"),
        ]
        loaded_custom = False
        curr_audio = []
        for custom_p in custom_transcript_candidates:
            if os.path.exists(custom_p):
                print(f"    - Loading transcript: {custom_p}")
                transcript = load_transcript(custom_p)
                for i, segment in enumerate(transcript):
                    curr_audio.append(Evidence(
                        id=f"{v_stem}_audio_{i}",
                        content=segment["text"],
                        modality="audio",
                        source=v_basename,
                        timestamp=segment["start"],
                        entities=extract_entities(segment["text"]),
                        confidence=0.95,
                        relationships=[],
                        metadata={"start": segment["start"], "end": segment["end"], "duration": segment.get("end", 0) - segment.get("start", 0)}
                    ))
                loaded_custom = True
                break

        if not loaded_custom:
            audio_ev = extract_audio_evidence(
                video_path,
                output_transcript_path=os.path.join(processed_dir, f"{v_stem}_transcript.json")
            )
            for a in audio_ev:
                a.source = v_basename
            curr_audio = audio_ev

        audio_evidences.extend(curr_audio)

        curr_frames = extract_video_evidence(
            video_path,
            output_dir=os.path.join(processed_dir, "frames"),
            interval_seconds=5
        )
        
        for f_ev in curr_frames:
            frame_img_path = f_ev.metadata.get("frame_path")
            if frame_img_path and os.path.exists(frame_img_path):
                img_extracted = extract_image(frame_img_path)
                ocr_text = img_extracted.metadata.get("ocr_text", "")
                ocr_status = img_extracted.metadata.get("ocr_status", "unavailable")

                # Update entities from OCR content
                f_ev.entities = extract_entities(img_extracted.content)

                # CRITICAL: write the actual OCR text into the frame's content and metadata
                # so the retrieval layer can search what was DISPLAYED ON SCREEN at this timestamp
                if ocr_text:
                    t = f_ev.timestamp
                    src = f_ev.source
                    f_ev.content = (
                        f"Screen content at {t:.1f}s in {src}:\n{ocr_text}"
                    )
                # Always persist OCR metadata on the frame node
                f_ev.metadata["ocr_text"] = ocr_text
                f_ev.metadata["ocr_status"] = ocr_status
            else:
                f_ev.entities = extract_entities(f_ev.content)
                f_ev.metadata["ocr_text"] = ""
                f_ev.metadata["ocr_status"] = "no_image"

        frame_evidences.extend(curr_frames)

    # 2. Ingest PDFs
    pdf_candidates = glob.glob(os.path.join(data_dir, "*.pdf")) + glob.glob(os.path.join(data_dir, "raw", "*.pdf"))
    print(f"\n[2/4] Ingesting PDFs: {len(pdf_candidates)} found.")
    for pdf_file in pdf_candidates:
        print(f"  - Processing PDF: {pdf_file}")
        extracted = extract_pdf(pdf_file)
        for ev in extracted:
            ev.entities = extract_entities(ev.content)
        pdf_evidences.extend(extracted)

    # 3. Ingest Images / Diagrams
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    image_candidates = []
    for ext in image_extensions:
        image_candidates.extend(glob.glob(os.path.join(data_dir, ext)))
        image_candidates.extend(glob.glob(os.path.join(data_dir, "raw", ext)))

    # Exclude extracted video frames from standalone image scan
    image_candidates = [img for img in image_candidates if "frames" not in img]
    print(f"\n[3/4] Ingesting Standalone Images: {len(image_candidates)} found.")
    for img_file in image_candidates:
        print(f"  - Processing Image: {img_file}")
        ev = extract_image(img_file)
        ev.entities = extract_entities(ev.content)
        image_evidences.append(ev)

    # 4. Build Relationships (Temporal + Entity Cross-Modal)
    print("\n[4/4] Building Cross-Modal & Temporal Relationships...")
    build_temporal_relationships(audio_evidences, frame_evidences, window_seconds=6.0)

    # Sort video evidence by source and timestamp so all nodes for a video (audio + frames) are chronologically grouped
    media_evidences = audio_evidences + frame_evidences
    media_evidences.sort(key=lambda x: (x.source, x.timestamp if x.timestamp is not None else 0.0, x.modality))

    # PDFs sorted by page, images by ID
    pdf_evidences.sort(key=lambda x: (x.source, x.page if x.page is not None else 0))
    image_evidences.sort(key=lambda x: x.id)

    all_evidences = media_evidences + pdf_evidences + image_evidences
    build_entity_relationships(all_evidences)

    # 5. Persist Unified Evidence JSON
    evidence_output_path = os.path.join(processed_dir, "evidence.json")
    serialized_evidence = [ev.to_dict() for ev in all_evidences]
    with open(evidence_output_path, "w", encoding="utf-8") as f:
        json.dump(serialized_evidence, f, indent=2, ensure_ascii=False)

    print(f"\n" + "=" * 55)
    print(f" Unified Ingestion Summary:")
    print(f"   Audio Segments : {len(audio_evidences)}")
    print(f"   Video Frames   : {len(frame_evidences)}")
    print(f"   PDF Pages      : {len(pdf_evidences)}")
    print(f"   Images         : {len(image_evidences)}")
    print(f"   Total Evidence : {len(all_evidences)}")
    print(f" Saved unified evidence graph to: {evidence_output_path}")
    print("=" * 55)

    return all_evidences


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ContextMesh Multimodal Ingestion Pipeline")
    parser.add_argument("--video", type=str, default=None, help="Path to single video for single-video ingestion validation")
    args = parser.parse_args()

    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found at {args.video}")
            sys.exit(1)
        v_basename = os.path.basename(args.video)
        v_stem = os.path.splitext(v_basename)[0]
        proc_dir = "data/processed"
        os.makedirs(proc_dir, exist_ok=True)
        
        print(f"\n=======================================================")
        print(f" Single-Video Ingestion Validation: {args.video}")
        print(f"=======================================================")
        
        # Audio
        audio_ev = extract_audio_evidence(args.video, output_transcript_path=os.path.join(proc_dir, f"{v_stem}_transcript.json"))
        # Video frames
        frame_ev = extract_video_evidence(args.video, output_dir=os.path.join(proc_dir, "frames"), interval_seconds=5)
        for f in frame_ev:
            frame_img_path = f.metadata.get("frame_path")
            if frame_img_path and os.path.exists(frame_img_path):
                img_ex = extract_image(frame_img_path)
                f.entities = extract_entities(img_ex.content)

        build_temporal_relationships(audio_ev, frame_ev, window_seconds=6.0)
        combined = audio_ev + frame_ev
        build_entity_relationships(combined)

        print(f"\nINGESTION DEBUG SUMMARY FOR: {v_basename}")
        print(f"  VIDEO SOURCE         : {args.video}")
        print(f"  AUDIO SEGMENTS       : {len(audio_ev)}")
        print(f"  FRAMES EXTRACTED     : {len(frame_ev)}")
        print(f"  TOTAL EVIDENCE NODES : {len(combined)}")
        print(f"  RELATIONSHIPS BUILT  : {sum(len(e.relationships) for e in combined)}")
        
        if audio_ev:
            print("\nSample Audio Evidence Items:")
            for a in audio_ev[:3]:
                print(f"  - [{a.id}] ({a.timestamp:.1f}s): \"{a.content}\" | Entities: {a.entities}")
        else:
            print("\n  Notice: No speech audio segments transcribed for this video.")

        if frame_ev:
            print("\nSample Video Frame Evidence Items:")
            for f in frame_ev[:3]:
                print(f"  - [{f.id}] ({f.timestamp:.1f}s): {f.metadata.get('frame_path')} | Entities: {f.entities}")

        print(f"=======================================================\n")
    else:
        ingest_all()
