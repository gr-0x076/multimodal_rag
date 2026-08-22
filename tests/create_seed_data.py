"""
ContextMesh — Seed Evidence Data Generator

Creates a realistic evidence.json for testing the retrieval pipeline
without needing to run the full ingestion (which requires video files, Whisper, etc.).

Run:  python -m tests.create_seed_data
"""

import json
import os
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence
from knowledge.relationships import build_relationship_graph


def create_seed_evidence():
    """Create realistic seed evidence covering audio, video_frame, pdf, image."""

    evidence = [
        # --- Audio transcript segments (meeting.mp4) ---
        Evidence(
            id="meeting_audio_0",
            content="Today we are going to discuss the system architecture for our web application.",
            modality="audio",
            source="meeting.mp4",
            timestamp=0.0,
            entities=["architecture", "system"],
            confidence=0.95,
            metadata={"start": 0.0, "end": 5.0},
        ),
        Evidence(
            id="meeting_audio_1",
            content="The main bottleneck we are seeing is database latency under high load.",
            modality="audio",
            source="meeting.mp4",
            timestamp=5.0,
            entities=["database", "latency", "load"],
            confidence=0.95,
            metadata={"start": 5.0, "end": 10.0},
        ),
        Evidence(
            id="meeting_audio_2",
            content="We propose using Redis caching to reduce database load and improve response times.",
            modality="audio",
            source="meeting.mp4",
            timestamp=10.0,
            entities=["Redis", "caching", "database", "load"],
            confidence=0.95,
            metadata={"start": 10.0, "end": 15.0},
        ),
        Evidence(
            id="meeting_audio_3",
            content="The API layer will handle request routing and load balancing across server nodes.",
            modality="audio",
            source="meeting.mp4",
            timestamp=15.0,
            entities=["Api", "load", "server", "node"],
            confidence=0.95,
            metadata={"start": 15.0, "end": 20.0},
        ),

        # --- Video frames (meeting.mp4) ---
        Evidence(
            id="meeting_frame_0",
            content="Title slide: System Architecture Review Meeting.",
            modality="video_frame",
            source="meeting.mp4",
            timestamp=0.0,
            entities=["architecture", "system"],
            confidence=0.85,
            metadata={"frame_path": "data/processed/frames/frame_0.jpg"},
        ),
        Evidence(
            id="meeting_frame_5",
            content="Slide showing database performance metrics and bottleneck analysis.",
            modality="video_frame",
            source="meeting.mp4",
            timestamp=5.0,
            entities=["database"],
            confidence=0.85,
            metadata={"frame_path": "data/processed/frames/frame_5.jpg"},
        ),
        Evidence(
            id="meeting_frame_10",
            content="Redis architecture diagram showing Application to Redis Cache to Database flow.",
            modality="video_frame",
            source="meeting.mp4",
            timestamp=10.0,
            entities=["Redis", "architecture", "database", "caching"],
            confidence=0.85,
            metadata={"frame_path": "data/processed/frames/frame_10.jpg"},
        ),
        Evidence(
            id="meeting_frame_15",
            content="API gateway and load balancer architecture slide.",
            modality="video_frame",
            source="meeting.mp4",
            timestamp=15.0,
            entities=["Api", "load", "architecture"],
            confidence=0.85,
            metadata={"frame_path": "data/processed/frames/frame_15.jpg"},
        ),

        # --- PDF pages (architecture.pdf) ---
        Evidence(
            id="architecture_page_1",
            content="Chapter 1: System Overview. This document describes the architecture of a high-performance web application designed to handle high traffic loads with minimal database latency.",
            modality="pdf",
            source="architecture.pdf",
            page=1,
            entities=["architecture", "system", "database", "latency", "load"],
            confidence=1.0,
            metadata={},
        ),
        Evidence(
            id="architecture_page_2",
            content="Chapter 2: Caching Strategy. Redis is deployed as an in-memory caching layer between the application and the primary database. This reduces database load by serving frequently accessed data from cache, cutting average query latency from 200ms to 5ms.",
            modality="pdf",
            source="architecture.pdf",
            page=2,
            entities=["Redis", "caching", "database", "latency", "load"],
            confidence=1.0,
            metadata={},
        ),
        Evidence(
            id="architecture_page_3",
            content="Chapter 3: API Design. The API layer uses a gateway pattern with load balancing across multiple server nodes. Each node runs an independent Python API server.",
            modality="pdf",
            source="architecture.pdf",
            page=3,
            entities=["Api", "load", "server", "node", "Python"],
            confidence=1.0,
            metadata={},
        ),

        # --- Standalone image (diagram.png) ---
        Evidence(
            id="diagram_img_0",
            content="Architecture diagram showing Client, API Gateway, Redis Cache, and PostgreSQL Database. Arrows indicate data flow with caching layer intercepting read queries.",
            modality="image",
            source="diagram.png",
            entities=["Redis", "architecture", "database", "caching", "Api"],
            confidence=0.9,
            metadata={"image_path": "data/diagram.png"},
        ),
    ]

    return evidence


def main():
    """Generate seed evidence.json with pre-computed relationships."""
    evidence = create_seed_evidence()

    # Build relationships across all evidence
    build_relationship_graph(evidence, temporal_window=6.0, entity_min_overlap=1)

    # Save to data/processed/evidence.json
    output_dir = os.path.join(root_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evidence.json")

    serialized = [ev.to_dict() for ev in evidence]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    print(f"Seed evidence written to: {output_path}")
    print(f"Total evidence objects: {len(evidence)}")
    print()

    # Summary
    modalities = {}
    for ev in evidence:
        modalities[ev.modality] = modalities.get(ev.modality, 0) + 1
    for mod, count in sorted(modalities.items()):
        print(f"  {mod:15s}: {count}")

    print()
    print("Relationships:")
    for ev in evidence:
        if ev.relationships:
            print(f"  {ev.id} -> {ev.relationships}")


if __name__ == "__main__":
    main()
