import streamlit as st
import os
import json

st.set_page_config(page_title="ContextMesh - Multimodal Ingestion", layout="wide")

st.title("🎥 ContextMesh — Video & Audio Ingestion")
st.markdown("Ingest MP4 video files into timestamped audio transcripts and visual video frames.")

video_file = st.sidebar.text_input("Video File Path", value="data/raw/meeting.mp4")

if st.sidebar.button("Run Ingestion Pipeline"):
    if os.path.exists(video_file):
        with st.spinner("Processing video and audio..."):
            from pipeline.ingest import main as run_ingest
            run_ingest()
        st.success("Ingestion complete!")
    else:
        st.error(f"Video file not found at: {video_file}")

st.header("Processed Evidence Explorer")

evidence_path = "data/processed/evidence.json"
if os.path.exists(evidence_path):
    with open(evidence_path, "r", encoding="utf-8") as f:
        evidences = json.load(f)

    st.write(f"Total Evidence Items: **{len(evidences)}**")

    audio_evs = [e for e in evidences if e.get("modality") == "audio"]
    frame_evs = [e for e in evidences if e.get("modality") == "video_frame"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🎤 Audio Transcripts ({len(audio_evs)})")
        for ev in audio_evs:
            with st.expander(f"[{ev.get('timestamp', 0):.1f}s] {ev.get('id')}"):
                st.write(f"**Content:** {ev.get('content')}")
                st.write(f"**Entities:** {', '.join(ev.get('entities', []))}")
                st.write(f"**Relationships:** {ev.get('relationships')}")

    with col2:
        st.subheader(f"🎥 Video Frames ({len(frame_evs)})")
        for ev in frame_evs:
            with st.expander(f"[{ev.get('timestamp', 0):.1f}s] {ev.get('id')}"):
                st.write(f"**Content:** {ev.get('content')}")
                frame_path = ev.get("metadata", {}).get("frame_path")
                if frame_path and os.path.exists(frame_path):
                    st.image(frame_path, caption=f"Frame at {ev.get('timestamp')}s")
else:
    st.info("No processed evidence found yet. Place a video in `data/raw/meeting.mp4` and run `python pipeline/ingest.py`.")
