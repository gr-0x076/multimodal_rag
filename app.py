"""
ContextMesh — End-to-End Multimodal RAG & Streamlit Web UI

Question → load_evidence → search_evidence → expand_relationships → Grounded Context → Groq → GroundedAnswer

Usage:
    streamlit run app.py                       # Launch interactive Streamlit Web UI
    python app.py                              # Run golden query via CLI
    python app.py "Your question here"         # Run custom query via CLI
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence, GroundedAnswer
from knowledge.relationships import expand_relationships
from retrieval.search import search_evidence
from retrieval.grounding import build_grounded_prompt, build_grounded_answer, format_provenance
from llm.groq_client import query_groq_with_engine
from tests.test_end_to_end import load_evidence


EVIDENCE_PATH = os.path.join(root_dir, "data", "processed", "evidence.json")

GOLDEN_QUERY = (
    "What architecture was proposed to reduce database load?"
)


# ---------------------------------------------------------------------------
# Core Pipeline (Backend Integration Contract)
# ---------------------------------------------------------------------------

def ask(
    question: str,
    top_k: int = 2,
    max_hops: int = 1,
    verbose: bool = False,
) -> GroundedAnswer:
    """
    Full ContextMesh pipeline matching Person 2's API contract:

        app.py
           ↓
        load evidence
           ↓
        search_evidence(query, all_evidence, top_k=2)
           ↓
        seed Evidence objects
           ↓
        expand_relationships(seed, all_evidence, max_hops=1)
           ↓
        build grounded context
           ↓
        Groq
           ↓
        GroundedAnswer
    """
    all_evidence = load_evidence()
    if not all_evidence:
        return GroundedAnswer(
            query=question,
            answer="No evidence found in data/processed/evidence.json. Please run pipeline/ingest.py first.",
            cited_evidence=[],
            modalities_used=[],
            metadata={"engine": "fallback", "status": "error_no_evidence"}
        )

    # 1. Direct Search Hits
    if verbose:
        print("\n[1/4] Running search_evidence (direct retrieval)...")

    seeds = search_evidence(question, all_evidence, top_k=top_k)

    if verbose:
        print(f"      Top direct hits (top_k={top_k}): {[ev.id for ev in seeds]}")
        for ev in seeds:
            print(f"        [{ev.modality:12s}] {ev.id} ({ev.source})")

    # 2. Relationship Expansion
    if verbose:
        print("\n[2/4] Running expand_relationships (graph traversal)...")

    expanded_evidence = expand_relationships(seeds, all_evidence, max_hops=max_hops)

    if verbose:
        print(f"      Expanded context ({len(expanded_evidence)} items): {[ev.id for ev in expanded_evidence]}")

    # 3. Build Grounded Context Prompt
    if verbose:
        print("\n[3/4] Building grounded context prompt...")

    system_prompt, user_prompt = build_grounded_prompt(question, expanded_evidence)

    # 4. Groq LLM Call
    if verbose:
        print("\n[4/4] Calling Groq LLM...")

    try:
        raw_answer, engine_used = query_groq_with_engine(system_prompt, user_prompt)
    except Exception as e:
        raw_answer = f"[LLM Error] {e}"
        engine_used = "fallback"

    # Handle insufficient evidence case
    if not seeds or not expanded_evidence or "insufficient" in raw_answer.lower():
        cited_evidence_list = [] if (not seeds or "insufficient" in raw_answer.lower()) else expanded_evidence
    else:
        cited_evidence_list = expanded_evidence

    # 5. Build & Return GroundedAnswer Contract Object
    grounded_answer = build_grounded_answer(
        query=question,
        answer_text=raw_answer,
        cited_evidence=cited_evidence_list,
        engine=engine_used,
        metadata={
            "seed_ids": [ev.id for ev in seeds],
            "expanded_ids": [ev.id for ev in expanded_evidence],
            "top_k": top_k,
            "max_hops": max_hops,
        }
    )

    return grounded_answer


# ---------------------------------------------------------------------------
# Pretty Printer for CLI
# ---------------------------------------------------------------------------

def print_result(result: GroundedAnswer) -> None:
    """Print the GroundedAnswer in a clean human-readable format."""
    print("\n" + "=" * 70)
    print("  ContextMesh -- Multimodal RAG Grounded Answer")
    print("=" * 70)

    print(f"\nQuestion: {result.query}")
    print(f"\n{'-' * 70}")
    print(f"\n{result.answer}")
    print(f"\n{'-' * 70}")

    print(f"\nMetadata:")
    print(f"  engine        : {result.metadata.get('engine', 'unknown')}")
    print(f"  seed_ids      : {result.metadata.get('seed_ids', [])}")
    print(f"  expanded_ids  : {result.metadata.get('expanded_ids', [])}")

    if result.cited_evidence:
        print(f"\nCited Evidence ({len(result.cited_evidence)} items, Modalities: {', '.join(result.modalities_used)}):")
        for ev in result.cited_evidence:
            line = f"  * {ev.source}"
            if ev.timestamp is not None:
                mins, secs = divmod(int(ev.timestamp), 60)
                line += f" -- {mins:02d}:{secs:02d}"
            if ev.page is not None:
                line += f" -- page {ev.page}"
            line += f"  [{ev.modality}] ({ev.id})"
            print(line)
    else:
        print("\n  (no cited evidence)")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Streamlit Web UI Implementation (Milestone 3 — Person 1)
# ---------------------------------------------------------------------------

def render_streamlit_ui():
    import streamlit as st

    st.set_page_config(
        page_title="ContextMesh - Multimodal RAG Explorer",
        page_icon="🕸️",
        layout="wide"
    )

    st.markdown("""
    <style>
        .main-header { font-size: 2.3rem; font-weight: 800; color: #0F172A; margin-bottom: 0.1rem; }
        .sub-header { font-size: 1.05rem; color: #475569; margin-bottom: 1.5rem; }
        .answer-box { background-color: #F8FAFC; border-left: 5px solid #2563EB; border-radius: 6px; padding: 18px; margin-bottom: 20px; font-size: 1.05rem; }
        .insufficient-box { background-color: #FFFBEB; border-left: 5px solid #F59E0B; border-radius: 6px; padding: 18px; margin-bottom: 20px; color: #92400E; }
        .badge-audio { background-color: #E0F2FE; color: #0369A1; font-weight: 700; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }
        .badge-frame { background-color: #FEF3C7; color: #B45309; font-weight: 700; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }
        .badge-pdf { background-color: #DCFCE7; color: #15803D; font-weight: 700; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }
        .badge-image { background-color: #F3E8FF; color: #7E22CE; font-weight: 700; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; }
        .card-header { font-weight: 700; font-size: 1.0rem; color: #1E293B; margin-bottom: 4px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🕸️ ContextMesh — Multimodal RAG Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions across Audio, Video Frames, PDF Documents, and Architectural Diagrams with verified provenance.</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Retrieval Parameters")
    top_k = st.sidebar.slider("Top Direct Seeds (top_k)", min_value=1, max_value=5, value=2)
    max_hops = st.sidebar.slider("Graph Expansion Hops (max_hops)", min_value=0, max_value=2, value=1)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔄 Dataset Ingestion")
    if st.sidebar.button("Re-run Unified Ingestion"):
        with st.spinner("Ingesting raw media into evidence store..."):
            from pipeline.ingest import ingest_all
            ingest_all()
        st.sidebar.success("Ingestion complete!")

    # Preset Sample Queries
    st.markdown("### 💡 Try Sample Queries")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    query_input = ""
    if preset_col1.button("🌟 Database Architecture Proposal"):
        query_input = GOLDEN_QUERY
    if preset_col2.button("⏱️ Peak Traffic Bottleneck"):
        query_input = "What is the main bottleneck during peak traffic?"
    if preset_col3.button("⚠️ Insufficient Evidence Query"):
        query_input = "What machine learning algorithm was used to train the model in France?"

    # User Query Input
    user_query = st.text_input("Enter your question:", value=query_input, placeholder="e.g. What architecture was proposed to reduce database load?")

    if user_query:
        with st.spinner("Searching multimodal evidence & querying Groq LLM..."):
            res: GroundedAnswer = ask(user_query, top_k=top_k, max_hops=max_hops, verbose=False)

        st.markdown("---")
        st.markdown(f"### ❓ Question: *{res.query}*")

        # Engine & Modality Badges
        engine_name = res.metadata.get("engine", "fallback")
        engine_label = "⚡ Groq (llama-3.3-70b-versatile)" if engine_name == "groq" else "🛡️ Grounded Fallback Engine"
        st.caption(f"Engine: **{engine_label}** | Modalities Utilized: **{', '.join(res.modalities_used) if res.modalities_used else 'None'}**")

        # Check Insufficient Evidence
        is_insufficient = (len(res.cited_evidence) == 0) or ("insufficient" in res.answer.lower())

        if is_insufficient:
            st.markdown(
                f'<div class="insufficient-box">'
                f'<h3>⚠️ Insufficient Evidence</h3>'
                f'<p>{res.answer}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            # Grounded Answer Display
            st.markdown(
                f'<div class="answer-box">'
                f'<h3>💬 Grounded Answer</h3>'
                f'{res.answer}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Cited Evidence Cards Breakdown
            st.markdown(f"### 📁 Supporting Evidence ({len(res.cited_evidence)} items)")
            
            # Group cited evidence by modality
            audio_items = [e for e in res.cited_evidence if e.modality == "audio"]
            frame_items = [e for e in res.cited_evidence if e.modality == "video_frame"]
            pdf_items = [e for e in res.cited_evidence if e.modality == "pdf"]
            image_items = [e for e in res.cited_evidence if e.modality == "image"]

            ev_col1, ev_col2 = st.columns(2)

            with ev_col1:
                if audio_items:
                    st.subheader(f"🎤 Spoken Audio ({len(audio_items)})")
                    for ev in audio_items:
                        mins, secs = divmod(int(ev.timestamp or 0), 60)
                        ts_formatted = f"{mins:02d}:{secs:02d}"
                        with st.expander(f"[{ts_formatted}] {ev.source} ({ev.id})", expanded=True):
                            st.markdown('<span class="badge-audio">MODALITY: AUDIO</span>', unsafe_allow_html=True)
                            st.write(f"**Source:** `{ev.source}` @ `{ts_formatted}` ({ev.timestamp}s)")
                            st.info(f"🎤 *\"{ev.content}\"*")
                            if ev.entities:
                                st.write(f"**Entities:** `{', '.join(ev.entities)}`")

                if pdf_items:
                    st.subheader(f"📄 PDF Documents ({len(pdf_items)})")
                    for ev in pdf_items:
                        with st.expander(f"[Page {ev.page}] {ev.source} ({ev.id})", expanded=True):
                            st.markdown('<span class="badge-pdf">MODALITY: PDF</span>', unsafe_allow_html=True)
                            st.write(f"**Source:** `{ev.source}` — **Page:** `{ev.page}`")
                            st.success(f"📄 *\"{ev.content}\"*")
                            if ev.entities:
                                st.write(f"**Entities:** `{', '.join(ev.entities)}`")

            with ev_col2:
                if frame_items:
                    st.subheader(f"🎥 Video Frames ({len(frame_items)})")
                    for ev in frame_items:
                        mins, secs = divmod(int(ev.timestamp or 0), 60)
                        ts_formatted = f"{mins:02d}:{secs:02d}"
                        with st.expander(f"[{ts_formatted}] {ev.source} ({ev.id})", expanded=True):
                            st.markdown('<span class="badge-frame">MODALITY: VIDEO_FRAME</span>', unsafe_allow_html=True)
                            st.write(f"**Source:** `{ev.source}` @ `{ts_formatted}` ({ev.timestamp}s)")
                            frame_path = ev.metadata.get("frame_path")
                            if frame_path and os.path.exists(frame_path):
                                st.image(frame_path, caption=f"Frame at {ts_formatted} ({ev.source})", use_container_width=True)
                            else:
                                st.write(f"*{ev.content}*")

                if image_items:
                    st.subheader(f"🖼️ Images & Diagrams ({len(image_items)})")
                    for ev in image_items:
                        with st.expander(f"{ev.source} ({ev.id})", expanded=True):
                            st.markdown('<span class="badge-image">MODALITY: IMAGE</span>', unsafe_allow_html=True)
                            st.write(f"**Source:** `{ev.source}`")
                            st.write(f"*{ev.content}*")
                            img_path = os.path.join("data", ev.source)
                            if os.path.exists(img_path):
                                st.image(img_path, caption=ev.source, use_container_width=True)


# ---------------------------------------------------------------------------
# Execution Dispatcher (CLI vs Streamlit)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Detect if running via Streamlit
    is_streamlit = (
        "streamlit" in sys.modules and
        any("streamlit" in arg for arg in sys.argv)
    ) or os.environ.get("STREAMLIT_SERVER_PORT") is not None

    if is_streamlit:
        render_streamlit_ui()
    else:
        if len(sys.argv) > 1:
            question = " ".join(sys.argv[1:])
        else:
            question = GOLDEN_QUERY

        result = ask(question, verbose=True)
        print_result(result)
