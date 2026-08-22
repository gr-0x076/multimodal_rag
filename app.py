"""
ContextMesh — Multimodal RAG Application
Interactive Streamlit Interface and CLI Grounded Answering Tool.

Usage:
    streamlit run app.py                      # Launch the interactive Multimodal UI
    python app.py                             # Run the Golden Demo Query via CLI
    python app.py "Your custom query here"    # Run a custom query via CLI
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path
root_dir = str(Path(__file__).resolve().parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from knowledge.schema import Evidence, GroundedAnswer
from knowledge.relationships import expand_relationships
from retrieval.search import search_evidence
from llm.groq_client import generate_grounded_answer, query_groq_with_engine
from pipeline.ingest import ingest_all

GOLDEN_QUERY = "What architecture was proposed to reduce database load, and what visual/document evidence supports it?"


# -----------------------------------------------------------------------------
# Core Pipeline Execution Function (CLI + Streamlit Backend)
# -----------------------------------------------------------------------------
def ask(
    question: str,
    top_k: int = 2,
    max_hops: int = 1,
    verbose: bool = False,
) -> GroundedAnswer:
    """
    Executes the full ContextMesh pipeline:
      1. Load unified evidence from data/processed/evidence.json
      2. Direct Search: search_evidence(question, all_evidence, top_k=2)
      3. Graph Traversal: expand_relationships(seeds, all_evidence, max_hops=1)
      4. Grounded Synthesis: generate_grounded_answer(question, expanded_context)
    """
    evidence_path = os.path.join(root_dir, "data", "processed", "evidence.json")
    if not os.path.exists(evidence_path):
        if verbose:
            print("evidence.json not found. Running multimodal ingestion...")
        ingest_all(data_dir=os.path.join(root_dir, "data"), processed_dir=os.path.join(root_dir, "data", "processed"))

    with open(evidence_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    all_evidence = [
        Evidence(
            id=item["id"],
            content=item["content"],
            modality=item["modality"],
            source=item["source"],
            timestamp=item.get("timestamp"),
            page=item.get("page"),
            entities=item.get("entities", []),
            confidence=item.get("confidence", 1.0),
            relationships=item.get("relationships", []),
            metadata=item.get("metadata", {})
        )
        for item in raw_items
    ]

    # 1. Direct Search
    seeds = search_evidence(question, all_evidence, top_k=top_k)
    if verbose:
        print(f"[Direct Hits (top_k={top_k})]: {[ev.id for ev in seeds]}")

    # 2. Relationship Expansion (1-hop graph traversal)
    expanded = expand_relationships(seeds, all_evidence, max_hops=max_hops)
    if verbose:
        print(f"[Expanded Graph ({len(expanded)} items)]: {[ev.id for ev in expanded]}")

    # 3. Grounded Answer Synthesis
    grounded_answer = generate_grounded_answer(question, expanded)
    grounded_answer.metadata.update({
        "seed_ids": [ev.id for ev in seeds],
        "expanded_ids": [ev.id for ev in expanded],
        "top_k": top_k,
        "max_hops": max_hops
    })
    return grounded_answer


def print_cli_result(result: GroundedAnswer) -> None:
    """Print the GroundedAnswer in a clean CLI format."""
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
    print("\n" + "=" * 70)


# -----------------------------------------------------------------------------
# Streamlit Interactive Application
# -----------------------------------------------------------------------------
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="ContextMesh — Multimodal RAG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
        .main-header {
            font-size: 2.3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #3B82F6, #8B5CF6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #64748B;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .answer-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1.5rem;
        }
        .badge-groq {
            background-color: #DCFCE7;
            color: #166534;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-fallback {
            background-color: #FEF3C7;
            color: #92400E;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .modality-badge {
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-right: 0.3rem;
        }
        .mod-audio { background: #E0E7FF; color: #3730A3; }
        .mod-video { background: #FCE7F3; color: #9D174D; }
        .mod-pdf { background: #FEF08A; color: #854D0E; }
        .mod-image { background: #CCFBF1; color: #115E59; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar: Ingestion & API Settings
    with st.sidebar:
        st.image("https://img.icons8.com/isometric/100/mind-map.png", width=60)
        st.markdown("### ContextMesh Pipeline")
        st.caption("Multimodal Graph-Augmented Retrieval")
        st.markdown("---")
        st.subheader("⚙️ Store & Ingestion")

        evidence_path = os.path.join(root_dir, "data", "processed", "evidence.json")
        ev_count = 0
        if os.path.exists(evidence_path):
            with open(evidence_path, "r", encoding="utf-8") as f:
                ev_count = len(json.load(f))
        st.write(f"Indexed Evidence Nodes: **{ev_count}**")

        if st.button("🔄 Re-run Ingestion Pipeline", use_container_width=True):
            with st.spinner("Processing Video, Audio, PDFs & Images..."):
                ingest_all(data_dir=os.path.join(root_dir, "data"), processed_dir=os.path.join(root_dir, "data", "processed"))
                st.rerun()

        st.markdown("---")
        st.subheader("🔑 Groq API Configuration")
        custom_key = st.text_input("Groq API Key (Optional)", type="password", help="Overrides GROQ_API_KEY from .env")
        if custom_key:
            os.environ["GROQ_API_KEY"] = custom_key.strip()
        st.markdown("---")
        st.caption("ContextMesh MVP • Phase 4")

    st.markdown('<div class="main-header">🧠 ContextMesh — Multimodal RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cross-Modal & Temporal Evidence Retrieval over Video, Audio, PDFs, and Diagrams</div>', unsafe_allow_html=True)

    # Demo Quick-Select Buttons
    st.markdown("##### ⚡ Quick Demo Scenarios for Evaluators:")
    col_d1, col_d2, col_d3 = st.columns(3)
    preset_query = ""
    if col_d1.button("🎯 Demo 1: Golden Redis Query", use_container_width=True):
        preset_query = GOLDEN_QUERY
    if col_d2.button("⚡ Demo 2: Peak Bottleneck", use_container_width=True):
        preset_query = "What was the primary bottleneck during peak traffic?"
    if col_d3.button("🛑 Demo 3: Unsupported Refusal", use_container_width=True):
        preset_query = "What machine-learning algorithm was used to train Redis caching?"

    # Query Input
    user_query = st.text_input(
        "Ask a multimodal question:",
        value=preset_query if preset_query else GOLDEN_QUERY,
        placeholder="e.g. What architecture was proposed to reduce database load?"
    )

    if user_query:
        with st.spinner("Searching multimodal evidence & expanding graph..."):
            grounded_ans = ask(user_query, top_k=2, max_hops=1)

        # Grounded Answer Display
        st.markdown("### 💬 Grounded Synthesis")
        engine_type = grounded_ans.metadata.get("engine", "fallback")
        if engine_type == "groq":
            badge_html = '<span class="badge-groq">⚡ Live Groq API (LLaMA-3.3-70B)</span>'
        else:
            badge_html = '<span class="badge-fallback">🛡️ Grounded Deterministic Engine (Zero Hallucination)</span>'

        st.markdown(f"""
        <div class="answer-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <span style="font-weight: 600; color: #1E293B;">Synthesized Answer</span>
                {badge_html}
            </div>
            <div style="font-size: 1.08rem; line-height: 1.6; color: #0F172A;">
                {grounded_ans.answer}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Multimodal Evidence Grid
        st.markdown(f"### 📍 Multimodal Supporting Evidence ({len(grounded_ans.cited_evidence)} Connected Nodes)")
        tab1, tab2 = st.tabs(["🧩 Evidence Cards", "🕸️ Cross-Modal Relationship Graph"])

        with tab1:
            cols = st.columns(min(len(grounded_ans.cited_evidence), 3) if grounded_ans.cited_evidence else 1)
            for idx, ev in enumerate(grounded_ans.cited_evidence):
                col = cols[idx % 3]
                with col:
                    mod_cls = f"mod-{ev.modality.replace('_frame', '')}"
                    mod_label = ev.modality.replace('_', ' ').upper()
                    loc_str = ""
                    if ev.timestamp is not None:
                        loc_str = f"⏱️ {int(ev.timestamp // 60):02d}:{int(ev.timestamp % 60):02d} ({ev.timestamp:.1f}s)"
                    elif ev.page is not None:
                        loc_str = f"📄 Page {ev.page}"
                    else:
                        loc_str = f"🖼️ Image"

                    with st.container(border=True):
                        st.markdown(f'<span class="modality-badge {mod_cls}">{mod_label}</span> <small style="color: #64748B;">{loc_str}</small>', unsafe_allow_html=True)
                        st.markdown(f"**Source:** `{ev.source}`")
                        
                        if ev.modality == "video_frame":
                            frame_path = ev.metadata.get("frame_path")
                            if frame_path and os.path.exists(frame_path):
                                st.image(frame_path, caption=f"Frame at {ev.timestamp:.1f}s", use_container_width=True)
                        elif ev.modality == "image":
                            img_path = os.path.join(root_dir, "data", ev.source)
                            if os.path.exists(img_path):
                                st.image(img_path, caption=ev.source, use_container_width=True)

                        st.write(f"_{ev.content[:140]}..._")
                        if ev.entities:
                            st.caption(f"🏷️ **Entities:** {', '.join(ev.entities)}")
                        if ev.relationships:
                            st.caption(f"🔗 **Linked to:** {', '.join(ev.relationships)}")

        with tab2:
            st.write("#### 🔗 Provenance & Cross-Modal Links")
            graph_data = []
            for ev in grounded_ans.cited_evidence:
                graph_data.append({
                    "Node ID": ev.id,
                    "Modality": ev.modality,
                    "Source": ev.source,
                    "Location": f"{ev.timestamp:.1f}s" if ev.timestamp is not None else f"Page {ev.page}" if ev.page is not None else "Image",
                    "Entities": ", ".join(ev.entities),
                    "Connected Edges": ", ".join(ev.relationships)
                })
            st.dataframe(graph_data, use_container_width=True)

    with st.expander("📂 Explore Raw Dataset Files in Repository"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Video Files (`data/raw/`):**")
            st.code("meeting.mp4 (15s Presentation + Audio)")
        with c2:
            st.markdown("**PDF Documents (`data/`):**")
            st.code("architecture.pdf (2 Pages Specification)")
        with c3:
            st.markdown("**Image Assets (`data/`):**")
            st.code("diagram.png (Architecture Topology)")


# -----------------------------------------------------------------------------
# Main Execution Guard (Runs Streamlit if called via streamlit run, CLI if python app.py)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Check if executed via `streamlit run`
    is_streamlit = any("streamlit" in arg for arg in sys.argv) or ("STREAMLIT_SERVER_PORT" in os.environ)
    if not is_streamlit:
        # Check if caller is streamlit main module
        try:
            import inspect
            stack = [frame.filename for frame in inspect.stack()]
            if any("streamlit" in fn for fn in stack):
                is_streamlit = True
        except Exception:
            pass

    if is_streamlit:
        run_streamlit_app()
    else:
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else GOLDEN_QUERY
        result = ask(q, verbose=True)
        print_cli_result(result)

