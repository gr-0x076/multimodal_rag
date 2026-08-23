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
# Helpers
# -----------------------------------------------------------------------------
def _fmt_ts(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02d}:{secs:02d}"


def _load_evidence_stats(evidence_path: str) -> Dict:
    if not os.path.exists(evidence_path):
        return {"total": 0, "audio": 0, "video_frame": 0, "pdf": 0, "image": 0, "sources": []}
    with open(evidence_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    stats = {"total": len(items), "audio": 0, "video_frame": 0, "pdf": 0, "image": 0, "sources": []}
    seen_sources = set()
    for it in items:
        mod = it.get("modality", "")
        if mod in stats:
            stats[mod] += 1
        src = it.get("source", "")
        if src and src not in seen_sources:
            seen_sources.add(src)
            stats["sources"].append(src)
    return stats


# -----------------------------------------------------------------------------
# Streamlit Interactive Application
# -----------------------------------------------------------------------------
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="ContextMesh — Multimodal RAG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # ── Global CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hide default Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Main background */
    .stApp { background: #0A0E1A; }
    .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

    /* ── Hero header ── */
    .cm-hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .cm-logo {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818CF8 0%, #C084FC 40%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .cm-tagline {
        color: #64748B;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* ── Status bar ── */
    .cm-status-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 0.9rem 2rem;
        margin: 1rem auto 2rem;
        max-width: 780px;
    }
    .cm-status-dot {
        width: 9px; height: 9px;
        background: #22C55E;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #22C55E;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .cm-status-item {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .cm-status-item strong { color: #E2E8F0; }

    /* ── Query section ── */
    .cm-query-label {
        font-size: 1.3rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 0.5rem;
    }

    /* Streamlit text_input override */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(129, 140, 248, 0.35) !important;
        border-radius: 12px !important;
        color: #E2E8F0 !important;
        font-size: 1.05rem !important;
        padding: 0.9rem 1.2rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.2s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #818CF8 !important;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
    }

    /* ── Quick query chips ── */
    .cm-chip-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .cm-try-label {
        color: #475569;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }

    /* Override Streamlit button for chips */
    div[data-testid="column"] .stButton > button {
        background: rgba(129,140,248,0.1);
        border: 1px solid rgba(129,140,248,0.3);
        border-radius: 999px;
        color: #A5B4FC;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.3rem 0.9rem;
        transition: all 0.2s;
        white-space: nowrap;
    }
    div[data-testid="column"] .stButton > button:hover {
        background: rgba(129,140,248,0.25);
        border-color: #818CF8;
        color: #E0E7FF;
    }

    /* Main search button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.7rem 2.5rem;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4);
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 25px rgba(99,102,241,0.5);
    }

    /* ── Answer card ── */
    .cm-answer-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.06) 100%);
        border: 1px solid rgba(129,140,248,0.25);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
    }
    .cm-answer-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .cm-answer-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #818CF8;
    }
    .cm-answer-text {
        color: #E2E8F0;
        font-size: 1.05rem;
        line-height: 1.75;
    }
    .cm-answer-meta {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    .cm-meta-pill {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 500;
    }

    /* ── Engine badges ── */
    .badge-groq {
        background: linear-gradient(135deg, #166534, #15803D);
        color: #BBF7D0;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid rgba(187,247,208,0.2);
    }
    .badge-fallback {
        background: linear-gradient(135deg, #92400E, #B45309);
        color: #FDE68A;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid rgba(253,230,138,0.2);
    }
    .badge-insufficient {
        background: linear-gradient(135deg, #7F1D1D, #991B1B);
        color: #FCA5A5;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* ── Evidence section label ── */
    .cm-section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #C4B5FD;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.8rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .cm-section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(196,181,253,0.3) 0%, transparent 100%);
        margin-left: 0.75rem;
    }

    /* ── Evidence cards ── */
    .cm-ev-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        transition: border-color 0.2s, background 0.2s;
    }
    .cm-ev-card:hover {
        background: rgba(255,255,255,0.055);
        border-color: rgba(255,255,255,0.13);
    }
    .cm-ev-card-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.7rem;
    }
    .mod-pill {
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .mod-audio  { background: rgba(99,102,241,0.2); color: #A5B4FC; border: 1px solid rgba(99,102,241,0.3); }
    .mod-video  { background: rgba(236,72,153,0.15); color: #F9A8D4; border: 1px solid rgba(236,72,153,0.25); }
    .mod-pdf    { background: rgba(234,179,8,0.15); color: #FDE68A; border: 1px solid rgba(234,179,8,0.25); }
    .mod-image  { background: rgba(20,184,166,0.15); color: #99F6E4; border: 1px solid rgba(20,184,166,0.25); }
    .cm-ev-ts {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }
    .cm-ev-src {
        font-size: 0.75rem;
        color: #475569;
        margin-left: auto;
        font-style: italic;
    }
    .cm-ev-quote {
        color: #CBD5E1;
        font-size: 0.92rem;
        line-height: 1.6;
        border-left: 3px solid rgba(129,140,248,0.5);
        padding-left: 0.8rem;
        margin: 0.5rem 0;
        font-style: italic;
    }
    .cm-ev-screen {
        background: rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 0.6rem;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        max-height: 100px;
        overflow: hidden;
    }
    .cm-ev-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin-top: 0.6rem;
    }
    .cm-tag {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 4px;
        padding: 0.1rem 0.4rem;
        font-size: 0.68rem;
        color: #64748B;
        font-weight: 500;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 0.2rem;
        gap: 0.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #64748B;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(129,140,248,0.15) !important;
        color: #A5B4FC !important;
    }

    /* ── Graph text ── */
    .cm-graph-node {
        font-family: 'Courier New', monospace;
        color: #94A3B8;
        font-size: 0.88rem;
        line-height: 2;
    }
    .cm-graph-node .node-audio  { color: #818CF8; font-weight: 700; }
    .cm-graph-node .node-video  { color: #F472B6; font-weight: 700; }
    .cm-graph-node .node-pdf    { color: #FCD34D; font-weight: 700; }
    .cm-graph-node .node-image  { color: #34D399; font-weight: 700; }

    /* ── Divider ── */
    .cm-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(129,140,248,0.2) 30%, rgba(196,181,253,0.2) 70%, transparent 100%);
        margin: 2rem 0;
    }

    /* ── Provenance table ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    .stDataFrame th { background: rgba(129,140,248,0.15) !important; color: #A5B4FC !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(10,14,26,0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Insufficient evidence card ── */
    .cm-insuf-card {
        background: rgba(153,27,27,0.15);
        border: 1px solid rgba(239,68,68,0.25);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        color: #FCA5A5;
    }
    .cm-insuf-title { font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem; }
    .cm-insuf-text { font-size: 0.95rem; color: #FCA5A5; opacity: 0.85; line-height: 1.6; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.3); border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Evidence stats ─────────────────────────────────────────────────────────
    evidence_path = os.path.join(root_dir, "data", "processed", "evidence.json")
    stats = _load_evidence_stats(evidence_path)
    groq_key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())

    # Try loading .env
    env_path = os.path.join(root_dir, ".env")
    if not groq_key_set and os.path.exists(env_path):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
            groq_key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())
        except ImportError:
            with open(env_path) as ef:
                for line in ef:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            os.environ["GROQ_API_KEY"] = val
                            groq_key_set = True
                        break

    engine_label = "⚡ Groq" if groq_key_set else "🛡️ Fallback"
    system_ready = stats["total"] > 0

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 ContextMesh")
        st.caption("Multimodal Evidence Explorer")
        st.markdown("---")
        st.markdown("### ⚙️ Pipeline")
        st.write(f"**Evidence Nodes:** {stats['total']}")
        st.write(f"**Sources:** {len(stats['sources'])}")
        for src in stats["sources"]:
            st.caption(f"  📁 {src}")
        st.markdown("---")
        if st.button("🔄 Re-run Ingestion"):
            with st.spinner("Processing all media…"):
                ingest_all(
                    data_dir=os.path.join(root_dir, "data"),
                    processed_dir=os.path.join(root_dir, "data", "processed")
                )
                st.rerun()
        st.markdown("---")
        st.markdown("### 🔑 Groq API Key")
        custom_key = st.text_input("Override key (optional)", type="password")
        if custom_key:
            os.environ["GROQ_API_KEY"] = custom_key.strip()
        st.markdown("---")
        st.caption("ContextMesh · Multimodal RAG")

    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cm-hero">
        <div class="cm-logo">ContextMesh</div>
        <div class="cm-tagline">Multimodal RAG — Connect speech, visuals &amp; documents into grounded answers.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status bar ────────────────────────────────────────────────────────────
    ready_dot = '<span class="cm-status-dot"></span>' if system_ready else '🔴'
    ready_text = "System Ready" if system_ready else "No Evidence"
    audio_chk = "✓" if stats["audio"] > 0 else "✗"
    video_chk = "✓" if stats["video_frame"] > 0 else "✗"
    pdf_chk   = "✓" if stats["pdf"] > 0 else "✗"
    img_chk   = "✓" if stats["image"] > 0 else "✗"

    st.markdown(f"""
    <div class="cm-status-bar">
        <span class="cm-status-item">{ready_dot} <strong>{ready_text}</strong></span>
        <span class="cm-status-item">📦 <strong>{stats['total']}</strong> nodes</span>
        <span class="cm-status-item">🎙️ Audio {audio_chk}</span>
        <span class="cm-status-item">🎥 Video {video_chk}</span>
        <span class="cm-status-item">📄 PDF {pdf_chk}</span>
        <span class="cm-status-item">🖼️ Images {img_chk}</span>
        <span class="cm-status-item">Engine: <strong>{engine_label}</strong></span>
    </div>
    """, unsafe_allow_html=True)

    # ── Query box ─────────────────────────────────────────────────────────────
    st.markdown('<div class="cm-query-label">🔍 Ask anything about your content</div>', unsafe_allow_html=True)

    # Quick demo chip buttons
    QUICK_QUERIES = [
        ("🎯 Redis Architecture",       "What architecture was proposed to reduce database load?"),
        ("⚡ Peak Bottleneck",           "What was the primary bottleneck during peak traffic?"),
        ("🐍 Python Functions",          "What was being explained about Python functions in the video?"),
        ("💡 Why Reuse Code?",           "Why did the speaker say we need to reuse code?"),
        ("🖥️ What was on screen?",      "What code was displayed on the screen?"),
        ("🛑 Unsupported Query",         "What machine-learning algorithm was used to train Redis caching?"),
    ]

    if "preset_query" not in st.session_state:
        st.session_state["preset_query"] = ""

    st.markdown('<div class="cm-try-label">Try a query:</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(QUICK_QUERIES))
    for i, (label, qtext) in enumerate(QUICK_QUERIES):
        with chip_cols[i]:
            if st.button(label, key=f"chip_{i}"):
                st.session_state["preset_query"] = qtext
                st.rerun()

    user_query = st.text_input(
        "Your question:",
        value=st.session_state.get("preset_query", ""),
        placeholder="e.g. What was being explained about Python functions?",
        label_visibility="collapsed"
    )

    search_col, _ = st.columns([1, 4])
    with search_col:
        search_clicked = st.button("🔍 Ask ContextMesh", type="primary", use_container_width=True)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if search_clicked and user_query.strip():
        st.session_state["last_query"] = user_query.strip()
        with st.spinner("🔍 Searching multimodal evidence & expanding graph…"):
            result = ask(user_query.strip(), top_k=2, max_hops=1)
        st.session_state["last_result"] = result

    result: Optional[GroundedAnswer] = st.session_state.get("last_result")
    if result is None:
        st.markdown('<hr class="cm-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0; color: #334155;">
            <div style="font-size:3rem; margin-bottom:0.5rem;">🔍</div>
            <div style="font-size:1rem; font-weight:600; color:#475569;">Enter a question above to explore your content</div>
            <div style="font-size:0.85rem; color:#334155; margin-top:0.3rem;">ContextMesh retrieves evidence across speech, video frames, PDFs and images</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Answer section ────────────────────────────────────────────────────────
    st.markdown('<hr class="cm-divider">', unsafe_allow_html=True)

    engine = result.metadata.get("engine", "fallback")
    is_insufficient = "insufficient" in result.answer.lower()
    num_ev = len(result.cited_evidence)
    modalities_str = " + ".join(
        {"audio": "🎙️ Audio", "video_frame": "🎥 Video", "pdf": "📄 PDF", "image": "🖼️ Image"}.get(m, m)
        for m in sorted(set(ev.modality for ev in result.cited_evidence))
    )

    if engine == "groq":
        badge = '<span class="badge-groq">⚡ Live Groq · LLaMA-3.3-70B</span>'
    else:
        badge = '<span class="badge-fallback">🛡️ Deterministic Fallback</span>'

    if is_insufficient:
        st.markdown(f"""
        <div class="cm-insuf-card">
            <div class="cm-insuf-title">🛡️ INSUFFICIENT EVIDENCE</div>
            <div class="cm-insuf-text">{result.answer}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        answer_html = result.answer.replace("\n", "<br>")
        st.markdown(f"""
        <div class="cm-answer-card">
            <div class="cm-answer-header">
                <span class="cm-answer-title">💡 Grounded Answer</span>
                {badge}
            </div>
            <div class="cm-answer-text">{answer_html}</div>
            <div class="cm-answer-meta">
                <span class="cm-meta-pill">📚 {num_ev} evidence nodes</span>
                <span class="cm-meta-pill">{modalities_str}</span>
                <span class="cm-meta-pill">🔗 1-hop expansion</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Evidence tabs ─────────────────────────────────────────────────────────
    if result.cited_evidence:
        tab_ev, tab_graph, tab_prov = st.tabs(["🔗 Supporting Evidence", "🕸️ Evidence Graph", "🔍 Provenance"])

        # ── Tab 1: Evidence Cards ─────────────────────────────────────────────
        with tab_ev:
            audio_ev  = [e for e in result.cited_evidence if e.modality == "audio"]
            frame_ev  = [e for e in result.cited_evidence if e.modality == "video_frame"]
            pdf_ev    = [e for e in result.cited_evidence if e.modality == "pdf"]
            image_ev  = [e for e in result.cited_evidence if e.modality == "image"]

            # ── Speech evidence ───────────────────────────────────────────────
            if audio_ev:
                st.markdown('<div class="cm-section-title">🎙️ Speech Evidence</div>', unsafe_allow_html=True)
                a_cols = st.columns(min(len(audio_ev), 2))
                for idx, ev in enumerate(audio_ev):
                    ts = _fmt_ts(ev.timestamp or 0)
                    meta = ev.metadata or {}
                    screen_snippet = meta.get("synchronized_screen_text", "")
                    with a_cols[idx % 2]:
                        tag_html = "".join(f'<span class="cm-tag">{e}</span>' for e in (ev.entities or [])[:6])
                        screen_html = ""
                        if screen_snippet:
                            short = screen_snippet[:200].replace("<","&lt;").replace(">","&gt;")
                            screen_html = f'<div class="cm-ev-screen">🖥️ On screen at {ts}:<br>{short}{"…" if len(screen_snippet)>200 else ""}</div>'

                        st.markdown(f"""
                        <div class="cm-ev-card">
                            <div class="cm-ev-card-header">
                                <span class="mod-pill mod-audio">🎙️ Audio</span>
                                <span class="cm-ev-ts">⏱️ {ts}</span>
                                <span class="cm-ev-src">📁 {ev.source}</span>
                            </div>
                            <div class="cm-ev-quote">{ev.content}</div>
                            {screen_html}
                            <div class="cm-ev-tags">{tag_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # ── Video frame evidence ──────────────────────────────────────────
            if frame_ev:
                st.markdown('<div class="cm-section-title">🎥 Video Frame Evidence</div>', unsafe_allow_html=True)
                f_cols = st.columns(min(len(frame_ev), 3))
                for idx, ev in enumerate(frame_ev):
                    ts = _fmt_ts(ev.timestamp or 0)
                    meta = ev.metadata or {}
                    frame_path = meta.get("frame_path", "")
                    ocr_text = meta.get("ocr_text", "")
                    audio_snippet = meta.get("synchronized_audio_transcript", "")

                    with f_cols[idx % 3]:
                        tag_html = "".join(f'<span class="cm-tag">{e}</span>' for e in (ev.entities or [])[:5])

                        # Show the frame image
                        if frame_path and os.path.exists(frame_path):
                            st.image(frame_path, use_container_width=True)

                        # Audio spoken at this timestamp
                        audio_html = ""
                        if audio_snippet:
                            short = audio_snippet[:120]
                            audio_html = f'<div class="cm-ev-quote" style="border-left-color:rgba(244,114,182,0.5);">🎙️ "{short}{"…" if len(audio_snippet)>120 else ""}"</div>'

                        # OCR snippet
                        ocr_html = ""
                        if ocr_text:
                            short_ocr = ocr_text.strip()[:180].replace("<","&lt;").replace(">","&gt;")
                            ocr_html = f'<div class="cm-ev-screen">{short_ocr}{"…" if len(ocr_text)>180 else ""}</div>'

                        st.markdown(f"""
                        <div class="cm-ev-card" style="margin-top:0.4rem;">
                            <div class="cm-ev-card-header">
                                <span class="mod-pill mod-video">🎥 Frame</span>
                                <span class="cm-ev-ts">⏱️ {ts}</span>
                                <span class="cm-ev-src">📁 {ev.source}</span>
                            </div>
                            {audio_html}
                            {ocr_html}
                            <div class="cm-ev-tags">{tag_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # ── PDF evidence ──────────────────────────────────────────────────
            if pdf_ev:
                st.markdown('<div class="cm-section-title">📄 Document Evidence</div>', unsafe_allow_html=True)
                for ev in pdf_ev:
                    tag_html = "".join(f'<span class="cm-tag">{e}</span>' for e in (ev.entities or [])[:8])
                    excerpt = ev.content[:300]
                    st.markdown(f"""
                    <div class="cm-ev-card">
                        <div class="cm-ev-card-header">
                            <span class="mod-pill mod-pdf">📄 PDF</span>
                            <span class="cm-ev-ts">Page {ev.page}</span>
                            <span class="cm-ev-src">📁 {ev.source}</span>
                        </div>
                        <div class="cm-ev-quote" style="border-left-color:rgba(250,204,21,0.5); font-style:normal; color:#D1D5DB; font-size:0.88rem;">{excerpt}{"…" if len(ev.content)>300 else ""}</div>
                        <div class="cm-ev-tags">{tag_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Image evidence ────────────────────────────────────────────────
            if image_ev:
                st.markdown('<div class="cm-section-title">🖼️ Visual Evidence</div>', unsafe_allow_html=True)
                for ev in image_ev:
                    img_path_candidates = [
                        os.path.join(root_dir, "data", ev.source),
                        os.path.join(root_dir, "data", "raw", ev.source),
                    ]
                    ocr_text = ev.metadata.get("ocr_text", "") if ev.metadata else ""
                    tag_html = "".join(f'<span class="cm-tag">{e}</span>' for e in (ev.entities or [])[:6])
                    for p in img_path_candidates:
                        if os.path.exists(p):
                            st.image(p, caption=f"📁 {ev.source}", use_container_width=True)
                            break
                    if ocr_text:
                        st.markdown(f"""
                        <div class="cm-ev-card">
                            <div class="cm-ev-card-header">
                                <span class="mod-pill mod-image">🖼️ Image</span>
                                <span class="cm-ev-src">📁 {ev.source}</span>
                            </div>
                            <div class="cm-ev-screen">{ocr_text[:250].replace("<","&lt;").replace(">","&gt;")}</div>
                            <div class="cm-ev-tags">{tag_html}</div>
                        </div>
                        """, unsafe_allow_html=True)

        # ── Tab 2: Evidence Graph ─────────────────────────────────────────────
        with tab_graph:
            st.markdown("### 🕸️ Evidence Relationship Graph")
            st.caption("How retrieved nodes are connected — the system didn't just find similar text, it followed relationships.")

            ev_by_id = {ev.id: ev for ev in result.cited_evidence}
            mod_icon = {"audio": "🎙️", "video_frame": "🎥", "pdf": "📄", "image": "🖼️"}
            mod_color = {"audio": "#818CF8", "video_frame": "#F472B6", "pdf": "#FCD34D", "image": "#34D399"}

            # Build graph as text tree
            visited = set()
            graph_lines = []
            graph_lines.append(f'<div style="color:#C084FC; font-weight:700; margin-bottom:0.5rem;">🔍 QUERY: "{st.session_state.get("last_query","")}"</div>')
            graph_lines.append('<div style="color:#475569; margin-left:1.5rem; margin-bottom:0.3rem;">│</div>')
            graph_lines.append('<div style="color:#475569; margin-left:1.5rem; margin-bottom:0.5rem;">▼  <span style="color:#64748B; font-size:0.85rem;">Retrieval + 1-hop expansion</span></div>')

            for ev in result.cited_evidence:
                if ev.id in visited:
                    continue
                visited.add(ev.id)
                icon = mod_icon.get(ev.modality, "●")
                color = mod_color.get(ev.modality, "#94A3B8")
                loc = f"@{_fmt_ts(ev.timestamp)}" if ev.timestamp is not None else f"Page {ev.page}" if ev.page else ""
                graph_lines.append(
                    f'<div class="cm-graph-node" style="margin-left:1rem;">'
                    f'├── <span style="color:{color}; font-weight:700;">{icon} {ev.id}</span>'
                    f'<span style="color:#475569;"> ({ev.source} {loc})</span></div>'
                )
                # Show connected nodes (that are in cited evidence)
                connected_in_result = [
                    rid for rid in (ev.relationships or [])
                    if rid in ev_by_id and rid not in visited
                ]
                for rid in connected_in_result[:4]:
                    rel = ev_by_id[rid]
                    rel_icon = mod_icon.get(rel.modality, "●")
                    rel_color = mod_color.get(rel.modality, "#94A3B8")
                    rel_loc = f"@{_fmt_ts(rel.timestamp)}" if rel.timestamp is not None else f"Page {rel.page}" if rel.page else ""
                    graph_lines.append(
                        f'<div class="cm-graph-node" style="margin-left:3rem;">'
                        f'└── <span style="color:{rel_color};">{rel_icon} {rel.id}</span>'
                        f'<span style="color:#334155;"> ({rel.source} {rel_loc})</span></div>'
                    )
                    visited.add(rid)

            st.markdown(
                f'<div style="background:rgba(0,0,0,0.3); border:1px solid rgba(255,255,255,0.06); '
                f'border-radius:12px; padding:1.5rem 1.5rem; line-height:1.9;">'
                + "\n".join(graph_lines)
                + "</div>",
                unsafe_allow_html=True
            )

            # Seed vs expanded distinction
            seed_ids = set(result.metadata.get("seed_ids", []))
            expanded_ids = set(result.metadata.get("expanded_ids", [])) - seed_ids
            st.markdown(f"""
            <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-top:1rem;">
                <span style="font-size:0.8rem; color:#94A3B8;">
                    🎯 <strong style="color:#818CF8;">{len(seed_ids)}</strong> direct hits (TF-IDF retrieval)
                </span>
                <span style="font-size:0.8rem; color:#94A3B8;">
                    🔗 <strong style="color:#C084FC;">{len(expanded_ids)}</strong> nodes via graph expansion
                </span>
                <span style="font-size:0.8rem; color:#94A3B8;">
                    📊 <strong style="color:#F472B6;">{len(result.cited_evidence)}</strong> total context nodes
                </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Tab 3: Provenance ─────────────────────────────────────────────────
        with tab_prov:
            st.markdown("### 🔍 Evidence Provenance")
            st.caption("Where every piece of the answer came from — exact source, modality, and location.")

            prov_rows = []
            for ev in result.cited_evidence:
                loc = _fmt_ts(ev.timestamp) if ev.timestamp is not None else (f"Page {ev.page}" if ev.page else "—")
                mod_display = {"audio": "🎙️ Audio", "video_frame": "🎥 Video Frame", "pdf": "📄 PDF", "image": "🖼️ Image"}.get(ev.modality, ev.modality)
                excerpt = ev.content[:90] + ("…" if len(ev.content) > 90 else "")
                prov_rows.append({
                    "Evidence ID": ev.id,
                    "Modality": mod_display,
                    "Source File": ev.source,
                    "Time / Page": loc,
                    "Content Excerpt": excerpt,
                    "Entities": ", ".join((ev.entities or [])[:5]),
                })

            import pandas as pd
            df = pd.DataFrame(prov_rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Evidence ID": st.column_config.TextColumn("Evidence ID", width="medium"),
                    "Modality": st.column_config.TextColumn("Modality", width="small"),
                    "Source File": st.column_config.TextColumn("Source", width="medium"),
                    "Time / Page": st.column_config.TextColumn("Time / Page", width="small"),
                    "Content Excerpt": st.column_config.TextColumn("Content", width="large"),
                    "Entities": st.column_config.TextColumn("Key Entities", width="medium"),
                }
            )

            # Copy-friendly JSON view
            with st.expander("📋 Raw evidence metadata (for debugging)"):
                prov_json = [
                    {
                        "id": ev.id, "modality": ev.modality, "source": ev.source,
                        "timestamp": ev.timestamp, "page": ev.page,
                        "entities": ev.entities, "relationships": ev.relationships[:6],
                    }
                    for ev in result.cited_evidence
                ]
                st.json(prov_json)


# -----------------------------------------------------------------------------
# Main Execution Guard
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Always run Streamlit UI when invoked via `streamlit run app.py`
    # Fall back to CLI when invoked via `python app.py`
    is_streamlit = (
        any("streamlit" in arg for arg in sys.argv)
        or "STREAMLIT_SERVER_PORT" in os.environ
    )
    if not is_streamlit:
        try:
            import inspect
            if any("streamlit" in fn.filename for fn in inspect.stack()):
                is_streamlit = True
        except Exception:
            pass

    if is_streamlit:
        run_streamlit_app()
    else:
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else GOLDEN_QUERY
        result = ask(q, verbose=True)
        print_cli_result(result)
