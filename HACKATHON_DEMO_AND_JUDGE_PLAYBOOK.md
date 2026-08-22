# 🏆 ContextMesh — Hackathon Demo & Judge Presentation Playbook

> **Tagline:** Cross-Modal & Temporal Evidence Retrieval over Video, Audio, PDFs, and Diagrams with Provable Grounding.

---

## ⏱️ The 3-Minute Winning Demo Script

### 0:00 – 0:30 | The Problem Hook (The Enterprise Blindspot)
> *"Judges, in every engineering organization, 80% of critical knowledge is fragmented across meeting recordings, architecture diagrams, and slide decks. Traditional RAG systems search text documents in isolation—if an engineer speaks about an architecture change in a meeting at 10 seconds while referencing a diagram, text-only RAG completely misses the visual and temporal context. Today, we present **ContextMesh**—a multimodal retrieval engine that treats spoken audio, keyframes, PDF specifications, and diagrams as a connected evidence graph."*

### 0:30 – 1:30 | Hero Demo 1: Cross-Modal Knowledge Synthesis
- **Action:** Click **"🎯 Demo 1: Golden Redis Query"** in the Streamlit UI or type:
  `"What architecture was proposed to reduce database load, and what visual/document evidence supports it?"`
- **What Appears:**
  1. Grounded synthesis citing Redis in-memory caching layer.
  2. Live Engine badge (`⚡ Live Groq LLaMA-3.3-70B` or `🛡️ Grounded Engine`).
  3. **Left Column:** 🎤 Spoken audio transcript (`meeting.mp4 @ 00:10`) + 📄 PDF document excerpt (`architecture.pdf Page 2`).
  4. **Right Column:** 🎥 Video keyframe showing the exact architecture slide (`meeting.mp4 @ 00:10`).
- **Speaking Script:**
  > *"Notice what happened in sub-second time: ContextMesh did not just find the word 'Redis'. It retrieved the direct spoken sentence from the audio, traversed our 1-hop relationship graph, pulled the synchronized video keyframe from the meeting at 10 seconds, and connected it to the formal caching specification in page 2 of our architecture PDF. All citations have exact timestamps and page provenance."*

### 1:30 – 2:10 | Demo 2 & Demo 3: Contrast & Refusal Behavior
- **Action 1:** Click **"⚡ Demo 2: Peak Bottleneck"**
  - Synthesizes database latency bottleneck from `meeting.mp4 @ 00:05` audio + load metric frame.
- **Action 2:** Click **"🛑 Demo 3: Unsupported Refusal"** (`"What machine-learning algorithm was used to train Redis caching?"`)
  - The system outputs: *"The available evidence is insufficient to fully answer this question."*
- **Speaking Script:**
  > *"Grounding is not just about finding answers—it's about knowing what NOT to answer. When we ask a hallucination-inducing query about non-existent ML training for Redis, ContextMesh strictly refuses to hallucinate, enforcing our deterministic grounding contract."*

### 2:10 – 3:00 | Architecture, Groq Speed & Vision
- **Action:** Switch to the **"🕸️ Cross-Modal Relationship Graph"** tab.
- **Speaking Script:**
  > *"Under the hood, ContextMesh is built on a clean 4-tier pipeline: Unified Ingestion $\rightarrow$ Query Token & Entity Retrieval $\rightarrow$ Graph Relationship Expansion $\rightarrow$ Groq LPU Synthesis using LLaMA-3.3-70B. We have 12 automated unit and integration tests verifying schema integrity and retrieval contracts. ContextMesh turns raw organizational media into verified, citation-backed truth."*

---

## 🎯 The 3 Live Demo Scenarios

| Scenario | Input Query | Supporting Modalities | Key Judge Takeaway |
| :--- | :--- | :--- | :--- |
| **1. Golden Cross-Modal Query** | `"What architecture was proposed to reduce database load, and what visual/document evidence supports it?"` | 🎤 Audio (`00:10`)<br>🎥 Video Frame (`00:10`)<br>📄 PDF (`Page 2`) | Demonstrates cross-modal entity and temporal linking between spoken speech and visual slides. |
| **2. Peak Bottleneck Query** | `"What was the primary bottleneck during peak traffic?"` | 🎤 Audio (`00:05`)<br>🎥 Video Frame (`00:05`) | Demonstrates temporal alignment between metric discussion and load graph visuals. |
| **3. Negative Refusal Query** | `"What machine-learning algorithm was used to train Redis caching?"` | Cross-modal context evaluated | Demonstrates refusal to hallucinate when evidence is missing. |

---

## 🏛️ Architecture & Technical Differentiation

```
  ┌────────────────────────────────────────────────────────┐
  │                 Unified Ingestion                      │
  │   Video (OpenCV) │ Audio (Whisper) │ PDF │ Images      │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │         Canonical Evidence Schema (evidence.json)      │
  │     ID • Modality • Provenance • Entities • Edges      │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │                Dual-Stage Retrieval                    │
  │   Stage 1: Direct Seed Search (Content + Entity Score) │
  │   Stage 2: 1-Hop Graph BFS Traversal (Temporal/Entity) │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │           Groq LPU Grounded Synthesis Engine           │
  │    Strict System Grounding Prompt + Provenance Check   │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │       Streamlit UI / CLI Output (GroundedAnswer)       │
  └────────────────────────────────────────────────────────┘
```

### Why ContextMesh vs. Flat Text-Only RAG?
1. **Multimodal Alignment**: Flat RAG chunks text into isolated vector buckets. ContextMesh preserves the temporal relation between speech timestamps and video frames ($\Delta t \approx 0$).
2. **Graph Expansion**: Direct keyword/vector search often returns 1 modality. 1-hop relationship expansion pulls associated visual diagrams and technical specs without needing huge LLM context windows.
3. **Provable Provenance**: Every claim includes `source`, `timestamp` (MM:SS), and `page`, preventing synthetic provenance fabrication.

---

## ⚡ Why Groq LPU Inference?
- **Speed**: Multimodal retrieval pipelines require low-latency LLM synthesis to feel interactive. Groq's LPU provides 300+ tokens/sec on LLaMA-3.3-70B.
- **Resilience**: The system features a built-in deterministic fallback engine that guarantees complete functionality even in offline or air-gapped environments.

---

## 🛡️ Judge Q&A Defense Playbook

### Q1: "How do you avoid hallucinations when synthesizing across modalities?"
> **Answer:** *"We enforce grounding at two levels: First, prompt-level strictness instructing the model to synthesize ONLY supplied context blocks. Second, application-level refusal: if retrieved seeds have zero entity overlap or if the user asks for unsupported attributes, the pipeline returns a standard refusal rather than generating unverified text."*

### Q2: "Why didn't you use a heavy Vector Database like Milvus or Pinecone?"
> **Answer:** *"For multimodal engineering intelligence, the primary bottleneck is not brute-force vector cosine similarity—it is representing and connecting cross-modal entities across time and space. Our lightweight canonical index with graph adjacency expansion solves the entity alignment problem deterministically with sub-millisecond overhead and zero external infrastructure dependencies."*

### Q3: "How does the system scale to thousands of hours of video?"
> **Answer:** *"Our Evidence schema is decoupled from storage. The ingestion pipeline can stream frames and transcripts into hierarchical time-indexed clusters (e.g., HNSW + Graph Adjacency index). Ingestion is parallelized per video chunk, and query-time BFS is $O(k + \text{deg}(v))$, which scales linearly."*

### Q4: "What happens if the network drops or the Groq API key is rate-limited?"
> **Answer:** *"ContextMesh has built-in offline failover. If the Groq API is unavailable, the system automatically runs the deterministic grounding engine, ensuring the live demo never crashes or leaves the user hanging."*

---

## 🚀 Quick Commands Checklist

```bash
# 1. Run all 12 automated unit & contract tests:
.venv\Scripts\python.exe -m pytest tests/ -v

# 2. Run CLI Golden Query:
.venv\Scripts\python.exe app.py

# 3. Launch Streamlit Web UI:
.venv\Scripts\streamlit.exe run app.py
```
