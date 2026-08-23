# 🧠 ContextMesh — Multimodal Video & Document RAG

ContextMesh is a multimodal Retrieval-Augmented Generation (RAG) system that connects **speech dialogue (Whisper audio transcription)**, **on-screen text (video keyframe OCR)**, **PDF document specifications**, and **visual diagrams** into a unified, cross-modal Evidence Graph with temporal and entity relationships.

---

## 🎬 Live Hackathon Demo — Quick Start with Your Own Video

Follow these **5 simple steps** to test ContextMesh on any fresh MP4 video:

### STEP 1: Clone & Setup Environment
```bash
git clone https://github.com/gr-0x076/multimodal_rag.git
cd multimodal_rag

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### STEP 2: Configure API Key (Optional for Live Groq LLM)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to add your Groq API key (get a free key at [console.groq.com](https://console.groq.com/keys)):
```env
GROQ_API_KEY=gsk_your_groq_key_here
```
*(Note: If no API key is provided, ContextMesh automatically switches to its deterministic evidence-driven synthesizer with zero hallucination!)*

---

### STEP 3: Reset Previous Data (Clean State)
Clear any previously generated evidence graph, extracted frames, or transcript files:
```bash
python pipeline/reset.py
```

---

### STEP 4: Add Your Video & Run Ingestion
Copy your fresh MP4 video file into the `test_data/` folder:
```bash
# Example:
cp /path/to/your_video.mp4 test_data/my_video.mp4

# Run the ingestion pipeline:
python pipeline/ingest.py
```
The ingestion pipeline automatically:
1. **Extracts Audio** & transcribes speech with exact timestamps using OpenAI Whisper
2. **Extracts Keyframes** at 5-second intervals and runs Tesseract OCR on screen content
3. **Ingests Documents** (PDFs & Images) if present in `test_data/`
4. **Builds Relationship Graph** linking spoken audio, on-screen text, and document pages
5. **Saves Unified Graph** to `data/processed/evidence.json`

---

### STEP 5: Launch the Interactive UI
Launch the Multimodal Evidence Explorer Streamlit web application:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to:
- Ask natural-language questions about your video
- See grounded LLM synthesis from **Live Groq**
- Inspect supporting **Speech Transcript Cards** with exact timestamps
- View actual **Video Frame Thumbnails** and on-screen code/text
- Explore the **Interactive Relationship Graph** and **Provenance Table**

---

## 🛠️ Repository Layout

```text
multimodal_rag/
├── app.py                      # Interactive Streamlit Web UI & Grounded Query Engine
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── README.md                   # Project documentation
│
├── test_data/                  # Input folder for fresh videos & documents
│   └── README.md
├── data/
│   ├── raw/                    # Alternative raw input folder
│   └── processed/              # Generated evidence graph, transcripts & frame images
│
├── pipeline/
│   ├── ingest.py               # Unified multimodal ingestion pipeline orchestrator
│   ├── reset.py                # Data reset utility script
│   ├── audio.py                # Audio extraction module wrapper
│   └── video.py                # Video extraction module wrapper
│
├── ingestion/                  # Modality extractors
│   ├── audio.py                # Whisper speech-to-text extractor & entity filter
│   ├── video.py                # OpenCV keyframe extractor
│   ├── image.py                # Tesseract OCR image extractor
│   └── pdf.py                  # PyPDF2 document extractor
│
├── knowledge/
│   ├── schema.py               # Evidence & GroundedAnswer schema definitions
│   └── relationships.py        # 1-hop relationship graph traversal engine
│
├── retrieval/
│   ├── search.py               # TF-IDF direct evidence retrieval engine
│   └── grounding.py            # Prompt construction & provenance formatter
│
├── llm/
│   └── groq_client.py          # Groq API client with evidence-driven fallback
│
└── tests/                      # Automated unit, evaluation, and end-to-end test suites
    ├── test_end_to_end.py      # End-to-end integration tests
    ├── test_evaluation_contract.py  # Evaluation & baseline comparison tests
    └── test_retrieval_grounding.py # Retrieval unit tests
```

---

## 🧪 Running Automated Tests

Run the full test suite across unit, contract, and integration tests:
```bash
python -m unittest discover tests
```
Or run individual test scripts:
```bash
python tests/test_end_to_end.py
python tests/test_evaluation_contract.py
python tests/test_retrieval_grounding.py
```

---

## 👥 Team & Component Architecture

| Component | Responsibility | Primary Module |
|---|---|---|
| **Video & Audio Ingestion** | Whisper speech transcription, OpenCV keyframes, Tesseract OCR | `ingestion/audio.py`, `ingestion/video.py` |
| **Document Ingestion** | Multi-page PDF parsing, standalone diagram OCR | `ingestion/pdf.py`, `ingestion/image.py` |
| **Knowledge Graph** | Unified `Evidence` schema, temporal & cross-modal edges | `knowledge/schema.py`, `knowledge/relationships.py` |
| **Multimodal Retrieval** | Direct TF-IDF search + 1-hop graph expansion | `retrieval/search.py`, `retrieval/grounding.py` |
| **Grounded LLM** | Groq LLaMA-3.3 / GPT-OSS 120B synthesis & fallback | `llm/groq_client.py` |
| **Interactive UI** | Multimodal Evidence Explorer web application | `app.py` |
