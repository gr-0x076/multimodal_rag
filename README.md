# ContextMesh — Multimodal RAG System

ContextMesh is a multimodal Retrieval-Augmented Generation (RAG) system capable of ingesting video, audio, PDFs, and images into a unified, cross-modal Evidence graph with temporal and entity-based relationship linking.

---

## 👥 Team Responsibilities & Ownership

| Component / File | Owner | Responsibility |
|---|---|---|
| `ingestion/video.py` | Person 1 | Video processing, frame extraction, visual OCR/descriptions with timestamps |
| `ingestion/audio.py` | Person 1 | Audio extraction, speech-to-text transcript generation with timestamps |
| `ingestion/pdf.py` | Person 2 (You) | Multi-page PDF text extraction and evidence structuring |
| `ingestion/image.py` | Person 2 (You) | Image OCR and visual evidence extraction |
| `knowledge/schema.py` | Person 2 / Shared | Core `Evidence` schema dataclass contract |
| `knowledge/relationships.py` | Person 3 | Cross-modal and temporal relationship linking (entity, timestamp, source) |
| `retrieval/search.py` | Person 3 | Semantic search over Chroma vector database |
| `llm/groq_client.py` | Person 3 | Groq LLM integration for grounded multimodal answer generation |
| `app.py` | Person 3 / Team | Interactive Streamlit user interface |

---

## 📜 Team Integration Contract

All ingestion modules **must** produce and return `Evidence` objects defined in `knowledge/schema.py`.

### Evidence Schema:
```python
from knowledge.schema import Evidence

item = Evidence(
    id="unique_evidence_id",
    content="Extracted text or description...",
    modality="pdf" | "image" | "audio" | "video_frame",
    source="filename.ext",
    timestamp=632.5,  # float in seconds (for audio/video) or None
    page=3,           # int page number (for PDFs) or None
    entities=["Redis", "Database"],
    confidence=0.95,
    relationships=[]
)
```

### Integration Guidelines:
1. **Ingestion Isolation**: Ingestion modules do not directly call the retrieval or LLM layers. They strictly extract and structure data into `Evidence` objects.
2. **Provenance Preservation**: Retain original sources, timestamps, and page numbers so retrieved answers can link directly back to exact multimodal evidence.
3. **Branch Workflow**:
   - `main`: Production-ready integrated code.
   - `person1-video`: Video/Audio ingestion branch.
   - `person2-documents`: PDF & Image ingestion branch.
   - `person3-retrieval`: Vector search, relationship builder, Groq LLM & Streamlit branch.

---

## 🚀 Quick Start (Local Setup)

1. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Ingestion Tests**:
   ```bash
   python test_ingestion.py
   ```
