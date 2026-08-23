# ContextMesh — Input Video Directory (`test_data/`)

Place any fresh MP4 video files into this directory for ingestion into ContextMesh.

### How to use with a fresh video:

1. Copy your MP4 file into this folder:
   ```bash
   cp /path/to/your_video.mp4 test_data/my_video.mp4
   ```

2. (Optional) Clear previous generated evidence:
   ```bash
   python pipeline/reset.py
   ```

3. Run ingestion:
   ```bash
   python pipeline/ingest.py
   ```
   The pipeline will automatically:
   - Extract audio and transcribe speech using Whisper with timestamps
   - Extract video keyframes at 5-second intervals
   - Perform OCR text extraction on screen content
   - Build a cross-modal relationship graph
   - Save the unified graph to `data/processed/evidence.json`

4. Launch the application:
   ```bash
   streamlit run app.py
   ```
