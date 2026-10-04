# AIC-2026-BuBuChaCha

Learning scripts and a multimedia search prototype developed with AI assistance while preparing for the Ho Chi Minh City AI Challenge 2026, Division A.

## What's here

The `learn/` directory contains 16 Python scripts covering:

- Gradient descent in Python, NumPy, and PyTorch; neural networks and CNNs on MNIST.
- CLIP image/text embeddings and FAISS similarity search.
- Video keyframe extraction, EasyOCR, Whisper transcription, YOLO detection, and ByteTrack tracking.
- Temporal search, an optional Gemini visual question-answering demo, and a Streamlit search interface.

The [Streamlit prototype](learn/ui1_app.py) supports separate CLIP, OCR, ASR, object-count, and temporal search modes. It is an experimental learning tool; retrieval quality has not been benchmarked and competition submission is not implemented.

## Setup

Run commands from the repository root. One setup option is a Python 3.11 conda environment:

```bash
conda create -n aic python=3.11
conda activate aic
python -m pip install -r requirements.txt
python verify_setup.py
```

`verify_setup.py` checks PyTorch, Transformers, CLIP model loading, and a small FAISS index. It may download CLIP weights and does not validate every demo. Whisper transcription also requires `ffmpeg` on your system.

## Try video search

Create `data/videos/` and place a video there, then extract keyframes and open the interface:

```bash
mkdir -p data/videos
python learn/keyframe1_extract.py
streamlit run learn/ui1_app.py
```

The extractor samples a frame every two seconds into `data/keyframes/`. The interface encodes these images with CLIP ViT-B/32 and builds an in-memory FAISS index. Enter an English scene description to search.

For the optional OCR, speech, or object-count modes, generate their metadata first:

```bash
python learn/ocr1_demo.py
python learn/asr1_demo.py
python learn/yolo1_detect.py
```

These scripts write JSON files under `data/`; reopen the interface after generating them. Model demos may download weights on first use. Video files, generated metadata, and model weights are excluded from Git.

The separate `learn/vqa1_demo.py` requires a `GEMINI_API_KEY` environment variable and access to its configured Gemini model to make API calls.

## Files

| Path | Purpose |
| --- | --- |
| [learn/](learn/) | Learning scripts and the search prototype |
| [learn/query_strategy.md](learn/query_strategy.md) | Notes on writing CLIP search queries |
| [requirements.txt](requirements.txt) | Python dependencies |
| [verify_setup.py](verify_setup.py) | Core environment check |
