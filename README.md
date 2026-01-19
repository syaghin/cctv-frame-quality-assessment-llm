# CCTV Frame Quality Assessment using Vision-Language Model

This project evaluates CCTV frame quality for vision analytics using a multimodal LLM (Qwen3-VL-8B).
It classifies frames into GOOD, NOT GOOD, or GLITCH based on visual usability.

## Features
- Folder-based CCTV frame evaluation
- Vision-Language Model (Qwen3-VL) inference
- Quality classification with human-readable reasoning
- Web-based review UI (FastAPI + Jinja2)
- Sanitized sample dataset for public demonstration
- Mock mode for lightweight execution without loading the full model

## Architecture
Image Folder → LLM Predictor → CSV → Web UI

## Sample Data
All sample images and CSV files are sanitized and do not represent live CCTV feeds.

## Tech Stack
- Python
- FastAPI
- Qwen3-VL-8B
- PyTorch
- OpenCV / FFmpeg

## Why LLM?
- Flexible rules without retraining
- Human-readable explanations
- Easier extension for new quality criteria
