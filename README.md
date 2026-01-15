# CCTV Frame Quality Assessment using Vision-Language Model

This project evaluates CCTV frame quality for vision analytics using a multimodal LLM (Qwen3-VL-8B).
It classifies frames into GOOD, NOT GOOD, or GLITCH based on visual usability.

## Features
- RTSP frame scraping with retry & timeout handling
- Vision-Language Model (Qwen3-VL) inference
- Quality classification with reasoning
- Web-based review UI (FastAPI + Jinja2)
- Sanitized sample dataset for public demonstration

## Architecture
CCTV Frame → LLM Predictor → CSV → Web UI

## Sample Data
All sample images and CSV files are sanitized and do not represent real locations or live CCTV feeds.

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

## Disclaimer
This repository contains no production credentials or internal datasets.
