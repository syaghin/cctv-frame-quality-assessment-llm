#!/usr/bin/env python3
import os
import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from PIL import Image
import torch

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent  
INPUT_DIR = BASE_DIR / "data" / "sample_images"
PRED_OUTPUT_CSV = BASE_DIR / "data" / "sample_predictions.csv"
SCRAP_ISSUES_CSV = BASE_DIR / "data" / "scrap_issues_sample.csv"

# Tuning
REAL_MODE = os.getenv("REAL_MODE", "0") == "1"  # set to "1" to load Qwen model
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

PROMPT_TEXT = (
    "Evaluate the quality of this CCTV frame.\n\n"
    "Important constraints:\n"
    "1) The analytics system counts not only vehicles (car, motorcycle, bus) "
    "but also people and small vendors (PKL). Do NOT mark a frame as NOT GOOD "
    "simply because few or no vehicles are present (e.g., a park or pedestrian-only scene is acceptable).\n\n"
    "2) Only judge the CAMERA/FRAME_QUALITY for analytics viewability, including:\n"
    "   - sharpness and focus\n"
    "   - obstructions (trees, poles, vendors blocking the view)\n"
    "   - lighting and exposure (dark but still visible is OK)\n"
    "   - color fidelity (colors should be reasonable for detection)\n"
    "   - noise / compression artifacts\n\n"
    "3) If the frame is corrupted in a way that makes the image unusable for analytics "
    "(e.g., severe vertical/horizontal banding, scan lines, heavy compression artifacts, "
    "striping, tearing, or other corruption), label it as GLITCH and give a short reason.\n\n"
    "Do NOT judge whether vehicles exist — judge whether the frame SHOWS a clear, usable view "
    "for counting objects (vehicles, people, PKL) and performing analytics.\n\n"
    "Answer in this format:\n"
    "Decision: GOOD / NOT GOOD/ GLITCH\n"
    "Reason: short explanation"
)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------- Helpers ----------------
def resize_max_side(img: Image.Image, max_size: int = 850) -> Image.Image:
    w, h = img.size
    max_side = max(w, h)
    if max_side <= max_size:
        return img
    scale = max_size / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.BILINEAR)

def load_images(paths):
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            img = resize_max_side(img, max_size=850)
            images.append(img)
        except Exception as e:
            logging.debug(f"[PREDICT] Failed to load image {p}: {e}")
            images.append(None)
    return images

def build_messages(images):
    msgs = []
    for img in images:
        if img is None:
            msgs.append(None)
        else:
            msgs.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": PROMPT_TEXT},
                ],
            }])
    return msgs

def run_batch(model, processor, messages):
    """
    Run a batch through model+processor and return decoded strings.
    """
    valid_idx = [i for i, m in enumerate(messages) if m is not None]
    if not valid_idx:
        return [None] * len(messages)
    compact = [messages[i] for i in valid_idx]
    inputs = processor.apply_chat_template(
        compact,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [ out[len(inp):] for inp, out in zip(inputs.input_ids, outputs) ]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    results = [None] * len(messages)
    for i, text in zip(valid_idx, decoded):
        results[i] = text
    return results

# ---------------- Main predictor ----------------
def run_predictor():
    # find image files
    files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    logging.info(f"[PREDICT] Found {len(files)} frames in {INPUT_DIR}")

    if not files:
        logging.info("[PREDICT] No frames found. Exiting predictor.")
        return

    # load model & processor if REAL_MODE
    model = None
    processor = None
    if REAL_MODE:
        logging.info("[PREDICT] Loading model (REAL_MODE)...")
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype="auto", device_map="auto")
        except Exception as e:
            logging.warning(f"[PREDICT] Model.load with device_map='auto' failed: {e}. Trying without device_map.")
            model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_NAME)
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        try:
            processor.tokenizer.padding_side = "left"
        except Exception:
            pass
        logging.info("[PREDICT] Model and processor loaded")
    else:
        logging.info("[PREDICT] MOCK mode (no model loaded). Results will be synthetic/sample.")

    rows = []
    for i in tqdm(range(0, len(files), BATCH_SIZE), desc="Predicting"):
        batch_files = files[i:i + BATCH_SIZE]
        images = load_images(batch_files)
        messages = build_messages(images)

        if REAL_MODE:
            try:
                outputs = run_batch(model, processor, messages)
            except RuntimeError as e:
                logging.warning(f"[PREDICT] Batch {i} failed: {e}")
                outputs = [None] * len(batch_files)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        else:
            # MOCK outputs: simple deterministic sample (you can customize)
            outputs = []
            for img in images:
                if img is None:
                    outputs.append(None)
                else:
                    # A naive "mock" decision: check brightness -> GOOD/NOT GOOD
                    try:
                        avg = sum(img.convert("L").getdata()) / (img.width * img.height)
                        if avg < 40:
                            outputs.append("Decision: NOT GOOD\nReason: Image too dark for reliable analytics")
                        else:
                            outputs.append("Decision: GOOD\nReason: Sample auto label")
                    except Exception:
                        outputs.append("Decision: GOOD\nReason: Sample auto label")

        for f, out in zip(batch_files, outputs):
            decision = ""
            reason = ""
            status = "OK"
            if out is None:
                status = "FAILED"
            else:
                for line in out.splitlines():
                    if line.lower().startswith("decision:"):
                        decision = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("reason:"):
                        reason = line.split(":", 1)[1].strip()
                if not decision and not reason:
                    reason = out.strip()
            rows.append({
                "file": Path(f).name if isinstance(f, (str, Path)) else str(f),
                "decision": decision,
                "reason": reason,
                "status": status,
            })

        time.sleep(0.3)

    # atomic write CSV
    PRED_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = PRED_OUTPUT_CSV.with_suffix(".tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "decision", "reason", "status"])
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(PRED_OUTPUT_CSV)
    logging.info(f"[PREDICT] Results saved to {PRED_OUTPUT_CSV}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    start = datetime.now()
    logging.info("Starting predictor (folder mode)...")
    logging.info("REAL_MODE=%s", REAL_MODE)
    run_predictor()
    elapsed = datetime.now() - start
    logging.info("Done. Elapsed: %s", elapsed)
