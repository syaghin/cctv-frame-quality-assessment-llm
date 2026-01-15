# FastAPI entry point
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from pathlib import Path
import csv
import os

app = FastAPI()

# ==========================
# CONFIG
# ==========================
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = Path(
    os.getenv("PREDICTION_CSV", BASE_DIR / "data/sample_predictions.csv")
)
IMAGE_DIR = Path(
    os.getenv("IMAGE_DIR", BASE_DIR / "data/sample_images")
)

templates = Jinja2Templates(directory=BASE_DIR / "app/templates")

# ==========================
# Helpers
# ==========================
def load_results():
    results = []
    with CSV_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def last_update_str():
    if not CSV_FILE.exists():
        return "N/A"

    ts = CSV_FILE.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
# ==========================
# HTML PAGE
# ==========================
@app.get("/", response_class=HTMLResponse)
def review_page(request: Request):
    data = load_results()
    items = []

    for row in data:
        decision = (row.get("decision") or "").upper()

        if decision in {"NOT GOOD", "GLITCH"} and row.get("status") == "OK":
            filename = Path(row["file"]).name
            name = filename.replace(".jpg", "").rsplit("_", 1)[0]

            items.append({
                "nama": name,
                "filename": filename,
                "reason": row.get("reason"),
                "decision": decision,
            })

    return templates.TemplateResponse(
        "quality_review.html",
        {
            "request": request,
            "items": items,
            "last_update": last_update_str()
        }
    )

# ==========================
# Serve image
# ==========================
@app.get("/api/gambar/{filename}")
def get_image(filename: str):
    image_path = IMAGE_DIR / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)
