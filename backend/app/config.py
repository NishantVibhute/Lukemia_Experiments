"""Paths and user settings (persisted to backend/settings.json)."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
SETTINGS_FILE = BASE / "settings.json"
MODELS.mkdir(exist_ok=True)

DEFAULTS = {
    # Folder that contains ALL_IDB1 and ALL_IDB2 (searched recursively).
    "dataset_root": str(BASE.parent / "data"),
    # Optional: ground-truth masks for ALL-IDB1, same file stem as the image.
    "masks_dir": "",
    # Optional: folder-per-class dataset (e.g. Normal/, L1/, L2/, L3/) for subtype work.
    "class_dir": "",
}


def load() -> dict:
    settings = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        settings.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    return settings


def save(update: dict) -> dict:
    settings = load()
    settings.update({k: v for k, v in update.items() if k in DEFAULTS and v is not None})
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings
