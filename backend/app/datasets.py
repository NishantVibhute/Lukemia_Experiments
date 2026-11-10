"""Locating ALL-IDB1 / ALL-IDB2 (and optional masks / subtype folders) on disk.

Expected, but not required, layout under the dataset root:
    ALL_IDB1/im/Im001_1.jpg     ALL_IDB1/xyc/Im001_1.xyc
    ALL_IDB2/img/Im001_1.tif
The trailing _1 / _0 in a file name is the ALL-IDB label (1 = lymphoblast).
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .imaging import read_image

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CLASSES = ["Normal", "Lymphoblast"]


def _key(part: str) -> str:
    return part.upper().replace("_", "").replace("-", "").replace(" ", "")


@lru_cache(maxsize=4)
def _scan(root: str):
    base = Path(root)
    found = {"idb1": [], "idb2": [], "xyc": {}}
    if not root or not base.is_dir():
        return found
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        parts = [_key(x) for x in p.relative_to(base).parts[:-1]] + [_key(base.name)]
        ext = p.suffix.lower()
        if ext == ".xyc":
            found["xyc"][p.stem] = p
        elif ext in IMG_EXT:
            if any("IDB1" in x for x in parts):
                found["idb1"].append(p)
            elif any("IDB2" in x for x in parts):
                found["idb2"].append(p)
    found["idb1"].sort()
    found["idb2"].sort()
    return found


def rescan():
    _scan.cache_clear()


def images(settings, subset):
    return _scan(settings["dataset_root"])[subset]


def find(settings, subset, name):
    for p in images(settings, subset):
        if p.name == name:
            return p
    raise FileNotFoundError(name)


def label_of(path: Path):
    """ALL-IDB label from the file name, or None when it carries none."""
    tail = path.stem.rsplit("_", 1)[-1]
    return int(tail) if tail in ("0", "1") else None


def centroids(settings, path: Path):
    """Blast centroids annotated in the matching ALL-IDB1 .xyc file."""
    xyc = _scan(settings["dataset_root"])["xyc"].get(path.stem)
    if xyc is None:
        return None
    pts = []
    for line in xyc.read_text(errors="ignore").splitlines():
        nums = line.replace(",", " ").split()
        try:
            pts.append((float(nums[0]), float(nums[1])))
        except (IndexError, ValueError):
            continue
    return pts


def truth_mask(settings, path: Path, shape):
    """Ground-truth nucleus mask for an image, if a masks folder is configured."""
    folder = Path(settings.get("masks_dir") or "")
    if not settings.get("masks_dir") or not folder.is_dir():
        return None
    for ext in (".png", ".tif", ".tiff", ".bmp", ".jpg"):
        candidate = folder / (path.stem + ext)
        if candidate.exists():
            mask = cv2.imdecode(np.fromfile(str(candidate), np.uint8), cv2.IMREAD_UNCHANGED)
            if mask.ndim == 3:
                mask = mask.max(axis=2)
            if mask.shape != tuple(shape):
                mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            return mask
    return None


def classification_samples(settings):
    """(paths, labels, class_names, source) for classifier training.

    A folder-per-class directory takes priority so subtype datasets (L1/L2/L3,
    B-ALL/T-ALL) can be used; otherwise ALL-IDB2's binary labels are used.
    """
    class_dir = Path(settings.get("class_dir") or "")
    if settings.get("class_dir") and class_dir.is_dir():
        names = sorted(d.name for d in class_dir.iterdir() if d.is_dir())
        paths, labels = [], []
        for i, name in enumerate(names):
            for p in sorted((class_dir / name).rglob("*")):
                if p.suffix.lower() in IMG_EXT:
                    paths.append(p)
                    labels.append(i)
        if len(names) >= 2 and paths:
            return paths, labels, names, f"class folders in {class_dir}"
    paths, labels = [], []
    for p in images(settings, "idb2"):
        y = label_of(p)
        if y is not None:
            paths.append(p)
            labels.append(y)
    return paths, labels, CLASSES, "ALL-IDB2"


def summary(settings):
    scan = _scan(settings["dataset_root"])
    paths, labels, names, source = classification_samples(settings)
    masks = 0
    if settings.get("masks_dir") and Path(settings["masks_dir"]).is_dir():
        stems = {p.stem for p in Path(settings["masks_dir"]).iterdir()}
        masks = sum(1 for p in scan["idb1"] if p.stem in stems)
    return {
        "root_exists": Path(settings["dataset_root"]).is_dir(),
        "idb1": len(scan["idb1"]),
        "idb1_with_centroids": sum(1 for p in scan["idb1"] if p.stem in scan["xyc"]),
        "idb1_with_masks": masks,
        "idb2": len(scan["idb2"]),
        "classification": {
            "source": source,
            "classes": names,
            "counts": [labels.count(i) for i in range(len(names))],
        },
    }


def load(path):
    return read_image(path)
