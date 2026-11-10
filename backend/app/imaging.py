"""Image I/O helpers. All images are BGR uint8 (OpenCV convention)."""
import base64
from pathlib import Path

import cv2
import numpy as np


def read_image(path) -> np.ndarray:
    # np.fromfile + imdecode handles Windows paths with spaces/unicode.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {Path(path).name}")
    return img


def decode_bytes(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image file")
    return img


def encode(img: np.ndarray, ext: str = ".jpg") -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError("Image encoding failed")
    return buf.tobytes()


def data_url(img: np.ndarray, ext: str = ".jpg") -> str:
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(encode(img, ext)).decode()}"


def resize_max(img: np.ndarray, max_side: int):
    """Downscale so the longest side is at most max_side. Returns (image, scale)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale
