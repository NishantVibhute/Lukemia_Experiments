"""Stage 3 - handcrafted morphological, texture and colour features
(synopsis Algorithm 3; deep features are added in `deep.py`)."""
import cv2
import numpy as np
from skimage import measure
from skimage.feature import graycomatrix, graycoprops

MORPH = ["area", "perimeter", "circularity", "eccentricity", "solidity", "nc_ratio"]
TEXTURE = ["glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity", "entropy"]
COLOUR = ["gray_mean", "gray_std", "hue_mean", "hue_std", "sat_mean", "sat_std", "val_mean", "val_std"]
NAMES = MORPH + TEXTURE + COLOUR

_LEVELS = 32


def _glcm(gray, mask):
    """GLCM statistics restricted to the masked region: level 0 is reserved for
    pixels outside the mask and its row/column are dropped."""
    q = (gray.astype(np.float32) / 256.0 * _LEVELS).astype(np.uint8) + 1
    q[~mask] = 0
    glcm = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=_LEVELS + 1, symmetric=True).astype(np.float64)
    glcm = glcm[1:, 1:]
    total = glcm.sum(axis=(0, 1), keepdims=True)
    total[total == 0] = 1
    glcm /= total
    return [float(np.nan_to_num(graycoprops(glcm, p)).mean())
            for p in ("contrast", "correlation", "energy", "homogeneity")]


def extract(img, labels, cell_labels, scale=1.0):
    """One feature row per nucleus instance.

    `scale` is the factor the image was downscaled by, so size features stay in
    original-image pixels. Returns (rows, regions).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cell_area = np.bincount(cell_labels.ravel(), minlength=labels.max() + 1)
    rows, regions = [], []
    for r in measure.regionprops(labels):
        y0, x0, y1, x1 = r.bbox
        mask = r.image
        g = gray[y0:y1, x0:x1]
        px = g[mask]
        perimeter = max(r.perimeter, 1.0)
        p = np.bincount(px, minlength=256) / px.size
        p = p[p > 0]
        h, s, v = (hsv[y0:y1, x0:x1, c][mask] for c in range(3))
        row = [
            r.area / scale ** 2,
            perimeter / scale,
            min(1.0, 4 * np.pi * r.area / perimeter ** 2),
            r.eccentricity,
            r.solidity,
            r.area / max(cell_area[r.label], r.area),  # nucleus share of the cell, 0-1
            *_glcm(g, mask),
            float(-(p * np.log2(p)).sum()),
            px.mean(), px.std(),
            h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std(),
        ]
        rows.append([float(x) for x in row])
        regions.append(r)
    return np.array(rows, dtype=np.float32).reshape(-1, len(NAMES)), regions


def crop(img, region, pad=0.35, size=None):
    """Square-ish crop around an instance, with context padding."""
    y0, x0, y1, x1 = region.bbox
    p = int(max(y1 - y0, x1 - x0) * pad)
    h, w = img.shape[:2]
    out = img[max(0, y0 - p):min(h, y1 + p), max(0, x0 - p):min(w, x1 + p)]
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA) if size else out
