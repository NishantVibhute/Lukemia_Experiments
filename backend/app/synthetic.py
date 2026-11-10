"""Synthetic blood-smear images with exact ground truth.

These are drawn shapes, not microscopy. They exist to exercise the pipeline
and the experiment code when no dataset is configured (and in the tests);
numbers measured on them say nothing about clinical performance.
"""
import cv2
import numpy as np

# BGR
_PLASMA = (205, 212, 222)
_RBC = (165, 150, 205)
_RBC_CORE = (190, 185, 220)
_CYTO = (205, 165, 170)
_NUCLEUS = (140, 45, 95)


def _texture(img, rng, sigma):
    noise = cv2.GaussianBlur(rng.normal(0, sigma, img.shape).astype(np.float32), (0, 0), 1.2)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _nucleus(img, labels, idx, centre, radius, rng):
    axes = (int(radius * rng.uniform(0.85, 1.1)), int(radius * rng.uniform(0.85, 1.1)))
    angle = float(rng.uniform(0, 180))
    cv2.ellipse(img, centre, axes, angle, 0, 360, _NUCLEUS, -1, cv2.LINE_AA)
    cv2.ellipse(labels, centre, axes, angle, 0, 360, int(idx), -1)


def smear(size=(600, 800), n_cells=8, n_overlap_pairs=3, n_rbc=60, seed=0):
    """Returns (image, nucleus_labels, centroids). Overlap pairs are two nuclei
    whose outlines intersect - the case the thesis targets."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.full((h, w, 3), _PLASMA, np.uint8)
    for _ in range(n_rbc):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        r = int(rng.integers(16, 22))
        cv2.circle(img, c, r, _RBC, -1, cv2.LINE_AA)
        cv2.circle(img, c, r // 3, _RBC_CORE, -1, cv2.LINE_AA)
    labels = np.zeros((h, w), np.int32)
    centres, idx = [], 0

    def place(centre, radius):
        nonlocal idx
        idx += 1
        _nucleus(img, labels, idx, centre, radius, rng)
        centres.append((float(centre[0]), float(centre[1])))

    margin = 70
    spots = []
    while len(spots) < n_cells + n_overlap_pairs:
        c = (int(rng.integers(margin, w - margin)), int(rng.integers(margin, h - margin)))
        if all((c[0] - s[0]) ** 2 + (c[1] - s[1]) ** 2 > 130 ** 2 for s in spots):
            spots.append(c)
    for i, c in enumerate(spots):
        r = int(rng.integers(24, 32))
        pair = i < n_overlap_pairs
        second = None
        if pair:
            a = rng.uniform(0, 2 * np.pi)
            second = (int(c[0] + np.cos(a) * r * 1.5), int(c[1] + np.sin(a) * r * 1.5))
        for centre in filter(None, (c, second)):
            cv2.circle(img, centre, int(r * 1.35), _CYTO, -1, cv2.LINE_AA)
        place(c, r)
        if second:
            place(second, r)
    img = _texture(cv2.GaussianBlur(img, (0, 0), 0.8), rng, 6)
    return img, labels, centres


def cell(blast, size=257, seed=0):
    """Single-cell crop in the style of ALL-IDB2. Blasts get a larger nucleus
    that nearly fills the cell; normal lymphocytes a smaller, rounder one."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), _PLASMA, np.uint8)
    for _ in range(6):
        c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        if abs(c[0] - size // 2) + abs(c[1] - size // 2) > 110:
            cv2.circle(img, c, int(rng.integers(18, 24)), _RBC, -1, cv2.LINE_AA)
    centre = (size // 2 + int(rng.integers(-8, 9)), size // 2 + int(rng.integers(-8, 9)))
    if blast:
        cyto, nuc = int(rng.integers(52, 62)), None
        nuc = int(cyto * rng.uniform(0.82, 0.92))
        axes = (nuc, int(nuc * rng.uniform(0.75, 0.95)))
    else:
        cyto = int(rng.integers(40, 50))
        nuc = int(cyto * rng.uniform(0.6, 0.72))
        axes = (nuc, int(nuc * rng.uniform(0.92, 1.0)))
    cv2.circle(img, centre, cyto, _CYTO, -1, cv2.LINE_AA)
    cv2.ellipse(img, centre, axes, float(rng.uniform(0, 180)), 0, 360, _NUCLEUS, -1, cv2.LINE_AA)
    return _texture(cv2.GaussianBlur(img, (0, 0), 0.8), rng, 9 if blast else 5)
