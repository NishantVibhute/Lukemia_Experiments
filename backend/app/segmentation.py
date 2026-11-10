"""Stage 2 - leukocyte nucleus segmentation.

Classical baselines (Otsu, adaptive threshold, K-Means, Fuzzy C-Means, basic
and marker-controlled watershed) work out of the box. U-Net and the proposed
AMIS network are served from `deep.py` once trained weights exist.

Every method returns an int32 label image: 0 = background, 1..N = one
leukocyte nucleus each.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.feature import peak_local_max
from skimage.segmentation import expand_labels, find_boundaries, watershed

CLASSICAL = {
    "otsu": "Otsu Thresholding",
    "adaptive": "Adaptive Thresholding",
    "kmeans": "K-Means (K=3, Lab)",
    "fcm": "Fuzzy C-Means",
    "watershed": "Watershed (basic)",
    "marker_watershed": "Marker-Controlled Watershed",
}
DEEP = {"unet": "U-Net", "amis": "AMIS (proposed)"}
METHODS = {**CLASSICAL, **DEEP}


def stain_map(img):
    """Purple-ness of each pixel (Lab a* minus b*), 0-255.

    Romanowsky-stained nuclei are deep purple (a* high, b* strongly negative),
    erythrocytes pink (b* near zero) and plasma neutral, so a* - b* isolates
    leukocyte nuclei far better than grey level does.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    return np.clip((lab[..., 1] - lab[..., 2]) * 2.5, 0, 255).astype(np.uint8)


# ------------------------------------------------------ foreground masks

def _otsu(m):
    t, mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # A field with no stained nuclei has no real second mode.
    if t < 12:
        return np.zeros_like(m, dtype=bool)
    return mask > 0


def _adaptive(m, offset=15):
    # The window must be wider than a nucleus or cell interiors fall below
    # their own local mean and come out hollow.
    block = max(31, (min(m.shape) // 3) | 1)
    mask = cv2.adaptiveThreshold(m, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, -float(offset)) > 0
    return ndi.binary_fill_holes(mask)


def _colour_samples(img, n=20000, seed=0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    idx = np.random.default_rng(seed).choice(len(lab), min(n, len(lab)), replace=False)
    return lab, lab[idx]


def _nucleus_cluster(lab, centres, shape):
    # Nearest centre per pixel, chunked to bound memory on full-resolution smears.
    assign = np.concatenate([((c[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
                             for c in np.array_split(lab, max(1, len(lab) // 500_000))])
    purple = centres[:, 1] - centres[:, 2]
    if purple.max() - np.median(purple) < 8:  # no distinct stained cluster
        return np.zeros(shape, dtype=bool)
    return (assign == purple.argmax()).reshape(shape)


def _kmeans(img, k=3):
    lab, sample = _colour_samples(img)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(0)
    _, _, centres = cv2.kmeans(sample, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return _nucleus_cluster(lab, centres, img.shape[:2])


def _fcm(img, c=3, m=2.0, iters=40, seed=0):
    lab, x = _colour_samples(img, seed=seed)
    rng = np.random.default_rng(seed)
    u = rng.dirichlet(np.ones(c), len(x)).astype(np.float32)
    for _ in range(iters):
        um = u ** m
        centres = (um.T @ x) / um.sum(0)[:, None]
        d = np.linalg.norm(x[:, None, :] - centres[None], axis=2) + 1e-6
        new_u = 1.0 / (d ** (2 / (m - 1)) * (d ** (-2 / (m - 1))).sum(1, keepdims=True))
        if np.abs(new_u - u).max() < 1e-4:
            u = new_u
            break
        u = new_u
    # Highest membership == nearest centre for m = 2.
    return _nucleus_cluster(lab, centres.astype(np.float32), img.shape[:2])


# ------------------------------------------------------------- instances

def _min_area(shape):
    return max(30, int(shape[0] * shape[1] * 0.0002))


def _clean(mask, shape):
    return morphology.remove_small_objects(mask, _min_area(shape))


def _components(mask):
    return measure.label(mask, connectivity=2).astype(np.int32)


def _typical_radius(mask):
    """Median inscribed radius over connected components."""
    dist = ndi.distance_transform_edt(mask)
    lab = _components(mask)
    if lab.max() == 0:
        return dist, 0.0
    radii = ndi.maximum(dist, lab, index=np.arange(1, lab.max() + 1))
    return dist, float(np.median(radii))


def _watershed_basic(mask, min_distance=3):
    dist, r = _typical_radius(mask)
    if r == 0:
        return _components(mask)
    peaks = peak_local_max(dist, min_distance=int(min_distance), labels=mask.astype(np.int32), exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dist, markers, mask=mask).astype(np.int32)


def _watershed_marker(mask, marker_spacing=0.7, min_peak=0.4):
    """Marker-controlled watershed: morphological cleaning, then one internal
    marker per nucleus taken from smoothed distance-transform peaks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=2) > 0
    mask = ndi.binary_fill_holes(mask)
    dist, r = _typical_radius(mask)
    if r < 2:
        return _components(mask)
    smooth = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), max(1.0, r * 0.25))
    peaks = peak_local_max(smooth, min_distance=max(3, int(r * marker_spacing)), threshold_abs=r * min_peak,
                           labels=mask.astype(np.int32), exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-smooth, markers, mask=mask).astype(np.int32)
    # Components that produced no peak keep their own label.
    rest = _components(mask & (labels == 0))
    labels[rest > 0] = rest[rest > 0] + labels.max()
    return labels


def segment(img, method="marker_watershed", **params):
    """Segment leukocyte nuclei. `img` should already be pre-processed.
    `params` are the settings of the chosen method (see phases.SEG_PARAMS)."""
    if method in DEEP:
        from . import deep
        return deep.segment(img, method, **params)
    shape = img.shape[:2]
    if method == "kmeans":
        labels = _components(_clean(_kmeans(img, int(params.get("clusters", 3))), shape))
    elif method == "fcm":
        labels = _components(_clean(_fcm(img, int(params.get("clusters", 3)), float(params.get("fuzziness", 2.0))), shape))
    elif method == "adaptive":
        labels = _components(_clean(_adaptive(stain_map(img), **params), shape))
    else:
        mask = _otsu(stain_map(img))
        if method == "otsu":
            labels = _components(_clean(mask, shape))
        elif method == "watershed":
            labels = _watershed_basic(mask, **params)
        elif method == "marker_watershed":
            labels = _watershed_marker(mask, **params)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
    return relabel(labels, _min_area(shape))


def relabel(labels, min_area=0):
    """Drop tiny instances and renumber 1..N."""
    out = np.zeros_like(labels)
    n = 0
    for region in measure.regionprops(labels):
        if region.area >= min_area:
            n += 1
            out[labels == region.label] = n
    return out


def cell_regions(img, labels):
    """Approximate whole-cell regions (nucleus + cytoplasm) by growing each
    nucleus into the surrounding lightly stained area."""
    if labels.max() == 0:
        return labels.copy()
    m = stain_map(img)
    t, _ = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cyto = m > max(8, 0.35 * t)
    areas = np.bincount(labels.ravel())[1:]
    radius = float(np.sqrt(np.median(areas) / np.pi))
    grown = expand_labels(labels, distance=max(2, radius * 0.6))
    return np.where((labels > 0) | cyto, grown, 0).astype(np.int32)


# ---------------------------------------------------------------- metrics

def mask_metrics(pred, truth):
    """Pixel-level agreement between two binary masks."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    tn = float((~pred & ~truth).sum())
    div = lambda a, b: a / b if b else 0.0
    return {
        "dice": div(2 * tp, 2 * tp + fp + fn),
        "iou": div(tp, tp + fp + fn),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "accuracy": div(tp + tn, tp + tn + fp + fn),
    }


def centroid_recall(labels, points):
    """Share of annotated cell centroids that fall inside a segmented instance,
    and the share whose instance is not shared with another centroid (i.e. the
    cell was separated from its neighbours)."""
    if not len(points):
        return None
    h, w = labels.shape
    hit = [int(labels[min(h - 1, max(0, round(y))), min(w - 1, max(0, round(x)))]) for x, y in points]
    found = [l for l in hit if l > 0]
    counts = {l: found.count(l) for l in set(found)}
    separated = sum(1 for l in found if counts[l] == 1)
    return {"detected": len(found) / len(points), "separated": separated / len(points)}


# -------------------------------------------------------------- rendering

def overlay(img, labels, colours=None, thickness=2):
    """Draw instance outlines. `colours` maps label -> BGR tuple."""
    out = img.copy()
    edges = find_boundaries(labels, mode="inner")
    if thickness > 1:
        k = np.ones((thickness, thickness), np.uint8)
        edge_labels = cv2.dilate((labels * edges).astype(np.float32), k).astype(np.int32)
    else:
        edge_labels = labels * edges
    if colours is None:
        rng = np.random.default_rng(7)
        palette = rng.integers(60, 256, size=(labels.max() + 1, 3))
        out[edge_labels > 0] = palette[edge_labels[edge_labels > 0]]
    else:
        for label, colour in colours.items():
            out[edge_labels == label] = colour
    return out
