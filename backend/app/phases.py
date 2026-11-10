"""Phase-by-phase experiments for the Streamlit study app.

Phases 1-3 work as trials: one method with one set of settings is measured on
every image and saved; the saved trials are then compared and one is carried
into the next phase. A method with its settings is a "spec":
    {"method": ..., "params": {...}, "label": "..."}
"""
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from . import classify, datasets, experiments, features, preprocessing as pp, segmentation as seg, synthetic

NOISE = {"salt_pepper": "Salt-and-pepper", "gaussian": "Gaussian", "speckle": "Speckle",
         "poisson": "Poisson (shot)", "mixed": "Mixed (Gaussian + salt-and-pepper)"}

# Tunable settings per method: ("select", options, default) or
# ("slider", min, max, default, step).
_KERNEL = ("select", [3, 5, 7], 3)
_SIGMA = ("slider", 0.5, 3.0, 1.0, 0.1)
DENOISE_PARAMS = {
    "Median": {"kernel": _KERNEL},
    "Gaussian": {"sigma": _SIGMA},
    "Bilateral": {"diameter": ("select", [5, 7, 9], 7), "sigma_color": ("slider", 10, 100, 40, 5),
                  "sigma_space": ("slider", 1, 15, 5, 1)},
    "Wiener": {"kernel": ("select", [3, 5, 7], 5)},
    "Adaptive Median": {"max_window": ("select", [5, 7, 9], 7)},
    "Median → Gaussian": {"kernel": _KERNEL, "sigma": _SIGMA},
    "Gaussian → Median": {"sigma": _SIGMA, "kernel": _KERNEL},
}
ENHANCE_PARAMS = {
    "None": {},
    "Histogram Equalization": {},
    "Contrast Stretching": {"low_percentile": ("slider", 0, 10, 2, 1), "high_percentile": ("slider", 90, 100, 98, 1)},
    "CLAHE": {"clip_limit": ("slider", 0.5, 8.0, 2.0, 0.5), "tiles": ("select", [4, 8, 16], 8)},
}
SEG_PARAMS = {
    "otsu": {},
    "adaptive": {"offset": ("slider", 2, 40, 15, 1)},
    "kmeans": {"clusters": ("select", [2, 3, 4], 3)},
    "fcm": {"clusters": ("select", [2, 3, 4], 3), "fuzziness": ("slider", 1.5, 3.0, 2.0, 0.1)},
    "watershed": {"min_distance": ("slider", 1, 25, 3, 1)},
    "marker_watershed": {"marker_spacing": ("slider", 0.3, 1.5, 0.7, 0.1), "min_peak": ("slider", 0.1, 0.9, 0.4, 0.1)},
    "unet": {},
    "amis": {"score": ("slider", 0.1, 0.9, 0.5, 0.05)},
}
FEATURE_SETS = {
    "Morphological": features.MORPH,
    "Texture (GLCM)": features.TEXTURE,
    "Colour / intensity": features.COLOUR,
    "Morphological + Texture": features.MORPH + features.TEXTURE,
    "All handcrafted": features.NAMES,
}
DEEP_SET = "All handcrafted + deep (ResNet-50, DenseNet-121)"


def defaults(schema):
    return {k: v[2] if v[0] == "select" else v[3] for k, v in schema.items()}


def spec(method, params=None, schema=None):
    """A method plus its settings, with a readable label."""
    params = dict(params if params is not None else defaults(schema[method]))
    shown = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
    name = seg.METHODS.get(method, method)
    return {"method": method, "params": params, "label": f"{name} ({shown})" if shown else name}


def denoise(img, s):
    m, p = s["method"], s["params"]
    if m == "Median":
        return pp.median(img, p["kernel"])
    if m == "Gaussian":
        return pp.gaussian(img, p["sigma"])
    if m == "Bilateral":
        return pp.bilateral(img, p["diameter"], p["sigma_color"], p["sigma_space"])
    if m == "Wiener":
        return pp.wiener(img, p["kernel"])
    if m == "Adaptive Median":
        return pp.adaptive_median(img, p["max_window"])
    if m == "Median → Gaussian":
        return pp.gaussian(pp.median(img, p["kernel"]), p["sigma"])
    if m == "Gaussian → Median":
        return pp.median(pp.gaussian(img, p["sigma"]), p["kernel"])
    raise ValueError(f"Unknown filter: {m}")


def enhance(img, s):
    m, p = s["method"], s["params"]
    if m == "None":
        return img
    if m == "Histogram Equalization":
        return pp.hist_eq(img)
    if m == "Contrast Stretching":
        return pp.contrast_stretch(img, p["low_percentile"], p["high_percentile"])
    if m == "CLAHE":
        return pp.clahe(img, p["clip_limit"], p["tiles"])
    raise ValueError(f"Unknown enhancement: {m}")


def preprocess(img, den, enh):
    return enhance(denoise(img, den), enh)


def segment(img, s):
    return seg.segment(img, s["method"], **s["params"])


def smears(settings, source, n, max_side=1024):
    """[(name, image, truth_mask, centroids, true_count)] for whole-smear phases."""
    subset = "idb1"
    if source == "dataset" and not datasets.images(settings, "idb1"):
        subset = "idb2"
    return list(experiments._samples(settings, source, subset, n, max_side))


def zoom_box(img, size=220):
    """A window centred on the most strongly stained region, so detail is visible."""
    m = cv2.GaussianBlur(seg.stain_map(img), (0, 0), 15)
    y, x = np.unravel_index(int(m.argmax()), m.shape)
    h, w = m.shape
    size = min(size, h, w)
    y0 = int(np.clip(y - size // 2, 0, h - size))
    x0 = int(np.clip(x - size // 2, 0, w - size))
    return slice(y0, y0 + size), slice(x0, x0 + size)


# ------------------------------------------------- phase 1: noise removal

def noise_trial(samples, den, kinds, sp_density=0.02, gauss_sigma=10.0):
    """Measure one filter setting under each noise type on every image.
    Rows carry the filtered score and the unfiltered (noisy) baseline."""
    rows, figs = [], {}
    for i, (name, img, *_r) in enumerate(samples):
        for kind in kinds:
            noisy = pp.add_noise(img, kind, sp_density, gauss_sigma, seed=i)
            out = denoise(noisy, den)
            q, base = pp.quality(img, out), pp.quality(img, noisy)
            rows.append({"image": name, "noise": NOISE[kind], **q, "psnr_noisy": base["psnr"]})
            if i == 0:
                box = zoom_box(img)
                figs[f"{NOISE[kind]} · BEFORE ({base['psnr']:.1f} dB)"] = noisy[box]
                figs[f"{NOISE[kind]} · AFTER ({q['psnr']:.1f} dB)"] = out[box]
    return pd.DataFrame(rows), figs


# ------------------------------------------ phase 2: contrast enhancement

def separability(img):
    """Otsu's criterion on the stain map: between-class / total variance (0-1).
    Higher means nuclei stand out more cleanly from everything else."""
    m = seg.stain_map(img).ravel()
    total = m.var()
    if total == 0:
        return 0.0
    t, _ = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = m > t
    w = fg.mean()
    if w in (0.0, 1.0):
        return 0.0
    return float(w * (1 - w) * (m[fg].mean() - m[~fg].mean()) ** 2 / total)


def contrast_scores(img, truth=None):
    row = {**pp.contrast_stats(img), "separability": separability(img)}
    if truth is not None:
        row["dice_otsu"] = seg.mask_metrics(seg.segment(img, "otsu") > 0, truth)["dice"]
    return row


def contrast_trial(samples, den, enh):
    rows, figs = [], {}
    for i, (name, img, truth, *_r) in enumerate(samples):
        base = denoise(img, den)
        out = enhance(base, enh)
        rows.append({"image": name, **contrast_scores(out, truth)})
        if i == 0:
            figs = {"BEFORE (denoised only)": base, "AFTER": out}
    return pd.DataFrame(rows), figs


# -------------------------------------------------- phase 3: segmentation

def segmentation_scores(labels, truth, pts, true_count):
    row = {"instances": int(labels.max())}
    if truth is not None:
        row.update(seg.mask_metrics(labels > 0, truth))
    if pts:
        row.update(seg.centroid_recall(labels, pts))
    if true_count is not None:
        row["count_error"] = abs(int(labels.max()) - true_count)
    return row


def segmentation_trial(samples, den, enh, sg):
    rows, figs = [], {}
    for i, (name, img, truth, pts, true_count) in enumerate(samples):
        for cond, im in (("Raw image", img), ("Preprocessed", preprocess(img, den, enh))):
            t0 = time.time()
            labels = segment(im, sg)
            rows.append({"image": name, "condition": cond, "seconds": time.time() - t0,
                         **segmentation_scores(labels, truth, pts, true_count)})
            if i == 0 and cond == "Preprocessed":
                figs = {"BEFORE (preprocessed image)": im,
                        "AFTER (nucleus outlines)": seg.overlay(im, labels, thickness=3)}
    return pd.DataFrame(rows), figs


# ---------------------------------------------- phase 4: feature extraction

def cell_images(settings, source, n_synthetic=40):
    """(images, labels, class_names, source_name) of single-cell images."""
    if source == "synthetic":
        imgs = [synthetic.cell(i % 2, seed=i) for i in range(2 * n_synthetic)]
        return imgs, [i % 2 for i in range(2 * n_synthetic)], datasets.CLASSES, "synthetic cells"
    paths, labels, names, src = datasets.classification_samples(settings)
    return [datasets.load(p) for p in paths], labels, names, src


def cell_dataset(settings, source, den, enh, sg, deep=False):
    """Feature matrix for the dominant leukocyte of every single-cell image."""
    imgs, labels, names, src = cell_images(settings, source)
    X, y, crops, skipped = [], [], [], 0
    for img, label in zip(imgs, labels):
        work = preprocess(img, den, enh)
        lab = segment(work, sg)
        if lab.max() == 0:
            skipped += 1
            continue
        rows, regions = features.extract(work, lab, seg.cell_regions(work, lab))
        k = int(np.argmax(rows[:, 0]))
        X.append(rows[k])
        y.append(label)
        crops.append(features.crop(work, regions[k]))
    X = np.array(X, np.float32).reshape(-1, len(features.NAMES))
    if deep and len(crops):
        from .deep import deep_features
        X = np.hstack([X, deep_features(crops)])
    return X, np.array(y, int), names, src, skipped


def feature_ranking(X, y):
    """Per-feature ANOVA F-score between classes (handcrafted features only)."""
    n = len(features.NAMES)
    f, p = f_classif(X[:, :n], y)
    group = {**{k: "Morphological" for k in features.MORPH}, **{k: "Texture" for k in features.TEXTURE},
             **{k: "Colour / intensity" for k in features.COLOUR}}
    df = pd.DataFrame({"feature": features.NAMES, "group": [group[k] for k in features.NAMES],
                       "F": np.nan_to_num(f), "p": np.nan_to_num(p, nan=1.0)})
    return df.sort_values("F", ascending=False).reset_index(drop=True)


def columns_for(feature_set, n_total):
    """(column indices, number of handcrafted columns) for a feature set."""
    if feature_set == DEEP_SET:
        return list(range(n_total)), len(features.NAMES)
    cols = [features.NAMES.index(k) for k in FEATURE_SETS[feature_set]]
    return cols, len(cols)


def feature_set_experiment(X, y, n_classes, folds=5):
    """Cross-validated Random Forest accuracy for each feature group."""
    sets = list(FEATURE_SETS) + ([DEEP_SET] if X.shape[1] > len(features.NAMES) else [])
    rows = []
    for name in sets:
        cols, n_hand = columns_for(name, X.shape[1])
        res, _, k = classify.cross_validate(X[:, cols], y, n_classes, ["rf"], folds, n_hand)
        for fold, acc in enumerate(res[0]["fold_accuracy"]):
            rows.append({"feature_set": name, "n_features": len(cols), "fold": fold + 1, "accuracy": acc})
    return pd.DataFrame(rows)


# ------------------------------------------------ phase 5: classification

def classification_experiment(X, y, n_classes, feature_set, models, folds=5):
    cols, n_hand = columns_for(feature_set, X.shape[1])
    results, fitted, k = classify.cross_validate(X[:, cols], y, n_classes, models, folds, n_hand)
    rows = [{"model": r["name"], **{m: v["mean"] for m, v in r["metrics"].items()},
             **{m + "_std": v["std"] for m, v in r["metrics"].items()}, "seconds": r["seconds"]} for r in results]
    confusion = {r["name"]: r["confusion"] for r in results}
    fitted = {classify.MODEL_NAMES[k_]: v for k_, v in fitted.items()}
    return pd.DataFrame(rows), confusion, fitted, cols, k


# ----------------------------------------------------- final pipeline run

def run_pipeline(img, den, enh, sg, model=None, cols=None, classes=None):
    """Apply the settings chosen in phases 1-5 to one image."""
    img, scale = experiments.resize_max(img, 2048)
    denoised = denoise(img, den)
    work = enhance(denoised, enh)
    labels = segment(work, sg)
    rows, regions = features.extract(work, labels, seg.cell_regions(work, labels), scale)
    table = pd.DataFrame(rows, columns=features.NAMES)
    table.insert(0, "cell", range(1, len(rows) + 1))
    colours = None
    if model is not None and len(rows):
        X = rows
        if cols is not None and max(cols) >= len(features.NAMES):
            from .deep import deep_features
            X = np.hstack([rows, deep_features([features.crop(work, r) for r in regions])])
        proba = model.predict_proba(X[:, cols])
        pred = proba.argmax(1)
        table.insert(1, "prediction", [classes[k] for k in pred])
        table.insert(2, "probability", proba.max(1))
        palette = [(110, 170, 40), (60, 60, 220), (30, 150, 240), (200, 110, 40), (170, 70, 170), (140, 140, 30)]
        colours = {r.label: palette[int(k) % len(palette)] for r, k in zip(regions, pred)}
    return {"original": img, "denoised": denoised, "enhanced": work,
            "overlay": seg.overlay(img, labels, colours, thickness=3), "cells": table}
