"""Batch experiments behind the Preprocessing and Segmentation labs."""
import time

import numpy as np
from scipy import stats

from . import datasets, deep, preprocessing as pp, segmentation as seg, synthetic
from .imaging import data_url, resize_max


def _pick(items, n):
    """n items spread evenly across the list."""
    if len(items) <= n:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def _agg(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def _samples(settings, source, subset, n, max_side):
    """Yields (name, image, truth_mask, centroids, true_count)."""
    if source == "synthetic":
        for i in range(n):
            img, labels, centres = synthetic.smear(seed=i)
            yield f"synthetic-{i + 1}", img, labels > 0, centres, len(centres)
        return
    paths = _pick(datasets.images(settings, subset), n)
    if not paths:
        raise ValueError(f"No {subset.upper()} images found under the dataset folder.")
    for path in paths:
        img, scale = resize_max(datasets.load(path), max_side)
        truth = datasets.truth_mask(settings, path, img.shape[:2])
        pts = datasets.centroids(settings, path) if subset == "idb1" else None
        if pts is not None:
            pts = [(x * scale, y * scale) for x, y in pts]
        yield path.name, img, (truth > 0) if truth is not None else None, pts, None


# ---------------------------------------------------------- preprocessing

def preprocessing(settings, source="dataset", subset="idb1", n=10, noise="mixed",
                  sp_density=0.02, gauss_sigma=10.0, max_side=1024):
    denoise = {name: [] for name in ["Noisy input (no filter)", *pp.DENOISERS, "DFAE (full)"]}
    contrast = {name: [] for name in ["Original", *pp.ENHANCERS, "DFAE (full)"]}
    example = None
    for i, (name, img, *_rest) in enumerate(_samples(settings, source, subset, n, max_side)):
        noisy = pp.add_noise(img, noise, sp_density, gauss_sigma, seed=i)
        denoise["Noisy input (no filter)"].append(pp.quality(img, noisy))
        for label, fn in pp.DENOISERS.items():
            denoise[label].append(pp.quality(img, fn(noisy)))
        full = pp.dfae_image(noisy)
        # CLAHE deliberately changes intensities, so the full framework is
        # scored against the CLAHE-enhanced clean image, not the raw one.
        denoise["DFAE (full)"].append(pp.quality(pp.clahe(img), full))
        contrast["Original"].append(pp.contrast_stats(img))
        for label, fn in pp.ENHANCERS.items():
            contrast[label].append(pp.contrast_stats(fn(img)))
        contrast["DFAE (full)"].append(pp.contrast_stats(pp.dfae_image(img)))
        if example is None:
            small = lambda im: data_url(resize_max(im, 640)[0])
            example = {"name": name, "original": small(img), "noisy": small(noisy),
                       "denoised": small(pp.DENOISERS["Median → Gaussian"](noisy)), "dfae": small(full)}

    table = [{"method": m, **{k: _agg([r[k] for r in rows]) for k in ("psnr", "snr", "mse", "ssim")}}
             for m, rows in denoise.items()]
    anova = None
    groups = [[r["psnr"] for r in denoise[m]] for m in pp.DENOISERS]
    if len(groups[0]) >= 3:
        f, p = stats.f_oneway(*groups)
        anova = {"f": float(f), "p": float(p), "df_between": len(groups) - 1,
                 "df_within": sum(map(len, groups)) - len(groups)}
    return {
        "source": source, "subset": subset, "n_images": len(denoise["Median"]), "noise": noise,
        "denoise": table,
        "contrast": [{"method": m, **{k: _agg([r[k] for r in rows]) for k in ("rms_contrast", "entropy")}}
                     for m, rows in contrast.items()],
        "anova_psnr": anova, "example": example,
    }


# ----------------------------------------------------------- segmentation

def segmentation(settings, source="dataset", subset="idb1", n=10, methods=None, max_side=1024):
    methods = methods or list(seg.CLASSICAL)
    ready = deep.available()
    rows = {m: {c: [] for c in ("raw", "dfae")} for m in methods}
    unavailable = {m: f"No trained weights (backend/models/{m}.pt)" for m in methods
                   if m in seg.DEEP and not ready[m]}
    panel, has_truth, has_points, n_images = None, False, False, 0
    for name, img, truth, pts, true_count in _samples(settings, source, subset, n, max_side):
        n_images += 1
        variants = {"raw": img, "dfae": pp.dfae_image(img)}
        first = panel is None
        if first:
            panel = {"name": name, "original": data_url(resize_max(img, 520)[0]), "outputs": []}
        for m in methods:
            if m in unavailable:
                continue
            for cond, im in variants.items():
                t0 = time.time()
                labels = seg.segment(im, m)
                row = {"instances": int(labels.max()), "seconds": time.time() - t0}
                if truth is not None:
                    has_truth = True
                    row.update(seg.mask_metrics(labels > 0, truth))
                if pts:
                    has_points = True
                    row.update(seg.centroid_recall(labels, pts))
                if true_count is not None:
                    row["count_error"] = abs(int(labels.max()) - true_count)
                rows[m][cond].append(row)
                if first and cond == "dfae":
                    panel["outputs"].append({"method": m, "name": seg.METHODS[m], "instances": row["instances"],
                                             "image": data_url(resize_max(seg.overlay(im, labels), 520)[0])})

    keys = ("dice", "iou", "precision", "recall", "accuracy", "detected", "separated",
            "count_error", "instances", "seconds")
    table = []
    for m in methods:
        entry = {"method": m, "name": seg.METHODS[m], "error": unavailable.get(m)}
        for cond in ("raw", "dfae"):
            entry[cond] = {k: _agg([r.get(k) for r in rows[m][cond]]) for k in keys}
        a = [r.get("dice") for r in rows[m]["raw"]]
        b = [r.get("dice") for r in rows[m]["dfae"]]
        entry["ttest"] = None
        if has_truth and len(a) >= 3 and np.std(np.subtract(b, a)) > 0:
            t, p = stats.ttest_rel(b, a)
            entry["ttest"] = {"t": float(t), "p": float(p), "diff": float(np.mean(b) - np.mean(a))}
        table.append(entry)

    anova = None
    groups = [[r["dice"] for r in rows[m]["dfae"]] for m in methods if has_truth and rows[m]["dfae"]]
    if len(groups) >= 2 and len(groups[0]) >= 3:
        f, p = stats.f_oneway(*groups)
        anova = {"f": float(f), "p": float(p), "df_between": len(groups) - 1,
                 "df_within": sum(map(len, groups)) - len(groups)}
    truth_kind = "masks" if has_truth else ("centroids" if has_points else "none")
    return {"source": source, "subset": subset, "n_images": n_images, "ground_truth": truth_kind,
            "table": table, "anova_dice": anova, "panel": panel}
