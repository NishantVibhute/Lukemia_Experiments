"""Stage 4 - classification: SVM / Random Forest / k-NN baselines and SADMC,
evaluated with stratified k-fold cross-validation."""
import time

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import datasets, features, preprocessing, segmentation
from .config import MODELS
from .sadmc import SADMCClassifier

BUNDLE = MODELS / "classifiers.joblib"
MODEL_NAMES = {
    "svm": "Support Vector Machine (RBF)",
    "rf": "Random Forest (200 trees)",
    "knn": "k-Nearest Neighbours (k=5)",
    "sadmc": "SADMC (proposed)",
}
N_HAND = len(features.NAMES)
_feature_cache = {}
_bundle = None


def _estimator(name, seed):
    return {
        "svm": lambda: SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=seed),
        "rf": lambda: RandomForestClassifier(200, criterion="gini", random_state=seed),
        "knn": lambda: KNeighborsClassifier(5),
        "sadmc": lambda: SADMCClassifier(seed=seed),
    }[name]()


def _pipeline(name, n_hand, n_total, seed=0):
    """The first n_hand columns are handcrafted features; any further columns
    are deep features, compressed with PCA before fusion."""
    blocks = [("hand", StandardScaler(), slice(0, n_hand))]
    if n_total > n_hand:
        blocks.append(("deep", make_pipeline(StandardScaler(), PCA(32, random_state=seed)),
                       slice(n_hand, n_total)))
    return Pipeline([("fuse", ColumnTransformer(blocks)), ("clf", _estimator(name, seed))])


# --------------------------------------------------------------- features

def cell_features(img, use_dfae=True, deep=False):
    """Features of the dominant leukocyte in a single-cell image, or None."""
    work = preprocessing.dfae_image(img) if use_dfae else img
    labels = segmentation.segment(work, "marker_watershed")
    if labels.max() == 0:
        return None
    rows, regions = features.extract(work, labels, segmentation.cell_regions(work, labels))
    i = int(np.argmax(rows[:, 0]))
    row = rows[i]
    if deep:
        from .deep import deep_features
        row = np.concatenate([row, deep_features([features.crop(work, regions[i])])[0]])
    return row


def build_dataset(settings, use_dfae=True, deep=False):
    paths, labels, names, source = datasets.classification_samples(settings)
    key = (source, len(paths), use_dfae, deep)
    if key not in _feature_cache:
        X, y, skipped = [], [], []
        for path, label in zip(paths, labels):
            row = cell_features(datasets.load(path), use_dfae, deep)
            if row is None:
                skipped.append(path.name)
            else:
                X.append(row)
                y.append(label)
        _feature_cache.clear()
        _feature_cache[key] = (np.array(X, np.float32), np.array(y), skipped)
    X, y, skipped = _feature_cache[key]
    return X, y, names, source, skipped


# ------------------------------------------------------------- evaluation

def _scores(y, pred, proba, n_classes):
    cm = confusion_matrix(y, pred, labels=range(n_classes))
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(tp / (tp + fp))
        recall = np.nan_to_num(tp / (tp + fn))
        specificity = np.nan_to_num(tn / (tn + fp))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    # Binary: report the positive (last) class; multi-class: macro average.
    pick = (lambda v: float(v[1])) if n_classes == 2 else (lambda v: float(v.mean()))
    try:
        auc = float(roc_auc_score(y, proba[:, 1]) if n_classes == 2
                    else roc_auc_score(y, proba, multi_class="ovr", labels=range(n_classes)))
    except ValueError:
        auc = float("nan")
    return {"accuracy": float(tp.sum() / cm.sum()), "precision": pick(precision),
            "recall": pick(recall), "specificity": pick(specificity), "f1": pick(f1), "auc": auc}, cm


def cross_validate(X, y, n_classes, models, folds=5, n_hand=None, seed=0):
    """Stratified k-fold CV of each model, then a final fit on all data.
    Returns (results, fitted_models, folds_used)."""
    n_hand = min(N_HAND, X.shape[1]) if n_hand is None else n_hand
    folds = int(min(folds, np.bincount(y, minlength=n_classes).min()))
    if folds < 2:
        raise ValueError("Each class needs at least two images for cross-validation.")
    splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
    results, fitted = [], {}
    for name in models:
        t0 = time.time()
        per_fold, total_cm = [], np.zeros((n_classes, n_classes), int)
        for train, test in splits:
            model = _pipeline(name, n_hand, X.shape[1], seed).fit(X[train], y[train])
            proba = model.predict_proba(X[test])
            scores, cm = _scores(y[test], proba.argmax(1), proba, n_classes)
            per_fold.append(scores)
            total_cm += cm
        metrics = {k: {"mean": float(np.nanmean([f[k] for f in per_fold])),
                       "std": float(np.nanstd([f[k] for f in per_fold]))} for k in per_fold[0]}
        results.append({"model": name, "name": MODEL_NAMES[name], "metrics": metrics,
                        "confusion": total_cm.tolist(), "seconds": round(time.time() - t0, 1),
                        "fold_accuracy": [f["accuracy"] for f in per_fold]})
        fitted[name] = _pipeline(name, n_hand, X.shape[1], seed).fit(X, y)  # final model on all data
    return results, fitted, folds


def evaluate(settings, models=("svm", "rf", "knn", "sadmc"), folds=5, use_dfae=True, deep=False, seed=0):
    X, y, names, source, skipped = build_dataset(settings, use_dfae, deep)
    if len(X) == 0:
        raise ValueError("No labelled single-cell images found. Set the dataset folder first.")
    n_classes = len(names)
    results, fitted, folds = cross_validate(X, y, n_classes, models, folds, seed=seed)

    global _bundle
    _bundle = {"models": fitted, "classes": names, "use_dfae": use_dfae, "deep": deep,
               "source": source, "n_samples": int(len(X)), "trained": time.strftime("%Y-%m-%d %H:%M"),
               "cv": {r["model"]: r["metrics"]["accuracy"]["mean"] for r in results}}
    joblib.dump(_bundle, BUNDLE)

    importance = None
    if "rf" in fitted:
        imp = fitted["rf"].named_steps["clf"].feature_importances_[:N_HAND]
        importance = sorted(({"feature": n, "importance": float(v)} for n, v in zip(features.NAMES, imp)),
                            key=lambda d: -d["importance"])
    return {"source": source, "classes": names, "counts": np.bincount(y, minlength=n_classes).tolist(),
            "folds": folds, "use_dfae": use_dfae, "deep": deep, "n_features": int(X.shape[1]),
            "skipped": skipped, "results": results, "feature_importance": importance}


# -------------------------------------------------------------- inference

def bundle():
    global _bundle
    if _bundle is None and BUNDLE.exists():
        _bundle = joblib.load(BUNDLE)
    return _bundle


def info():
    b = bundle()
    if b is None:
        return None
    return {k: b[k] for k in ("classes", "use_dfae", "deep", "source", "n_samples", "trained", "cv")} | {
        "models": list(b["models"])}


def predict(X, model):
    """Class probabilities for feature rows, using a trained model."""
    b = bundle()
    if b is None or model not in b["models"]:
        raise ValueError("That classifier has not been trained yet - run the Classification lab first.")
    return b["models"][model].predict_proba(X)
