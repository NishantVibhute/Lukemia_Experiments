"""Smoke tests on synthetic images (drawn shapes with exact ground truth)."""
import numpy as np
import pytest

from app import features, preprocessing as pp, segmentation as seg, synthetic
from app.sadmc import SADMCClassifier


@pytest.fixture(scope="module")
def smear():
    return synthetic.smear(seed=1)


def test_dfae_improves_on_noisy_input(smear):
    img = smear[0]
    noisy = pp.add_noise(img, "mixed", seed=0)
    before = pp.quality(img, noisy)["psnr"]
    after = pp.quality(img, pp.DENOISERS["Median → Gaussian"](noisy))["psnr"]
    assert after > before
    out = pp.dfae(noisy)
    assert out["enhanced"].shape == img.shape
    assert pp.contrast_stats(out["enhanced"])["rms_contrast"] > 0


@pytest.mark.parametrize("method", list(seg.CLASSICAL))
def test_classical_segmentation_finds_nuclei(smear, method):
    img, truth, _ = smear
    labels = seg.segment(pp.dfae_image(img), method)
    # Local thresholding also picks up cytoplasm; it is the weak baseline.
    floor = 0.5 if method == "adaptive" else 0.8
    assert seg.mask_metrics(labels > 0, truth > 0)["dice"] > floor


def test_marker_watershed_separates_overlapped_cells(smear):
    img, truth, centres = smear
    work = pp.dfae_image(img)
    merged = seg.centroid_recall(seg.segment(work, "otsu"), centres)["separated"]
    split = seg.centroid_recall(seg.segment(work, "marker_watershed"), centres)["separated"]
    assert split > merged
    assert split >= 0.9


def test_blank_field_has_no_cells():
    blank = np.full((200, 200, 3), 215, np.uint8)
    for method in seg.CLASSICAL:
        assert seg.segment(blank, method).max() == 0


def test_features_shape_and_ranges(smear):
    img, truth, _ = smear
    rows, regions = features.extract(img, truth, seg.cell_regions(img, truth))
    assert rows.shape == (truth.max(), len(features.NAMES))
    assert np.isfinite(rows).all()
    col = dict(zip(features.NAMES, rows.T))
    assert ((col["circularity"] > 0) & (col["circularity"] <= 1)).all()
    assert ((col["nc_ratio"] > 0) & (col["nc_ratio"] <= 1)).all()


def test_sadmc_learns_separable_classes():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-1, 1, (60, 19)), rng.normal(1, 1, (60, 19))]).astype(np.float32)
    y = np.repeat([0, 1], 60)
    model = SADMCClassifier(epochs=40).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    assert np.allclose(model.predict_proba(X).sum(1), 1, atol=1e-5)
