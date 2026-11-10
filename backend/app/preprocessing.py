"""Stage 1 - noise removal, contrast enhancement and the DFAE framework.

DFAE (Dual-Filter Adaptive Enhancement, synopsis Algorithm 1):
    median filter -> Gaussian smoothing -> CLAHE on 8x8 tiles.
"""
import cv2
import numpy as np
from scipy.signal import wiener as _wiener
from skimage.metrics import structural_similarity


# ---------------------------------------------------------------- filters

def median(img, k=3):
    return cv2.medianBlur(img, k)


def gaussian(img, sigma=1.0):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def bilateral(img, d=7, sigma_color=40, sigma_space=5):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def wiener(img, k=5):
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        with np.errstate(divide="ignore", invalid="ignore"):
            ch = _wiener(img[..., c].astype(np.float64), (k, k))
        out[..., c] = np.clip(np.nan_to_num(ch), 0, 255).astype(np.uint8)
    return out


def adaptive_median(img, max_k=7):
    """Replace only impulse pixels (local min/max), growing the window when the
    median itself is an impulse."""
    out = img.copy()
    pending = np.ones(img.shape, dtype=bool)
    for k in range(3, max_k + 1, 2):
        kernel = np.ones((k, k), np.uint8)
        lo, hi, med = cv2.erode(img, kernel), cv2.dilate(img, kernel), cv2.medianBlur(img, k)
        med_ok = (med > lo) & (med < hi)
        impulse = (img <= lo) | (img >= hi)
        fix = pending & med_ok & impulse
        out[fix] = med[fix]
        pending &= ~med_ok
    return out


# ---------------------------------------------------- contrast enhancement

def _on_lightness(img, fn):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = fn(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe(img, clip=2.0, tiles=8):
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return _on_lightness(img, op.apply)


def hist_eq(img):
    return _on_lightness(img, cv2.equalizeHist)


def contrast_stretch(img, lo_pct=2, hi_pct=98):
    def stretch(l):
        lo, hi = np.percentile(l, (lo_pct, hi_pct))
        if hi <= lo:
            return l
        return np.clip((l.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return _on_lightness(img, stretch)


# ------------------------------------------------------------------- DFAE

def auto_sigma(img):
    """Chapter 3.3.3.1: Gaussian sigma tuned from the median-filtered image
    variance - busier (denser) fields get stronger smoothing."""
    std = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())
    return float(np.clip(0.5 + std / 64.0, 0.5, 2.0))


def dfae(img, k=3, sigma=1.0, clip=2.0, tiles=8, adaptive=False):
    """Returns the intermediate and final images of the DFAE framework."""
    im = median(img, k)
    if adaptive:
        sigma = auto_sigma(im)
    ig = gaussian(im, sigma)
    ie = clahe(ig, clip, tiles)
    return {"median": im, "gaussian": ig, "enhanced": ie, "sigma": sigma}


def dfae_image(img, **kw):
    return dfae(img, **kw)["enhanced"]


DENOISERS = {
    "Median": median,
    "Gaussian": gaussian,
    "Bilateral": bilateral,
    "Wiener": wiener,
    "Adaptive Median": adaptive_median,
    "Median → Gaussian": lambda im: gaussian(median(im)),
    "Gaussian → Median": lambda im: median(gaussian(im)),
}

ENHANCERS = {
    "Histogram Equalization": hist_eq,
    "Contrast Stretching": contrast_stretch,
    "CLAHE": clahe,
}


# ------------------------------------------------------------------ noise

def add_noise(img, kind="mixed", sp_density=0.02, gauss_sigma=10.0, seed=0):
    """Degrade a clean image with a known noise model.

    salt_pepper - impulse noise (dust, dead pixels)
    gaussian    - additive sensor / electronic noise
    speckle     - multiplicative noise, stronger in bright regions
    poisson     - photon shot noise, set by the light level itself
    mixed       - Gaussian followed by salt-and-pepper
    """
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    if kind in ("gaussian", "mixed"):
        out += rng.normal(0, gauss_sigma, img.shape)
    elif kind == "speckle":
        out *= 1 + rng.normal(0, gauss_sigma / 100.0, img.shape)
    elif kind == "poisson":
        # Fewer photons per grey level = noisier; tied to the same strength slider.
        photons = max(0.05, 4.0 / max(gauss_sigma, 1e-6))
        out = rng.poisson(out * photons) / photons
    out = np.clip(out, 0, 255).astype(np.uint8)
    if kind in ("salt_pepper", "mixed"):
        r = rng.random(img.shape[:2])
        out[r < sp_density / 2] = 0
        out[r > 1 - sp_density / 2] = 255
    return out


# ---------------------------------------------------------------- metrics

def quality(reference, test):
    """MSE on the [0, 1] intensity scale, PSNR and SNR in dB, plus SSIM."""
    ref = reference.astype(np.float64) / 255.0
    x = test.astype(np.float64) / 255.0
    err = float(np.mean((ref - x) ** 2))
    eps = 1e-12
    return {
        "mse": err,
        "psnr": float(10 * np.log10(1.0 / max(err, eps))),
        "snr": float(10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - x) ** 2), eps))),
        "ssim": float(structural_similarity(reference, test, channel_axis=2, data_range=255)),
    }


def contrast_stats(img):
    """RMS contrast and Shannon entropy of the lightness channel."""
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    p = np.bincount(l.ravel(), minlength=256) / l.size
    p = p[p > 0]
    return {"rms_contrast": float(l.std() / 255.0), "entropy": float(-(p * np.log2(p)).sum())}
