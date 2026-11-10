"""Deep models: U-Net, the AMIS network and deep feature extractors.

AMIS (Attention-Mask Instance Segmentation, synopsis Algorithm 2) is a
Mask R-CNN whose FPN feature maps pass through a spatial attention module
before the RPN / RoIAlign heads.

Segmentation models need trained weights in backend/models/ - produce them
with `python train_segmentation.py`.
"""
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch import nn

from .config import MODELS
from .segmentation import relabel

WEIGHTS = {"unet": MODELS / "unet.pt", "amis": MODELS / "amis.pt"}
_cache = {}


def available():
    return {name: path.exists() for name, path in WEIGHTS.items()}


# ------------------------------------------------------------------ U-Net

def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Four-level encoder-decoder with skip connections (Chapter 3.4.5.1)."""

    def __init__(self, base=32):
        super().__init__()
        chans = [base, base * 2, base * 4, base * 8]
        self.down = nn.ModuleList()
        cin = 3
        for c in chans:
            self.down.append(_block(cin, c))
            cin = c
        self.bottleneck = _block(cin, cin * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin *= 2
        for c in reversed(chans):
            self.up.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.dec.append(_block(c * 2, c))
            cin = c
        self.out = nn.Conv2d(cin, 1, 1)

    def forward(self, x):
        skips = []
        for d in self.down:
            x = d(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.out(x)


# ------------------------------------------------------------------- AMIS

class SpatialAttention(nn.Module):
    """Weights each location by a map computed from channel-wise mean and max,
    emphasising nuclei / chromatin and damping background."""

    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2)

    def forward(self, x):
        desc = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(desc))


class AttentionBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.body = backbone
        self.out_channels = backbone.out_channels
        self.attention = nn.ModuleDict({k: SpatialAttention() for k in ("0", "1", "2", "3", "pool")})

    def forward(self, x):
        return OrderedDict((k, self.attention[k](v)) for k, v in self.body(x).items())


def build_amis(pretrained=False, **kw):
    """Mask R-CNN (ResNet-50 + FPN) with spatial attention on the pyramid.
    `pretrained=True` starts from COCO weights (downloaded by torchvision)."""
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None, **kw)
    model.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)       # background + leukocyte
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    model.backbone = AttentionBackbone(model.backbone)
    return model


# -------------------------------------------------------------- inference

def to_tensor(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def _load(name):
    if name not in _cache:
        if not WEIGHTS[name].exists():
            raise FileNotFoundError(
                f"No trained weights for {name.upper()} at backend/models/{WEIGHTS[name].name}. "
                f"Train with: python train_segmentation.py --model {name}")
        ckpt = torch.load(WEIGHTS[name], map_location="cpu")
        model = UNet(ckpt["meta"].get("base", 32)) if name == "unet" else build_amis()
        model.load_state_dict(ckpt["state"])
        _cache[name] = model.eval()
    return _cache[name]


def reload():
    _cache.clear()


@torch.no_grad()
def segment(img, method, score=0.5):
    model = _load(method)
    h, w = img.shape[:2]
    if method == "unet":
        side = 512
        x = to_tensor(cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA))[None]
        prob = torch.sigmoid(model(x))[0, 0].numpy()
        mask = cv2.resize(prob, (w, h)) > 0.5
        from skimage import measure
        labels = measure.label(mask, connectivity=2).astype(np.int32)
    else:
        out = model([to_tensor(img)])[0]
        labels = np.zeros((h, w), np.int32)
        keep = out["scores"] >= score
        masks = out["masks"][keep, 0].numpy() > 0.5
        # Lowest score first so confident instances win contested pixels.
        for i in reversed(range(len(masks))):
            labels[masks[i]] = i + 1
    return relabel(labels, max(30, int(h * w * 0.0002)))


# ---------------------------------------------------------- deep features

def _extractors():
    if "features" not in _cache:
        from torchvision import models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        resnet.fc = nn.Identity()
        dense = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        dense.classifier = nn.Identity()
        _cache["features"] = (resnet.eval(), dense.eval())
    return _cache["features"]


@torch.no_grad()
def deep_features(crops, batch=16):
    """ResNet-50 (2048-d) + DenseNet-121 (1024-d) globally pooled features.
    Downloads ImageNet weights on first use."""
    resnet, dense = _extractors()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(crops), batch):
        x = torch.stack([to_tensor(cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA))
                         for c in crops[i:i + batch]])
        x = (x - mean) / std
        out.append(torch.cat([resnet(x), dense(x)], dim=1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 3072), np.float32)
