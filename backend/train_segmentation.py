"""Train U-Net or the AMIS network on ALL-IDB1 and save weights for the app.

    python train_segmentation.py --model amis --epochs 30 --pretrained
    python train_segmentation.py --model unet --epochs 60

Targets come from the masks folder set in the app (one mask per image, same
file stem; an instance-label image, or a binary mask that is split into
connected components). With --pseudo-labels, marker-controlled watershed
output is used instead (Chapter 3.4.6) - the network then learns to imitate
that method, so treat such weights as a starting point, not a result.

A CUDA GPU is strongly recommended; AMIS on CPU is very slow.
"""
import argparse
import random

import numpy as np
import torch
from skimage import measure

from app import config, datasets, deep, preprocessing as pp, segmentation as seg
from app.imaging import resize_max


def load_samples(settings, max_side, pseudo, limit):
    samples = []
    for path in datasets.images(settings, "idb1")[:limit or None]:
        img, _ = resize_max(datasets.load(path), max_side)
        img = pp.dfae_image(img)
        if pseudo:
            labels = seg.segment(img, "marker_watershed")
        else:
            mask = datasets.truth_mask(settings, path, img.shape[:2])
            if mask is None:
                continue
            # Binary masks become one instance per connected component.
            labels = mask.astype(np.int32) if len(np.unique(mask)) > 2 else measure.label(mask > 0)
            labels = seg.relabel(labels)
        samples.append((img, labels))
    return samples


def augment(img, labels):
    if random.random() < 0.5:
        img, labels = img[:, ::-1], labels[:, ::-1]
    if random.random() < 0.5:
        img, labels = img[::-1], labels[::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(labels)


def amis_target(labels, device):
    ids = [i for i in np.unique(labels) if i > 0]
    masks = np.stack([labels == i for i in ids]) if ids else np.zeros((0, *labels.shape), bool)
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return {"boxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
            "labels": torch.ones(len(ids), dtype=torch.int64, device=device),
            "masks": torch.tensor(masks, dtype=torch.uint8, device=device)}


def dice_bce(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + target.sum() + 1)
    return dice + torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def train(model_name, samples, epochs, lr, batch, pretrained, device, unet_side=512, **amis_kw):
    import cv2
    if model_name == "amis":
        model = deep.build_amis(pretrained, **amis_kw)
        batch = min(batch, 2)
    else:
        model = deep.UNet()
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        random.shuffle(samples)
        total, steps = 0.0, 0
        for i in range(0, len(samples), batch):
            items = [augment(*s) for s in samples[i:i + batch]]
            if model_name == "amis":
                items = [(im, lb) for im, lb in items if lb.max() > 0]
                if not items:
                    continue
                images = [deep.to_tensor(im).to(device) for im, _ in items]
                loss = sum(model(images, [amis_target(lb, device) for _, lb in items]).values())
            else:
                x = torch.stack([deep.to_tensor(cv2.resize(im, (unet_side, unet_side))) for im, _ in items])
                y = torch.stack([torch.from_numpy(cv2.resize((lb > 0).astype(np.float32), (unet_side, unet_side),
                                                             interpolation=cv2.INTER_NEAREST))[None] for _, lb in items])
                loss = dice_bce(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
        print(f"epoch {epoch + 1}/{epochs}  loss {total / max(steps, 1):.4f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["unet", "amis"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images")
    ap.add_argument("--pseudo-labels", action="store_true")
    ap.add_argument("--pretrained", action="store_true", help="AMIS: start from COCO weights (download)")
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    settings = config.load()
    samples = load_samples(settings, args.max_side, args.pseudo_labels, args.limit)
    if not samples:
        raise SystemExit("No training samples. Set the dataset and masks folders in the app, "
                         "or pass --pseudo-labels.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(samples)} images, device={device}")
    lr = args.lr or (5e-5 if args.model == "amis" else 1e-4)
    model = train(args.model, samples, args.epochs, lr, args.batch, args.pretrained, device)
    meta = {"model": args.model, "epochs": args.epochs, "images": len(samples),
            "pseudo_labels": args.pseudo_labels, "base": 32}
    torch.save({"state": model.cpu().state_dict(), "meta": meta}, deep.WEIGHTS[args.model])
    print(f"saved {deep.WEIGHTS[args.model]}")


if __name__ == "__main__":
    main()
