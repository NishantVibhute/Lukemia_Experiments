# ALL Research Workbench

Research application for the thesis *"A Robust Segmentation Algorithm for Detection of
Overlapped Cells in Blood Smear Images and Automated Acute Lymphoblastic Leukemia
Classification"*. It implements the four-stage pipeline from the synopsis and lets you
rerun the comparative experiments on your own copy of ALL-IDB.

Research tool only. Not a medical device and not for diagnosis.

## Quick start (Windows)

1. Install Python 3.11–3.13 from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"**.
2. Click **Code → Download ZIP** above and extract it (or `git clone` this repository).
3. Get the ALL-IDB dataset from its owner (http://www.dti.unimi.it/fscotti/all/) and put the
   `ALL_IDB1` and `ALL_IDB2` folders in the `data` folder. Without it the app runs in demo mode.
4. Double-click **`start_app.bat`**. The first start installs everything (5–15 minutes); then the
   app opens at http://localhost:8501.

Step-by-step instructions and a troubleshooting table are in
[HOW_TO_RUN.md](HOW_TO_RUN.md). Mac/Linux:

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m streamlit run streamlit_app.py

The dataset is not included in this repository; it is distributed by its owner under their own terms.

## How the study works

The app walks through the thesis one phase at a time. Each phase runs every available
method, shows the measurements, writes a conclusion and records which method is carried
into the next phase. Confirming a phase unlocks the next one; changing an earlier
decision clears the later phases, because they depended on it.

| Phase | Methods compared | Decided by |
|---|---|---|
| Dataset | - | Set the ALL-IDB folder (or demo mode on synthetic images) |
| 1 Noise removal | Median, Gaussian, Bilateral, Wiener, Adaptive Median, Median → Gaussian, Gaussian → Median, under salt-and-pepper, Gaussian and mixed noise | PSNR, SNR, MSE, SSIM against the clean image; ANOVA |
| 2 Contrast enhancement | None, Histogram Equalization, Contrast Stretching, CLAHE at four clip limits | Dice of Otsu segmentation if masks exist, otherwise nucleus separability |
| 3 Segmentation | Otsu, adaptive threshold, K-Means, Fuzzy C-Means, watershed, marker-controlled watershed (+ U-Net, AMIS once trained), each on raw and preprocessed images | Cells separated, Dice, IoU; paired t-test for the preprocessing effect |
| 4 Feature extraction | Morphological, texture, colour, combinations (+ deep features optionally) | ANOVA F-score per feature; cross-validated accuracy per feature group |
| 5 Classification | SVM, Random Forest, k-NN, SADMC | k-fold accuracy, sensitivity, specificity, F1, ROC-AUC, confusion matrices |
| Summary | - | Chosen pipeline, conclusions, downloadable report, try it on one image |

Everything is written to `results/`: one CSV of raw per-image rows per phase, the example
figures, `state.json` with the decisions and conclusions, and the trained models.



## How the thesis maps to the code

| Thesis component | File | State |
|---|---|---|
| DFAE (median → Gaussian → CLAHE) and baseline filters | `backend/app/preprocessing.py` | Working |
| Otsu, adaptive, K-Means, FCM, watershed, marker-controlled watershed | `backend/app/segmentation.py` | Working |
| U-Net, AMIS (Mask R-CNN + spatial attention) | `backend/app/deep.py`, `backend/train_segmentation.py` | Implemented; **needs training** |
| Morphological, GLCM and colour features | `backend/app/features.py` | Working |
| ResNet-50 + DenseNet-121 deep features | `backend/app/deep.py` | Working (optional, downloads ImageNet weights) |
| SVM / RF / k-NN baselines, SADMC | `backend/app/classify.py`, `backend/app/sadmc.py` | Working |

## Things to know before quoting numbers

- **Ground truth.** ALL-IDB1 ships blast centroids (`.xyc`), not pixel masks. Without masks
  the Segmentation lab reports "cells found / cells separated" instead of Dice and IoU.
  Put masks (same file stem as the image) in a folder and set it under Dataset & models.
- **Subtypes.** ALL-IDB2 is labelled normal vs lymphoblast only, so classification is
  binary. For L1/L2/L3 or B-ALL/T-ALL, set "Class folders" to a folder with one
  sub-folder of single-cell images per class.
- **AMIS and U-Net** are untrained. `python train_segmentation.py --model amis --epochs 30
  --pretrained` trains from the masks folder; a CUDA GPU is strongly recommended.
  `--pseudo-labels` trains on watershed output instead, which only teaches the network to
  imitate watershed.
- **Synthetic test images** are drawn shapes for checking the software. Never report them.
- The "reported in the synopsis" rows in each lab are copied from the synopsis for
  reference; the app does not reproduce them automatically.

## Tests

    cd backend
    python -m pytest
