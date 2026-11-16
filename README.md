# ALL Research Workbench

Research application for the thesis *"A Robust Segmentation Algorithm for Detection of
Overlapped Cells in Blood Smear Images and Automated Acute Lymphoblastic Leukemia
Classification"*. It implements the four-stage pipeline from the synopsis and lets you
rerun the comparative experiments on your own copy of ALL-IDB.

Research tool only. Not a medical device and not for diagnosis.

## Quick start (Windows)

1. Install Python 3.11–3.13 from https://www.python.org/downloads/ and tick **"Add python.exe to PATH"**.
2. Install Git, then run `git clone https://github.com/NishantVibhute/Lukemia_Experiments.git`
   (later, `git pull` in that folder fetches updates).
3. Get the ALL-IDB dataset from its owner (https://scotti.di.unimi.it/all/) and put the
   `ALL_IDB1` and `ALL_IDB2` folders in the `data` folder. Received the dataset files from someone?
   See [HOW_TO_RUN.md](HOW_TO_RUN.md#if-you-were-given-the-dataset-files). Without it the app runs in demo mode.
4. Double-click **`start_app.bat`**. The first start installs everything (5–15 minutes); then the
   app opens at http://localhost:8501.

Step-by-step instructions and a troubleshooting table are in
[HOW_TO_RUN.md](HOW_TO_RUN.md). Mac/Linux:

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m streamlit run streamlit_app.py

The dataset is not included in this repository; it is distributed by its owner under their own terms.

## How the study works

The app walks through the thesis one phase at a time. Each phase feeds the next: the
method you confirm in one phase is applied to the images before the next phase runs.
Changing an earlier decision clears the later phases, because they were measured on the
old input.

```
Phase 1 filter (or your own chain) → Phase 2 contrast → Phase 3 segmentation
    → Phase 4 features → Phase 5 classifier
```

### Phases 1–3: try, save, compare, decide

Each of these phases has two tabs:

1. **① Try and save** – pick an image and a method, tune its settings (every setting has a
   **?** tooltip), and see BEFORE / AFTER with a plain-language verdict. Switch on
   **"Show all images at once"** to see the same method on every image in the set.
   Click **Save this trial** to measure that exact setting on all study images.
2. **② Compare and decide** – every saved trial ranked best first, with a chart, the saved
   images, an editable conclusion, and **Confirm and continue** to carry the best one forward.

Nothing is saved unless you click **Save this trial**. The number of images per trial is set
on the Dataset page (use a small number while exploring, all 108 for final results).

| Phase | Methods you can try | Judged by |
|---|---|---|
| 1 Noise removal | Median, Gaussian, Bilateral, Wiener, Adaptive Median, Median → Gaussian, Gaussian → Median, each with its own settings, under salt-and-pepper, Gaussian, speckle, Poisson or mixed noise | Image quality against the clean image: PSNR (main), SNR, MSE, RMSE, MAE, SSIM, edge preservation, correlation |
| 2 Contrast enhancement | None, Histogram Equalization, Contrast Stretching, CLAHE (clip limit and tile size adjustable) | Nucleus separability, or Dice of Otsu segmentation if masks exist |
| 3 Segmentation | Otsu, adaptive threshold, K-Means, Fuzzy C-Means, watershed, marker-controlled watershed (+ U-Net, AMIS once trained), each on raw and preprocessed images | Annotated cells found and separated; Dice / IoU with masks; paired t-test for the preprocessing effect |

### Preprocessing playground

A free-form page for building your own chain of steps: **Add noise**, **Filter**, filter the
result again, **Enhance contrast**, in any order. Every image is shown after every step with
its quality measurements against the original, plus the average per step and a chart of any
measurement. A chain with its noise steps at the
top can be saved as a Phase 1 trial and chosen there like any single filter; only its filter
and enhancement steps are carried forward.

### Image quality measurements

Phase 1 and the playground compare each image with the clean original using eight
measurements (↑ higher is better, ↓ lower is better):

| Measurement | What it tells you |
|---|---|
| PSNR (dB) ↑ | How close the image is to the original; the main number |
| SNR (dB) ↑ | Signal strength compared with the noise that is left |
| MSE ↓ | Average squared difference from the original (0–1 intensity scale) |
| RMSE ↓ | Square root of MSE, in pixel units |
| MAE ↓ | Average absolute difference; less affected by a few extreme pixels |
| SSIM ↑ | How well structure (shapes, edges, texture) is kept, 0–1 |
| Edge preservation (EPI) ↑ | How well cell edges survive, 0–1; low means edges were blurred |
| Correlation (NCC) ↑ | How closely the brightness pattern follows the original, −1 to 1 |

Choose which ones to display in the **Measurements to show** box, or tick **Show all**. They
are listed one per line under each image. Trials saved before a measurement existed show "–"
for it; save them again to fill it in.

### Phases 4–5 and Summary

| Phase | What runs | Judged by |
|---|---|---|
| 4 Feature extraction | Morphological, texture and colour features of each ALL-IDB2 cell (+ ResNet-50 / DenseNet-121 features optionally); each feature group is tested | ANOVA F-score per feature; cross-validated accuracy per group |
| 5 Classification | SVM, Random Forest, k-NN, SADMC on the chosen features | k-fold accuracy, sensitivity, specificity, F1, ROC-AUC, confusion matrices |
| Summary | The chosen pipeline and all conclusions; download a report; run the pipeline on one image | – |

**Reading the numbers.** PSNR only says how well an image was cleaned. Whether the whole
pipeline works is shown by the end result: Phase 3 cell separation and Phase 5 accuracy.

### Where results are stored

Everything goes to `results/` (never committed): per-trial CSVs with one row per image, the
before/after figures, `state.json` with your decisions and conclusions, `playground_chain.json`,
and the trained models. **Start the study again** in the sidebar clears it.

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
  Phase 3 reports "cells found / cells separated" instead of Dice and IoU. Put masks (same
  file name as the image) in a folder and set it on the Dataset page.
- **Over-segmentation.** "Cells separated" does not penalise cutting one cell into pieces;
  check "instances per image" and the images as well.
- **Comparable trials.** Only compare Phase 1 trials saved with the same noise type and
  strength, and trials measured on the same number of images.
- **Subtypes.** ALL-IDB2 is labelled normal vs lymphoblast only, so classification is
  binary. For L1/L2/L3 or B-ALL/T-ALL, set "Class folders" to a folder with one
  sub-folder of single-cell images per class.
- **AMIS and U-Net** are untrained. `python backend/train_segmentation.py --model amis
  --epochs 30 --pretrained` trains from the masks folder; a CUDA GPU is strongly recommended.
  `--pseudo-labels` trains on watershed output instead, which only teaches the network to
  imitate watershed.
- **Synthetic test images** (demo mode) are drawn shapes for checking the software.
  Never report them.

## Tests

    cd backend
    python -m pytest
