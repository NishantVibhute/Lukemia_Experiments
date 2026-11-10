# ALL Study App — how to run it

A research app that studies acute lymphoblastic leukaemia (ALL) detection in blood smear
images, one phase at a time: noise removal, contrast enhancement, cell segmentation,
feature extraction and classification. Research use only; it is not a diagnostic tool.

## What you need

- A Windows computer (Mac/Linux: see the end of this file)
- Python 3.11, 3.12 or 3.13 — download from https://www.python.org/downloads/
  **During installation, tick "Add python.exe to PATH".**
- Git — https://git-scm.com/download/win
- About 3 GB of free disk space and an internet connection for the first start

## Step 1 · Get the app

You need **Git** (https://git-scm.com/download/win, default options are fine). Open a terminal
(Command Prompt or PowerShell) in the folder where you want the app and run:

```
git clone https://github.com/NishantVibhute/Lukemia_Experiments.git
cd Lukemia_Experiments
```

**Getting updates later:** in that folder run `git pull`. Your dataset and saved results are
not touched, because they are never part of the repository.

## Step 2 · Add the dataset

The app uses the public **ALL-IDB** dataset (University of Milan). It is not included in
the download. Request it from the dataset owner at http://www.dti.unimi.it/fscotti/all/ (the address given in the dataset's own readme) and
accept their terms of use.

Put the downloaded folders inside the `data` folder of the app, so it looks like this:

```
ALL_Study_App\
└── data\
    ├── ALL_IDB1\   (108 .jpg smear images + .xyc files; sub-folders are fine)
    └── ALL_IDB2\   (260 .tif single-cell images)
```

The folder names only need to contain `IDB1` and `IDB2`. Keep the original file names.

No dataset yet? The app still runs in **demo mode** on synthetic test images, so you can
try every screen. Demo numbers are not real results.

## Step 3 · Start the app

Double-click **`start_app.bat`**.

- The **first start** installs everything (about 1 GB download, 5–15 minutes).
- After that it starts in a few seconds.
- Your browser opens at http://localhost:8501. If it doesn't, open that address yourself.
- Keep the black window open while you use the app. Close it to stop the app.

If Windows shows "Windows protected your PC", click **More info → Run anyway**.

## Step 4 · Use it

1. **Dataset** page (left sidebar): check that it found 108 + 260 images. If not, enter the
   dataset folder path and click **Save and scan**.
2. Work through **Phase 1 → 5** in order. In each phase:
   - Tab **①**: pick a method and its settings, look at BEFORE / AFTER, and click
     **Save this trial** for results worth keeping.
   - Tab **②**: see all saved trials ranked, pick the best and click **Confirm and continue**.
3. **Summary** shows the chosen pipeline and lets you download a report.

Hover the **?** next to any setting for an explanation, and open
**"What do these numbers mean?"** under the results.

Everything you save is stored in the `results` folder (tables as CSV, images as JPG).

## Problems

| What you see | What to do |
|---|---|
| "Python was not found" | Install Python and tick "Add python.exe to PATH", then restart the computer |
| Installation failed | Check the internet connection, delete the `.venv` folder, double-click `start_app.bat` again |
| Browser says "can't connect" | Wait 20 seconds and refresh; the first load is slow |
| Dataset page shows 0 images | Check the folder layout in Step 2, or paste the dataset path and click **Save and scan** |
| Want to start over | Click **Start the study again** in the sidebar (deletes saved trials) |

## Mac or Linux

Open a terminal in the app folder and run:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m streamlit run streamlit_app.py
```

Next time, only the last line is needed.
