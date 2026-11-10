"""ALL study, phase by phase.

    streamlit run streamlit_app.py

Phases 1-3 are trial based: pick a method, tune its settings, look at the
before/after images, save the trial; then compare all saved trials and carry
the best one into the next phase. Everything is saved under results/.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "backend"))

from app import config, datasets, deep, features, phases, preprocessing as pp, segmentation as seg  # noqa: E402
from app.classify import MODEL_NAMES  # noqa: E402
from app.imaging import decode_bytes, resize_max  # noqa: E402

RESULTS = ROOT / "results"
STATE = RESULTS / "state.json"
VERSION = 2
PHASES = ["Dataset", "1 · Noise removal", "2 · Contrast enhancement", "3 · Segmentation",
          "4 · Feature extraction", "5 · Classification", "Summary"]

st.set_page_config(page_title="ALL study", layout="wide")


# ------------------------------------------------------------------ state

def load_state():
    if "study" not in st.session_state:
        state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
        if state.get("version") != VERSION:  # results from an older layout cannot be reused
            keep = {k: state[k] for k in ("source", "n") if k in state}
            shutil.rmtree(RESULTS, ignore_errors=True)
            state = {"source": "dataset", "n": 10, **keep, "version": VERSION, "phases": {}}
        st.session_state.study = state
    return st.session_state.study


def save_state():
    RESULTS.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st.session_state.study, indent=2), encoding="utf-8")


def read_csv(path):
    if not path.exists():
        return None
    # Only empty cells are missing values; a method called "None" is a real label.
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def save_df(key, df):
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"{key}.csv", index=False)


def load_df(key):
    return read_csv(RESULTS / f"{key}.csv")


def save_figs(folder, images):
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for i, (name, img) in enumerate(images.items()):
        safe = "".join(c if c.isalnum() else "_" for c in name)[:60]
        cv2.imencode(".jpg", img)[1].tofile(str(folder / f"{i:02d}_{safe}.jpg"))
    (folder / "captions.json").write_text(json.dumps(list(images)), encoding="utf-8")


def show_figs(folder, columns=4):
    if not (folder / "captions.json").exists():
        return
    captions = json.loads((folder / "captions.json").read_text(encoding="utf-8"))
    cols = st.columns(columns)
    for i, (f, cap) in enumerate(zip(sorted(folder.glob("*.jpg")), captions)):
        cols[i % columns].image(str(f), caption=cap, use_container_width=True)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------- trials

def trials_file(phase):
    return RESULTS / f"phase{phase}_trials.json"


def trials(phase):
    f = trials_file(phase)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def trial_rows(phase):
    return read_csv(RESULTS / f"phase{phase}_trials.csv")


def add_trial(phase, spec, rows, figs, context=""):
    """Save one measured trial. A trial with the same settings replaces the old one."""
    RESULTS.mkdir(exist_ok=True)
    label = spec["label"] + (f" | {context}" if context else "")
    old = [t for t in trials(phase) if t["label"] != label]
    tid = max([t["id"] for t in trials(phase)], default=0) + 1
    trials_file(phase).write_text(json.dumps(old + [{"id": tid, "label": label, "spec": spec}], indent=2),
                                  encoding="utf-8")
    rows = rows.copy()
    rows.insert(0, "trial", label)
    prev = trial_rows(phase)
    if prev is not None:
        rows = pd.concat([prev[prev["trial"] != label], rows], ignore_index=True)
    rows.to_csv(RESULTS / f"phase{phase}_trials.csv", index=False)
    save_figs(RESULTS / "figures" / f"phase{phase}" / f"trial_{tid}", figs)


def remove_trials(phase, labels):
    kept = [t for t in trials(phase) if t["label"] not in labels]
    for t in trials(phase):
        if t["label"] in labels:
            shutil.rmtree(RESULTS / "figures" / f"phase{phase}" / f"trial_{t['id']}", ignore_errors=True)
    trials_file(phase).write_text(json.dumps(kept, indent=2), encoding="utf-8")
    rows = trial_rows(phase)
    if rows is not None:
        rows[~rows["trial"].isin(labels)].to_csv(RESULTS / f"phase{phase}_trials.csv", index=False)


def param_widgets(schema, key):
    """Render one control per tunable setting and return the chosen values."""
    params = {}
    if not schema:
        st.caption("This method has no settings to tune.")
        return params
    for col, (name, cfg) in zip(st.columns(len(schema)), schema.items()):
        title = name.replace("_", " ").capitalize()
        if cfg[0] == "select":
            params[name] = col.selectbox(title, cfg[1], index=cfg[1].index(cfg[2]), key=f"{key}_{name}",
                                         help=PARAM_HELP.get(name))
        else:
            _, lo, hi, default, step = cfg
            params[name] = col.slider(title, lo, hi, default, step, key=f"{key}_{name}", help=PARAM_HELP.get(name))
    return params


PARAM_HELP = {
    "kernel": "Size of the window the filter looks at. Bigger removes more noise but blurs small details.",
    "sigma": "How strongly the image is smoothed. Higher = smoother, but edges get softer.",
    "diameter": "Size of the neighbourhood used around each pixel.",
    "sigma_color": "How different two colours can be and still be averaged together. Higher = more smoothing.",
    "sigma_space": "How far apart two pixels can be and still affect each other.",
    "max_window": "Largest window the filter may grow to when the noise is heavy.",
    "low_percentile": "Darkest share of pixels pushed to black.",
    "high_percentile": "Brightest share of pixels pushed to white.",
    "clip_limit": "How much local contrast is boosted. Higher = stronger, but can exaggerate noise.",
    "tiles": "The image is split into this many tiles per side; contrast is adjusted in each one.",
    "offset": "How much darker than its surroundings a pixel must be to count as nucleus.",
    "clusters": "Number of colour groups the pixels are sorted into (nucleus, cytoplasm, background...).",
    "fuzziness": "How soft the group boundaries are. 2 is the usual value.",
    "min_distance": "Smallest allowed gap between two cell centres, in pixels. Too small splits one cell into pieces.",
    "marker_spacing": "Gap between cell centres, as a share of a typical nucleus radius. Lower splits more, higher merges more.",
    "min_peak": "How pronounced a centre must be to count as a separate cell. Higher = fewer splits.",
    "score": "How confident the network must be to keep a detection.",
}


def guide(*steps):
    """A short 'what to do on this page' box."""
    st.info("**What to do here**\n\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(steps, 1)))


def step(n, title):
    st.markdown(f"#### Step {n} · {title}")


def verdict(kind, text):
    {"good": st.success, "bad": st.error, "flat": st.warning}[kind](text)


def glossary(items):
    with st.expander("What do these numbers mean?"):
        for name, meaning in items.items():
            st.markdown(f"- **{name}** — {meaning}")


def trial_gallery(phase, columns=4):
    saved = trials(phase)
    pick = st.selectbox("Show the saved before/after images of", [t["label"] for t in saved], key=f"gallery{phase}")
    tid = next(t["id"] for t in saved if t["label"] == pick)
    show_figs(RESULTS / "figures" / f"phase{phase}" / f"trial_{tid}", columns)


def trial_manager(phase):
    with st.expander("Remove saved trials"):
        gone = st.multiselect("Trials to remove", [t["label"] for t in trials(phase)], key=f"remove{phase}")
        if st.button("Remove", key=f"removebtn{phase}", disabled=not gone):
            remove_trials(phase, gone)
            st.rerun()


# -------------------------------------------------------------- decisions

def chosen(phase):
    return study["phases"].get(str(phase), {}).get("choice")


def chosen_spec(phase):
    return study["phases"][str(phase)]["spec"]


def require(*needed):
    missing = [p for p in needed if not chosen(p)]
    if missing:
        st.warning(f"Finish phase {missing[0]} first: save trials there and confirm a choice. Later phases build on it.")
        st.stop()


def tie_note(scores, best, tol=1e-3):
    same = [str(k) for k, v in scores.items() if k != best and abs(v - scores[best]) < tol]
    if not same:
        return ""
    return (f"- {', '.join(same)} scored the same as {best} to three decimals, so this experiment does not "
            "separate them; the choice between them needs another criterion.\n")


def decision(phase, options, best, auto_text, specs=None, label="Carried into the next phase"):
    """Conclusion text, the choice to carry forward and the confirm button."""
    st.subheader("Conclusion")
    st.markdown(auto_text)
    saved = study["phases"].get(phase, {})
    notes = st.text_area("Your conclusion for the thesis (edit freely)", saved.get("conclusion", auto_text),
                         height=120, key=f"notes{phase}")
    default = saved.get("choice", best)
    choice = st.selectbox(label, options, index=options.index(default) if default in options else 0,
                          key=f"choice{phase}")
    if best and choice != best:
        st.caption(f"The measurements favour “{best}”. You can still carry “{choice}” forward; say why in the conclusion.")
    if st.button("Confirm and continue", type="primary", key=f"confirm{phase}"):
        if saved.get("choice") != choice:
            clear_after(int(phase))  # later phases were measured with the old choice
        study["phases"][phase] = {"choice": choice, "conclusion": notes, "spec": (specs or {}).get(choice)}
        save_state()
        st.success(f"Saved. “{choice}” is carried into the next phase.")


def clear_after(phase):
    """Drop the decisions, trials and results of every phase after `phase`."""
    for k in range(phase + 1, 6):
        study["phases"].pop(str(k), None)
        shutil.rmtree(RESULTS / "figures" / f"phase{k}", ignore_errors=True)
        for f in RESULTS.glob(f"phase{k}_*"):
            f.unlink()
    if phase < 4:
        (RESULTS / "features.npz").unlink(missing_ok=True)
    if phase < 5:
        (RESULTS / "models.joblib").unlink(missing_ok=True)


def summary(df, by, cols, digits=3):
    """(numeric means, 'mean ± std' text table) grouped by `by`."""
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(by, sort=False)[cols]
    mean, std = g.mean(), g.std(ddof=0)
    text = mean.copy().astype(object)
    for c in cols:
        text[c] = [f"{m:.{digits}f} ± {s:.{digits}f}" for m, s in zip(mean[c], std[c])]
    return mean, text


def synthetic_note():
    if study["source"] == "synthetic":
        st.warning("Demo mode: these are synthetic test images (drawn shapes). The numbers show that each "
                   "phase works; they are not results and must not go into the thesis.")


@st.cache_data(show_spinner="Loading images…")
def load_smears(source, n, root, masks):
    return phases.smears(config.load(), source, n)


def get_smears():
    return load_smears(study["source"], study["n"], settings["dataset_root"], settings["masks_dir"])


MANY = 10  # images shown at once in the "several images" view


def small(img, side=480):
    """Downscale for display so a grid of images stays fast."""
    return rgb(resize_max(img, side)[0])


def grid(cells, columns=4):
    """Show (image, caption) pairs in rows of `columns`."""
    for start in range(0, len(cells), columns):
        cols = st.columns(columns)
        for col, (im, cap) in zip(cols, cells[start:start + columns]):
            col.image(im, caption=cap, use_container_width=True)


def preview_picker(samples, key):
    names = [s[0] for s in samples]
    return samples[names.index(st.selectbox("Preview image", names, key=key))]


study = load_state()
settings = config.load()

# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("ALL study")
    st.caption("Overlapped-cell segmentation and ALL classification, one phase at a time")
    # Labels stay fixed: changing them would reset the selection on rerun.
    page = st.radio("Phase", PHASES, key="page")
    st.divider()
    st.markdown("**Decisions so far**")
    for k, name in {"1": "Denoising", "2": "Contrast", "3": "Segmentation", "4": "Features", "5": "Classifier"}.items():
        st.caption(f"{'✅' if chosen(k) else '⬜'} {name}: {chosen(k) or 'not decided'}")
    nxt = next((k for k in "12345" if not chosen(k)), None)
    st.info(f"Next: finish phase {nxt}" if nxt else "All phases done. Open the Summary.")
    st.divider()
    if st.button("Start the study again"):
        shutil.rmtree(RESULTS, ignore_errors=True)
        st.session_state.pop("study", None)
        st.rerun()
    st.caption("Research tool only. Not for diagnosis.")

# ---------------------------------------------------------------- dataset

if page == "Dataset":
    st.header("Dataset")
    st.write("Point the study at your copy of ALL-IDB. The folder is searched recursively for "
             "`ALL_IDB1` (whole smears) and `ALL_IDB2` (single cells).")
    with st.form("paths"):
        root = st.text_input("Dataset folder", settings["dataset_root"])
        masks = st.text_input("Ground-truth masks for ALL-IDB1 (optional; same file names as the images)",
                              settings["masks_dir"])
        classes = st.text_input("Class folders for subtype labels (optional; one sub-folder per class)",
                                settings["class_dir"])
        if st.form_submit_button("Save and scan"):
            settings = config.save({"dataset_root": root.strip('" '), "masks_dir": masks.strip('" '),
                                    "class_dir": classes.strip('" ')})
            datasets.rescan()
    info = datasets.summary(settings)
    c = st.columns(4)
    c[0].metric("ALL-IDB1 smears", info["idb1"])
    c[1].metric("… with centroid files", info["idb1_with_centroids"])
    c[2].metric("… with pixel masks", info["idb1_with_masks"])
    c[3].metric("ALL-IDB2 single cells", info["idb2"])
    cl = info["classification"]
    st.caption(f"Classifier labels from {cl['source']}: " +
               ", ".join(f"{n} {name}" for n, name in zip(cl["counts"], cl["classes"])))
    if not info["idb1_with_masks"]:
        st.info("No pixel masks: phase 3 will report how many annotated cells were found and separated, "
                "not Dice / IoU. Add a masks folder above to get those.")

    st.subheader("Images used in the experiments")
    have = info["idb1"] or info["idb2"]
    options = ["dataset", "synthetic"]
    src = st.radio("Source", options, index=options.index(study["source"]) if have else 1, horizontal=True,
                   format_func=lambda s: "My ALL-IDB images" if s == "dataset" else "Synthetic test images (demo mode)",
                   disabled=not have)
    n = st.number_input("Smear images per trial (phases 1–3)", 3, 200, study["n"])
    if (src, n) != (study["source"], study["n"]):
        study["source"], study["n"] = src, int(n)
        save_state()
    if not have:
        st.warning("No ALL-IDB images found, so the study runs in demo mode on synthetic images.")
    synthetic_note()
    for col, s in zip(st.columns(4), get_smears()[:4]):
        col.image(rgb(s[1]), caption=s[0], use_container_width=True)

# ---------------------------------------------------------------- phase 1

elif page.startswith("1"):
    st.header("Phase 1 · Noise removal")
    st.markdown("**Goal:** find the filter that cleans a noisy image best without blurring the cells.")
    guide("In tab ①, add noise to a clean image and pick a filter. Look at BEFORE and AFTER.",
          "When a result looks good, click **Save this trial**. Repeat for other filters or settings.",
          "In tab ②, see all saved trials ranked, pick the best one and click **Confirm and continue**.")
    synthetic_note()
    samples = get_smears()
    try_tab, cmp_tab = st.tabs(["① Try a filter and save it", "② Compare saved trials and decide"])

    with try_tab:
        step(1, "Choose an image and the noise to add")
        c = st.columns(4)
        with c[0]:
            name, img, *_ = preview_picker(samples, "p1img")
        kind = c[1].selectbox("Noise type", list(phases.NOISE), format_func=phases.NOISE.get)
        # Only the controls that affect the chosen noise type are shown.
        sp, sigma, parts = 0.02, 10.0, []
        if kind in ("salt_pepper", "mixed"):
            sp = c[2].slider("Salt-and-pepper density", 0.005, 0.10, 0.02, 0.005, format="%.3f")
            parts.append(f"density {sp:g}")
        if kind != "salt_pepper":
            title = {"gaussian": "Gaussian σ (0–255 scale)", "mixed": "Gaussian σ (0–255 scale)",
                     "speckle": "Speckle strength", "poisson": "Poisson strength"}[kind]
            sigma = c[3 if kind == "mixed" else 2].slider(title, 2.0, 40.0, 10.0, 1.0)
            parts.append(f"strength {sigma:g}")
        step(2, "Choose a filter and its settings")
        method = st.selectbox("Filter", list(phases.DENOISE_PARAMS))
        den = phases.spec(method, param_widgets(phases.DENOISE_PARAMS[method], f"p1{method}"))
        step(3, "Look at the result")

        noisy = pp.add_noise(img, kind, sp, sigma, seed=0)
        out = phases.denoise(noisy, den)
        q, base = pp.quality(img, out), pp.quality(img, noisy)
        zoom = st.toggle("Zoom in on a cell (noise is hard to see at full size)", True)
        box = phases.zoom_box(img) if zoom else (slice(None), slice(None))
        v = st.columns(3)
        v[0].image(rgb(img[box]), caption="Clean original", use_container_width=True)
        v[1].image(rgb(noisy[box]), caption=f"BEFORE · {phases.NOISE[kind]} noise", use_container_width=True)
        v[2].image(rgb(out[box]), caption=f"AFTER · {den['label']}", use_container_width=True)
        m = st.columns(4)
        m[0].metric("PSNR (dB)", f"{q['psnr']:.2f}", f"{q['psnr'] - base['psnr']:+.2f} vs noisy")
        m[1].metric("SNR (dB)", f"{q['snr']:.2f}", f"{q['snr'] - base['snr']:+.2f}")
        m[2].metric("MSE", f"{q['mse']:.5f}", f"{q['mse'] - base['mse']:+.5f}", delta_color="inverse")
        m[3].metric("SSIM", f"{q['ssim']:.3f}", f"{q['ssim'] - base['ssim']:+.3f}")
        gain = q["psnr"] - base["psnr"]
        if gain > 0.5:
            verdict("good", f"The filter improved this image: PSNR went from {base['psnr']:.1f} to {q['psnr']:.1f} dB "
                            f"(+{gain:.1f}). Higher PSNR means closer to the clean original.")
        elif gain < -0.5:
            verdict("bad", f"The filter made this image worse: PSNR fell from {base['psnr']:.1f} to {q['psnr']:.1f} dB. "
                           "It is probably blurring more than it cleans; try a smaller setting or another filter.")
        else:
            verdict("flat", f"The filter made almost no difference (PSNR {base['psnr']:.1f} → {q['psnr']:.1f} dB).")
        glossary({
            "PSNR (dB)": "how close the filtered image is to the clean original. **Higher is better.** This is the main number.",
            "SNR (dB)": "signal strength compared with remaining noise. Higher is better.",
            "MSE": "average squared error against the clean original. **Lower is better.**",
            "SSIM": "how well shapes and edges are preserved, from 0 to 1. Higher is better.",
            "Small number under each value": "the change compared with the noisy image, before filtering.",
        })
        if st.toggle(f"Show {min(MANY, len(samples))} images at once", False, key="many1",
                      help="Applies the same noise and filter to several images, so you can see whether "
                           "the result holds up beyond the one preview image."):
            cells, gains = [], []
            with st.spinner("Filtering…"):
                for i, (n_, im, *_r) in enumerate(samples[:MANY]):
                    nz = pp.add_noise(im, kind, sp, sigma, seed=i)  # same noise as when the trial is saved
                    out_ = phases.denoise(nz, den)
                    qa, qb = pp.quality(im, out_), pp.quality(im, nz)
                    gains.append(qa["psnr"] - qb["psnr"])
                    bx = phases.zoom_box(im) if zoom else (slice(None), slice(None))
                    cells += [(small(nz[bx]), f"{n_} · BEFORE · {qb['psnr']:.1f} dB"),
                              (small(out_[bx]), f"{n_} · AFTER · {qa['psnr']:.1f} dB")]
            better = sum(g > 0.5 for g in gains)
            verdict("good" if better == len(gains) else ("bad" if better == 0 else "flat"),
                    f"The filter improved {better} of {len(gains)} images; average PSNR change "
                    f"{np.mean(gains):+.1f} dB (lowest {min(gains):+.1f}, highest {max(gains):+.1f}).")
            st.caption("Each pair is one image: BEFORE (with noise) on the left, AFTER (filtered) on the right.")
            grid(cells, 4)
        step(4, "Save it if you want to compare it later")
        st.caption(f"The numbers above are for this one image. Saving repeats the test on {len(samples)} images and stores the average.")

        context = f"{phases.NOISE[kind]} noise, {', '.join(parts)}"
        if st.button(f"Save this trial (measure on {len(samples)} images)", type="primary"):
            with st.spinner("Measuring…"):
                add_trial(1, den, *phases.noise_trial(samples, den, [kind], sp, sigma), context)
            st.success(f"Saved: {den['label']} on {context}. Change the filter, its settings or the noise and save "
                       "again, or open the Compare tab.")
        with st.expander("Shortcut: let the app try every filter for me"):
            st.write(f"Tests all seven filters at their standard settings on **{context}** and saves each as a trial. "
                     "A quick way to see which filters are worth tuning.")
            if st.button("Try all seven filters"):
                bar = st.progress(0.0)
                for i, name_ in enumerate(phases.DENOISE_PARAMS):
                    d = phases.spec(name_, schema=phases.DENOISE_PARAMS)
                    add_trial(1, d, *phases.noise_trial(samples, d, [kind], sp, sigma), context)
                    bar.progress((i + 1) / len(phases.DENOISE_PARAMS))
                st.success("Done. Open tab ② to see them ranked.")

    with cmp_tab:
        rows = trial_rows(1)
        if rows is None or rows.empty:
            st.info("No saved trials yet. Save some in the first tab.")
        else:
            rows["gain"] = rows["psnr"] - rows["psnr_noisy"]
            st.subheader(f"{rows['trial'].nunique()} saved trials · {rows['image'].nunique()} images each")
            mean, text = summary(rows, "trial", ["psnr", "gain", "snr", "mse", "ssim"])
            text["mse"] = [f"{x:.5f}" for x in mean["mse"]]
            order = mean["psnr"].sort_values(ascending=False).index
            st.success(f"🏆 Best so far: **{order[0]}** — average PSNR {mean.loc[order[0], 'psnr']:.2f} dB, "
                       f"{mean.loc[order[0], 'gain']:.2f} dB better than the noisy images.")
            st.caption("Ranked best first. Each value is the average over the images ± how much it varied.")
            detail = st.toggle("Show all measurements", False, key="detail1")
            keep = ["psnr", "gain", "snr", "mse", "ssim"] if detail else ["psnr", "gain", "ssim"]
            st.dataframe(text.loc[order, keep].rename(columns={
                "psnr": "PSNR (dB) ↑", "gain": "Improvement over noisy (dB) ↑", "snr": "SNR (dB) ↑",
                "mse": "MSE ↓", "ssim": "SSIM ↑"}), use_container_width=True)
            wide = rows.pivot_table(index="trial", columns="noise", values="psnr", sort=False).loc[order]
            st.markdown("**PSNR of each trial (longer bar = better)**")
            st.bar_chart(wide, horizontal=True, stack=False)
            if detail:
                st.markdown("**PSNR by trial and noise type (dB)**")
                st.dataframe(wide.style.format("{:.2f}").highlight_max(axis=0), use_container_width=True)
            trial_gallery(1)
            trial_manager(1)

            best = order[0]
            per_noise = "; ".join(f"{n}: {wide[n].idxmax().split(' | ')[0]} ({wide[n].max():.2f} dB)" for n in wide.columns)
            text = (f"- Across the noise types measured, **{best.split(' | ')[0]}** gave the highest PSNR "
                    f"(**{mean.loc[best, 'psnr']:.2f} dB**, a gain of {mean.loc[best, 'gain']:.2f} dB over the "
                    f"unfiltered noisy image; SSIM {mean.loc[best, 'ssim']:.3f}).\n"
                    f"- Best trial per noise type — {per_noise}.\n")
            groups = [g["psnr"].values for _, g in rows.groupby("trial", sort=False)]
            if len(groups) > 1 and min(map(len, groups)) >= 3:
                f, p = stats.f_oneway(*groups)
                text += (f"- The trials differ {'significantly' if p < 0.05 else 'but not significantly'} in PSNR "
                         f"(one-way ANOVA, F = {f:.2f}, p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}).\n")
            if rows["noise"].nunique() > 1:
                text += ("- The trials cover more than one noise type. PSNR is only comparable between trials saved "
                         "with the same noise type and strength; use the per-noise table above for that.\n")
            text += tie_note(mean["psnr"], best)
            specs = {t["label"]: t["spec"] for t in trials(1)}
            decision("1", list(order), best, text, specs, label="Filter and settings carried into phase 2")

# ---------------------------------------------------------------- phase 2

elif page.startswith("2"):
    st.header("Phase 2 · Contrast enhancement")
    require(1)
    den = chosen_spec(1)
    st.markdown(f"**Goal:** make the cell nuclei stand out more clearly, after denoising with **{den['label']}**.")
    guide("In tab ①, pick an enhancement method and look at BEFORE and AFTER.",
          "Save the ones worth comparing. Also save **None** once, so you have a baseline.",
          "In tab ②, pick the best one and click **Confirm and continue**.")
    synthetic_note()
    samples = get_smears()
    try_tab, cmp_tab = st.tabs(["① Try a method and save it", "② Compare saved trials and decide"])

    with try_tab:
        step(1, "Choose an image and an enhancement method")
        c = st.columns(2)
        with c[0]:
            name, img, truth, *_ = preview_picker(samples, "p2img")
        method = c[1].selectbox("Enhancement method", list(phases.ENHANCE_PARAMS))
        enh = phases.spec(method, param_widgets(phases.ENHANCE_PARAMS[method], f"p2{method}"))
        before = phases.denoise(img, den)
        after = phases.enhance(before, enh)
        step(2, "Look at the result")
        view = st.radio("Show", ["Image", "Stain map (what segmentation sees)"], horizontal=True)
        show = (lambda im: rgb(im)) if view == "Image" else seg.stain_map
        v = st.columns(2)
        v[0].image(show(before), caption="BEFORE · denoised only", use_container_width=True)
        v[1].image(show(after), caption=f"AFTER · {enh['label']}", use_container_width=True)
        a, b0 = phases.contrast_scores(after, truth), phases.contrast_scores(before, truth)
        titles = {"rms_contrast": "RMS contrast", "entropy": "Entropy (bits)", "separability": "Nucleus separability",
                  "dice_otsu": "Dice (Otsu)"}
        for col, k in zip(st.columns(len(a)), a):
            col.metric(titles[k], f"{a[k]:.3f}", f"{a[k] - b0[k]:+.3f} vs before")
        main = "dice_otsu" if "dice_otsu" in a else "separability"
        change = a[main] - b0[main]
        what = "segmentation accuracy (Dice)" if main == "dice_otsu" else "nucleus separability"
        if change > 0.005:
            verdict("good", f"Nuclei are easier to pick out after this enhancement: {what} rose from "
                            f"{b0[main]:.3f} to {a[main]:.3f}.")
        elif change < -0.005:
            verdict("bad", f"Nuclei are harder to pick out after this enhancement: {what} fell from "
                           f"{b0[main]:.3f} to {a[main]:.3f}. The image may look sharper, but segmentation gets harder.")
        else:
            verdict("flat", f"This enhancement changes how the image looks but not how easily nuclei separate "
                            f"({what} {b0[main]:.3f} → {a[main]:.3f}).")
        glossary({
            "Nucleus separability": "how cleanly nuclei stand apart from everything else, from 0 to 1. "
                                    "**Higher is better. This is the number that matters for the next phase.**",
            "RMS contrast": "overall contrast of the image. Higher looks punchier, but is not automatically better.",
            "Entropy (bits)": "amount of detail in the image. Higher means more variation, including noise.",
            "Dice (Otsu)": "only shown when you have ground-truth masks: how well a simple segmentation matches them.",
        })
        if st.toggle(f"Show {min(MANY, len(samples))} images at once", False, key="many2",
                      help="Applies the same enhancement to several images."):
            cells, changes = [], []
            with st.spinner("Enhancing…"):
                for n_, im, tr, *_r in samples[:MANY]:
                    b_ = phases.denoise(im, den)
                    a_ = phases.enhance(b_, enh)
                    sb, sa = phases.separability(b_), phases.separability(a_)
                    changes.append(sa - sb)
                    cells += [(show(resize_max(b_, 480)[0]), f"{n_} · BEFORE · separability {sb:.3f}"),
                              (show(resize_max(a_, 480)[0]), f"{n_} · AFTER · separability {sa:.3f}")]
            better = sum(c_ > 0.005 for c_ in changes)
            verdict("good" if better == len(changes) else ("bad" if better == 0 else "flat"),
                    f"Nuclei became easier to pick out in {better} of {len(changes)} images; average separability "
                    f"change {np.mean(changes):+.3f}.")
            st.caption("Each pair is one image: BEFORE (denoised only) on the left, AFTER on the right.")
            grid(cells, 4)
        step(3, "Save it if you want to compare it later")
        st.caption(f"The numbers above are for this one image. Saving repeats the test on {len(samples)} images and stores the average.")
        if st.button(f"Save this trial (measure on {len(samples)} images)", type="primary"):
            with st.spinner("Measuring…"):
                add_trial(2, enh, *phases.contrast_trial(samples, den, enh))
            st.success(f"Saved: {enh['label']}. Try another method or setting, or open the Compare tab.")

    with cmp_tab:
        rows = trial_rows(2)
        if rows is None or rows.empty:
            st.info("No saved trials yet. Save some in the first tab.")
        else:
            key = "dice_otsu" if "dice_otsu" in rows and rows["dice_otsu"].notna().all() else "separability"
            st.subheader(f"{rows['trial'].nunique()} saved trials · {rows['image'].nunique()} images each")
            mean, text = summary(rows, "trial", ["rms_contrast", "entropy", "separability", "dice_otsu"])
            order = mean[key].sort_values(ascending=False).index
            st.success(f"🏆 Best so far: **{order[0]}** — {'Dice' if key == 'dice_otsu' else 'nucleus separability'} "
                       f"{mean.loc[order[0], key]:.3f}.")
            st.caption("Ranked best first. Each value is the average over the images ± how much it varied.")
            detail = st.toggle("Show all measurements", False, key="detail2")
            keep = list(text.columns) if detail else [key]
            st.dataframe(text.loc[order, keep].rename(columns={
                "rms_contrast": "RMS contrast", "entropy": "Entropy (bits)", "separability": "Nucleus separability ↑",
                "dice_otsu": "Dice (Otsu) ↑"}), use_container_width=True)
            st.markdown(f"**{'Dice of Otsu segmentation' if key == 'dice_otsu' else 'Nucleus separability'} by trial**")
            st.bar_chart(mean.loc[order, [key]], horizontal=True)
            trial_gallery(2, columns=2)
            trial_manager(2)
            best = order[0]
            top_contrast = mean["rms_contrast"].idxmax()
            text = (f"- Judged by {'Dice against the ground-truth masks' if key == 'dice_otsu' else 'nucleus separability'}, "
                    f"**{best}** scored highest (**{mean.loc[best, key]:.3f}**).\n")
            if "None" in mean.index:
                text += f"- With no enhancement the score was {mean.loc['None', key]:.3f}.\n"
            else:
                text += "- No trial without enhancement was saved, so there is no baseline; save “None” to get one.\n"
            text += (f"- {top_contrast} produced the highest raw contrast ({mean.loc[top_contrast, 'rms_contrast']:.3f}); "
                     "high global contrast alone is not the goal, because it also amplifies background and red cells.\n")
            if key == "separability":
                text += "- No pixel masks were available, so this ranking uses separability rather than segmentation accuracy.\n"
            text += tie_note(mean[key], best)
            specs = {t["label"]: t["spec"] for t in trials(2)}
            decision("2", list(order), best, text, specs, label="Enhancement and settings carried into phase 3")

# ---------------------------------------------------------------- phase 3

elif page.startswith("3"):
    st.header("Phase 3 · Segmentation of overlapped cells")
    require(1, 2)
    den, enh = chosen_spec(1), chosen_spec(2)
    st.markdown("**Goal:** outline every white-cell nucleus, and split cells that touch or overlap into separate ones.")
    st.caption(f"Images are first prepared with your earlier choices: {den['label']}, then {enh['label']}.")
    guide("In tab ①, pick a segmentation method. Each coloured outline in AFTER is one cell the method found.",
          "Check the touching cells: one outline around two cells means they were **not** separated.",
          "Save the methods worth comparing, then pick the best in tab ② and click **Confirm and continue**.")
    synthetic_note()
    weights = deep.available()
    available = list(seg.CLASSICAL) + [m for m in seg.DEEP if weights[m]]
    missing = [seg.METHODS[m] for m in seg.DEEP if not weights[m]]
    if missing:
        st.info(f"{' and '.join(missing)} cannot be tried yet: no trained weights in backend/models/. "
                "Train with `python backend/train_segmentation.py --model amis` (needs pixel masks and ideally a GPU).")
    samples = get_smears()
    try_tab, cmp_tab = st.tabs(["① Try a method and save it", "② Compare saved trials and decide"])

    with try_tab:
        step(1, "Choose an image and a segmentation method")
        c = st.columns(2)
        with c[0]:
            name, img, truth, pts, true_count = preview_picker(samples, "p3img")
        method = c[1].selectbox("Segmentation method", available, format_func=seg.METHODS.get,
                                index=available.index("marker_watershed"))
        sg = phases.spec(method, param_widgets(phases.SEG_PARAMS[method], f"p3{method}"))
        work = phases.preprocess(img, den, enh)
        labels = phases.segment(work, sg)
        step(2, "Look at the result")
        v = st.columns(2)
        v[0].image(rgb(work), caption="BEFORE · preprocessed image", use_container_width=True)
        v[1].image(rgb(seg.overlay(work, labels, thickness=3)), caption=f"AFTER · {sg['label']}", use_container_width=True)
        sc = phases.segmentation_scores(labels, truth, pts, true_count)
        titles = {"instances": "Nuclei found", "detected": "Annotated cells found", "separated": "Annotated cells separated",
                  "dice": "Dice", "iou": "IoU", "count_error": "Count error"}
        shown = [k for k in titles if k in sc]
        for col, k in zip(st.columns(len(shown)), shown):
            col.metric(titles[k], f"{sc[k]:.0%}" if k in ("detected", "separated") else
                       (f"{sc[k]:.3f}" if k in ("dice", "iou") else sc[k]))
        if pts:
            n_sep, n_found = round(sc["separated"] * len(pts)), round(sc["detected"] * len(pts))
            msg = (f"The dataset marks {len(pts)} leukaemia cells in this image. The method found {n_found} of them "
                   f"and gave {n_sep} their own outline. It drew {sc['instances']} outlines in total.")
            if sc["separated"] >= 0.9 and sc["instances"] <= 2.5 * len(pts):
                verdict("good", msg)
            elif sc["instances"] > 2.5 * len(pts):
                verdict("flat", msg + " That is far more outlines than cells, so it may be cutting cells into pieces "
                                      "or outlining other things. Check the image.")
            else:
                verdict("bad", msg + " Cells that share an outline were merged, which is the overlap problem.")
        else:
            verdict("flat", f"The method drew {sc['instances']} outlines. This image has no marked cells to check "
                            "against (healthy smear), so judge by eye or pick an image whose name ends in _1.")
        glossary({
            "Nuclei found": "number of outlines the method drew.",
            "Annotated cells found": "share of the dataset's marked leukaemia cells that fall inside an outline. Higher is better.",
            "Annotated cells separated": "share of marked cells that got an outline of their own, not shared with a "
                                         "neighbour. **Higher is better. This measures overlap handling.**",
            "Dice / IoU": "only shown when you have ground-truth masks: overlap between the outlines and the true cell area.",
        })
        if st.toggle(f"Show {min(MANY, len(samples))} images at once", False, key="many3",
                      help="Runs the same segmentation on several images."):
            cells, marked, separated = [], 0, 0
            with st.spinner("Segmenting…"):
                for n_, im, tr, pts_, tc in samples[:MANY]:
                    w_ = phases.preprocess(im, den, enh)
                    lab_ = phases.segment(w_, sg)
                    sc_ = phases.segmentation_scores(lab_, tr, pts_, tc)
                    cap = f"{n_} · {sc_['instances']} outlines"
                    if pts_:
                        marked += len(pts_)
                        separated += round(sc_["separated"] * len(pts_))
                        cap += f" · {round(sc_['separated'] * len(pts_))}/{len(pts_)} marked cells separated"
                    cells.append((small(seg.overlay(w_, lab_, thickness=3)), cap))
            if marked:
                verdict("good" if separated / marked >= 0.9 else "flat",
                        f"Across these images the method gave {separated} of {marked} marked leukaemia cells "
                        f"their own outline ({separated / marked:.0%}).")
            else:
                verdict("flat", "None of these images have marked cells to check against, so judge by eye.")
            grid(cells, 3)
        step(3, "Save it if you want to compare it later")
        st.caption(f"The numbers above are for this one image. Saving repeats the test on {len(samples)} images and stores the average.")
        if st.button(f"Save this trial (measure on {len(samples)} images)", type="primary"):
            with st.spinner("Segmenting…"):
                add_trial(3, sg, *phases.segmentation_trial(samples, den, enh, sg))
            st.success(f"Saved: {sg['label']}. Try another method or setting, or open the Compare tab.")

    with cmp_tab:
        rows = trial_rows(3)
        if rows is None or rows.empty:
            st.info("No saved trials yet. Save some in the first tab.")
        else:
            has_dice = "dice" in rows and rows["dice"].notna().any()
            has_pts = "separated" in rows and rows["separated"].notna().any()
            pre = rows[rows["condition"] == "Preprocessed"]
            st.subheader(f"{rows['trial'].nunique()} saved trials · {rows['image'].nunique()} images each · preprocessed images")
            cols = ["dice", "iou", "precision", "recall", "accuracy", "detected", "separated", "count_error", "instances", "seconds"]
            mean, text = summary(pre, "trial", [c_ for c_ in cols if c_ in pre and pre[c_].notna().any()])
            key = "separated" if has_pts else ("dice" if has_dice else None)
            order = (mean.sort_values([key] + (["dice"] if has_dice and key != "dice" else []), ascending=False).index
                     if key else mean.index)
            st.dataframe(text.loc[order].rename(columns={
                "dice": "Dice", "iou": "IoU (Jaccard)", "precision": "Precision", "recall": "Recall",
                "accuracy": "Pixel accuracy", "detected": "Cells found", "separated": "Cells separated",
                "count_error": "Count error", "instances": "Instances / image", "seconds": "Seconds / image"}),
                use_container_width=True)
            if not has_dice:
                st.info("No pixel masks for these images, so Dice / IoU are not available. “Cells found” is the share of "
                        "annotated blast centroids inside a segmented nucleus; “cells separated” is the share that got an "
                        "instance of their own. Neither penalises a method for cutting one cell into pieces, so also "
                        "check “instances / image” against the images." if has_pts else
                        "No ground truth for these images: only instance counts and timings can be shown, so the "
                        "measurements cannot rank the methods. Judge from the images.")
            for k, title in (("dice", "Dice"), ("separated", "Share of annotated cells separated")):
                if k in rows and rows[k].notna().any():
                    st.markdown(f"**{title}: raw image vs preprocessed**")
                    st.bar_chart(rows.pivot_table(index="trial", columns="condition", values=k, sort=False).loc[order],
                                 horizontal=True, stack=False)
            trial_gallery(3, columns=2)
            trial_manager(3)

            best, text = None, "- Without ground truth the measurements cannot pick a winner.\n"
            if key:
                best = order[0]
                text = f"- **{best}** performed best"
                if has_pts:
                    text += f", separating **{mean.loc[best, 'separated']:.1%}** of annotated cells into their own instance"
                if has_dice:
                    text += f" with Dice **{mean.loc[best, 'dice']:.3f}** and IoU {mean.loc[best, 'iou']:.3f}"
                text += f" ({mean.loc[best, 'instances']:.1f} instances per image).\n"
                if has_pts and len(mean) > 1:
                    worst = mean["separated"].idxmin()
                    text += f"- The weakest at separating cells was {worst} ({mean.loc[worst, 'separated']:.1%}).\n"
                raw = rows[rows["condition"] == "Raw image"].set_index(["trial", "image"])[key]
                diff = (pre.set_index(["trial", "image"])[key] - raw).loc[best].dropna()
                if len(diff) >= 3 and diff.std() > 0:
                    t, p = stats.ttest_1samp(diff, 0)
                    text += (f"- Preprocessing changed its score by {diff.mean():+.3f} "
                             f"(paired t-test, t = {t:.2f}, p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}).\n")
                else:
                    text += f"- Preprocessing changed its score by {diff.mean():+.3f}.\n"
                text += tie_note(mean[key], best)
            specs = {t["label"]: t["spec"] for t in trials(3)}
            decision("3", list(order), best, text, specs, label="Segmentation method and settings carried into phase 4")

# ---------------------------------------------------------------- phase 4

elif page.startswith("4"):
    st.header("Phase 4 · Feature extraction")
    require(1, 2, 3)
    den, enh, sg = chosen_spec(1), chosen_spec(2), chosen_spec(3)
    st.markdown(
        f"**Question:** which features describe a cell well enough to tell the classes apart?  \n"
        f"**Method:** each single-cell image is preprocessed (**{den['label']}**, **{enh['label']}**), segmented with "
        f"**{sg['label']}**, and described by morphological, texture and colour features. Each feature is "
        "ranked by its ANOVA F-score between classes, and each feature group is tested with a Random Forest "
        "under 5-fold cross-validation.")
    synthetic_note()
    use_deep = st.checkbox("Also extract deep features (ResNet-50 + DenseNet-121). Downloads about 130 MB of "
                           "ImageNet weights the first time and takes several minutes on CPU.")
    if st.button("Extract features and test every feature group", type="primary"):
        with st.spinner("Extracting features…"):
            X, y, names, src, skipped = phases.cell_dataset(settings, study["source"], den, enh, sg, use_deep)
        if len(X) < 10:
            st.error("Too few single-cell images with a detected nucleus. Check the dataset folder (ALL-IDB2).")
            st.stop()
        RESULTS.mkdir(exist_ok=True)
        np.savez(RESULTS / "features.npz", X=X, y=y, names=np.array(names))
        table = pd.DataFrame(X[:, :len(features.NAMES)], columns=features.NAMES)
        table.insert(0, "class", [names[i] for i in y])
        save_df("phase4_features", table)
        save_df("phase4_feature_ranking", phases.feature_ranking(X, y))
        with st.spinner("Cross-validating feature groups…"):
            save_df("phase4_feature_sets", phases.feature_set_experiment(X, y, len(names)))
        study["phase4_info"] = {"source": src, "skipped": int(skipped), "n": int(len(X))}
        save_state()
    table, ranking, sets = load_df("phase4_features"), load_df("phase4_feature_ranking"), load_df("phase4_feature_sets")
    if sets is not None:
        info = study.get("phase4_info", {})
        st.subheader(f"Results · {info.get('n', len(table))} cells from {info.get('source', '')}")
        if info.get("skipped"):
            st.caption(f"{info['skipped']} image(s) skipped because no nucleus was found.")
        st.markdown("**Mean feature value per class**")
        st.dataframe(table.groupby("class").mean().T.style.format("{:.3f}"), use_container_width=True)
        a, b = st.columns(2)
        a.markdown("**Features ranked by ANOVA F-score**")
        a.dataframe(ranking.style.format({"F": "{:.1f}", "p": "{:.2g}"}), use_container_width=True, hide_index=True)
        mean, text = summary(sets, "feature_set", ["accuracy"])
        text.insert(0, "features", sets.groupby("feature_set", sort=False)["n_features"].first())
        b.markdown("**Accuracy by feature group (Random Forest, cross-validated)**")
        b.dataframe(text.rename(columns={"accuracy": "Accuracy"}), use_container_width=True)
        b.bar_chart(mean, horizontal=True)
        best = mean["accuracy"].idxmax()
        top = ranking.head(3)["feature"].tolist()
        concl = (f"- **{best}** gave the highest cross-validated accuracy (**{mean.loc[best, 'accuracy']:.1%}**).\n"
                 f"- The single most discriminative features were {', '.join(top)}.\n"
                 f"- Per group: " + "; ".join(f"{k} {v:.1%}" for k, v in mean["accuracy"].items()) + ".\n")
        concl += tie_note(mean["accuracy"], best)
        decision("4", list(mean.index), best, concl, label="Feature set carried into phase 5")
        st.download_button("Download the feature table (CSV)", table.to_csv(index=False), "features.csv")

# ---------------------------------------------------------------- phase 5

elif page.startswith("5"):
    st.header("Phase 5 · Classification")
    require(1, 2, 3, 4)
    if not (RESULTS / "features.npz").exists():
        st.warning("Run phase 4 again to rebuild the feature matrix.")
        st.stop()
    data = np.load(RESULTS / "features.npz")
    X, y, names = data["X"], data["y"], list(data["names"])
    st.markdown(
        f"**Question:** which classifier separates {' / '.join(names)} best on the **{chosen(4)}** features?  \n"
        "**Method:** stratified k-fold cross-validation with identical folds for every model. "
        "SADMC is the proposed classifier (multi-scale convolutions + squeeze-and-excitation attention).")
    synthetic_note()
    if len(names) == 2:
        st.info("The labels are binary (normal vs lymphoblast). Subtype classification (L1/L2/L3, B-ALL/T-ALL) "
                "needs a dataset with those labels, set as “Class folders” on the Dataset page.")
    c = st.columns([3, 1])
    models = c[0].multiselect("Classifiers", list(MODEL_NAMES), list(MODEL_NAMES), format_func=MODEL_NAMES.get)
    folds = c[1].number_input("Folds", 2, 10, 5)
    if st.button("Train and cross-validate all classifiers", type="primary", disabled=not models):
        with st.spinner("Training… SADMC takes the longest."):
            df, confusion, fitted, cols, k = phases.classification_experiment(X, y, len(names), chosen(4), models, folds)
        save_df("phase5_classification", df)
        joblib.dump({"models": fitted, "cols": cols, "classes": names}, RESULTS / "models.joblib")
        study["phase5_info"] = {"confusion": confusion, "folds": int(k), "n": int(len(X))}
        save_state()
    df = load_df("phase5_classification")
    if df is not None:
        info = study.get("phase5_info", {})
        st.subheader(f"Results · {info.get('folds')}-fold cross-validation on {info.get('n')} cells")
        metrics = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]
        text = pd.DataFrame({m: [f"{a:.3f} ± {s:.3f}" for a, s in zip(df[m], df[m + "_std"])] for m in metrics},
                            index=df["model"])
        st.dataframe(text.rename(columns={"accuracy": "Accuracy", "precision": "Precision", "recall": "Sensitivity",
                                          "specificity": "Specificity", "f1": "F1", "auc": "ROC-AUC"}),
                     use_container_width=True)
        st.markdown("**Accuracy by classifier**")
        st.bar_chart(df.set_index("model")[["accuracy"]], horizontal=True)
        st.markdown("**Confusion matrices** (all folds pooled; rows = true class, columns = predicted)")
        for col, (model, cm) in zip(st.columns(len(info.get("confusion", {})) or 1), info.get("confusion", {}).items()):
            col.caption(model)
            col.dataframe(pd.DataFrame(cm, index=names, columns=names))
        ranked = df.sort_values(["accuracy", "f1"], ascending=False)
        best, top = ranked.iloc[0]["model"], ranked.iloc[0]
        concl = (f"- **{best}** achieved the highest accuracy: **{top['accuracy']:.1%} ± {top['accuracy_std']:.1%}** "
                 f"(sensitivity {top['recall']:.3f}, specificity {top['specificity']:.3f}, F1 {top['f1']:.3f}).\n"
                 "- All models: " + "; ".join(f"{r['model']} {r['accuracy']:.1%}" for _, r in ranked.iterrows()) + ".\n")
        if len(ranked) > 1 and ranked.iloc[0]["accuracy"] - ranked.iloc[1]["accuracy"] < ranked.iloc[0]["accuracy_std"]:
            concl += ("- The gap to the runner-up is smaller than the fold-to-fold spread, so the ranking between "
                      "them is not conclusive on this dataset.\n")
        concl += tie_note(df.set_index("model")["accuracy"], best)
        decision("5", list(df["model"]), best, concl, label="Final classifier")

# ---------------------------------------------------------------- summary

else:
    st.header("Summary · the pipeline this study arrived at")
    synthetic_note()
    titles = {"1": "Noise removal", "2": "Contrast enhancement", "3": "Segmentation",
              "4": "Feature extraction", "5": "Classification"}
    report = ["# ALL study: phase-by-phase results\n"]
    if study["source"] == "synthetic":
        report.append("> Demo run on synthetic test images. Not thesis results.\n")
    for k, title in titles.items():
        ph = study["phases"].get(k)
        st.subheader(f"Phase {k} · {title}")
        if not ph:
            st.caption("Not completed yet.")
            continue
        st.markdown(f"**Chosen: {ph['choice']}**")
        st.markdown(ph["conclusion"])
        report += [f"## Phase {k}: {title}\n", f"**Chosen:** {ph['choice']}\n", ph["conclusion"], ""]
    if all(chosen(k) for k in titles):
        st.success(f"Final pipeline: {chosen(1)} → {chosen(2)} → {chosen(3)} → {chosen(4)} features → {chosen(5)}")
    st.download_button("Download the report (Markdown)", "\n".join(report), "all_study_report.md")
    st.caption(f"Raw tables (CSV) and figures for every phase are in {RESULTS}")

    st.subheader("Try the chosen pipeline on one image")
    require(1, 2, 3)
    up = st.file_uploader("Blood smear image", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
    demo = st.button("Use a sample image instead")
    img = None
    if up is not None:
        img = decode_bytes(up.getvalue())
    elif demo:
        img = get_smears()[0][1]
    if img is not None:
        model = cols = classes = None
        if chosen(5) and (RESULTS / "models.joblib").exists():
            bundle = joblib.load(RESULTS / "models.joblib")
            model, cols, classes = bundle["models"].get(chosen(5)), bundle["cols"], bundle["classes"]
        with st.spinner("Running the pipeline…"):
            out = phases.run_pipeline(img, chosen_spec(1), chosen_spec(2), chosen_spec(3), model, cols, classes)
        c = st.columns(3)
        c[0].image(rgb(out["original"]), caption="Original", use_container_width=True)
        c[1].image(rgb(out["enhanced"]), caption="After denoising and enhancement", use_container_width=True)
        c[2].image(rgb(out["overlay"]), caption=f"{len(out['cells'])} nuclei"
                   + (" (green = first class, red = second)" if model is not None else ""), use_container_width=True)
        if "prediction" in out["cells"]:
            st.write(out["cells"]["prediction"].value_counts().rename("cells").to_frame().T)
        st.dataframe(out["cells"], use_container_width=True, hide_index=True)
