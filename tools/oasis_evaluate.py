r"""
oasis_evaluate.py - cross-dataset evaluation: ADNI-trained v3 models tested on OASIS-1.

Run from the project folder:
    python tools\oasis_evaluate.py --check     # 1) picture of OASIS slices after re-orientation (no evaluation)
    python tools\oasis_evaluate.py             # 2) full evaluation (after the picture is confirmed)

NO training happens. The saved v3 models (supervised, SSL fine-tuned, SSL linear probe, per run)
are loaded from outputs/checkpoints and evaluated on:
  * the ADNI test split of their own run (recomputed with the same code, plus ROC-AUC), and
  * all OASIS-1 subjects aged >= 60 with a CDR score (first scan MR1 only),
using exactly the same slice selection, normalisation and input transform as in training.

OASIS labels: CDR 0 -> CN, CDR 0.5 -> MCI (approximation: "very mild dementia"), CDR >= 1 -> AD.
OASIS images: RAW/<ID>_mpr-1_anon (single raw MPRAGE, like ADNI), sagittal along array axis 2,
1.25 mm slices -> resampled to 1 mm; in-plane re-orientation OASIS_ORIENTATION (checked visually).

Writes NEW files only:
    outputs/plots/oasis_orientation_final_check.png        (--check)
    outputs/results/oasis_eval_v3_<timestamp>.json          (evaluation)
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scipy.ndimage import zoom
from scipy.stats import rankdata

import config
from dataset import MRIDataset, _normalize, extract_2d_slices, LABEL_MAP

OASIS = ROOT / "data" / "raw" / "OASIS"
VERSION = "v3_f1select_reg_probe"
OASIS_ORIENTATION = (1, True)     # rotate 90 degrees (k=1), then mirror left-right - chosen visually 18 Oct 2026
CLASSES = ["CN", "MCI", "AD"]


# =============================================================================
# OASIS dataset with the same interface as MRIDataset
# =============================================================================
def cdr_to_diagnosis(cdr):
    if cdr == 0:
        return "CN"
    if cdr == 0.5:
        return "MCI"
    if cdr >= 1:
        return "AD"
    return None


class OASISDataset:
    def __init__(self, min_age: float = 60):
        import pandas as pd
        df = pd.read_excel(OASIS / "oasis_cross-sectional.xlsx")
        df = df[df["ID"].astype(str).str.endswith("_MR1") & (df["Age"] >= min_age) & df["CDR"].notna()]
        raw_files = {p.name.replace("_mpr-1_anon.hdr", ""): p
                     for p in (OASIS / "extracted").rglob("*_mpr-1_anon.hdr")}
        self.subjects = {}
        for _, r in df.iterrows():
            sid, dx = str(r["ID"]), cdr_to_diagnosis(float(r["CDR"]))
            if dx is None or sid not in raw_files:
                continue
            self.subjects[sid] = {"subject_id": sid, "diagnosis": dx, "label": LABEL_MAP[dx],
                                  "cdr": float(r["CDR"]), "age": float(r["Age"]),
                                  "path": str(raw_files[sid]), "dataset": "OASIS"}
        counts = {c: sum(1 for s in self.subjects.values() if s["diagnosis"] == c) for c in CLASSES}
        print(f"[OASIS] {len(self.subjects)} subjects (age >= {min_age}, MR1, with CDR): {counts}")

    def get_subject_ids(self):
        return list(self.subjects)

    def get_subject_info(self, sid):
        return dict(self.subjects[sid])

    def load_subject(self, sid):
        import nibabel as nib
        info = self.subjects[sid]
        img = nib.load(info["path"])
        vol = np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
        spacing = float(img.header.get_zooms()[2])
        vol = np.moveaxis(vol, 2, 0)                                  # sagittal slices first
        if abs(spacing - 1.0) > 0.05:
            vol = zoom(vol, (spacing, 1.0, 1.0), order=1)             # 1 mm, like ADNI
        k, mirror = OASIS_ORIENTATION
        vol = np.rot90(vol, k, axes=(1, 2))
        if mirror:
            vol = vol[:, :, ::-1]
        return _normalize(np.ascontiguousarray(vol)), info["diagnosis"], sid


class AllTestSplit:
    """Splitter stand-in: every subject is in the 'test' split."""
    def __init__(self, ids):
        self.ids = list(ids)

    def get_split_subjects(self, split):
        return self.ids if split == "test" else []


# =============================================================================
# Metrics
# =============================================================================
def auc_binary(y, score):
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics_from_probs(y_true, probs):
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(1)
    cm = np.zeros((3, 3), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    prec, rec, f1 = [], [], []
    for c in range(3):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        prec.append(p_); rec.append(r_); f1.append(2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0)
    present = [c for c in range(3) if (y_true == c).any()]
    aucs = [auc_binary(y_true == c, probs[:, c]) for c in present]
    m = {
        "n_subjects": int(len(y_true)),
        "class_counts": {CLASSES[c]: int((y_true == c).sum()) for c in range(3)},
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(np.mean([rec[c] for c in present])),
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "f1_macro": float(np.mean(f1)),
        "roc_auc_macro_ovr": float(np.nanmean(aucs)),
        "roc_auc_per_class": {CLASSES[c]: a for c, a in zip(present, aucs)},
        "confusion_matrix": cm.tolist(),
    }
    sel = (y_true == 0) | (y_true == 2)                      # CN vs AD, ignoring MCI
    if sel.sum() and (y_true[sel] == 2).any() and (y_true[sel] == 0).any():
        score = probs[sel, 2] / (probs[sel, 2] + probs[sel, 0] + 1e-12)
        m["roc_auc_cn_vs_ad"] = auc_binary(y_true[sel] == 2, score)
    else:
        m["roc_auc_cn_vs_ad"] = float("nan")
    return m


@torch.no_grad()
def subject_probs(model, loader):
    model.eval()
    by_s, lab = {}, {}
    for x, y, sids in loader:
        p = torch.softmax(model(x), 1).numpy()
        for pi, yi, s in zip(p, y.numpy(), sids):
            by_s.setdefault(s, []).append(pi)
            lab[s] = int(yi)
    subs = sorted(by_s)
    return np.array([lab[s] for s in subs]), np.array([np.mean(by_s[s], 0) for s in subs])


# =============================================================================
# Models
# =============================================================================
def load_models(run_dir: Path):
    from models import get_model
    from train import FrozenEncoderClassifier
    out = {}
    for arm, fname in (("supervised", "supervised.pt"), ("ssl", "ssl_finetune.pt"),
                       ("ssl_linear_probe", "ssl_linear_probe.pt")):
        f = run_dir / fname
        if not f.exists():
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        if not ck.get("completed"):
            print(f"  [skip] {run_dir.name}/{fname} is not a finished stage")
            continue
        model = (FrozenEncoderClassifier(get_model("encoder"), 3) if arm == "ssl_linear_probe"
                 else get_model("classifier", num_classes=3))
        model.load_state_dict(ck["model"])
        out[arm] = model.eval()
    return out


# =============================================================================
# Main
# =============================================================================
def orientation_check(oasis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = []
    for dx in CLASSES:
        picks += [s for s in oasis.get_subject_ids() if oasis.subjects[s]["diagnosis"] == dx][:3]
    fig, ax = plt.subplots(2, 6, figsize=(18, 7))
    try:
        d = MRIDataset("adni")
        sid = d.get_subject_ids()[0]
        sl = extract_2d_slices(d.load_subject(sid)[0])
        ax[0][0].imshow(sl[4], cmap="gray", vmin=0, vmax=1); ax[0][0].set_title(f"ADNI {sid}", color="blue", fontsize=9)
        ax[1][0].imshow(sl[20], cmap="gray", vmin=0, vmax=1); ax[1][0].set_title("ADNI other side", color="blue", fontsize=9)
    except Exception as e:
        ax[0][0].text(0.05, 0.5, f"ADNI not available\n{e}", fontsize=7)
    cells = [ax[r][c] for r in range(2) for c in range(1, 6)]
    for a, sid in zip(cells, picks):
        sl = extract_2d_slices(oasis.load_subject(sid)[0])
        a.imshow(sl[4 if len(cells) else 0], cmap="gray", vmin=0, vmax=1)
        a.set_title(f"OASIS {sid} ({oasis.subjects[sid]['diagnosis']})", fontsize=9)
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"OASIS after re-orientation {OASIS_ORIENTATION} - every head should look like ADNI "
                 f"(face LEFT, top of head UP)", fontsize=12)
    out = ROOT / "outputs" / "plots" / "oasis_orientation_final_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    print(f"\nSaved picture: {out}\nCheck it before running the full evaluation.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only save the orientation picture")
    args = ap.parse_args()

    from dataloader import MRISliceDataset
    from torch.utils.data import DataLoader

    oasis = OASISDataset()
    if args.check:
        orientation_check(oasis)
        return

    run_dirs = sorted(d for d in config.CHECKPOINTS_DIR.iterdir()
                      if d.is_dir() and d.name.endswith(f"_adni_{VERSION}"))
    if not run_dirs:
        sys.exit(f"No checkpoints for {VERSION} found in {config.CHECKPOINTS_DIR}")
    print(f"Runs found: {[d.name for d in run_dirs]}")

    print("\n[OASIS] Preparing slices (same selection + normalisation as ADNI)...")
    oasis_ds = MRISliceDataset(oasis, AllTestSplit(oasis.get_subject_ids()), split="test", augment=False)
    oasis_loader = DataLoader(oasis_ds, batch_size=32, shuffle=False, num_workers=0)

    adni = MRIDataset("adni")
    from splitter import SubjectSplitter

    results = {"version": VERSION, "created": datetime.now().isoformat(timespec="seconds"),
               "oasis_label_mapping": "CDR 0 -> CN, 0.5 -> MCI, >=1 -> AD; age >= 60; MR1; RAW mpr-1",
               "oasis_orientation": list(OASIS_ORIENTATION), "runs": {}}
    for run_dir in run_dirs:
        seed = int(re.search(r"seed_(\d+)", run_dir.name).group(1))
        print("\n" + "=" * 72 + f"\n{run_dir.name}\n" + "=" * 72)
        models = load_models(run_dir)
        splitter = SubjectSplitter(adni, random_state=seed)
        adni_ds = MRISliceDataset(adni, splitter, split="test", augment=False)
        adni_loader = DataLoader(adni_ds, batch_size=32, shuffle=False, num_workers=0)

        saved = {}
        run_json = config.RESULTS_DIR / "runs" / f"{run_dir.name}.json"
        if run_json.exists():
            saved = json.loads(run_json.read_text(encoding="utf-8"))

        results["runs"][run_dir.name] = {}
        for arm, model in models.items():
            ya, pa = subject_probs(model, adni_loader)
            yo, po = subject_probs(model, oasis_loader)
            m_adni, m_oasis = metrics_from_probs(ya, pa), metrics_from_probs(yo, po)
            gap = {k: m_adni[k] - m_oasis[k] for k in
                   ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc_macro_ovr", "roc_auc_cn_vs_ad")}
            check = ""
            if arm in saved:
                check = f"  (saved ADNI acc {saved[arm]['accuracy']:.3f} -> recomputed {m_adni['accuracy']:.3f})"
            print(f"  {arm:<17} ADNI F1 {m_adni['f1_macro']:.3f} AUC {m_adni['roc_auc_macro_ovr']:.3f} | "
                  f"OASIS F1 {m_oasis['f1_macro']:.3f} AUC {m_oasis['roc_auc_macro_ovr']:.3f} "
                  f"CN-vs-AD AUC {m_oasis['roc_auc_cn_vs_ad']:.3f}{check}")
            results["runs"][run_dir.name][arm] = {"adni_test": m_adni, "oasis": m_oasis, "gap_adni_minus_oasis": gap}

    # ---- summary across runs
    print("\n" + "=" * 72 + "\nSUMMARY (mean over runs)  -  gap = ADNI test minus OASIS\n" + "=" * 72)
    summary = {}
    for arm in ("supervised", "ssl", "ssl_linear_probe"):
        rows = [r[arm] for r in results["runs"].values() if arm in r]
        if not rows:
            continue
        s = {}
        for key in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc_macro_ovr", "roc_auc_cn_vs_ad"):
            a = [r["adni_test"][key] for r in rows]
            o = [r["oasis"][key] for r in rows]
            s[key] = {"adni_mean": float(np.nanmean(a)), "oasis_mean": float(np.nanmean(o)),
                      "gap_mean": float(np.nanmean(np.array(a) - np.array(o))),
                      "adni_values": a, "oasis_values": o}
        summary[arm] = s
        print(f"\n{arm}  ({len(rows)} runs)")
        for key, v in s.items():
            print(f"  {key:<20} ADNI {v['adni_mean']:.3f}   OASIS {v['oasis_mean']:.3f}   gap {v['gap_mean']:+.3f}")
    results["summary"] = summary

    out = config.RESULTS_DIR / f"oasis_eval_{VERSION}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
