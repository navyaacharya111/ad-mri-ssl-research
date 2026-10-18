r"""
evaluate_saved_models.py - CORRECTED evaluation of saved models (no training).

Run from the project folder:
    python tools\evaluate_saved_models.py --version v3_f1select_reg_probe
    python tools\evaluate_saved_models.py --version v2_hippocampal

Why: the original evaluate.py compared labels and predictions of DIFFERENT subjects
(order bug), so all test metrics saved during training are invalid. This script re-scores
the saved (finished) models with labels matched by subject ID, for:
  * the ADNI test split of each run (same subjects as during training), and
  * OASIS-1 (age >= 60, MR1, CDR -> CN/MCI/AD), same preprocessing + re-orientation as
    tools/oasis_evaluate.py.
It also runs a PAIRED BOOTSTRAP on OASIS (same 198 subjects for all models) to give a
95% confidence interval for "SSL minus supervised" within each run.

Writes ONE new file: outputs/results/evaluation_corrected_<version>_<timestamp>.json
(includes every subject's predicted probabilities). Changes nothing else.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import MRIDataset
from oasis_evaluate import (OASISDataset, AllTestSplit, metrics_from_probs, auc_binary,
                            load_models, OASIS_ORIENTATION, CLASSES)

ARMS = ["supervised", "ssl", "ssl_linear_probe"]
KEYS = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_macro_ovr", "roc_auc_cn_vs_ad"]


@torch.no_grad()
def predict(model, loader):
    """Return (subject_ids sorted, labels, mean slice probabilities) - labels matched by subject ID."""
    model.eval()
    probs, lab = {}, {}
    for x, y, sids in loader:
        p = torch.softmax(model(x), 1).numpy()
        for pi, yi, s in zip(p, y.numpy(), sids):
            probs.setdefault(s, []).append(pi)
            lab[s] = int(yi)
    subs = sorted(probs)
    return subs, np.array([lab[s] for s in subs]), np.array([np.mean(probs[s], 0) for s in subs])


# ---------------------------------------------------------------- bootstrap metrics (fast)
def _f1_macro(y, p):
    pred = p.argmax(1)
    f1 = []
    for c in range(3):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1))


def _auc_macro(y, p):
    vals = [auc_binary(y == c, p[:, c]) for c in range(3) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(vals)) if vals else float("nan")


def _auc_cn_ad(y, p):
    s = (y == 0) | (y == 2)
    if not ((y[s] == 0).any() and (y[s] == 2).any()):
        return float("nan")
    return auc_binary(y[s] == 2, p[s, 2] / (p[s, 2] + p[s, 0] + 1e-12))


BOOT_METRICS = {"roc_auc_cn_vs_ad": _auc_cn_ad, "roc_auc_macro_ovr": _auc_macro, "f1_macro": _f1_macro}


def paired_bootstrap(y, p_a, p_b, n_boot=2000, seed=0):
    """Difference (a - b) for each metric, resampling the SAME subjects for both models."""
    rng = np.random.RandomState(seed)
    out = {}
    for name, fn in BOOT_METRICS.items():
        observed = fn(y, p_a) - fn(y, p_b)
        diffs = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(y), len(y))
            d = fn(y[idx], p_a[idx]) - fn(y[idx], p_b[idx])
            if not np.isnan(d):
                diffs.append(d)
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_two = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
        out[name] = {"difference": float(observed), "ci95": [float(lo), float(hi)], "p_bootstrap": p_two,
                     "n_boot": int(len(diffs))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3_f1select_reg_probe")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    from dataloader import MRISliceDataset
    from splitter import SubjectSplitter

    run_dirs = sorted(d for d in config.CHECKPOINTS_DIR.iterdir()
                      if d.is_dir() and d.name.endswith(f"_adni_{args.version}"))
    if not run_dirs:
        sys.exit(f"No checkpoints for version '{args.version}' in {config.CHECKPOINTS_DIR}")
    print(f"Version {args.version}: runs {[d.name for d in run_dirs]}")

    oasis = OASISDataset()
    oasis_loader = DataLoader(MRISliceDataset(oasis, AllTestSplit(oasis.get_subject_ids()), split="test", augment=False),
                              batch_size=32, shuffle=False, num_workers=0)
    adni = MRIDataset("adni")

    res = {"version": args.version, "created": datetime.now().isoformat(timespec="seconds"),
           "note": "Corrected evaluation: labels matched to predictions by subject ID.",
           "oasis_label_mapping": "CDR 0 -> CN, 0.5 -> MCI, >=1 -> AD; age >= 60; MR1; RAW mpr-1",
           "oasis_orientation": list(OASIS_ORIENTATION), "runs": {}}

    for run_dir in run_dirs:
        seed = int(re.search(r"seed_(\d+)", run_dir.name).group(1))
        print("\n" + "=" * 76 + f"\n{run_dir.name}\n" + "=" * 76)
        models = load_models(run_dir)
        adni_loader = DataLoader(MRISliceDataset(adni, SubjectSplitter(adni, random_state=seed), split="test", augment=False),
                                 batch_size=32, shuffle=False, num_workers=0)
        run = {}
        oasis_probs = {}
        for arm in ARMS:
            if arm not in models:
                continue
            sa, ya, pa = predict(models[arm], adni_loader)
            so, yo, po = predict(models[arm], oasis_loader)
            ma, mo = metrics_from_probs(ya, pa), metrics_from_probs(yo, po)
            run[arm] = {"adni_test": ma, "oasis": mo,
                        "gap_adni_minus_oasis": {k: ma[k] - mo[k] for k in KEYS},
                        "adni_predictions": {"subjects": sa, "labels": ya.tolist(), "probs": pa.round(5).tolist()},
                        "oasis_predictions": {"subjects": so, "labels": yo.tolist(), "probs": po.round(5).tolist()}}
            oasis_probs[arm] = (yo, po)
            print(f"\n  {arm}")
            print(f"    ADNI : acc {ma['accuracy']:.3f}  bal.acc {ma['balanced_accuracy']:.3f}  F1 {ma['f1_macro']:.3f}  "
                  f"AUC {ma['roc_auc_macro_ovr']:.3f}  CN-vs-AD AUC {ma['roc_auc_cn_vs_ad']:.3f}   CM {ma['confusion_matrix']}")
            print(f"    OASIS: acc {mo['accuracy']:.3f}  bal.acc {mo['balanced_accuracy']:.3f}  F1 {mo['f1_macro']:.3f}  "
                  f"AUC {mo['roc_auc_macro_ovr']:.3f}  CN-vs-AD AUC {mo['roc_auc_cn_vs_ad']:.3f}   CM {mo['confusion_matrix']}")

        # paired bootstrap on OASIS: each SSL arm vs supervised
        boot = {}
        if "supervised" in oasis_probs:
            y_sup, p_sup = oasis_probs["supervised"]
            for arm in ("ssl", "ssl_linear_probe"):
                if arm in oasis_probs:
                    y_arm, p_arm = oasis_probs[arm]
                    assert np.array_equal(y_arm, y_sup)
                    boot[f"{arm}_minus_supervised"] = paired_bootstrap(y_sup, p_arm, p_sup, args.n_boot, seed)
            for comp, mets in boot.items():
                print(f"\n  OASIS paired bootstrap ({args.n_boot}x): {comp}")
                for k, v in mets.items():
                    print(f"    {k:<18} diff {v['difference']:+.3f}   95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]   p = {v['p_bootstrap']:.3f}")
        run["oasis_bootstrap"] = boot
        res["runs"][run_dir.name] = run

    # ---- summary across runs
    print("\n" + "=" * 76 + "\nSUMMARY (mean over runs)   gap = ADNI test minus OASIS\n" + "=" * 76)
    summary = {}
    for arm in ARMS:
        rows = [r[arm] for r in res["runs"].values() if arm in r]
        if not rows:
            continue
        summary[arm] = {}
        print(f"\n{arm} ({len(rows)} runs)")
        for k in KEYS:
            a = [r["adni_test"][k] for r in rows]; o = [r["oasis"][k] for r in rows]
            summary[arm][k] = {"adni_mean": float(np.nanmean(a)), "oasis_mean": float(np.nanmean(o)),
                               "gap_mean": float(np.nanmean(np.array(a) - np.array(o))),
                               "adni_values": a, "oasis_values": o}
            print(f"  {k:<20} ADNI {np.nanmean(a):.3f}   OASIS {np.nanmean(o):.3f}   gap {np.nanmean(np.array(a) - np.array(o)):+.3f}"
                  f"   (per run ADNI {[round(x, 3) for x in a]}, OASIS {[round(x, 3) for x in o]})")
    res["summary"] = summary

    out = config.RESULTS_DIR / f"evaluation_corrected_{args.version}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[Saved] {out}")


if __name__ == "__main__":
    main()
