r"""
make_paper_figures.py - paper tables and figures, rebuilt ONLY from the corrected evaluation files.

Run from the project folder:
    python tools\make_paper_figures.py
    python tools\make_paper_figures.py --splits      (also counts CN/MCI/AD in each ADNI train/val/test split)

Reads  : the newest outputs/results/evaluation_corrected_v3_*.json  (the six v3 models)
         the newest outputs/results/evaluation_corrected_v2_*.json  (sensitivity check, optional)
Writes : outputs/paper/tables/*.csv  and  outputs/paper/figures/*.png  (new folder; changes nothing else)

Nothing is typed in by hand: every number comes from the JSON files, so the figures can always be
regenerated from the experiment outputs.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results"
OUT = ROOT / "outputs" / "paper"

CLASSES = ["CN", "MCI", "AD"]
ARMS = ["supervised", "ssl", "ssl_linear_probe"]
ARM_LABEL = {"supervised": "Supervised", "ssl": "SimCLR fine-tuned", "ssl_linear_probe": "SimCLR linear probe"}
ARM_COLOR = {"supervised": "#2a78d6", "ssl": "#eb6834", "ssl_linear_probe": "#1baf7a"}   # validated palette slots 1-3
ARM_MARKER = {"supervised": "o", "ssl": "s", "ssl_linear_probe": "^"}                   # second cue besides colour
METRICS = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("f1_macro", "Macro-F1"),
           ("roc_auc_macro_ovr", "Macro ROC-AUC (OvR)"), ("roc_auc_cn_vs_ad", "CN-vs-AD ROC-AUC")]
CHANCE = {"balanced_accuracy": 1 / 3, "roc_auc_macro_ovr": 0.5, "roc_auc_cn_vs_ad": 0.5}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
})


# ------------------------------------------------------------------ loading
def newest(pattern):
    files = sorted(RESULTS.glob(pattern), key=lambda p: p.name)
    return files[-1] if files else None


def load(pattern, required=True):
    f = newest(pattern)
    if f is None:
        if required:
            sys.exit(f"No file matching {pattern} in {RESULTS}")
        return None, None
    return json.loads(f.read_text(encoding="utf-8")), f.name


def runs_of(res):
    """[(run_name, run_dict)] sorted by run name (run_1 ..., run_2 ...)."""
    return sorted(res["runs"].items())


def run_label(name):
    return "Run " + name.split("_")[1] + " (seed " + name.split("seed_")[1].split("_")[0] + ")"


# ------------------------------------------------------------------ helpers
def per_class(cm):
    cm = np.asarray(cm, float)
    out = []
    for c in range(3):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out.append((p, r, f, int(cm[c, :].sum()), int(cm[:, c].sum())))
    return out


def check_cm(m):
    """Recompute accuracy / balanced accuracy / macro-F1 from the confusion matrix (consistency check)."""
    cm = np.asarray(m["confusion_matrix"], float)
    acc = np.trace(cm) / cm.sum()
    pc = per_class(cm)
    bal = np.mean([r for _, r, _, _, _ in pc]); f1 = np.mean([f for _, _, f, _, _ in pc])
    bad = [k for k, v in (("accuracy", acc), ("balanced_accuracy", bal), ("f1_macro", f1)) if abs(v - m[k]) > 0.0015]
    return bad


def write_csv(name, header, rows):
    p = OUT / "tables" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  table  {p.relative_to(ROOT)}")


def save(fig, name):
    p = OUT / "figures" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    print(f"  figure {p.relative_to(ROOT)}")


def r3(x):
    return f"{x:.3f}"


# ------------------------------------------------------------------ tables
def tables(v3, v2, split_counts):
    runs = runs_of(v3)

    # Table: subject distribution
    rows = []
    if split_counts:
        for run, d in split_counts.items():
            for split in ("train", "val", "test"):
                c = d[split]
                rows.append([f"ADNI {split}", run_label(run), c.get("CN", 0), c.get("MCI", 0), c.get("AD", 0), sum(c.values())])
    else:
        for name, r in runs:
            cm = np.asarray(r["supervised"]["adni_test"]["confusion_matrix"])
            n = cm.sum(1)
            rows.append(["ADNI test", run_label(name), *map(int, n), int(n.sum())])
    cm = np.asarray(runs[0][1]["supervised"]["oasis"]["confusion_matrix"])
    n = cm.sum(1)
    rows.append(["OASIS-1 (external test)", "all runs", *map(int, n), int(n.sum())])
    write_csv("table_subject_distribution.csv", ["Cohort / split", "Run", "CN", "MCI (OASIS: CDR 0.5)", "AD", "Total"], rows)

    # Table: performance of all six models + method means
    rows = []
    for arm in ARMS:
        for name, r in runs:
            for ds in ("adni_test", "oasis"):
                m = r[arm][ds]
                rows.append([ARM_LABEL[arm], run_label(name), "ADNI test" if ds == "adni_test" else "OASIS-1",
                             *[r3(m[k]) for k, _ in METRICS]])
        for ds in ("adni_test", "oasis"):
            rows.append([ARM_LABEL[arm], "Mean of 2 runs", "ADNI test" if ds == "adni_test" else "OASIS-1",
                         *[r3(np.mean([r[arm][ds][k] for _, r in runs])) for k, _ in METRICS]])
    write_csv("table_performance_v3.csv", ["Method", "Run", "Test set", *[l for _, l in METRICS]], rows)

    # Table: generalization gap (ADNI test - OASIS), per run and mean; accuracy-type metrics in percentage points
    rows = []
    for arm in ARMS:
        for label, sel in [(run_label(n), [(n, r)]) for n, r in runs] + [("Mean of 2 runs", runs)]:
            g = [np.mean([r[arm]["adni_test"][k] - r[arm]["oasis"][k] for _, r in sel]) for k, _ in METRICS]
            rows.append([ARM_LABEL[arm], label,
                         *[f"{100 * x:+.1f}" for x in g[:3]], *[f"{x:+.3f}" for x in g[3:]]])
    write_csv("table_generalization_gap_v3.csv",
              ["Method", "Run", "Accuracy gap (pp)", "Balanced-accuracy gap (pp)", "Macro-F1 gap (pp)",
               "Macro-AUC gap", "CN-vs-AD AUC gap"], rows)

    # Table: per-class precision / recall / F1 (from confusion matrices)
    rows = []
    for arm in ARMS:
        for name, r in runs:
            for ds in ("adni_test", "oasis"):
                for c, (p, rc, f, n_true, n_pred) in zip(CLASSES, per_class(r[arm][ds]["confusion_matrix"])):
                    rows.append([ARM_LABEL[arm], run_label(name), "ADNI test" if ds == "adni_test" else "OASIS-1",
                                 c, r3(p), r3(rc), r3(f), n_true, n_pred])
    write_csv("table_per_class_v3.csv",
              ["Method", "Run", "Test set", "Class", "Precision", "Recall", "F1", "True subjects", "Predicted subjects"], rows)

    # Table: paired bootstrap on OASIS
    rows = []
    for tag, res in (("v3", v3), ("v2", v2)):
        if res is None:
            continue
        for name, r in runs_of(res):
            for comp, mets in r.get("oasis_bootstrap", {}).items():
                for k, v in mets.items():
                    rows.append([tag, run_label(name), comp.replace("_minus_", " - ").replace("ssl_linear_probe", "linear probe"),
                                 k, f"{v['difference']:+.3f}", f"{v['ci95'][0]:+.3f}", f"{v['ci95'][1]:+.3f}", f"{v['p_bootstrap']:.3f}"])
    write_csv("table_bootstrap_oasis.csv",
              ["Version", "Run", "Comparison", "Metric", "Difference", "CI 2.5%", "CI 97.5%", "p (bootstrap)"], rows)

    # Table: v2 sensitivity check
    if v2 is not None:
        rows = []
        for arm in ("supervised", "ssl"):
            for name, r in runs_of(v2):
                if arm not in r:
                    continue
                for ds in ("adni_test", "oasis"):
                    m = r[arm][ds]
                    rows.append([ARM_LABEL[arm], run_label(name), "ADNI test" if ds == "adni_test" else "OASIS-1",
                                 *[r3(m[k]) for k, _ in METRICS]])
        write_csv("table_performance_v2.csv", ["Method", "Run", "Test set", *[l for _, l in METRICS]], rows)


# ------------------------------------------------------------------ figures
def fig_slopes(v3):
    """Line (slope) graphs: each method's score on ADNI test -> OASIS. Lines join the SAME models on two test sets."""
    runs = runs_of(v3)
    keys = [("balanced_accuracy", "Balanced accuracy"), ("f1_macro", "Macro-F1"),
            ("roc_auc_macro_ovr", "Macro ROC-AUC"), ("roc_auc_cn_vs_ad", "CN-vs-AD ROC-AUC")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4), sharex=True)
    jit = {"supervised": -0.06, "ssl": 0.0, "ssl_linear_probe": 0.06}
    for ax, (k, title) in zip(axes, keys):
        for arm in ARMS:
            a = [r[arm]["adni_test"][k] for _, r in runs]; o = [r[arm]["oasis"][k] for _, r in runs]
            x = np.array([0, 1]) + jit[arm]
            for ai, oi in zip(a, o):  # individual runs: thin, faint
                ax.plot(x, [ai, oi], color=ARM_COLOR[arm], lw=0.8, alpha=0.35, zorder=2)
            ax.plot(x, [np.mean(a), np.mean(o)], color=ARM_COLOR[arm], lw=2, marker=ARM_MARKER[arm], ms=7,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        if k in CHANCE:
            ax.axhline(CHANCE[k], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
            ax.text(-0.28, CHANCE[k] + 0.015, "chance", color=INK2, fontsize=7, va="bottom", ha="left")
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["ADNI test", "OASIS-1"])
        ax.set_xlim(-0.3, 1.45); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score (subject level)")
    handles = [Line2D([], [], color=ARM_COLOR[a], marker=ARM_MARKER[a], lw=2, ms=7, markeredgecolor="white",
                      label=ARM_LABEL[a]) for a in ARMS]
    handles.append(Line2D([], [], color=INK2, lw=0.8, alpha=0.5, label="individual runs (thin)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, "fig_slope_adni_to_oasis_v3.png")


def fig_gap(v3):
    """Dot plot of the generalization gap per metric (a line across metrics would imply an order that does not exist)."""
    runs = runs_of(v3)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    labels = [l for _, l in METRICS]
    off = {"supervised": -0.22, "ssl": 0.0, "ssl_linear_probe": 0.22}
    for arm in ARMS:
        for i, (k, _) in enumerate(METRICS):
            g = [r[arm]["adni_test"][k] - r[arm]["oasis"][k] for _, r in runs]
            y = len(METRICS) - 1 - i + off[arm]
            ax.plot([min(g), max(g)], [y, y], color=ARM_COLOR[arm], lw=1.2, alpha=0.6)   # range of the 2 runs
            ax.plot(np.mean(g), y, marker=ARM_MARKER[arm], color=ARM_COLOR[arm], ms=7, markeredgecolor="white",
                    markeredgewidth=1.2, ls="none", label=ARM_LABEL[arm] if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(range(len(METRICS))); ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Gap = ADNI test minus OASIS-1 (score units; x100 = percentage points)\npositive = worse on OASIS; dot = mean of 2 runs, line = range of the runs")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_generalization_gap_v3.png")


def fig_forest(v3, v2):
    """Paired-bootstrap differences on OASIS with 95% CI (point + whisker, no bars)."""
    rows = []
    for tag, res in (("v3", v3), ("v2", v2)):
        if res is None:
            continue
        for name, r in runs_of(res):
            for comp, mets in r.get("oasis_bootstrap", {}).items():
                arm = "ssl" if comp.startswith("ssl_minus") else "ssl_linear_probe"
                rows.append((f"{tag} {run_label(name).split(' (')[0].lower()}: {ARM_LABEL[arm]} - supervised", arm, mets))
    order = sorted(rows, key=lambda t: (t[1] != "ssl", not t[0].startswith("v3"), t[0]))
    keys = [("roc_auc_cn_vs_ad", "CN-vs-AD ROC-AUC"), ("roc_auc_macro_ovr", "Macro ROC-AUC"), ("f1_macro", "Macro-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.42 * len(order) + 1.3), sharey=True)
    for ax, (k, title) in zip(axes, keys):
        for i, (lab, arm, mets) in enumerate(order):
            v = mets[k]; y = len(order) - 1 - i
            sig = v["p_bootstrap"] < 0.05
            ax.plot(v["ci95"], [y, y], color=ARM_COLOR[arm], lw=1.6)
            ax.plot(v["difference"], y, marker=ARM_MARKER[arm], ms=7, color=ARM_COLOR[arm] if sig else "white",
                    markeredgecolor=ARM_COLOR[arm], markeredgewidth=1.5, ls="none")
        ax.axvline(0, color=INK2, lw=0.9)
        ax.set_title(title, fontsize=9.5, loc="left"); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Difference on OASIS-1 (95% CI)")
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([o[0] for o in order][::-1], fontsize=8)
    fig.text(0.99, -0.02, "Filled marker: p < 0.05 (paired bootstrap, 2000 resamples). Hollow: not significant.",
             ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout()
    save(fig, "fig_bootstrap_forest_oasis.png")


def _cm_panel(ax, cm, title):
    cm = np.asarray(cm, float)
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{100 * norm[i, j]:.0f}%\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9, loc="left")


def fig_confusion(v3):
    runs = runs_of(v3)
    # pooled over the two runs (counts summed; ADNI test sets differ between runs, OASIS is the same 198 subjects twice)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for j, arm in enumerate(ARMS):
        for i, (ds, dl) in enumerate((("adni_test", "ADNI test"), ("oasis", "OASIS-1"))):
            cm = sum(np.asarray(r[arm][ds]["confusion_matrix"]) for _, r in runs)
            _cm_panel(axes[i, j], cm, f"{ARM_LABEL[arm]} - {dl}")
    fig.text(0.5, -0.01, "Row-normalised (% of true class); counts in brackets, summed over the two runs. OASIS 'MCI' = CDR 0.5.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig_confusion_matrices_v3_pooled.png")

    fig, axes = plt.subplots(4, 3, figsize=(10, 13))
    for j, arm in enumerate(ARMS):
        for i, (name, r) in enumerate(runs):
            for d, (ds, dl) in enumerate((("adni_test", "ADNI test"), ("oasis", "OASIS-1"))):
                _cm_panel(axes[2 * d + i, j], r[arm][ds]["confusion_matrix"],
                          f"{ARM_LABEL[arm]}\n{dl}, {run_label(name)}")
    fig.tight_layout()
    save(fig, "fig_confusion_matrices_v3_per_run.png")


def fig_pred_distribution(v3):
    """Share of OASIS subjects predicted as each class (dot plot: classes are categories, so no connecting lines)."""
    runs = runs_of(v3)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    true = np.asarray(runs[0][1]["supervised"]["oasis"]["confusion_matrix"]).sum(1)
    n = true.sum()
    series = [("True OASIS labels", true / n, INK, "D")] + [
        (ARM_LABEL[a], np.mean([np.asarray(r[a]["oasis"]["confusion_matrix"]).sum(0) / n for _, r in runs], 0),
         ARM_COLOR[a], ARM_MARKER[a]) for a in ARMS]
    off = np.linspace(0.27, -0.27, len(series))
    for (lab, share, col, mk), o in zip(series, off):
        for c in range(3):
            y = 2 - c + o
            ax.plot([0, share[c]], [y, y], color=col, lw=1, alpha=0.35)
            ax.plot(share[c], y, marker=mk, color=col, ms=7, markeredgecolor="white", ls="none",
                    label=lab if c == 0 else None)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["CN", "MCI (CDR 0.5)", "AD"])
    ax.set_ylabel("Class"); ax.set_xlim(0, 1)
    ax.set_xlabel(f"Share of the {n} OASIS-1 subjects (true, or predicted by each method; mean of 2 runs)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "fig_oasis_predicted_class_share_v3.png")


# ------------------------------------------------------------------ optional split counts (needs the data)
def count_splits(v3):
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
    from dataset import MRIDataset
    from splitter import SubjectSplitter
    adni = MRIDataset("adni")
    out = {}
    for name, _ in runs_of(v3):
        seed = int(name.split("seed_")[1].split("_")[0])
        sp = SubjectSplitter(adni, random_state=seed)
        out[name] = {}
        for split in ("train", "val", "test"):
            c = {}
            for s in sp.get_split_subjects(split):
                lab = adni.get_subject_info(s)["diagnosis"]
                c[lab] = c.get(lab, 0) + 1
            out[name][split] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", action="store_true", help="also count classes per ADNI split (reads the ADNI data)")
    args = ap.parse_args()

    v3, f3 = load("evaluation_corrected_v3_*.json")
    v2, f2 = load("evaluation_corrected_v2_*.json", required=False)
    print(f"Using {f3}" + (f" and {f2}" if f2 else ""))

    problems = 0
    for tag, res in (("v3", v3), ("v2", v2)):
        if res is None:
            continue
        for name, r in runs_of(res):
            for arm in ARMS:
                for ds in ("adni_test", "oasis"):
                    if arm in r:
                        bad = check_cm(r[arm][ds])
                        if bad:
                            problems += 1
                            print(f"  WARNING {tag} {name} {arm} {ds}: {bad} disagree with the confusion matrix")
    print("Consistency check: " + ("all confusion matrices agree with the saved metrics" if not problems else f"{problems} problems"))

    split_counts = None
    if args.splits:
        try:
            split_counts = count_splits(v3)
        except Exception as e:  # data not available -> fall back to test-set counts from the confusion matrices
            print(f"  (could not count splits: {e}; using test-set counts only)")

    tables(v3, v2, split_counts)
    fig_slopes(v3)
    fig_gap(v3)
    fig_forest(v3, v2)
    fig_confusion(v3)
    fig_pred_distribution(v3)
    print(f"\nDone. Everything is in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
