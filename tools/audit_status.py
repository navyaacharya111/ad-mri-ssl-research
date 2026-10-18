r"""
audit_status.py - READ-ONLY status report for the AD-MRI SSL project.

Run from the project folder:   python tools\audit_status.py
It only reads files and prints a report. It changes, deletes and trains nothing.
"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOW = time.time()


def when(p: Path) -> str:
    t = p.stat().st_mtime
    age_h = (NOW - t) / 3600
    return f"{datetime.fromtimestamp(t):%Y-%m-%d %H:%M}  ({age_h:.1f} h ago)"


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main():
    print(f"AUDIT of {ROOT}   ({datetime.now():%Y-%m-%d %H:%M})")

    # ---------------------------------------------------------------- code
    section("1. CODE VERSION")
    exp = ROOT / "src" / "experiment.py"
    train = ROOT / "src" / "train.py"
    ds = ROOT / "src" / "dataset.py"
    dl = ROOT / "src" / "dataloader.py"
    cfg = ROOT / "config.py"
    exp_t = exp.read_text(encoding="utf-8") if exp.exists() else ""
    m = re.search(r'EXPERIMENT_VERSION = "([^"]+)"', exp_t)
    version = m.group(1) if m else "none (pre-v2)"
    print(f"experiment.py EXPERIMENT_VERSION : {version}")
    tr_t = train.read_text(encoding="utf-8") if train.exists() else ""
    checks = {
        "real SimCLR two-view augmentation (train.py)": "simclr_augment" in tr_t,
        "checkpoint/resume (train.py)": "save_checkpoint" in tr_t,
        "v3 subject-F1 model selection + linear probe (train.py)": "FrozenEncoderClassifier" in tr_t,
        "hippocampal slice selection (dataset.py)": ds.exists() and "select_slice_indices" in ds.read_text(encoding="utf-8"),
        "normalisation fix max_pixel_value=1.0 (dataloader.py)": dl.exists() and "max_pixel_value=1.0" in dl.read_text(encoding="utf-8"),
    }
    for k, v in checks.items():
        print(f"  [{'x' if v else ' '}] {k}")
    if cfg.exists():
        c = cfg.read_text(encoding="utf-8")
        for key in ("pretrain_epochs", "temperature"):
            mm = re.search(rf'"{key}"\s*:\s*([0-9.]+)', c)
            print(f"  config.py {key}: {mm.group(1) if mm else '?'}")
    if "V3_SETTINGS" in exp_t:
        mm = re.search(r"V3_SETTINGS = \{(.*?)\}", exp_t, re.S)
        print("  V3_SETTINGS:", " ".join(mm.group(1).split()) if mm else "?")

    # ---------------------------------------------------------------- labels
    section("2. LABELS")
    raw = ROOT / "data" / "raw"
    for name in ("ADNI_diagnoses.csv", "ADNI_diagnoses_full.csv", "ADNI_diagnoses_real.csv",
                 "ADNI_diagnoses_PLACEHOLDER_random.csv", "ADNI_collection.csv"):
        p = raw / name
        if not p.exists():
            print(f"  {name:<40} MISSING")
            continue
        lines = p.read_text(encoding="utf-8-sig").splitlines()
        header = lines[0].split(",")
        counts = {}
        if "DIAGNOSIS" in header:
            i = header.index("DIAGNOSIS")
            for ln in lines[1:]:
                parts = ln.split(",")
                if len(parts) > i:
                    counts[parts[i]] = counts.get(parts[i], 0) + 1
        print(f"  {name:<40} {len(lines) - 1:>5} rows  {counts if counts else ''}")
    used = raw / "ADNI_diagnoses.csv"
    full = raw / "ADNI_diagnoses_full.csv"
    if used.exists() and full.exists():
        same = used.read_bytes() == full.read_bytes()
        print(f"  -> training labels = real labels (ADNI_diagnoses_full.csv): {'YES' if same else 'NO - CHECK!'}")

    # ---------------------------------------------------------------- runs
    section("3. FINISHED RUNS (outputs/results/runs)")
    runs_dir = ROOT / "outputs" / "results" / "runs"
    run_files = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime) if runs_dir.exists() else []
    if not run_files:
        print("  none")
    for p in run_files:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            arms = [a for a in ("supervised", "ssl", "ssl_linear_probe") if a in r]
            summary = "  ".join(f"{a}: acc {r[a]['accuracy']:.3f} F1 {r[a]['f1_macro']:.3f}" for a in arms)
            print(f"  {p.name}\n      saved {when(p)}\n      {summary}")
        except Exception as e:
            print(f"  {p.name}: could not read ({e})")

    section("4. FINAL STATISTICS FILES")
    res = ROOT / "outputs" / "results"
    stats = sorted(res.glob("statistics*.json"), key=lambda p: p.stat().st_mtime) if res.exists() else []
    if not stats:
        print("  none")
    for p in stats:
        print(f"  {p.name:<62} {when(p)}")

    # ---------------------------------------------------------------- checkpoints
    section("5. CHECKPOINTS (stage status per run)")
    ck = ROOT / "outputs" / "checkpoints"
    dirs = sorted([d for d in ck.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime) if ck.exists() else []
    if not dirs:
        print("  none")
    torch = None
    try:
        import torch  # noqa
    except Exception:
        print("  (torch not importable - activate the venv to see stage details)")
    for d in dirs:
        print(f"  {d.name}")
        for stage in ("supervised", "simclr_pretrain", "ssl_linear_probe", "ssl_finetune"):
            f = d / f"{stage}.pt"
            if not f.exists():
                continue
            status = ""
            if torch is not None:
                try:
                    s = torch.load(f, map_location="cpu", weights_only=False)
                    status = "FINISHED" if s.get("completed") else f"in progress, last finished epoch {s.get('epoch', -1) + 1}"
                except Exception as e:
                    status = f"unreadable ({e.__class__.__name__})"
            print(f"      {stage:<18} {status:<38} {when(f)}")

    # ---------------------------------------------------------------- logs
    section("6. LATEST LOG")
    logs = sorted((ROOT / "outputs" / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime) \
        if (ROOT / "outputs" / "logs").exists() else []
    if not logs:
        print("  none")
    else:
        latest = logs[-1]
        age_min = (NOW - latest.stat().st_mtime) / 60
        print(f"  {latest.name}   last written {when(latest)}")
        text = latest.read_text(encoding="utf-8", errors="replace")
        ver = re.search(r"Experiment version: (\S+)", text)
        print(f"  experiment version in log: {ver.group(1) if ver else '?'}")
        done = "EXPERIMENT COMPLETED" in text or "[Results] Saved statistics" in text
        if done:
            print("  STATUS: this run COMPLETED")
        elif age_min < 30:
            print("  STATUS: probably RUNNING NOW (log written in the last 30 min)")
        else:
            print("  STATUS: STOPPED before finishing (not completed, no recent writes) - rerun to resume")
        tail = [ln for ln in text.splitlines() if ln.strip()][-8:]
        print("  last lines:")
        for ln in tail:
            print("    " + ln[:110])

    # ---------------------------------------------------------------- oasis
    section("7. OASIS DATA")
    hits = [p for p in (ROOT / "data").rglob("*") if p.is_dir() and "oasis" in p.name.lower()] \
        if (ROOT / "data").exists() else []
    print("  found: " + ", ".join(str(h.relative_to(ROOT)) for h in hits[:5]) if hits else "  not found in data/")

    print("\n[Audit finished - nothing was changed]")


if __name__ == "__main__":
    main()
