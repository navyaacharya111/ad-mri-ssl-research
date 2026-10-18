r"""
select_balanced_subjects.py - pick a class-balanced, scanner-matched ADNI subject list.

Run from the project folder:
    python tools\select_balanced_subjects.py                 (all AD subjects, same number of CN and MCI)
    python tools\select_balanced_subjects.py --per-class 250 (250 subjects per class)

Reads (read-only):
    data/raw/ADNI_search_*.csv      IDA "CSV Download" files of the advanced searches
                                    (columns: Subject ID, Research Group, Visit, Study Date,
                                    Imaging Protocol, ...)
    data/raw/ADNI/ADNI/<SUBJECT>/   subjects already on disk (preferred, so less to download)

Writes (NEW files, never overwrites):
    data/raw/ADNI_selection_<N>_<timestamp>.csv      chosen subjects with class, scanner, visit
    data/raw/ADNI_selection_<N>_<timestamp>_TO_DOWNLOAD.txt
        subject IDs not yet on disk, comma-separated in blocks of 100 for the IDA Subject ID field

Method:
    - one scan date per subject: the earliest study date in the search results
    - group mapping: CN -> CN, MCI / EMCI / LMCI -> MCI, AD -> AD
    - stratum = scanner manufacturer x field strength (1.5T / 3T)
    - every class gets the same stratum proportions as AD, so the scanner cannot
      predict the diagnosis; within a stratum, subjects already on disk come first,
      the rest are drawn at random (seed 42)
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
IMAGE_DIR = RAW / "ADNI" / "ADNI"
GROUP_MAP = {"CN": "CN", "MCI": "MCI", "EMCI": "MCI", "LMCI": "MCI", "AD": "AD"}
CLASSES = ["CN", "MCI", "AD"]
SEED = 42


def load_searches(files):
    frames = []
    for f in files:
        df = pd.read_csv(f)
        proto = (df["Imaging Protocol"].str.extractall(r"(\w[\w ]*)=([^;]*)")
                 .reset_index().pivot(index="level_0", columns=0, values=1))
        frames.append(df.join(proto).assign(source=f.name))
    d = pd.concat(frames, ignore_index=True)
    d["Class"] = d["Research Group"].str.strip().str.upper().map(GROUP_MAP)
    d = d[d["Class"].notna()].copy()
    d["date"] = pd.to_datetime(d["Study Date"], format="%m/%d/%Y")
    d["Field"] = np.where(d["Field Strength"].astype(float) < 2, "1.5T", "3T")
    d["Maker"] = (d["Manufacturer"].str.upper()
                  .str.extract(r"(SIEMENS|PHILIPS|GE)")[0].fillna("OTHER"))
    d["Stratum"] = d["Maker"] + " " + d["Field"]
    return d


def earliest_per_subject(d):
    conflicts = d.groupby("Subject ID")["Class"].nunique()
    conflicts = conflicts[conflicts > 1].index
    if len(conflicts):
        print(f"Excluded {len(conflicts)} subjects listed under more than one class: "
              f"{', '.join(conflicts[:10])}")
        d = d[~d["Subject ID"].isin(conflicts)]
    return (d.sort_values(["Subject ID", "date"])
             .groupby("Subject ID", as_index=False).head(1)
             [["Subject ID", "Class", "Research Group", "Visit", "Study Date",
               "Stratum", "Description", "source"]])


def select(subjects, per_class, on_disk, rng):
    ad = subjects[subjects["Class"] == "AD"]
    share = ad["Stratum"].value_counts(normalize=True)
    # largest-remainder rounding so the targets add up to per_class exactly
    raw = share * per_class
    target = np.floor(raw).astype(int)
    target[(raw - target).sort_values(ascending=False).index[:per_class - target.sum()]] += 1

    chosen, shortfalls = [], []
    for cls in CLASSES:
        pool = subjects[subjects["Class"] == cls]
        for stratum, n in target.items():
            cand = pool[pool["Stratum"] == stratum].copy()
            cand["on_disk"] = cand["Subject ID"].isin(on_disk)
            cand["r"] = rng.random(len(cand))
            cand = cand.sort_values(["on_disk", "r"], ascending=[False, True])
            if len(cand) < n:
                shortfalls.append(f"{cls} {stratum}: wanted {n}, only {len(cand)} available")
            chosen.append(cand.head(n).drop(columns="r"))
    return pd.concat(chosen, ignore_index=True), target, shortfalls


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=None,
                    help="subjects per class (default: number of AD subjects available)")
    args = ap.parse_args()

    files = sorted(RAW.glob("ADNI_search_*.csv"))
    if not files:
        sys.exit(f"No ADNI_search_*.csv files found in {RAW}")
    print("Search files:\n  " + "\n  ".join(f.name for f in files))

    subjects = earliest_per_subject(load_searches(files))
    on_disk = {p.name for p in IMAGE_DIR.iterdir() if p.is_dir()} if IMAGE_DIR.exists() else set()
    print(f"\nSubjects in searches: {subjects['Class'].value_counts().reindex(CLASSES).to_dict()}")
    print(f"Subject folders already on disk: {len(on_disk)}")

    n_ad = int((subjects["Class"] == "AD").sum())
    per_class = args.per_class or n_ad
    if per_class > n_ad:
        sys.exit(f"--per-class {per_class} is more than the {n_ad} AD subjects available")

    chosen, target, shortfalls = select(subjects, per_class, on_disk, np.random.default_rng(SEED))

    print(f"\nTarget per class and scanner group (matches AD proportions), {per_class} per class:")
    table = pd.crosstab(chosen["Stratum"], chosen["Class"]).reindex(columns=CLASSES, fill_value=0)
    table["target"] = target
    print(table.to_string())
    for s in shortfalls:
        print("  SHORTFALL " + s)

    summary = pd.crosstab(chosen["Class"], chosen["on_disk"].map({True: "on disk", False: "to download"}))
    print("\n" + summary.reindex(CLASSES).fillna(0).astype(int).to_string())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = RAW / f"ADNI_selection_{per_class}_{stamp}.csv"
    out_txt = RAW / f"ADNI_selection_{per_class}_{stamp}_TO_DOWNLOAD.txt"
    for f in (out_csv, out_txt):
        if f.exists():
            sys.exit(f"{f} already exists - not overwriting")
    chosen.to_csv(out_csv, index=False)
    todo = sorted(chosen.loc[~chosen["on_disk"], "Subject ID"])
    blocks = [",".join(todo[i:i + 100]) for i in range(0, len(todo), 100)]
    out_txt.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"\nWrote {out_csv.relative_to(ROOT)}")
    print(f"Wrote {out_txt.relative_to(ROOT)}  ({len(todo)} subjects, {len(blocks)} block(s) of up to 100)")


if __name__ == "__main__":
    main()
