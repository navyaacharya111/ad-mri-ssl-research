r"""
oasis_inspect.py - READ-ONLY look at the extracted OASIS-1 data.

Run from the project folder:   python tools\oasis_inspect.py
Prints: subject folders, the files inside one subject, the spreadsheet columns
and CDR counts, and the shape / voxel size / orientation of the image files.
Changes nothing.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OASIS = ROOT / "data" / "raw" / "OASIS"
EXTRACTED = OASIS / "extracted"


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    import numpy as np
    try:
        import nibabel as nib
    except ImportError:
        sys.exit("nibabel is not installed. Run:  pip install nibabel openpyxl")

    # ------------------------------------------------------------ subjects
    section("1. SUBJECT FOLDERS")
    subj_dirs = sorted(p for p in EXTRACTED.rglob("OAS1_*_MR*") if p.is_dir())
    sessions = Counter(p.name.split("_")[-1] for p in subj_dirs)
    print(f"Session folders found: {len(subj_dirs)}   {dict(sessions)}")
    print(f"Unique subjects: {len({'_'.join(p.name.split('_')[:2]) for p in subj_dirs})}")
    print("Disc folders:", sorted({p.parent.name for p in subj_dirs}))
    if not subj_dirs:
        sys.exit("No OAS1_* folders found under data/raw/OASIS/extracted")

    # ------------------------------------------------------------ one subject
    section(f"2. FILES IN ONE SUBJECT ({subj_dirs[0].name})")
    for f in sorted(subj_dirs[0].rglob("*")):
        if f.is_file():
            print(f"  {str(f.relative_to(subj_dirs[0])):<70} {f.stat().st_size / 1e6:7.1f} MB")

    # ------------------------------------------------------------ spreadsheet
    section("3. SPREADSHEET")
    sheets = sorted(list(OASIS.glob("*.xlsx")) + list(OASIS.glob("*.csv")), key=lambda p: -p.stat().st_size)
    if not sheets:
        print("  No .xlsx/.csv in data/raw/OASIS")
    else:
        import pandas as pd
        for s in sheets:
            print(f"  {s.name}  ({s.stat().st_size / 1e3:.0f} KB)")
        s = sheets[0]
        df = pd.read_excel(s) if s.suffix == ".xlsx" else pd.read_csv(s)
        print(f"\n  Using largest: {s.name}  ->  {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False))
        if "CDR" in df.columns:
            print("\n  CDR counts (all rows):")
            print("   ", df["CDR"].value_counts(dropna=False).sort_index().to_dict())
            if "Age" in df.columns:
                old = df[df["Age"] >= 60]
                print("  CDR counts (Age >= 60):")
                print("   ", old["CDR"].value_counts(dropna=False).sort_index().to_dict())
                print("  Rows with no CDR - age range:",
                      df[df["CDR"].isna()]["Age"].min(), "to", df[df["CDR"].isna()]["Age"].max())
        id_col = next((c for c in df.columns if str(c).strip().upper() == "ID"), None)
        if id_col:
            ids = set(df[id_col].astype(str))
            on_disk = {p.name for p in subj_dirs}
            print(f"\n  Spreadsheet IDs matching a folder on disk: {len(ids & on_disk)} of {len(ids)}")

    # ------------------------------------------------------------ images
    section("4. IMAGE FILES (first subject)")
    for img in sorted(subj_dirs[0].rglob("*.hdr")) + sorted(subj_dirs[0].rglob("*.nii*")):
        try:
            im = nib.load(str(img))
            zooms = tuple(round(float(z), 3) for z in im.header.get_zooms()[:3])
            print(f"  {str(img.relative_to(subj_dirs[0])):<60}")
            print(f"      shape {im.shape}  voxel mm {zooms}  axes {''.join(nib.aff2axcodes(im.affine))}  "
                  f"dtype {im.get_data_dtype()}")
        except Exception as e:
            print(f"  {img.name}: could not read ({e})")

    print("\n[Inspection finished - nothing was changed]")


if __name__ == "__main__":
    main()
