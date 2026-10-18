r"""
oasis_orientation_check.py - READ-ONLY picture to compare OASIS and ADNI image orientation.

Run from the project folder:   python tools\oasis_orientation_check.py
Writes ONE new image: outputs/plots/oasis_orientation_check.png
  Top row:    an OASIS raw scan (RAW/..._mpr-1_anon) cut along each of its 3 axes, exactly as stored
  Bottom row: an ADNI slice exactly as the trained models see it (for comparison)
Changes nothing else.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OASIS = ROOT / "data" / "raw" / "OASIS"


def main():
    import nibabel as nib
    import pandas as pd

    sheet = OASIS / "oasis_cross-sectional.xlsx"
    df = pd.read_excel(sheet)
    df = df[df["ID"].astype(str).str.endswith("_MR1") & (df["Age"] >= 60) & (df["CDR"] >= 1)]
    sid = str(df.iloc[0]["ID"]) if len(df) else "OAS1_0001_MR1"
    raw = next((OASIS / "extracted").rglob(f"{sid}_mpr-1_anon.hdr"))
    img = nib.load(str(raw))
    vol = np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
    lo, hi = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)

    hdr = img.header
    try:
        orient = int(hdr["orient"])
    except Exception:
        orient = "n/a"
    print(f"OASIS subject: {sid}  (CDR >= 1)")
    print(f"File:          {raw.relative_to(ROOT)}")
    print(f"Shape:         {vol.shape}   voxel mm {tuple(float(z) for z in hdr.get_zooms()[:3])}")
    print(f"Analyze 'orient' field: {orient}  (0 = transverse, 1 = coronal, 2 = sagittal, 3-5 = flipped versions)")

    fig, ax = plt.subplots(2, 3, figsize=(13, 9))
    s0, s1, s2 = [n // 2 for n in vol.shape]
    views = [(vol[s0, :, :], f"OASIS axis 0 = {s0}\n(array[{s0}, :, :])"),
             (vol[:, s1, :], f"OASIS axis 1 = {s1}\n(array[:, {s1}, :])"),
             (vol[:, :, s2 - 28], f"OASIS axis 2 = {s2 - 28}  (~28 slices off centre)\n(array[:, :, {s2 - 28}])")]
    for a, (im, title) in zip(ax[0], views):
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.set_title(title, fontsize=9)
        a.set_xlabel("columns ->"); a.set_ylabel("rows ->")
        a.set_xticks([]); a.set_yticks([])

    try:
        from dataset import MRIDataset, extract_2d_slices
        d = MRIDataset("adni")
        ad = next(s for s in d.get_subject_ids() if d.get_subject_info(s)["diagnosis"] == "AD")
        v, _, _ = d.load_subject(ad)
        sl = extract_2d_slices(v)
        ax[1][0].imshow(sl[4], cmap="gray", vmin=0, vmax=1)
        ax[1][0].set_title(f"ADNI {ad}: slice as the model sees it", fontsize=9)
        ax[1][1].imshow(sl[20], cmap="gray", vmin=0, vmax=1)
        ax[1][1].set_title("ADNI: slice from the other hemisphere", fontsize=9)
    except Exception as e:
        ax[1][0].text(0.05, 0.5, f"ADNI reference not available:\n{e}", fontsize=8)
    for a in ax[1]:
        a.axis("off")

    fig.suptitle(f"Orientation check: OASIS {sid} (top, as stored) vs ADNI (bottom, as trained)", fontsize=12)
    out = ROOT / "outputs" / "plots" / "oasis_orientation_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    print(f"\nSaved picture: {out}")


if __name__ == "__main__":
    main()
