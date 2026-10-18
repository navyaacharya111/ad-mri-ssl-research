r"""
oasis_orientation_pick.py - READ-ONLY: find how OASIS sagittal slices must be rotated/mirrored
to match the ADNI slices the models were trained on.

Run from the project folder:   python tools\oasis_orientation_pick.py

Method: OASIS raw scans are sagittal along array axis 2 (128 slices x 1.25 mm). For several
OASIS subjects, the hippocampal-region slices are extracted exactly like ADNI (same code), then
each of the 8 possible rotations/mirrors is compared (correlation) with the average ADNI slice.
The transform with the highest correlation, consistent across subjects, is the answer.

Writes ONE new image: outputs/plots/oasis_orientation_pick.png   (changes nothing else)
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

OASIS = ROOT / "data" / "raw" / "OASIS"
TRANSFORMS = [(k, m) for m in (False, True) for k in range(4)]  # (number of 90-degree turns, mirror)


def name(t):
    k, m = t
    return f"rotate {90 * k:>3} deg" + (" + mirror" if m else "")


def apply2d(img, t):
    k, m = t
    out = np.rot90(img, k)
    return np.fliplr(out) if m else out


def load_oasis_sagittal(hdr_path):
    """OASIS raw scan -> (sagittal slices along axis 0, rows, cols) at 1 mm, normalized like ADNI."""
    import nibabel as nib
    from dataset import _normalize
    img = nib.load(str(hdr_path))
    vol = np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
    spacing = float(img.header.get_zooms()[2])
    vol = np.moveaxis(vol, 2, 0)                     # sagittal axis first
    if abs(spacing - 1.0) > 0.05:
        vol = zoom(vol, (spacing, 1.0, 1.0), order=1)
    return _normalize(vol)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def main():
    import pandas as pd
    from dataset import MRIDataset, extract_2d_slices

    # ---- ADNI template: mean model-input slice over 10 subjects
    d = MRIDataset("adni")
    ids = d.get_subject_ids()[:10]
    adni_slices = []
    for s in ids:
        v, _, _ = d.load_subject(s)
        adni_slices.extend(extract_2d_slices(v))
    template = np.mean(adni_slices, axis=0)
    print(f"ADNI template from {len(ids)} subjects ({len(adni_slices)} slices)")

    # ---- OASIS subjects (age >= 60, first scans), mixed diagnoses
    df = pd.read_excel(OASIS / "oasis_cross-sectional.xlsx")
    df = df[df["ID"].astype(str).str.endswith("_MR1") & (df["Age"] >= 60) & df["CDR"].notna()]
    pick = pd.concat([df[df["CDR"] == c].head(2) for c in (0, 0.5, 1)])
    scores, winners, example = [], [], None
    for sid in pick["ID"].astype(str):
        hdr = next((OASIS / "extracted").rglob(f"{sid}_mpr-1_anon.hdr"), None)
        if hdr is None:
            continue
        vol = load_oasis_sagittal(hdr)
        mean_slice = np.mean(extract_2d_slices(vol), axis=0)
        row = [corr(apply2d(mean_slice, t), template) for t in TRANSFORMS]
        scores.append(row)
        winners.append(TRANSFORMS[int(np.argmax(row))])
        if example is None:
            example = (sid, extract_2d_slices(vol)[4])
        print(f"  {sid}: best = {name(winners[-1])}  (r = {max(row):.3f})")

    scores = np.array(scores)
    mean_scores = scores.mean(axis=0)
    best = TRANSFORMS[int(np.argmax(mean_scores))]
    agree = Counter(winners)[best]
    print("\nMean correlation with the ADNI template for each transform:")
    for t, sc in sorted(zip(TRANSFORMS, mean_scores), key=lambda x: -x[1]):
        print(f"  {name(t):<26} r = {sc:.3f}" + ("   <-- BEST" if t == best else ""))
    second = sorted(mean_scores)[-2]
    print(f"\nRESULT: {name(best)}  (k={best[0]}, mirror={best[1]})")
    print(f"        chosen by {agree} of {len(winners)} subjects; margin over 2nd best: {mean_scores.max() - second:.3f}")

    # ---- picture: ADNI reference + the 8 candidates for one OASIS subject
    sid, sl = example
    fig, ax = plt.subplots(2, 5, figsize=(16, 7))
    ax[0][0].imshow(adni_slices[4], cmap="gray", vmin=0, vmax=1)
    ax[0][0].set_title("ADNI (as trained)", fontsize=10, color="blue")
    ax[1][0].imshow(template, cmap="gray")
    ax[1][0].set_title("ADNI average slice", fontsize=10, color="blue")
    for i, t in enumerate(TRANSFORMS):
        a = ax[i // 4][1 + i % 4]
        a.imshow(apply2d(sl, t), cmap="gray", vmin=0, vmax=1)
        a.set_title(f"{name(t)}\nr = {mean_scores[i]:.3f}", fontsize=9,
                    color="green" if t == best else "black", fontweight="bold" if t == best else "normal")
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"OASIS {sid}: the 8 possible orientations vs ADNI - best match in green", fontsize=12)
    out = ROOT / "outputs" / "plots" / "oasis_orientation_pick.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    print(f"\nSaved picture: {out}")


if __name__ == "__main__":
    main()
