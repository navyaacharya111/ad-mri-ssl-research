r"""
check_slices.py - save a picture of exactly what the model sees for one subject.

Run from the project folder:
    python tools\check_slices.py              (first AD subject)
    python tools\check_slices.py 002_S_0559   (a specific subject)

Reads one subject's scan (read-only) and writes ONE new image:
    outputs/plots/slice_check_<SUBJECT>.png
Top:    the slices the model is trained on (with distance from the midline)
Bottom: the whole head cut two other ways; the chosen slices are marked in red,
        the detected midline in yellow
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
import pydicom
import config
import dataset as ds_module
from dataset import MRIDataset, extract_2d_slices


def acquisition_plane(dicom_dir: Path):
    f = next(Path(dicom_dir).glob("*.dcm"))
    ds = pydicom.dcmread(str(f), stop_before_pixels=True)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None:
        return "unknown (no ImageOrientationPatient tag)"
    normal = np.abs(np.cross(np.array(iop[:3], float), np.array(iop[3:], float)))
    return ["SAGITTAL (side view)", "CORONAL (front view)", "AXIAL (top view)"][int(normal.argmax())]


def main():
    dataset = MRIDataset("adni")
    ids = dataset.get_subject_ids()
    if len(sys.argv) > 1:
        sid = sys.argv[1]
        if sid not in dataset.subjects:
            sys.exit(f"Subject {sid} not found among loaded subjects.")
    else:
        sid = next((s for s in ids if dataset.get_subject_info(s)["diagnosis"] == "AD"), ids[0])

    info = dataset.get_subject_info(sid)
    volume, diagnosis, _ = dataset.load_subject(sid)
    plane = acquisition_plane(Path(info["path"]))
    slices = extract_2d_slices(volume, target_size=config.MRI_SIZE)

    has_selector = hasattr(ds_module, "select_slice_indices")
    if has_selector:
        idx = ds_module.select_slice_indices(volume)
        centre = ds_module.find_midline(volume)
        mode = f"{len(idx) // 2} slices per side, {ds_module.LATERAL_RANGE_MM[0]:.0f}-{ds_module.LATERAL_RANGE_MM[1]:.0f} mm from the midline"
    else:  # older dataset.py: middle block
        n = len(slices)
        start = (volume.shape[0] - n) // 2
        idx = list(range(start, start + n))
        centre = (volume.shape[0] - 1) / 2
        mode = f"middle {n} slices"

    print(f"\nSubject:            {sid} ({diagnosis})")
    print(f"Volume shape:       {volume.shape}  (slices, rows, cols)" +
          ("  - resampled to 1 mm slices" if has_selector else ""))
    print(f"Acquisition plane:  {plane}")
    print(f"Detected midline:   slice {centre:.1f}")
    print(f"Model uses:         {mode}")
    print(f"Slice indices:      {idx}")

    fig = plt.figure(figsize=(16, 13))
    cols = 8
    rows_top = int(np.ceil(len(slices) / cols))
    gs = fig.add_gridspec(rows_top + 3, 24)
    for i, (s, z) in enumerate(zip(slices, idx)):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c * 3:(c + 1) * 3])
        ax.imshow(s, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"slice {z} ({z - centre:+.0f} mm)", fontsize=7)
        ax.axis("off")

    mid1, mid2 = volume.shape[1] // 2, volume.shape[2] // 2
    first = int(round(centre))
    views = [
        (volume[idx[len(idx) // 4]], f"slice {idx[len(idx) // 4]} (one the model sees)", False),
        (volume[:, mid1, :], "head cut the other way (rows view)", True),
        (volume[:, :, mid2], "head cut the other way (columns view)", True),
    ]
    for j, (img, title, mark) in enumerate(views):
        ax = fig.add_subplot(gs[rows_top:, j * 8:(j + 1) * 8])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
        if mark:
            for z in idx:
                ax.axhline(z, color="red", linewidth=0.6, alpha=0.8)
            ax.axhline(first, color="yellow", linewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{sid} ({diagnosis}) - {plane} - model uses {mode} "
                 f"(red lines; yellow = midline)", fontsize=12)
    out = config.PLOTS_DIR / f"slice_check_{sid}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out = out.with_name(f"slice_check_{sid}_v2.png")
    fig.savefig(out, dpi=80, bbox_inches="tight")
    print(f"\nSaved picture: {out}")


if __name__ == "__main__":
    main()
