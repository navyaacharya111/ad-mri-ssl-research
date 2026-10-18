"""
apply_fix2.py - switch slice selection to the hippocampal region (option A).

Run from the project folder:   python apply_fix2.py

What it does (nothing is deleted):
  1. Copies src/dataset.py, src/experiment.py and tools/check_slices.py into
     a new folder backup_before_fix2_<date_time>/
  2. Writes the new src/dataset.py: 16 slices per hemisphere, 20-38 mm from the
     detected midline, after resampling the slice axis to 1 mm
  3. Edits src/experiment.py so new runs are saved as "v2_hippocampal" and the
     earlier (midline-slice) runs are never reused or overwritten
  4. Writes the new tools/check_slices.py
  5. Runs a quick self-check (no real data, no training)
"""
import shutil, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NEW_DATASET = r'''"""
MRI Dataset: subject index + on-demand volume loading for ADNI (DICOM) or mock data.

Interface used by the rest of the pipeline:
  splitter.py   -> get_subject_ids(), get_subject_info(subject_id)
  dataloader.py -> load_subject(subject_id), extract_2d_slices(volume, target_size),
                   get_subject_info(subject_id)

ADNI layout:  data/raw/ADNI/ADNI/<SUBJECT_ID>/<SERIES>/<DATE>/<IMAGE_ID>/*.dcm
Labels:       data/raw/ADNI_diagnoses.csv  (columns: SUBJECT_ID, DIAGNOSIS in {CN, MCI, AD})

Slice selection (v2): ADNI MPRAGE scans are sagittal, so the middle slices show
only the brain midline. extract_2d_slices() therefore takes half of the slices
from each hemisphere, LATERAL_RANGE_MM (default 20-38 mm) either side of the
detected midline - the region containing the hippocampi. Volumes are resampled
to 1 mm along the slice axis first, so the mm range is the same for every scan.
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

warnings.filterwarnings("ignore")

LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
NUM_SLICES_PER_SUBJECT = getattr(config, "NUM_SLICES_PER_SUBJECT", 32)
MIN_SLICES = 40  # a folder needs at least this many slices to count as a 3D scan
LATERAL_RANGE_MM = getattr(config, "LATERAL_RANGE_MM", (20.0, 38.0))  # distance from midline


def _normalize(volume: np.ndarray) -> np.ndarray:
    """Clip to 1st-99th percentile and scale to [0, 1]."""
    volume = volume.astype(np.float32)
    lo, hi = np.percentile(volume, [1, 99])
    if hi <= lo:
        return np.zeros_like(volume)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def find_midline(volume_3d: np.ndarray) -> float:
    """
    Estimate the left-right centre of the head along axis 0: the weighted
    centre of all voxels brighter than background. The head is roughly
    symmetric, so this lands on the midline even if the scan is off-centre.
    """
    mask = volume_3d > 0.15
    weight = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(np.float64)
    if weight.sum() == 0:
        return (volume_3d.shape[0] - 1) / 2.0
    return float((np.arange(len(weight)) * weight).sum() / weight.sum())


def select_slice_indices(volume_3d: np.ndarray, num_slices: int = None) -> List[int]:
    """
    Indices of the slices used for training: num_slices // 2 per hemisphere,
    evenly spread between LATERAL_RANGE_MM from the midline (volume at 1 mm).
    """
    num_slices = num_slices or NUM_SLICES_PER_SUBJECT
    depth = volume_3d.shape[0]
    centre = find_midline(volume_3d)
    half = max(num_slices // 2, 1)
    offsets = np.linspace(LATERAL_RANGE_MM[0], LATERAL_RANGE_MM[1], half)
    idx = [centre - o for o in offsets[::-1]] + [centre + o for o in offsets]
    return [int(np.clip(round(i), 0, depth - 1)) for i in idx]


def extract_2d_slices(volume_3d: np.ndarray, target_size: int = None,
                      num_slices: int = None) -> List[np.ndarray]:
    """
    Take the hippocampal-region slices from both hemispheres (see
    select_slice_indices) and resize each to target_size x target_size.
    """
    target_size = target_size or config.MRI_SIZE

    volume_3d = np.squeeze(volume_3d)
    if volume_3d.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume_3d.shape}")

    slices = []
    for z in select_slice_indices(volume_3d, num_slices):
        s = volume_3d[z]
        s = zoom(s, (target_size / s.shape[0], target_size / s.shape[1]), order=1)
        slices.append(np.clip(s, 0.0, 1.0).astype(np.float32))
    return slices


class MRIDataset:
    """Index of subjects; volumes are read from disk only when load_subject() is called."""

    def __init__(self, dataset_type: str = "adni", data_dir: Path = None):
        self.dataset_type = dataset_type
        self.data_dir = Path(data_dir) if data_dir else Path(getattr(config, "RAW_DATA_DIR", "data/raw"))
        self.subjects: Dict[str, Dict] = {}

        print(f"\n[Dataset] Initializing {dataset_type.upper()} dataset...")
        if dataset_type == "adni":
            self._load_real_dataset()
        else:
            self._load_mock_dataset()

        counts = {d: 0 for d in LABEL_MAP}
        for info in self.subjects.values():
            counts[info["diagnosis"]] += 1
        print(f"[Dataset] Loaded {len(self.subjects)} subjects  "
              f"(CN={counts['CN']}, MCI={counts['MCI']}, AD={counts['AD']})")

    # ------------------------------------------------------------------ ADNI
    @staticmethod
    def _num_frames(dcm_file: Path) -> int:
        import pydicom
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            return int(getattr(ds, "NumberOfFrames", 1) or 1)
        except Exception:
            return 0

    def _find_dicom_dir(self, subject_folder: Path):
        """
        Return the folder holding this subject's 3D scan: among all folders that
        contain .dcm files, prefer MPRAGE-type series, then the most slices.
        Folders with fewer than MIN_SLICES slices (localizers etc.) are ignored.
        """
        counts = Counter(f.parent for f in subject_folder.rglob("*.dcm"))
        candidates = []
        for folder, n_files in counts.items():
            n_slices = n_files
            if n_files < MIN_SLICES:  # could be one multi-frame file holding the whole volume
                n_slices = sum(self._num_frames(f) for f in folder.glob("*.dcm"))
            if n_slices >= MIN_SLICES:
                is_rage = "RAGE" in str(folder.relative_to(subject_folder)).upper()
                candidates.append((is_rage, n_slices, folder))
        if not candidates:
            return None
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return candidates[0][2]

    def _load_real_dataset(self):
        labels_file = self.data_dir / "ADNI_diagnoses.csv"
        adni_root = self.data_dir / "ADNI" / "ADNI"
        print(f"[Dataset] Labels: {labels_file}")
        print(f"[Dataset] Images: {adni_root}")

        if not labels_file.exists() or not adni_root.exists():
            print("[Dataset] ERROR: labels file or image folder not found")
            return

        df = pd.read_csv(labels_file)
        labels = {str(s).strip(): str(d).strip().upper()
                  for s, d in zip(df["SUBJECT_ID"], df["DIAGNOSIS"])}

        no_label, no_dicom = 0, 0
        for folder in sorted(p for p in adni_root.iterdir() if p.is_dir()):
            sid = folder.name
            diagnosis = labels.get(sid)
            if diagnosis not in LABEL_MAP:
                no_label += 1
                continue
            dicom_dir = self._find_dicom_dir(folder)
            if dicom_dir is None:
                no_dicom += 1
                continue
            self.subjects[sid] = {
                "subject_id": sid,
                "diagnosis": diagnosis,
                "label": LABEL_MAP[diagnosis],
                "path": str(dicom_dir),
                "dataset": "ADNI",
            }
        if no_label or no_dicom:
            print(f"[Dataset] Skipped {no_label} folders with no valid label, "
                  f"{no_dicom} with no 3D scan (fewer than {MIN_SLICES} slices)")

    def _read_dicom_volume(self, dicom_dir: Path) -> np.ndarray:
        import pydicom

        frames = []  # (instance_number, frame_index, 2D image)
        positions = {}  # instance_number -> ImagePositionPatient (single-frame files)
        thickness = None
        for f in Path(dicom_dir).glob("*.dcm"):
            ds = pydicom.dcmread(str(f))
            order = float(getattr(ds, "InstanceNumber", 0) or 0)
            if thickness is None and getattr(ds, "SliceThickness", None):
                thickness = float(ds.SliceThickness)
            if getattr(ds, "ImagePositionPatient", None) is not None:
                positions[order] = np.array(ds.ImagePositionPatient, dtype=float)
            arr = ds.pixel_array.astype(np.float32)
            n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
            is_color = int(getattr(ds, "SamplesPerPixel", 1) or 1) > 1
            if is_color:                      # RGB -> grayscale
                arr = arr.mean(axis=-1)
            if n_frames > 1 or arr.ndim == 3:  # multi-frame file: one file = whole volume
                for i, frame in enumerate(arr):
                    frames.append((order, i, frame))
            else:
                frames.append((order, 0, arr))
        if not frames:
            raise ValueError(f"No DICOM files in {dicom_dir}")

        frames.sort(key=lambda t: (t[0], t[1]))
        shape = Counter(img.shape for _, _, img in frames).most_common(1)[0][0]
        stack = [img for _, _, img in frames if img.shape == shape]
        volume = np.stack(stack, axis=0)  # (z, y, x) in acquisition plane

        # Slice spacing in mm: from slice positions if available, else SliceThickness
        spacing = None
        if len(positions) > 2:
            pos = np.array([positions[k] for k in sorted(positions)])
            steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            steps = steps[steps > 1e-3]
            if len(steps):
                spacing = float(np.median(steps))
        spacing = spacing or thickness or 1.0

        # Resample the slice axis to 1 mm so mm-based slice selection is consistent
        if abs(spacing - 1.0) > 0.05:
            volume = zoom(volume, (spacing, 1.0, 1.0), order=1)
        return volume

    # ------------------------------------------------------------------ mock
    def _load_mock_dataset(self):
        rng = np.random.RandomState(config.RANDOM_SEED)
        for i in range(50):
            diagnosis = list(LABEL_MAP)[rng.randint(0, 3)]
            sid = f"MOCK_{i:03d}"
            self.subjects[sid] = {
                "subject_id": sid,
                "diagnosis": diagnosis,
                "label": LABEL_MAP[diagnosis],
                "path": None,
                "dataset": "MOCK",
            }

    # ------------------------------------------------------------------ API
    def __len__(self) -> int:
        return len(self.subjects)

    def get_subject_ids(self) -> List[str]:
        return list(self.subjects.keys())

    def get_subject_info(self, subject_id: str) -> Dict:
        return dict(self.subjects[subject_id])

    def load_subject(self, subject_id: str) -> Tuple[np.ndarray, str, str]:
        """Return (volume normalized to [0,1] with shape (z, y, x), diagnosis, subject_id)."""
        info = self.subjects[subject_id]
        if self.dataset_type == "adni":
            try:
                volume = self._read_dicom_volume(Path(info["path"]))
            except Exception as e:
                raise RuntimeError(f"Could not read {subject_id} at {info['path']}: {e}") from e
        else:
            rng = np.random.RandomState(abs(hash(subject_id)) % (2**31))
            volume = rng.rand(64, 128, 128).astype(np.float32)
        return _normalize(volume), info["diagnosis"], subject_id
'''

NEW_CHECK = r'''r"""
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
'''

EXP_EDITS = [('LOGS_DIR = config.OUTPUTS_DIR / "logs"\n', 'LOGS_DIR = config.OUTPUTS_DIR / "logs"\n# Change this whenever the method changes, so earlier runs are never reused on resume.\nEXPERIMENT_VERSION = "v2_hippocampal"\n'), ('return f"run_{run_id}_seed_{random_seed}_{self.dataset_type}"', 'return f"run_{run_id}_seed_{random_seed}_{self.dataset_type}_{EXPERIMENT_VERSION}"'), ('            "dataset_type": self.dataset_type,\n', '            "dataset_type": self.dataset_type,\n            "experiment_version": EXPERIMENT_VERSION,\n'), ('stamp = f"{self.dataset_type}_{datetime.now():%Y%m%d_%H%M%S}"', 'stamp = f"{self.dataset_type}_{EXPERIMENT_VERSION}_{datetime.now():%Y%m%d_%H%M%S}"'), ('        print(f"Dataset: {self.dataset_type}")\n', '        print(f"Dataset: {self.dataset_type}")\n        print(f"Experiment version: {EXPERIMENT_VERSION}")\n')]


def main():
    ds_path, exp_path, chk_path = ROOT / "src" / "dataset.py", ROOT / "src" / "experiment.py", ROOT / "tools" / "check_slices.py"
    for p in (ds_path, exp_path):
        if not p.exists():
            sys.exit(f"[Stop] {p} not found. Run this from the project folder. Nothing was changed.")

    exp_text = exp_path.read_text(encoding="utf-8")
    if "EXPERIMENT_VERSION" in exp_text:
        sys.exit("[Stop] experiment.py already has EXPERIMENT_VERSION - this fix seems to be installed already. Nothing was changed.")
    for old, new in EXP_EDITS:
        if exp_text.count(old) != 1:
            sys.exit(f"[Stop] experiment.py does not look as expected (could not find: {old.strip()[:60]}...). "
                     f"Was apply_fixes.py run first? Nothing was changed.")

    backup = ROOT / f"backup_before_fix2_{datetime.now():%Y%m%d_%H%M%S}"
    (backup / "src").mkdir(parents=True)
    (backup / "tools").mkdir()
    shutil.copy2(ds_path, backup / "src" / "dataset.py")
    shutil.copy2(exp_path, backup / "src" / "experiment.py")
    if chk_path.exists():
        shutil.copy2(chk_path, backup / "tools" / "check_slices.py")
    print(f"[1/5] Backed up dataset.py, experiment.py, check_slices.py -> {backup}")

    ds_path.write_text(NEW_DATASET, encoding="utf-8")
    print("[2/5] Wrote new src/dataset.py (hippocampal-region slices)")

    for old, new in EXP_EDITS:
        exp_text = exp_text.replace(old, new)
    exp_path.write_text(exp_text, encoding="utf-8")
    print("[3/5] Updated src/experiment.py (new runs saved as v2_hippocampal)")

    chk_path.parent.mkdir(exist_ok=True)
    chk_path.write_text(NEW_CHECK, encoding="utf-8")
    print("[4/5] Wrote new tools/check_slices.py")

    print("[5/5] Self-check...")
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
    import numpy as np
    import dataset, experiment  # noqa: F401
    vol = np.zeros((200, 64, 64), np.float32)
    vol[30:170] = 0.8                       # "head" spanning slices 30-169, centre 99.5
    idx = dataset.select_slice_indices(vol, 32)
    centre = dataset.find_midline(vol)
    assert abs(centre - 99.5) < 1 and len(idx) == 32
    print(f"      midline found at {centre:.1f} (expected 99.5); slices {idx[0]}-{idx[15]} and {idx[16]}-{idx[31]}")
    print(f"      new runs will be named like: run_1_seed_43_adni_{experiment.EXPERIMENT_VERSION}")
    print("\n[OK] Fix installed. Your old files are in", backup)


if __name__ == "__main__":
    main()
