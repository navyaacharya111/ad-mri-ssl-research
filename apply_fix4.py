"""
apply_fix4.py - fix the label/prediction order bug in src/evaluate.py.

Bug: evaluate_subject_level() stored true labels in data-loader order but built
predictions in sorted subject order, so labels and predictions were compared for
DIFFERENT subjects. Fix: keep each subject's label by subject id and put the labels
in the same sorted order as the predictions.

Run from the project folder:   python apply_fix4.py
Backs up src/evaluate.py first; changes 3 lines; nothing is deleted.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "src" / "evaluate.py"

EDITS = [
    ("        subject_probs = {}  # subject_id -> list of probability vectors\n",
     "        subject_probs = {}  # subject_id -> list of probability vectors\n"
     "        label_of = {}  # subject_id -> true label (fix: keeps labels aligned with sorted predictions)\n"),
    ("                        subject_probs[subject_id] = []\n"
     "                        true_labels.append(label)\n",
     "                        subject_probs[subject_id] = []\n"
     "                        label_of[subject_id] = int(label)\n"),
    ("        true_labels = np.array(true_labels)\n",
     "        true_labels = np.array([label_of[s] for s in unique_subjects])\n"),
]


def main():
    if not EVAL.exists():
        sys.exit(f"[Stop] {EVAL} not found. Run this from the project folder. Nothing was changed.")
    text = EVAL.read_text(encoding="utf-8")
    if "label_of[subject_id]" in text:
        sys.exit("[Stop] evaluate.py is already fixed. Nothing was changed.")
    for old, _ in EDITS:
        if text.count(old) != 1:
            sys.exit(f"[Stop] evaluate.py does not look as expected (could not find exactly once:\n{old}"
                     f"). Nothing was changed.")
    backup = ROOT / f"backup_before_fix4_{datetime.now():%Y%m%d_%H%M%S}" / "src"
    backup.mkdir(parents=True)
    shutil.copy2(EVAL, backup / "evaluate.py")
    print(f"[1/2] Backed up evaluate.py -> {backup}")
    for old, new in EDITS:
        text = text.replace(old, new)
    EVAL.write_text(text, encoding="utf-8")
    print("[2/2] Fixed src/evaluate.py (labels now aligned with predictions)")

    # self-check: perfect model on subjects visited in non-sorted order must score 100%
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
    import numpy as np, torch
    from evaluate import Evaluator

    class Perfect(torch.nn.Module):
        def forward(self, x):
            return x[:, 0, 0, :3] * 10.0   # the label is encoded in the input

    ids = ["S3", "S1", "S2", "S0", "S5", "S4"]
    labels = [2, 0, 1, 1, 2, 0]
    batches = []
    for sid, y in zip(ids, labels):
        x = torch.zeros(4, 3, 8, 8); x[:, 0, 0, y] = 1.0
        batches.append((x, torch.tensor([y] * 4), [sid] * 4))
    m = Evaluator(Perfect(), device="cpu").evaluate_subject_level(batches)
    assert abs(float(m["accuracy"]) - 1.0) < 1e-9, m["accuracy"]
    print("      self-check: a perfect model now scores accuracy 1.000 on out-of-order subjects")
    print("\n[OK] evaluate.py fixed.")


if __name__ == "__main__":
    main()
