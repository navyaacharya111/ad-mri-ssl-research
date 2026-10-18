"""
apply_fix3.py - v3 training changes for the AD-MRI SSL project.

Run from the project folder:   python apply_fix3.py

Changes:
  * best epoch chosen by subject-level validation macro-F1 (not validation loss)
  * regularisation: AdamW weight decay 0.01, label smoothing 0.1, supervised lr 3e-4
  * SimCLR pretraining 30 epochs (loss had plateaued by ~30 in v2)
  * new arm: linear probe (frozen SimCLR encoder + linear layer)
  * new runs saved as "v3_f1select_reg_probe" - earlier runs are kept and never reused

What it does (nothing is deleted):
  1. Copies src/train.py and src/experiment.py into backup_before_fix3_<date_time>/
  2. Writes the new src/train.py and src/experiment.py
  3. Runs a quick self-check (no real data, no training)
"""
import shutil, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

NEW_TRAIN = r'''"""
Training Loops: Supervised, SSL Pretraining, Fine-tuning and Linear Probe

Three training pipelines:
1. Supervised - Train classifier directly on labels
2. SSL Pretraining - Train SimCLR encoder with contrastive loss
3. Fine-tuning - Take pretrained encoder, train classifier on labels

Crash safety: when a checkpoint_path is given, the full training state is
saved after every epoch (written to a temp file, then renamed, so a power cut
can never leave a half-written checkpoint). Re-running with the same path
resumes from the last finished epoch; a finished stage is not retrained.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional
import config
from models import get_model, ResNet18Classifier, SimCLRModel


# ============================================================================
# Checkpoint helpers
# ============================================================================

def save_checkpoint(state: Dict, path: Path):
    """Write atomically: save to <path>.tmp, then rename over <path>."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Optional[Path]) -> Optional[Dict]:
    if path is None or not Path(path).exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


# ============================================================================
# SimCLR augmentations (applied on the batch, independently per image)
# ============================================================================

def simclr_augment(x: torch.Tensor) -> torch.Tensor:
    """
    Return a randomly augmented copy of a batch of images in [-1, 1],
    shape (B, C, H, W). Every image gets its own random parameters:
      - random resized crop (scale 0.6-1.0 of each side) with small translation
      - horizontal flip (p=0.5)
      - rotation (+/- 15 degrees)
      - brightness / contrast jitter
      - Gaussian blur (p=0.5)
      - light Gaussian noise
    Two calls on the same batch give two different "views" for SimCLR.
    """
    b, c, h, w = x.shape
    dev = x.device

    # --- geometric: one affine matrix per image (crop + flip + rotation) ---
    scale = torch.empty(b, device=dev).uniform_(0.6, 1.0)
    angle = torch.empty(b, device=dev).uniform_(-15, 15) * math.pi / 180
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    max_shift = 1.0 - scale
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.stack([
        torch.stack([cos * flip, -sin, tx], dim=1),
        torch.stack([sin * flip, cos, ty], dim=1),
    ], dim=1)  # (B, 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # --- intensity: brightness / contrast ---
    contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    brightness = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.2, 0.2)
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    out = (out - mean) * contrast + mean + brightness

    # --- Gaussian blur on a random half of the batch ---
    blur_mask = torch.rand(b, device=dev) < 0.5
    if blur_mask.any():
        sigma = float(np.random.uniform(0.5, 1.5))
        k = torch.arange(-2, 3, device=dev, dtype=x.dtype)
        k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
        k = k / k.sum()
        kernel = (k[:, None] * k[None, :]).expand(c, 1, 5, 5)
        blurred = F.conv2d(F.pad(out[blur_mask], (2, 2, 2, 2), mode="replicate"), kernel, groups=c)
        out = out.clone()
        out[blur_mask] = blurred

    # --- light noise ---
    out = out + torch.randn_like(out) * 0.02
    return out.clamp(-1.0, 1.0)


# ============================================================================
# Losses
# ============================================================================

class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR
    (Chen et al., 2020).

    For a batch of N images there are 2N views. For each view the positive is
    the other view of the same image; the other 2N-2 views are negatives.

    L = -log( exp(sim(z_i, z_j) / tau) / sum_{k != i} exp(sim(z_i, z_k) / tau) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        n = z_i.shape[0]
        z = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)      # (2N, D)
        sim = torch.mm(z, z.t()) / self.temperature                # (2N, 2N)
        sim.fill_diagonal_(float("-inf"))                          # exclude self-similarity
        targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(sim, targets)


# ============================================================================
# Supervised training (baseline and fine-tuning)
# ============================================================================

def subject_level_scores(probs: np.ndarray, labels: np.ndarray, subject_ids: List[str],
                         num_classes: int = 3) -> Tuple[float, float]:
    """
    Average slice probabilities per subject, predict argmax, and return
    (macro-F1, balanced accuracy) over subjects - the same aggregation the
    test evaluation uses.
    """
    by_subject, label_of = {}, {}
    for p, y, s in zip(probs, labels, subject_ids):
        by_subject.setdefault(s, []).append(p)
        label_of[s] = int(y)
    subs = sorted(by_subject)
    y_true = np.array([label_of[s] for s in subs])
    y_pred = np.array([int(np.mean(by_subject[s], axis=0).argmax()) for s in subs])
    f1s, recalls = [], []
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        if (y_true == c).any():
            recalls.append(rec)
    return float(np.mean(f1s)), float(np.mean(recalls)) if recalls else 0.0


class FrozenEncoderClassifier(nn.Module):
    """
    Linear probe: a frozen (pretrained) encoder followed by one trainable
    linear layer. The encoder's weights and BatchNorm statistics never change,
    so this measures how useful the pretrained features are on their own.
    """

    def __init__(self, encoder: nn.Module, num_classes: int = 3, feature_dim: int = 512):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(feature_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()  # keep BatchNorm statistics frozen
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(x)
        return self.fc(features.flatten(1))


class SupervisedTrainer:
    """
    Trainer for supervised learning (baseline, fine-tuning and linear probe).

    Model selection: after every epoch the validation set is evaluated at the
    SUBJECT level (slice probabilities averaged per subject, as in the test
    evaluation). The epoch with the highest subject-level macro-F1 is kept and
    restored at the end; early stopping counts epochs without F1 improvement.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                        "val_subject_f1": [], "val_subject_bal_acc": []}

    def train_epoch(self, train_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module) -> Tuple[float, float]:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch_images, batch_labels, _ in tqdm(train_loader, desc="Training"):
            batch_images = batch_images.to(self.device)
            batch_labels = batch_labels.to(self.device)

            logits = self.model(batch_images)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_images.size(0)
            total_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            total_samples += batch_images.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Slice-level loss/accuracy plus subject-level macro-F1 and balanced accuracy."""
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_probs, all_labels, all_subjects = [], [], []

        with torch.no_grad():
            for batch_images, batch_labels, batch_subjects in tqdm(val_loader, desc="Validating"):
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_images)
                loss = criterion(logits, batch_labels)

                total_loss += loss.item() * batch_images.size(0)
                total_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                total_samples += batch_images.size(0)
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
                all_subjects.extend(list(batch_subjects))

        f1, bal_acc = subject_level_scores(np.concatenate(all_probs), np.concatenate(all_labels),
                                           all_subjects, num_classes=all_probs[0].shape[1])
        return {"loss": total_loss / total_samples, "acc": total_correct / total_samples,
                "subject_f1": f1, "subject_bal_acc": bal_acc}

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, learning_rate: float = 0.001,
              early_stopping_patience: int = 10,
              class_weights: Optional[torch.Tensor] = None,
              checkpoint_path: Optional[Path] = None,
              stage_name: str = "SUPERVISED TRAINING",
              weight_decay: float = 0.0,
              label_smoothing: float = 0.0) -> Dict:
        """
        Train the model.

        Args:
            train_loader, val_loader: data
            num_epochs: maximum number of epochs
            learning_rate: learning rate (AdamW)
            early_stopping_patience: stop if subject-level val macro-F1 doesn't improve for N epochs
            class_weights: optional tensor (num_classes,) for weighted cross-entropy
            checkpoint_path: if given, save state every epoch and resume from it
            stage_name: heading printed in the log
            weight_decay: AdamW weight decay (regularisation)
            label_smoothing: cross-entropy label smoothing (regularisation)

        Returns:
            Training history
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        weight = class_weights.to(self.device) if class_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        val_criterion = nn.CrossEntropyLoss(weight=weight)
        has_val = val_loader is not None and len(val_loader) > 0

        start_epoch = 0
        best_score = -1.0
        best_epoch = None
        best_state = None
        patience_counter = 0

        print("\n" + "=" * 70)
        print(stage_name)
        print("=" * 70)
        print(f"  lr={learning_rate}, weight_decay={weight_decay}, label_smoothing={label_smoothing}, "
              f"selection = subject-level val macro-F1")

        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            self.model.load_state_dict(ckpt["model"])
            self.history = ckpt["history"]
            if ckpt.get("completed"):
                print(f"[Checkpoint] Stage already finished - loaded weights from {checkpoint_path}")
                return self.history
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_score = ckpt.get("best_score", -1.0)
            best_epoch = ckpt.get("best_epoch")
            best_state = ckpt["best_state"]
            patience_counter = ckpt["patience_counter"]
            print(f"[Checkpoint] Resuming from epoch {start_epoch + 1} ({checkpoint_path})")

        stopped_early = False
        for epoch in range(start_epoch, num_epochs):
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")

            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

            if has_val:
                v = self.validate(val_loader, val_criterion)
                self.history["val_loss"].append(v["loss"])
                self.history["val_acc"].append(v["acc"])
                self.history["val_subject_f1"].append(v["subject_f1"])
                self.history["val_subject_bal_acc"].append(v["subject_bal_acc"])
                print(f"  Val Loss:   {v['loss']:.4f}, Acc: {v['acc']:.4f} | "
                      f"subject macro-F1: {v['subject_f1']:.4f}, balanced acc: {v['subject_bal_acc']:.4f}")

                if v["subject_f1"] > best_score:
                    best_score = v["subject_f1"]
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                    print("  (new best model)")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        stopped_early = True
            else:
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1

            if checkpoint_path is not None and not stopped_early:
                save_checkpoint({
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "history": self.history,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "patience_counter": patience_counter,
                    "completed": False,
                }, checkpoint_path)

            if stopped_early:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\n[Best model] Restored epoch {best_epoch} (subject-level val macro-F1 {best_score:.4f})")
        self.history["best_epoch"] = best_epoch
        self.history["best_val_subject_f1"] = best_score

        if checkpoint_path is not None:
            save_checkpoint({"model": self.model.state_dict(), "history": self.history,
                             "completed": True}, checkpoint_path)

        print("\n" + "=" * 70)
        print("[OK] Training completed!")
        print("=" * 70)
        return self.history


# ============================================================================
# SimCLR pretraining
# ============================================================================

class SimCLRTrainer:
    """
    Trainer for self-supervised learning (SimCLR pretraining).

    Each batch is augmented twice with independent random augmentations
    (simclr_augment); the encoder learns to match the two views of the same
    image against all other views in the batch. Labels are never used.
    """

    def __init__(self, model: SimCLRModel, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.history = {"train_loss": []}

    def train_epoch(self, train_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: ContrastiveLoss) -> float:
        self.model.train()
        total_loss, total_samples = 0.0, 0

        for batch_images, _, _ in tqdm(train_loader, desc="SSL Pretraining"):
            if batch_images.size(0) < 2:  # need at least one negative
                continue
            batch_images = batch_images.to(self.device)

            view_1 = simclr_augment(batch_images)
            view_2 = simclr_augment(batch_images)
            z_i = self.model(view_1)
            z_j = self.model(view_2)

            loss = criterion(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_images.size(0)
            total_samples += batch_images.size(0)

        return total_loss / max(total_samples, 1)

    def pretrain(self, train_loader: DataLoader, num_epochs: int = 100,
                 learning_rate: float = 0.001, temperature: float = 0.07,
                 checkpoint_path: Optional[Path] = None) -> Dict:
        """
        Pretrain the model with contrastive learning.

        Args:
            train_loader: Training data (labels are ignored)
            num_epochs: Number of pretraining epochs
            learning_rate: Learning rate
            temperature: Temperature for NT-Xent loss
            checkpoint_path: if given, save state every epoch and resume from it

        Returns:
            Training history
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = ContrastiveLoss(temperature=temperature)
        start_epoch = 0

        print("\n" + "=" * 70)
        print("SELF-SUPERVISED PRETRAINING (SimCLR)")
        print("=" * 70)

        ckpt = load_checkpoint(checkpoint_path)
        if ckpt is not None:
            self.model.load_state_dict(ckpt["model"])
            self.history = ckpt["history"]
            if ckpt.get("completed"):
                print(f"[Checkpoint] Pretraining already finished - loaded weights from {checkpoint_path}")
                return self.history
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            print(f"[Checkpoint] Resuming from epoch {start_epoch + 1} ({checkpoint_path})")

        for epoch in range(start_epoch, num_epochs):
            print(f"\n[Epoch {epoch+1}/{num_epochs}]")
            loss = self.train_epoch(train_loader, optimizer, criterion)
            self.history["train_loss"].append(loss)
            print(f"  Loss: {loss:.4f}")

            if checkpoint_path is not None:
                save_checkpoint({
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "history": self.history,
                    "completed": False,
                }, checkpoint_path)

        if checkpoint_path is not None:
            save_checkpoint({"model": self.model.state_dict(), "history": self.history,
                             "completed": True}, checkpoint_path)

        print("\n" + "=" * 70)
        print("[OK] Pretraining completed!")
        print("=" * 70)
        return self.history

    def get_encoder(self):
        """Get the pretrained encoder for downstream tasks."""
        return self.model.get_encoder()


# ============================================================================
# TEST THE TRAINING LOOPS
# ============================================================================

if __name__ == "__main__":
    from dataset import MRIDataset
    from splitter import SubjectSplitter
    from dataloader import create_dataloaders

    print("\n" + "=" * 70)
    print("TESTING TRAINING LOOPS")
    print("=" * 70)

    dataset = MRIDataset(dataset_type="mock")
    splitter = SubjectSplitter(dataset)
    train_loader, val_loader, test_loader = create_dataloaders(dataset, splitter, batch_size=4)
    device = torch.device(config.DEVICE)

    print("\n" + "-" * 70)
    print("TEST 1: Supervised Training")
    print("-" * 70)
    trainer_sup = SupervisedTrainer(get_model("classifier", num_classes=3, pretrained=False), device=device)
    history_sup = trainer_sup.train(train_loader, val_loader if len(val_loader) > 0 else None,
                                    num_epochs=2, learning_rate=0.001, early_stopping_patience=5)
    print(f"Final train loss: {history_sup['train_loss'][-1]:.4f}")

    print("\n" + "-" * 70)
    print("TEST 2: SSL Pretraining")
    print("-" * 70)
    trainer_ssl = SimCLRTrainer(get_model("simclr", pretrained=False), device=device)
    history_ssl = trainer_ssl.pretrain(train_loader, num_epochs=2, learning_rate=0.001, temperature=0.07)
    print(f"Final pretrain loss: {history_ssl['train_loss'][-1]:.4f}")

    print("\n" + "=" * 70)
    print("[OK] Training loops working correctly!")
    print("=" * 70)
'''

NEW_EXPERIMENT = r'''"""
End-to-End Experiment Runner: Complete research pipeline

This script:
1. Runs multiple experiments (independent runs with different seeds)
2. For each run:
   - Creates train/val/test splits (subject-level)
   - Trains supervised baseline
   - Trains SSL (SimCLR pretraining + fine-tuning)
   - Evaluates both on test set
   - Saves the run's results immediately
3. Computes statistics (mean, std, confidence intervals)
4. Compares supervised vs SSL
5. Tests statistical significance

Crash safety:
- Each finished run is saved to outputs/results/runs/run_<id>_seed_<seed>.json.
  On restart, finished runs are loaded from disk and not retrained.
- Every training stage checkpoints after each epoch to
  outputs/checkpoints/run_<id>_seed_<seed>/<stage>.pt and resumes from there.
- Final results go to date-stamped files and never overwrite older results.
- All console output is also written to outputs/logs/experiment_<timestamp>.log.

Research Question:
"Does self-supervised contrastive pretraining improve cross-dataset
generalization of AD-MRI classification?"
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import copy
import json
from datetime import datetime
import config
from dataset import MRIDataset
from splitter import SubjectSplitter
from dataloader import create_dataloaders
from models import get_model
from train import SupervisedTrainer, SimCLRTrainer, FrozenEncoderClassifier
from evaluate import Evaluator, compare_models


RUNS_DIR = config.RESULTS_DIR / "runs"
LOGS_DIR = config.OUTPUTS_DIR / "logs"
# Change this whenever the method changes, so earlier runs are never reused on resume.
EXPERIMENT_VERSION = "v3_f1select_reg_probe"

# v3 training settings (override config.py where they differ; all are written to each run file).
# Changes vs v2: best epoch chosen by subject-level validation macro-F1 instead of
# validation loss; AdamW weight decay + label smoothing; lower supervised learning
# rate; 30 SimCLR epochs (loss had plateaued by ~30 in v2); extra linear-probe arm.
V3_SETTINGS = {
    "supervised_learning_rate": 3e-4,
    "finetune_learning_rate": config.SSL_CONFIG["finetune_learning_rate"],
    "weight_decay": 1e-2,
    "label_smoothing": 0.1,
    "pretrain_epochs": 30,
    "linear_probe_learning_rate": 1e-3,
    "linear_probe_epochs": 50,
}
ARMS = ["supervised", "ssl", "ssl_linear_probe"]


# ============================================================================
# Helpers: JSON that survives numpy types, atomic writes, console log file
# ============================================================================

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def write_json_atomic(data, path: Path):
    """Write to <path>.tmp then rename, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2)
    os.replace(tmp, path)


class _Tee:
    """Send everything printed to the console to a log file as well."""

    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log_file() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    return log_path


# ============================================================================
# Experiment runner
# ============================================================================

class ExperimentRunner:
    """
    Run complete experiment: supervised vs SSL.
    """

    def __init__(self, num_runs: int = 5, dataset_type: str = "adni"):
        """
        Initialize experiment runner.

        Args:
            num_runs: Number of independent runs
            dataset_type: "mock" or "adni"
        """
        self.num_runs = num_runs
        self.dataset_type = dataset_type
        self.device = torch.device(config.DEVICE)

        # Store results from all runs
        self.results = {arm: [] for arm in ARMS}
        self.results["run_ids"] = []

    def _run_tag(self, run_id: int, random_seed: int) -> str:
        return f"run_{run_id}_seed_{random_seed}_{self.dataset_type}_{EXPERIMENT_VERSION}"

    def _class_weights(self, dataset, splitter) -> torch.Tensor:
        """
        Inverse-frequency class weights from the TRAINING subjects only:
        w_c = N / (K * n_c). Every subject contributes the same number of
        slices, so subject counts give the slice-level class balance.
        """
        train_ids = splitter.get_split_subjects("train")
        counts = np.bincount([dataset.get_subject_info(s)["label"] for s in train_ids], minlength=3)
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (len(counts) * counts)
        print(f"[ClassWeights] Train subjects per class (CN, MCI, AD): {counts.tolist()}")
        print(f"[ClassWeights] Loss weights (CN, MCI, AD): {[round(float(w), 3) for w in weights]}")
        return torch.tensor(weights, dtype=torch.float32)

    def run_single_experiment(self, run_id: int, random_seed: int) -> Dict:
        """
        Run one complete experiment with a specific random seed.

        Returns:
            Results dictionary with supervised and SSL metrics
        """
        print("\n" + "=" * 70)
        print(f"RUN {run_id}/{self.num_runs} (Seed: {random_seed})")
        print("=" * 70)

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        ckpt_dir = config.CHECKPOINTS_DIR / self._run_tag(run_id, random_seed)

        # ===== STEP 1: Load data and create splits =====
        print("\n[Step 1] Loading dataset and creating splits...")
        dataset = MRIDataset(dataset_type=self.dataset_type)
        splitter = SubjectSplitter(dataset, random_state=random_seed)
        train_loader, val_loader, test_loader = create_dataloaders(dataset, splitter, batch_size=16)
        class_weights = self._class_weights(dataset, splitter)

        run_results = {
            "run_id": run_id,
            "seed": random_seed,
            "dataset_type": self.dataset_type,
            "experiment_version": EXPERIMENT_VERSION,
            "settings": V3_SETTINGS,
            "class_weights": class_weights.tolist(),
            "split_sizes": {s: len(splitter.get_split_subjects(s)) for s in ("train", "val", "test")},
        }

        # ===== STEP 2: Train supervised baseline =====
        print("\n[Step 2] Training supervised baseline...")
        model_sup = get_model("classifier", num_classes=3, pretrained=False)
        trainer_sup = SupervisedTrainer(model_sup, device=self.device)

        history_sup = trainer_sup.train(
            train_loader,
            val_loader if len(val_loader) > 0 else None,
            num_epochs=config.SUPERVISED_CONFIG["epochs"],
            learning_rate=V3_SETTINGS["supervised_learning_rate"],
            early_stopping_patience=config.SUPERVISED_CONFIG["early_stopping_patience"],
            class_weights=class_weights,
            checkpoint_path=ckpt_dir / "supervised.pt",
            stage_name="SUPERVISED TRAINING (baseline)",
            weight_decay=V3_SETTINGS["weight_decay"],
            label_smoothing=V3_SETTINGS["label_smoothing"],
        )

        print("\n[Step 2] Evaluating supervised model...")
        evaluator_sup = Evaluator(model_sup, device=self.device)
        metrics_sup = evaluator_sup.evaluate_subject_level(test_loader)
        run_results["supervised"] = metrics_sup
        run_results["history_supervised"] = history_sup

        # ===== STEP 3: SSL pretraining =====
        print("\n[Step 3] Training SSL (SimCLR pretraining)...")
        model_ssl = get_model("simclr", pretrained=False)
        trainer_ssl = SimCLRTrainer(model_ssl, device=self.device)

        history_ssl = trainer_ssl.pretrain(
            train_loader,
            num_epochs=V3_SETTINGS["pretrain_epochs"],
            learning_rate=config.SSL_CONFIG["pretrain_learning_rate"],
            temperature=config.SSL_CONFIG["temperature"],
            checkpoint_path=ckpt_dir / "simclr_pretrain.pt",
        )
        run_results["history_pretrain"] = history_ssl

        # ===== STEP 3b: Linear probe on a frozen COPY of the pretrained encoder =====
        print("\n[Step 3b] Linear probe (frozen SimCLR encoder + linear layer)...")
        model_probe = FrozenEncoderClassifier(copy.deepcopy(trainer_ssl.get_encoder()), num_classes=3)
        trainer_probe = SupervisedTrainer(model_probe, device=self.device)
        history_probe = trainer_probe.train(
            train_loader,
            val_loader if len(val_loader) > 0 else None,
            num_epochs=V3_SETTINGS["linear_probe_epochs"],
            learning_rate=V3_SETTINGS["linear_probe_learning_rate"],
            early_stopping_patience=config.SSL_CONFIG["early_stopping_patience"],
            class_weights=class_weights,
            checkpoint_path=ckpt_dir / "ssl_linear_probe.pt",
            stage_name="LINEAR PROBE (frozen SimCLR encoder)",
            weight_decay=V3_SETTINGS["weight_decay"],
            label_smoothing=V3_SETTINGS["label_smoothing"],
        )
        print("\n[Step 3b] Evaluating linear probe...")
        run_results["ssl_linear_probe"] = Evaluator(model_probe, device=self.device).evaluate_subject_level(test_loader)
        run_results["history_linear_probe"] = history_probe

        # ===== STEP 4: Fine-tune pretrained encoder =====
        print("\n[Step 4] Fine-tuning pretrained encoder...")
        pretrained_encoder = trainer_ssl.get_encoder()
        model_ssl_finetune = get_model("classifier", num_classes=3, pretrained=False)

        # Replace backbone layers with the pretrained encoder layers
        # (encoder = conv1, bn1, relu, maxpool, layer1..layer4, avgpool)
        model_ssl_finetune.backbone.conv1 = pretrained_encoder.encoder[0]
        model_ssl_finetune.backbone.bn1 = pretrained_encoder.encoder[1]
        model_ssl_finetune.backbone.layer1 = pretrained_encoder.encoder[4]
        model_ssl_finetune.backbone.layer2 = pretrained_encoder.encoder[5]
        model_ssl_finetune.backbone.layer3 = pretrained_encoder.encoder[6]
        model_ssl_finetune.backbone.layer4 = pretrained_encoder.encoder[7]

        trainer_finetune = SupervisedTrainer(model_ssl_finetune, device=self.device)
        history_finetune = trainer_finetune.train(
            train_loader,
            val_loader if len(val_loader) > 0 else None,
            num_epochs=config.SSL_CONFIG["finetune_epochs"],
            learning_rate=V3_SETTINGS["finetune_learning_rate"],
            early_stopping_patience=config.SSL_CONFIG["early_stopping_patience"],
            class_weights=class_weights,
            checkpoint_path=ckpt_dir / "ssl_finetune.pt",
            stage_name="FINE-TUNING (SimCLR-pretrained encoder)",
            weight_decay=V3_SETTINGS["weight_decay"],
            label_smoothing=V3_SETTINGS["label_smoothing"],
        )

        print("\n[Step 4] Evaluating SSL model...")
        evaluator_ssl = Evaluator(model_ssl_finetune, device=self.device)
        metrics_ssl = evaluator_ssl.evaluate_subject_level(test_loader)
        run_results["ssl"] = metrics_ssl
        run_results["history_finetune"] = history_finetune

        # ===== STEP 5: Compare results =====
        print("\n[Step 5] Comparing supervised vs SSL (fine-tuned)...")
        run_results["comparison"] = compare_models(metrics_sup, metrics_ssl)
        print("\n[Step 5] Comparing supervised vs SSL (linear probe)...")
        run_results["comparison_linear_probe"] = compare_models(metrics_sup, run_results["ssl_linear_probe"])

        return run_results

    def run_all_experiments(self):
        """
        Run all experiments with different random seeds. Runs that already
        have a saved result file are loaded instead of retrained.
        """
        print("\n" + "=" * 70)
        print("STARTING EXPERIMENT SUITE")
        print("=" * 70)
        print(f"Number of runs: {self.num_runs}")
        print(f"Dataset: {self.dataset_type}")
        print(f"Experiment version: {EXPERIMENT_VERSION}")
        print(f"Settings: {V3_SETTINGS}")
        print(f"Device: {self.device}")
        print(f"Per-run results: {RUNS_DIR}")
        print(f"Checkpoints:     {config.CHECKPOINTS_DIR}")

        for run_id in range(1, self.num_runs + 1):
            random_seed = config.RANDOM_SEED + run_id  # Different seed each run
            run_file = RUNS_DIR / f"{self._run_tag(run_id, random_seed)}.json"

            if run_file.exists():
                print(f"\n[Resume] Run {run_id} already finished - loading {run_file.name}")
                with open(run_file, encoding="utf-8") as f:
                    run_results = json.load(f)
            else:
                run_results = self.run_single_experiment(run_id, random_seed)
                run_results["finished_at"] = datetime.now().isoformat(timespec="seconds")
                write_json_atomic(run_results, run_file)
                print(f"\n[Saved] Run {run_id} results -> {run_file}")

            self.results["run_ids"].append(run_id)
            for arm in ARMS:
                self.results[arm].append(run_results[arm])

        self.compute_statistics()

    def compute_statistics(self):
        """
        Mean, std and 95% CI per arm; each SSL arm compared with supervised
        (difference and paired t-test across runs).
        """
        from scipy import stats

        print("\n" + "=" * 70)
        print("EXPERIMENT STATISTICS")
        print("=" * 70)

        metrics_to_analyze = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        labels = {"supervised": "Supervised", "ssl": "SSL fine-tuned", "ssl_linear_probe": "SSL linear probe"}
        self.statistics = {arm: {} for arm in ARMS}
        self.statistics["comparison"] = {}               # ssl (fine-tuned) vs supervised
        self.statistics["comparison_linear_probe"] = {}  # ssl linear probe vs supervised

        for metric in metrics_to_analyze:
            print(f"\n{metric.upper()}:")
            for arm in ARMS:
                values = [float(r[metric]) for r in self.results[arm]]
                mean, std = float(np.mean(values)), float(np.std(values))
                ci = 1.96 * std / np.sqrt(len(values))
                self.statistics[arm][metric] = {"mean": mean, "std": std, "ci": ci, "values": values}
                print(f"  {labels[arm]:<17} {mean:.4f} ± {ci:.4f}")

            sup_values = self.statistics["supervised"][metric]["values"]
            sup_mean = self.statistics["supervised"][metric]["mean"]
            for arm, key in (("ssl", "comparison"), ("ssl_linear_probe", "comparison_linear_probe")):
                arm_values = self.statistics[arm][metric]["values"]
                arm_mean = self.statistics[arm][metric]["mean"]
                diff = arm_mean - sup_mean
                entry = {"supervised_mean": sup_mean, "ssl_mean": arm_mean, "improvement": diff,
                         "improvement_percent": (diff / sup_mean * 100) if sup_mean != 0 else 0}
                if len(sup_values) > 1:
                    t_stat, p_value = stats.ttest_rel(arm_values, sup_values)
                    entry["t_statistic"], entry["p_value"] = float(t_stat), float(p_value)
                    sig = "significant (p < 0.05)" if p_value < 0.05 else "not significant"
                    print(f"  {labels[arm]} - Supervised: {diff:+.4f}  (paired t-test p={p_value:.4f}, {sig})")
                self.statistics[key][metric] = entry

        print("\n" + "=" * 70)

    def save_results(self, output_dir: Path = None):
        """
        Save all results to NEW date-stamped JSON files (never overwrites).
        """
        if output_dir is None:
            output_dir = config.RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{self.dataset_type}_{EXPERIMENT_VERSION}_{datetime.now():%Y%m%d_%H%M%S}"

        results_file = output_dir / f"experiment_results_{stamp}.json"
        write_json_atomic(self.results, results_file)
        print(f"\n[Results] Saved raw results to {results_file}")

        stats_file = output_dir / f"statistics_{stamp}.json"
        write_json_atomic(self.statistics, stats_file)
        print(f"[Results] Saved statistics to {stats_file}")


# ============================================================================
# RUN THE EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    log_path = start_log_file()

    print("\n" + "=" * 70)
    print("AD-MRI SSL RESEARCH EXPERIMENT")
    print("=" * 70)
    print(f"Log file: {log_path}")
    print("\nResearch Question:")
    print("Does self-supervised pretraining improve cross-dataset generalization")
    print("of Alzheimer's disease MRI classification?")
    print("\n" + "=" * 70)

    runner = ExperimentRunner(num_runs=2, dataset_type="adni")  # 2 runs
    runner.run_all_experiments()
    runner.save_results()

    print("\n" + "=" * 70)
    print("[OK] EXPERIMENT COMPLETED!")
    print("=" * 70)
'''


def main():
    train_path, exp_path = ROOT / "src" / "train.py", ROOT / "src" / "experiment.py"
    for p in (train_path, exp_path):
        if not p.exists():
            sys.exit(f"[Stop] {p} not found. Run this from the project folder. Nothing was changed.")
    exp_text = exp_path.read_text(encoding="utf-8")
    train_text = train_path.read_text(encoding="utf-8")
    if "FrozenEncoderClassifier" in train_text or "v3_f1select_reg_probe" in exp_text:
        sys.exit("[Stop] v3 is already installed. Nothing was changed.")
    if 'EXPERIMENT_VERSION = "v2_hippocampal"' not in exp_text or "simclr_augment" not in train_text:
        sys.exit("[Stop] train.py / experiment.py are not the expected v2 versions "
                 "(run apply_fixes.py and apply_fix2.py first). Nothing was changed.")

    backup = ROOT / f"backup_before_fix3_{datetime.now():%Y%m%d_%H%M%S}"
    (backup / "src").mkdir(parents=True)
    shutil.copy2(train_path, backup / "src" / "train.py")
    shutil.copy2(exp_path, backup / "src" / "experiment.py")
    print(f"[1/3] Backed up train.py, experiment.py -> {backup}")

    train_path.write_text(NEW_TRAIN, encoding="utf-8")
    exp_path.write_text(NEW_EXPERIMENT, encoding="utf-8")
    print("[2/3] Wrote new src/train.py and src/experiment.py (v3)")

    print("[3/3] Self-check...")
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
    import numpy as np, torch
    import experiment
    from train import FrozenEncoderClassifier, subject_level_scores
    from models import get_model
    enc = get_model("simclr").get_encoder()
    probe = FrozenEncoderClassifier(enc, 3)
    out = probe.train()(torch.rand(2, 3, 128, 128))
    assert out.shape == (2, 3) and not any(p.requires_grad for p in probe.encoder.parameters())
    f1, _ = subject_level_scores(np.eye(3)[[0, 0, 1, 1, 2, 2]], np.array([0, 0, 1, 1, 2, 2]), list("aabbcc"))
    assert abs(f1 - 1.0) < 1e-9
    print("      linear probe: encoder frozen, output shape OK; subject-level F1 OK")
    print(f"      new runs will be named like: run_1_seed_43_adni_{experiment.EXPERIMENT_VERSION}")
    print(f"      settings: {experiment.V3_SETTINGS}")
    print("\n[OK] v3 installed. Your old files are in", backup)


if __name__ == "__main__":
    main()
