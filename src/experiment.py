"""
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
