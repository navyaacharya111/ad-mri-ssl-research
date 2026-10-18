"""
Evaluation & Metrics: Compute performance metrics with subject-level aggregation

CRITICAL: We aggregate predictions at the SUBJECT level.
Each subject gets ONE prediction (average of their slices).
This prevents artificially inflated accuracy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import config


class Evaluator:
    """
    Evaluate model performance with subject-level aggregation.
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model to evaluate
            device: "cpu" or "cuda"
        """
        self.model = model.to(device)
        self.device = device
        self.slice_predictions = []  # For debugging
    
    def predict_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Get predictions for a batch of images.
        
        Args:
            images: (batch, 3, 128, 128)
        
        Returns:
            Class predictions, shape (batch,)
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            if isinstance(self.model, nn.Module):
                # Check if model outputs logits or probabilities
                output = self.model(images)
                
                # If output is 2D, it's logits (batch, num_classes)
                if len(output.shape) == 2:
                    predictions = output.argmax(dim=1)
                else:
                    predictions = output
            else:
                predictions = self.model(images).argmax(dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_probabilities(self, images: torch.Tensor) -> np.ndarray:
        """
        Get probability predictions for a batch of images.
        
        Args:
            images: (batch, 3, 128, 128)
        
        Returns:
            Class probabilities, shape (batch, num_classes)
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            output = self.model(images)
            
            # Convert logits to probabilities
            if len(output.shape) == 2:
                probabilities = torch.softmax(output, dim=1)
            else:
                probabilities = output
        
        return probabilities.cpu().numpy()
    
    def evaluate_subject_level(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model at SUBJECT level.
        
        This is the most important evaluation:
        1. Get predictions for all slices
        2. Aggregate by subject (average probabilities)
        3. Compute metrics on subject-level predictions
        
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        # Collect all predictions
        subject_ids = []
        true_labels = []
        subject_probs = {}  # subject_id -> list of probability vectors
        label_of = {}  # subject_id -> true label (fix: keeps labels aligned with sorted predictions)
        
        print("\n[Evaluate] Making predictions on test set...")
        
        with torch.no_grad():
            for batch_images, batch_labels, batch_subject_ids in tqdm(test_loader, desc="Predicting"):
                # Get probability predictions
                batch_probs = self.predict_probabilities(batch_images)
                
                # Store per subject
                for subject_id, probs, label in zip(batch_subject_ids, batch_probs, batch_labels.numpy()):
                    if subject_id not in subject_probs:
                        subject_probs[subject_id] = []
                        label_of[subject_id] = int(label)
                    
                    subject_probs[subject_id].append(probs)
        
        # Aggregate at subject level
        print("\n[Evaluate] Aggregating predictions to subject level...")
        
        unique_subjects = sorted(subject_probs.keys())
        subject_level_preds = []
        subject_level_probs = []
        
        for subject_id in unique_subjects:
            # Average probabilities across all slices
            subject_slice_probs = np.array(subject_probs[subject_id])  # (num_slices, 3)
            avg_probs = subject_slice_probs.mean(axis=0)  # (3,)
            pred = avg_probs.argmax()
            
            subject_level_preds.append(pred)
            subject_level_probs.append(avg_probs)
        
        subject_level_preds = np.array(subject_level_preds)
        subject_level_probs = np.array(subject_level_probs)
        true_labels = np.array([label_of[s] for s in unique_subjects])
        
        print(f"[Evaluate] Evaluated {len(unique_subjects)} subjects")
        
        # Compute metrics
        metrics = self._compute_metrics(true_labels, subject_level_preds, subject_level_probs)
        
        # Store for visualization
        metrics["subject_ids"] = unique_subjects
        metrics["true_labels"] = true_labels
        metrics["predictions"] = subject_level_preds
        metrics["probabilities"] = subject_level_probs
        
        return metrics
    
    def _compute_metrics(self, true_labels: np.ndarray, 
                        predictions: np.ndarray,
                        probabilities: np.ndarray = None) -> Dict:
        """
        Compute classification metrics.
        
        Args:
            true_labels: Ground truth labels
            predictions: Model predictions
            probabilities: Probability predictions (for ROC-AUC)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(true_labels, predictions)
        metrics["precision_macro"] = precision_score(true_labels, predictions, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(true_labels, predictions, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(true_labels, predictions, average="macro", zero_division=0)
        
        # Per-class metrics
        metrics["precision_per_class"] = precision_score(true_labels, predictions, average=None, zero_division=0)
        metrics["recall_per_class"] = recall_score(true_labels, predictions, average=None, zero_division=0)
        metrics["f1_per_class"] = f1_score(true_labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(true_labels, predictions)
        
        # ROC-AUC (if we have probability predictions)
        if probabilities is not None and len(np.unique(true_labels)) > 1:
            try:
                metrics["roc_auc_macro"] = roc_auc_score(true_labels, probabilities, average="macro", multi_class="ovr")
            except:
                metrics["roc_auc_macro"] = None
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in readable format."""
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)
        
        print(f"\nAccuracy:         {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"F1 (macro):        {metrics['f1_macro']:.4f}")
        
        if metrics.get("roc_auc_macro"):
            print(f"ROC-AUC (macro):  {metrics['roc_auc_macro']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics["confusion_matrix"])
        
        print("\nPer-class metrics:")
        for class_idx, class_name in config.CLASS_NAMES.items():
            print(f"\n  {class_name}:")
            print(f"    Precision: {metrics['precision_per_class'][class_idx]:.4f}")
            print(f"    Recall:    {metrics['recall_per_class'][class_idx]:.4f}")
            print(f"    F1:        {metrics['f1_per_class'][class_idx]:.4f}")
        
        print("\n" + "=" * 70)
    
    def plot_confusion_matrix(self, metrics: Dict, output_path: Path = None):
        """Plot and save confusion matrix."""
        cm = metrics["confusion_matrix"]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(config.CLASS_NAMES.values()),
                    yticklabels=list(config.CLASS_NAMES.values()))
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        
        if output_path is None:
            output_path = config.PLOTS_DIR / "confusion_matrix.png"
        
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"\n[Evaluate] Confusion matrix saved to {output_path}")
        plt.close()


def compare_models(supervised_metrics: Dict, ssl_metrics: Dict) -> Dict:
    """
    Compare supervised vs SSL results.
    
    This is the core research question:
    Does SSL improve cross-dataset generalization?
    
    Returns:
        Comparison dictionary
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: Supervised vs SSL")
    print("=" * 70)
    
    comparison = {
        "metric": [],
        "supervised": [],
        "ssl": [],
        "difference": [],
    }
    
    metrics_to_compare = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    
    for metric in metrics_to_compare:
        sup_val = supervised_metrics.get(metric, 0)
        ssl_val = ssl_metrics.get(metric, 0)
        diff = ssl_val - sup_val
        
        comparison["metric"].append(metric)
        comparison["supervised"].append(sup_val)
        comparison["ssl"].append(ssl_val)
        comparison["difference"].append(diff)
        
        symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"\n{metric}:")
        print(f"  Supervised: {sup_val:.4f}")
        print(f"  SSL:        {ssl_val:.4f}")
        print(f"  Difference: {diff:+.4f} {symbol}")
    
    print("\n" + "=" * 70)
    
    return comparison


# ============================================================================
# TEST THE EVALUATOR
# ============================================================================

if __name__ == "__main__":
    from dataset import MRIDataset
    from splitter import SubjectSplitter
    from dataloader import create_dataloaders
    from models import get_model
    from train import SupervisedTrainer
    
    print("\n" + "=" * 70)
    print("TESTING EVALUATOR")
    print("=" * 70)
    
    # Create dataset and dataloaders
    dataset = MRIDataset(dataset_type="mock")
    splitter = SubjectSplitter(dataset)
    train_loader, val_loader, test_loader = create_dataloaders(dataset, splitter, batch_size=16)
    
    device = torch.device(config.DEVICE)
    
    # Quick train
    print("\n[Test] Training supervised model...")
    model = get_model("classifier", num_classes=3, pretrained=False)
    trainer = SupervisedTrainer(model, device=device)
    trainer.train(train_loader, None, num_epochs=1, learning_rate=0.001)
    
    # Evaluate
    print("\n[Test] Evaluating model...")
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate_subject_level(test_loader)
    
    # Print results
    evaluator.print_metrics(metrics)
    
    print("\n" + "=" * 70)
    print("✓ Evaluator working correctly!")
    print("=" * 70)