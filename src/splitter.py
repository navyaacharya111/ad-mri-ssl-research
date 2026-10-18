"""
Subject-Level Splitter: Prevent data leakage by splitting at subject level

CRITICAL PRINCIPLE: 
We split SUBJECTS, not slices.
All slices from Subject #001 stay together in the same split.
Subject #001 can NEVER appear in both training and testing.

Why? If Subject #001 is in training AND testing, the model has effectively
"seen" that patient before. The test result is misleading - it's not truly
independent.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json
import config


class SubjectSplitter:
    """
    Split subjects into train/validation/test while preventing leakage.
    
    This ensures:
    - Subjects don't overlap across splits
    - Class balance is maintained
    - Results are reproducible
    """
    
    def __init__(self, dataset, train_ratio: float = None, val_ratio: float = None, 
                 test_ratio: float = None, random_state: int = None):
        """
        Initialize splitter and create train/val/test splits.
        
        Args:
            dataset: MRIDataset object with subjects
            train_ratio: Fraction of subjects for training (default from config)
            val_ratio: Fraction for validation (default from config)
            test_ratio: Fraction for testing (default from config)
            random_state: Random seed for reproducibility (default from config)
        """
        self.dataset = dataset
        self.train_ratio = train_ratio or config.TRAIN_RATIO
        self.val_ratio = val_ratio or config.VAL_RATIO
        self.test_ratio = test_ratio or config.TEST_RATIO
        self.random_state = random_state or config.RANDOM_SEED
        
        # Verify ratios sum to 1.0
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total}")
        
        # Initialize split assignments
        self.train_subjects = set()
        self.val_subjects = set()
        self.test_subjects = set()
        
        # Perform the split
        self._create_splits()
    
    def _create_splits(self):
        """
        Create train/val/test splits at the SUBJECT level.
        
        Process:
        1. Get all unique subjects
        2. Get their diagnoses
        3. Use stratified shuffle to maintain class balance
        4. Assign to train/val/test
        """
        print("\n[Splitter] Creating subject-level splits...")
        
        # Get all subjects
        all_subjects = self.dataset.get_subject_ids()
        num_subjects = len(all_subjects)
        
        print(f"[Splitter] Total subjects: {num_subjects}")
        print(f"[Splitter] Split ratios: train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}")
        
        # Get diagnosis label for each subject (for stratification)
        subject_diagnoses = []
        for subject_id in all_subjects:
            subject_info = self.dataset.get_subject_info(subject_id)
            subject_diagnoses.append(subject_info["label"])
        
        subject_diagnoses = np.array(subject_diagnoses)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Create stratified splits
        # We do this in two steps: first split into train+val vs test, then split train+val into train vs val
        
        # Step 1: Split into train+val and test
        train_val_size = int(num_subjects * (self.train_ratio + self.val_ratio))
        test_size = num_subjects - train_val_size
        
        # Shuffle indices while maintaining class balance
        indices = np.arange(num_subjects)
        np.random.shuffle(indices)
        
        # Get class distribution
        unique_classes, class_counts = np.unique(subject_diagnoses, return_counts=True)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        # Stratified split: maintain class distribution in each split
        for class_label in unique_classes:
            class_mask = subject_diagnoses == class_label
            class_indices = np.where(class_mask)[0]
            np.random.shuffle(class_indices)
            
            # Calculate how many of this class go to each split
            n_class = len(class_indices)
            n_train = int(n_class * self.train_ratio)
            n_val = int(n_class * self.val_ratio)
            n_test = n_class - n_train - n_val
            
            train_indices.extend(class_indices[:n_train])
            val_indices.extend(class_indices[n_train:n_train+n_val])
            test_indices.extend(class_indices[n_train+n_val:])
        
        # Assign subjects to splits
        self.train_subjects = set([all_subjects[i] for i in train_indices])
        self.val_subjects = set([all_subjects[i] for i in val_indices])
        self.test_subjects = set([all_subjects[i] for i in test_indices])
        
        # Print summary
        print(f"\n[Splitter] Split assignments:")
        print(f"  Train: {len(self.train_subjects)} subjects ({len(self.train_subjects)/num_subjects*100:.1f}%)")
        print(f"  Val:   {len(self.val_subjects)} subjects ({len(self.val_subjects)/num_subjects*100:.1f}%)")
        print(f"  Test:  {len(self.test_subjects)} subjects ({len(self.test_subjects)/num_subjects*100:.1f}%)")
        
        # Verify no overlap
        self.verify_no_leakage()
    
    def verify_no_leakage(self) -> bool:
        """
        Verify that subjects don't overlap across splits.
        
        This is a safety check to catch bugs.
        
        Returns:
            True if splits are clean (no overlap)
            False if leakage detected
        """
        train_val_overlap = self.train_subjects & self.val_subjects
        train_test_overlap = self.train_subjects & self.test_subjects
        val_test_overlap = self.val_subjects & self.test_subjects
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("\n[Splitter] ❌ LEAKAGE DETECTED!")
            if train_val_overlap:
                print(f"  Train-Val overlap: {train_val_overlap}")
            if train_test_overlap:
                print(f"  Train-Test overlap: {train_test_overlap}")
            if val_test_overlap:
                print(f"  Val-Test overlap: {val_test_overlap}")
            return False
        else:
            print("[Splitter] ✓ No leakage detected - splits are clean!")
            return True
    
    def get_split(self, subject_id: str) -> str:
        """
        Get which split a subject belongs to.
        
        Args:
            subject_id: Subject identifier
        
        Returns:
            "train", "val", or "test"
        """
        if subject_id in self.train_subjects:
            return "train"
        elif subject_id in self.val_subjects:
            return "val"
        elif subject_id in self.test_subjects:
            return "test"
        else:
            raise ValueError(f"Subject {subject_id} not found in any split")
    
    def get_split_subjects(self, split: str) -> List[str]:
        """
        Get all subjects in a specific split.
        
        Args:
            split: "train", "val", or "test"
        
        Returns:
            List of subject IDs in that split
        """
        if split == "train":
            return list(self.train_subjects)
        elif split == "val":
            return list(self.val_subjects)
        elif split == "test":
            return list(self.test_subjects)
        else:
            raise ValueError(f"Invalid split: {split}")
    
    def save_splits(self, output_file: Path = None):
        """
        Save split assignments to a JSON file for reproducibility.
        
        Args:
            output_file: Where to save (default: data/splits/split_assignments.json)
        """
        if output_file is None:
            output_file = config.SPLITS_DIR / "split_assignments.json"
        
        split_dict = {
            "train": sorted(list(self.train_subjects)),
            "val": sorted(list(self.val_subjects)),
            "test": sorted(list(self.test_subjects)),
            "random_state": self.random_state,
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(split_dict, f, indent=2)
        
        print(f"\n[Splitter] Split assignments saved to {output_file}")
    
    def load_splits(self, input_file: Path):
        """
        Load split assignments from a JSON file.
        
        Args:
            input_file: Path to split file
        """
        with open(input_file, 'r') as f:
            split_dict = json.load(f)
        
        self.train_subjects = set(split_dict["train"])
        self.val_subjects = set(split_dict["val"])
        self.test_subjects = set(split_dict["test"])
        
        print(f"[Splitter] Split assignments loaded from {input_file}")


# ============================================================================
# TEST THE SPLITTER
# ============================================================================

if __name__ == "__main__":
    from dataset import MRIDataset
    
    print("\n" + "=" * 70)
    print("TESTING SUBJECT-LEVEL SPLITTER")
    print("=" * 70)
    
    # Load dataset
    dataset = MRIDataset(dataset_type="mock")
    
    # Create splitter
    splitter = SubjectSplitter(dataset)
    
    # Show split assignments
    print(f"\nTrain subjects: {sorted(list(splitter.train_subjects))[:3]}...")
    print(f"Val subjects:   {sorted(list(splitter.val_subjects))}")
    print(f"Test subjects:  {sorted(list(splitter.test_subjects))}")
    
    # Test getting split for a subject
    train_subject = list(splitter.train_subjects)[0]
    split = splitter.get_split(train_subject)
    print(f"\n{train_subject} is in: {split}")
    
    # Save splits for reproducibility
    splitter.save_splits()
    
    print("\n" + "=" * 70)
    print("✓ Splitter module working correctly!")
    print("=" * 70)