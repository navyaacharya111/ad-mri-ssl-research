"""
PyTorch DataLoader: Create batches of MRI data for training/validation/testing

This module bridges the gap between:
- Raw data on disk (handled by dataset.py)
- Subjects split into train/val/test (handled by splitter.py)
- Neural network training (needs batches of data)

Key principle: Apply augmentations ONLY to training data.
Validation and test data should be realistic, not augmented.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
from dataset import MRIDataset, extract_2d_slices


class MRISliceDataset(Dataset):
    """
    PyTorch Dataset that returns individual MRI slices.
    
    This dataset:
    1. Loads 3D MRI volumes for subjects in a specific split
    2. Extracts 2D slices from each volume
    3. Applies augmentations (only for training)
    4. Returns slices as PyTorch tensors
    """
    
    def __init__(self, mri_dataset: MRIDataset, splitter, split: str = "train",
                 augment: bool = True):
        """
        Initialize slice dataset.
        
        Args:
            mri_dataset: MRIDataset object
            splitter: SubjectSplitter object with train/val/test assignments
            split: "train", "val", or "test"
            augment: Whether to apply augmentations (should only be True for train)
        """
        self.mri_dataset = mri_dataset
        self.splitter = splitter
        self.split = split
        self.augment = augment
        
        # Get subjects for this split
        self.subject_ids = splitter.get_split_subjects(split)
        
        print(f"\n[DataLoader] Creating {split} dataset with {len(self.subject_ids)} subjects")
        
        # Pre-load all slices (for faster training)
        # In real scenario with huge datasets, you'd load on-the-fly instead
        self.slices = []
        self.labels = []
        self.subject_ids_per_slice = []
        
        for subject_id in self.subject_ids:
            # Load subject's 3D volume
            volume, diagnosis, _ = mri_dataset.load_subject(subject_id)
            
            # Extract 2D slices
            slices = extract_2d_slices(volume, target_size=config.MRI_SIZE)
            
            # Get label
            subject_info = mri_dataset.get_subject_info(subject_id)
            label = subject_info["label"]
            
            # Store slices and labels
            for slice_2d in slices:
                self.slices.append(slice_2d)
                self.labels.append(label)
                self.subject_ids_per_slice.append(subject_id)
        
        print(f"[DataLoader] Extracted {len(self.slices)} slices from {len(self.subject_ids)} subjects")
        
        # Set up augmentation pipeline
        if augment:
            self.transforms = self._get_train_transforms()
        else:
            self.transforms = self._get_val_transforms()
    
    def _get_train_transforms(self):
        """
        Augmentation pipeline for TRAINING data.
        
        Augmentations make training data more varied, which helps the model
        learn more robust features. We apply rotation, elastic deformation, etc.
        """
        return A.Compose([
            A.Rotate(limit=config.AUGMENTATION_CONFIG["rotation_range"], p=0.5),
            A.ElasticTransform(alpha=30, sigma=5, p=0.3) if config.AUGMENTATION_CONFIG["elastic_deformation"] else A.NoOp(),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.3),  # light noise (std 1-3% of intensity range)
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0),  # slices are 0-1 -> [-1, 1]
            ToTensorV2(),  # Convert to PyTorch tensor
        ], is_check_shapes=False)
    
    def _get_val_transforms(self):
        """
        Augmentation pipeline for VALIDATION/TEST data.
        
        NO augmentations - just normalize and convert to tensor.
        We want test data to be realistic, not augmented.
        """
        return A.Compose([
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0),  # slices are 0-1 -> [-1, 1]
            ToTensorV2(),  # Convert to PyTorch tensor
        ], is_check_shapes=False)
    
    def __len__(self) -> int:
        """Return total number of slices."""
        return len(self.slices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single slice.
        
        Args:
            idx: Index of slice
        
        Returns:
            (image_tensor, label, subject_id)
            image_tensor: Shape (3, 128, 128) - 3-channel 128x128 image
            label: 0=CN, 1=MCI, 2=AD
            subject_id: Subject identifier (for subject-level aggregation later)
        """
        slice_2d = self.slices[idx]  # (128, 128) grayscale
        label = self.labels[idx]
        subject_id = self.subject_ids_per_slice[idx]
        
        # Convert grayscale to 3-channel (RGB)
        # PyTorch pretrained models expect 3 channels
        slice_3ch = np.stack([slice_2d, slice_2d, slice_2d], axis=2)  # (128, 128, 3)
        
        # Apply augmentations and convert to tensor
        augmented = self.transforms(image=slice_3ch)
        image_tensor = augmented['image']  # (3, 128, 128) tensor
        
        return image_tensor, label, subject_id


def create_dataloaders(mri_dataset: MRIDataset, splitter, 
                       batch_size: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test.
    
    Args:
        mri_dataset: MRIDataset object
        splitter: SubjectSplitter object
        batch_size: Batch size (default from config)
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.SUPERVISED_CONFIG["batch_size"]
    
    print("\n[DataLoader] Creating PyTorch DataLoaders...")
    
    # Create datasets
    train_dataset = MRISliceDataset(mri_dataset, splitter, split="train", augment=True)
    val_dataset = MRISliceDataset(mri_dataset, splitter, split="val", augment=False)
    test_dataset = MRISliceDataset(mri_dataset, splitter, split="test", augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=0,  # Set to 0 for Windows
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=0,
        pin_memory=False
    )
    
    print(f"[DataLoader] Train loader: {len(train_loader)} batches")
    print(f"[DataLoader] Val loader:   {len(val_loader)} batches")
    print(f"[DataLoader] Test loader:  {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# TEST THE DATALOADER
# ============================================================================

if __name__ == "__main__":
    from dataset import MRIDataset
    from splitter import SubjectSplitter
    
    print("\n" + "=" * 70)
    print("TESTING PYTORCH DATALOADER")
    print("=" * 70)
    
    # Create dataset and splitter
    dataset = MRIDataset(dataset_type="mock")
    splitter = SubjectSplitter(dataset)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(dataset, splitter, batch_size=2)
    
    # Get one batch from train loader
    print("\n[Test] Getting one batch from training set...")
    batch_images, batch_labels, batch_subject_ids = next(iter(train_loader))
    
    print(f"[Test] Batch shape: {batch_images.shape}")  # Should be (batch_size, 3, 128, 128)
    print(f"[Test] Batch labels: {batch_labels}")
    print(f"[Test] Batch subject IDs: {batch_subject_ids}")
    print(f"[Test] Image dtype: {batch_images.dtype}")
    print(f"[Test] Image range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
    
    print("\n" + "=" * 70)
    print("✓ DataLoader module working correctly!")
    print("=" * 70)