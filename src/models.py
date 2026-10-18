"""
Neural Network Models: ResNet-18 for supervised and self-supervised learning

Two architectures:
1. ResNet18Classifier - Outputs class probabilities (CN/MCI/AD)
2. ResNet18Encoder - Outputs feature embeddings (for SSL pretraining)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torchvision.models as models
import config


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 for supervised classification.
    
    Input: (batch, 3, 128, 128) - 3-channel MRI images
    Output: (batch, 3) - Logits for CN/MCI/AD
    
    This is used for:
    - Supervised baseline training
    - Fine-tuning after SSL pretraining
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = False):
        """
        Initialize ResNet-18 classifier.
        
        Args:
            num_classes: Number of output classes (3 for CN/MCI/AD)
            pretrained: If True, use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace the final classification layer
        # ResNet-18 outputs 512-dimensional features by default
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, 128, 128)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.backbone(x)


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 encoder for self-supervised learning.
    
    Input: (batch, 3, 128, 128) - 3-channel MRI images
    Output: (batch, 512) - Feature embeddings
    
    This is used for:
    - Pretraining with SimCLR (learns representations without labels)
    - Extracting features for downstream tasks
    """
    
    def __init__(self, pretrained: bool = False):
        """
        Initialize ResNet-18 encoder.
        
        Args:
            pretrained: If True, use ImageNet pretrained weights
        """
        super().__init__()
        
        # Load ResNet-18
        backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        # We only want the feature extraction part
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        
        # Output dimension (ResNet-18 outputs 512 features)
        self.feature_dim = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - extract features.
        
        Args:
            x: Input tensor of shape (batch, 3, 128, 128)
        
        Returns:
            Feature tensor of shape (batch, 512)
        """
        # Pass through encoder
        features = self.encoder(x)  # (batch, 512, 1, 1)
        
        # Flatten to (batch, 512)
        features = features.view(features.size(0), -1)
        
        return features


class SimCLRProjectionHead(nn.Module):
    """
    Projection head for SimCLR self-supervised learning.
    
    SimCLR uses a 2-layer MLP projection head:
    - Input: 512-dim features from ResNet-18
    - Hidden: 512-dim
    - Output: 128-dim projections
    
    The contrastive loss is computed in this 128-dim space.
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, 
                 output_dim: int = 128):
        """
        Initialize projection head.
        
        Args:
            input_dim: Input dimension (512 from ResNet-18)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (typically 128)
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 512)
        
        Returns:
            Projections of shape (batch, 128)
        """
        return self.net(x)


class SimCLRModel(nn.Module):
    """
    SimCLR model = Encoder + Projection Head.
    
    During pretraining:
    - Input: Two augmented views of the same image
    - Encoder: Extracts 512-dim features
    - Projection Head: Projects to 128-dim space
    - Loss: Contrastive loss in projection space
    
    After pretraining:
    - Drop the projection head
    - Use encoder to extract features for downstream tasks
    """
    
    def __init__(self, encoder: ResNet18Encoder = None, projection_dim: int = 128):
        """
        Initialize SimCLR model.
        
        Args:
            encoder: ResNet18Encoder (if None, creates new one)
            projection_dim: Dimension of projection head output
        """
        super().__init__()
        
        if encoder is None:
            self.encoder = ResNet18Encoder(pretrained=False)
        else:
            self.encoder = encoder
        
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=self.encoder.feature_dim,
            output_dim=projection_dim
        )
        
        self.projection_dim = projection_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - project features to lower-dim space.
        
        Args:
            x: Input tensor of shape (batch, 3, 128, 128)
        
        Returns:
            Projected features of shape (batch, projection_dim)
        """
        # Extract features
        features = self.encoder(x)  # (batch, 512)
        
        # Project to lower-dim space
        projections = self.projection_head(features)  # (batch, 128)
        
        return projections
    
    def get_encoder(self) -> ResNet18Encoder:
        """Get the encoder (used for downstream tasks after pretraining)."""
        return self.encoder


def get_model(model_type: str, num_classes: int = 3, 
              pretrained: bool = False) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: "classifier", "encoder", or "simclr"
        num_classes: Number of classes (for classifier)
        pretrained: Use ImageNet pretrained weights
    
    Returns:
        PyTorch model
    """
    if model_type == "classifier":
        return ResNet18Classifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "encoder":
        return ResNet18Encoder(pretrained=pretrained)
    elif model_type == "simclr":
        encoder = ResNet18Encoder(pretrained=pretrained)
        return SimCLRModel(encoder=encoder, projection_dim=config.SSL_CONFIG["projection_dim"])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# TEST THE MODELS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING NEURAL NETWORK MODELS")
    print("=" * 70)
    
    # Dummy input
    device = torch.device(config.DEVICE)
    batch_size = 2
    x = torch.randn(batch_size, 3, 128, 128).to(device)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Device: {device}")
    
    # Test Classifier
    print("\n--- ResNet18Classifier ---")
    classifier = get_model("classifier", num_classes=3, pretrained=False).to(device)
    logits = classifier(x)
    print(f"Output shape: {logits.shape}")  # Should be (2, 3)
    print(f"Output: {logits}")
    
    # Test Encoder
    print("\n--- ResNet18Encoder ---")
    encoder = get_model("encoder", pretrained=False).to(device)
    features = encoder(x)
    print(f"Output shape: {features.shape}")  # Should be (2, 512)
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    # Test SimCLR
    print("\n--- SimCLRModel ---")
    simclr = get_model("simclr", pretrained=False).to(device)
    projections = simclr(x)
    print(f"Output shape: {projections.shape}")  # Should be (2, 128)
    print(f"Projection range: [{projections.min():.3f}, {projections.max():.3f}]")
    
    # Count parameters
    print("\n--- Model Sizes ---")
    classifier_params = sum(p.numel() for p in classifier.parameters())
    encoder_params = sum(p.numel() for p in encoder.parameters())
    simclr_params = sum(p.numel() for p in simclr.parameters())
    
    print(f"Classifier params: {classifier_params:,}")
    print(f"Encoder params: {encoder_params:,}")
    print(f"SimCLR params: {simclr_params:,}")
    
    print("\n" + "=" * 70)
    print("✓ All models working correctly!")
    print("=" * 70)