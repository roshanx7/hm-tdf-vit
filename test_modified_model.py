#!/usr/bin/env python3
"""
Quick test script to verify the modified MFF-KAN architecture.
Run this to ensure the model can instantiate and perform forward passes.
"""

import sys
import os
import torch
import torch.nn as nn

# Add models path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

def test_model():
    """Test the modified MFF-KAN model"""
    
    print("=" * 60)
    print("Testing Modified MFF-KAN Architecture")
    print("=" * 60)
    
    # Test parameters
    num_features = 27  # Number of indicator features (from dataset)
    num_labels = 3     # Number of classes (non-FLD, mild, moderate/severe)
    drop_rate = 0.1
    batch_size = 4
    image_size = (224, 224)  # TinyViT expects 224x224 (was 448x448)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nDevice: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_labels}")
    
    # Import model
    try:
        from models import Mffkan as model_module
        print("\n✓ Successfully imported models.Mffkan")
    except ImportError as e:
        print(f"\n✗ Failed to import models: {e}")
        return False
    
    # Create model
    try:
        net = model_module.get_net(num_features, num_labels, drop_rate)
        print("✓ Successfully created model")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False
    
    # Move to device
    net = net.to(device)
    print(f"✓ Moved model to {device}")
    
    # Create dummy inputs
    try:
        X = torch.randn(batch_size, 3, image_size[0], image_size[1]).to(device)
        Xf = torch.randn(batch_size, num_features).to(device)
        print(f"✓ Created dummy inputs")
        print(f"  - Image tensor shape: {X.shape}")
        print(f"  - Feature tensor shape: {Xf.shape}")
    except Exception as e:
        print(f"✗ Failed to create inputs: {e}")
        return False
    
    # Forward pass
    try:
        net.eval()
        with torch.no_grad():
            # Enable debug output to verify dimensions
            logits = net(X, Xf, debug=True)
        print(f"✓ Forward pass successful")
        print(f"  - Output shape: {logits.shape}")
        print(f"  - Expected shape: ({batch_size}, {num_labels})")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify output shape
    if logits.shape != (batch_size, num_labels):
        print(f"✗ Output shape mismatch! Got {logits.shape}, expected ({batch_size}, {num_labels})")
        return False
    print("✓ Output shape verified")
    
    # Test loss calculation
    try:
        y = torch.randint(0, num_labels, (batch_size,)).to(device)
        loss = torch.nn.functional.cross_entropy(logits, y)
        print(f"✓ Cross-entropy loss calculation successful: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Loss calculation failed: {e}")
        return False
    
    # Test backward pass
    try:
        net.train()
        X.requires_grad = True
        Xf.requires_grad = True
        logits = net(X, Xf)
        y = torch.randint(0, num_labels, (batch_size,)).to(device)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        print(f"✓ Backward pass successful")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test regularization loss
    try:
        reg_loss = net.regularization_loss()
        print(f"✓ Regularization loss calculation successful: {reg_loss.item():.4f}")
    except Exception as e:
        print(f"✗ Regularization loss calculation failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)
    
    print("\nLayers in model:")
    for name, module in net.named_children():
        print(f"  - {name}: {module.__class__.__name__}")
        
        # Count parameters
        if hasattr(module, 'parameters'):
            params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"    Total params: {params:,}, Trainable: {trainable_params:,}")
    
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nTotal model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
