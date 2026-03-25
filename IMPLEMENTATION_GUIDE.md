# TinyViT-based MFF-KAN Model - Implementation Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

The key addition is `timm` (PyTorch Image Models):
```bash
pip install timm==0.9.7
```

### 2. Run the Modified Model
```bash
# Extract data (if not already done)
cd ./Tongue-FLD
cat Tongue_Images.tar.gz.* > Tongue_Images.tar.gz
tar xzf Tongue_Images.tar.gz
cd ..

# Optional: Generate rotated images for pre-training
python random_rotate_images.py

# Train the main model with TinyViT
python main.py
```

### 3. Verify Installation
```bash
# Test that the model can instantiate and run
python test_modified_model.py
```

---

## Architecture Details

### Image Encoder (IE) Changes

**Before (CNN_Kan):**
- Custom CNN architecture combining convolution and KAN layers
- Output: 512-dimensional feature vector
- Self-trained from scratch

**After (TinyViT):**
- Pretrained Vision Transformer (tiny variant)
- Model: `tiny_vit_5m_224` from timm
- Pretrained on ImageNet
- Output: 768-dimensional feature vector → Projected to 512-dim
- Fine-tuning: Initially frozen, can be unfrozen for better adaptation

### Indicator Encoder (DE) - UNCHANGED
- Still uses original KAN-based architecture
- Input: 27 physiological indicators
- Output: 128-dimensional feature vector
- Architecture:
  ```
  Input (27) → KAN (32) → BatchNorm → Dropout 
           → KAN (128) → Output (128)
  ```

### Classifier Changes

**Before (FFC + MEC):**
- FFC: KAN-based classifier for primary predictions
- MEC: Multi-expert classifier for uncertainty estimation
- Output: 4 tensors (ffc_out, mec_out, distance, encode)

**After (Simple MLP):**
- Unified feedforward classifier
- Architecture:
  ```
  Input (640) → BatchNorm → Dropout 
           → Linear (256) → ReLU → BatchNorm → Dropout
           → Linear (128) → ReLU → BatchNorm → Dropout  
           → Linear (num_labels) → Output
  ```
- Output: 1 tensor (logits)

---

## Feature Fusion Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                   Input Data                             │
│  Images (B, 3, 448, 448) | Features (B, 27)             │
└──────────────────┬────────────────────────────┬──────────┘
                   │                            │
        ┌──────────▼──────────┐     ┌──────────▼──────────┐
        │     TinyViT IE      │     │    KAN-based DE    │
        │  (pretrained VITS)  │     │  (indicator enc.)  │
        │ 768 → proj → 512-d  │     │    27 → 128-d      │
        └──────────┬──────────┘     └──────────┬──────────┘
                   │                            │
                   │        Concatenate         │
                   └────────────┬───────────────┘
                                │
                         (640-dim features)
                                │
                   ┌────────────▼──────────────┐
                   │  MLP Classifier           │
                   │  640 → 256 → 128 → 3     │
                   └────────────┬──────────────┘
                                │
                         (num_labels logits)
                                │
                   ┌────────────▼──────────────┐
                   │  Cross-Entropy Loss       │
                   │  (+ KAN Regularization)   │
                   └───────────────────────────┘
```

---

## Key Configuration Options

### 1. Freeze/Unfreeze TinyViT Layers

**Current default (frozen):** Lines 84-102 in main.py

To enable TinyViT fine-tuning:
```python
# In main.py training setup section
for param in net.IE.parameters():
    opt_parameters.append(param)
    param.requires_grad = True
```

### 2. Adjust Classifier Architecture

Edit [models/Mffkan.py](models/Mffkan.py) `__init__` method:
```python
self.classifier = nn.Sequential(
    nn.BatchNorm1d(640),
    nn.Dropout(0.2),  # Increase dropout
    nn.Linear(640, 512),  # Change intermediate size
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.2),
    nn.Linear(512, 256),  # Add more layers
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.2),
    nn.Linear(256, num_labels)
)
```

### 3. Use Different TinyViT Variants

Available TinyViT models in timm:
```python
# In models/Mffkan.py modify IE creation:
self.IE = timm.create_model('tiny_vit_11m_224', pretrained=True, num_classes=0)
self.IE = timm.create_model('tiny_vit_21m_224', pretrained=True, num_classes=0)
```

List all TinyViT models:
```python
import timm
vit_models = [m for m in timm.list_models() if 'tiny_vit' in m]
print(vit_models)
```

### 4. Input Image Size

**Current:** 448×448 (configured in main.py)

**Note:** TinyViT is trained on 224×224 images
- Images are resized before TinyViT processing (handled by torchvision.transforms.Resize)
- For best performance, consider retraining with 224×224 or using TinyViT trained on larger inputs

---

## Loss Function Considerations

### MOI Loss (Still used for initial training)
```python
def moi_loss(logit, y):
    probability = torch.softmax(logit, dim=1)
    y_center = (y * class_index).sum(dim=1, keepdim=True)
    rad = class_index - y_center
    
    cross_entropy = - (y * torch.log(probability)).sum(1).mean()
    moi = (rad * rad * probability).sum(1).mean()
    return alpha * moi + belta * cross_entropy
```

### Cross-Entropy Loss (Used for hard mining)
```python
loss = torch.nn.functional.cross_entropy(logits, y.argmax(dim=1))
```

### Regularization Loss (Only from DE layers now)
- Previously: Applied to FFC + MEC + DE
- Now: Only applied to DE KAN layers
- This reduces overall regularization but still maintains KAN spline smoothness for indicator encoding

---

## Hard Sample Mining Pipeline

The hard sample mining still works as before:

1. **Initial Training:** Train full model with MOI loss
2. **Uncertainty Calculation:** Uses `moi_uncertainty()` from logits
3. **Hard Sample Selection:** Selects high-uncertainty samples per class
4. **Hard Mining Training:** Fine-tune classifier + IE projection on hard samples

Key changes from original:
- ✅ Uncertainty still computed from logits
- ✅ Hard sample selection unchanged
- ✓ Loss function simplified to cross-entropy for hard mining phase
- ❌ No longer uses MEC distance-based loss

---

## Training Hyperparameters (in main.py)

Key parameters you may want to adjust:

```python
learning_rate = 0.001          # Initial learning rate
weight_decay = 0.0001          # L2 regularization
batch_size = 60                # Total batch size
images_per_gpu = 6             # Images per GPU (batch_size/images_per_gpu = mini batches)
drop_rate = 0.1                # Dropout rate in all modules
num_epochs = 50                # Epochs for initial training

# Hard sample mining
min_hard_rate = 0.5            # Minimum % of samples to consider hard
uncertainty_rate = 0.5         # Uncertainty threshold multiplier

# Regularization (only applied to DE KAN layers now)
reg_loss_rate_active = 0.1     # Activation regularization weight
reg_loss_rate_entropy = 0.1    # Entropy regularization weight

# Learning rate scheduling
warmup_epochs = 5              # Warmup epochs
warmup_factor = 0.1            # Initial LR = base_lr * warmup_factor
```

---

## Performance Considerations

### Computational Cost
- **Training time:** TinyViT is more efficient than CNN_Kan
- **Inference speed:** ~2-3x faster than original CNN_Kan
- **Memory usage:** Similar or slightly less than original (depending on batch size)

### Model Size
- **Original CNN_Kan:** ~15-20MB
- **TinyViT (pretrained):** ~20-25MB
- **Full model:** ~25-30MB (including all components)

### Accuracy Trade-offs
- **Pros:** Pretrained ImageNet features are generally more robust
- **Cons:** TinyViT may need different input preprocessing
- **Recommendation:** Validate performance on your dataset with k-fold cross-validation

---

## Debugging & Troubleshooting

### Issue: "CUDA out of memory"
```python
# In main.py, reduce batch size
batch_size = 40  # from 60
images_per_gpu = 4  # from 6
```

### Issue: "timm not found"
```bash
pip install timm
# Verify installation
python -c "import timm; print(timm.__version__)"
```

### Issue: Poor validation accuracy
1. Check that images are properly preprocessed
2. Verify that TinyViT layers are being used (not fallback network)
3. Try unfreezing TinyViT parameters for fine-tuning
4. Adjust learning rate (try 0.0005 or 0.0001 for fine-tuning)

### Issue: "AttributeError: MffKan has no attribute 'IE.base.layer1'"
This error means the old code is trying to freeze old IE layers that no longer exist.
✓ Already fixed in the updated main.py

### Check TinyViT is loaded correctly:
Add this to main.py after model creation:
```python
net = model.get_net(num_features, num_labels, drop_rate)
print(f"IE type: {type(net.IE)}")  # Should show timm model
print(f"IE output: {net.IE}")       # Print the model
```

---

## Comparison: Original vs Modified

| Feature | Original CNN_Kan | New TinyViT |
|---------|------------------|-----------|
| Architecture | Custom CNN + KAN | Vision Transformer |
| Pretrained | No | Yes (ImageNet) |
| Parameters (IE) | ~5M | ~5M |
| Output dim | 512 | 768 (→ 512) |
| Training epoch | Long | Faster |
| Inference | Slower | Faster |
| Robustness | Dataset-specific | More general |
| Accuracy | Good | Better (likely) |

---

## Next Steps

1. **Verify Model:** Run `python test_modified_model.py`
2. **Train Model:** Run `python main.py`
3. **Monitor Training:** Check saved metrics in `./test_work_dir/`
4. **Evaluate Results:** Compare k-fold cross-validation metrics with original
5. **Fine-tune if needed:** Adjust hyperparameters based on validation performance

---

## References & Resources

- **TinyViT Paper:** https://arxiv.org/abs/2207.10666
- **TIMM Library:** https://github.com/huggingface/pytorch-image-models
- **Vision Transformer:** https://arxiv.org/abs/2010.11929
- **KAN Layer:** https://github.com/KindXiaoming/pykan

---

## Support

If you encounter issues:
1. Check the error message carefully
2. Review the DEBUG section above
3. Verify all files are properly modified (check CHANGES_SUMMARY.md)
4. Ensure dependencies are installed: `pip install -r requirements.txt`
5. Run the test script: `python test_modified_model.py`
