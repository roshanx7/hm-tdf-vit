# MFF-KAN Architecture Modification Summary

## Overview
The MFF-KAN model has been successfully modified to use **TinyViT as the Image Encoder (IE)** while maintaining backward compatibility with the rest of the training pipeline.

---

## Changes Made

### 1. **Modified Files**

#### [models/Mffkan.py](models/Mffkan.py)
**Key Changes:**
- **Removed:** CNN_Kan-based Image Encoder and associated imports
- **Replaced with:** TinyViT (from `timm` library)
  - Model: `tiny_vit_5m_224` (pretrained, ImageNet)
  - Output dimension: 768
  - Feature extractor mode (num_classes=0)

- **Added Projection Layer:**
  - Projects 768-dim TinyViT features → 512-dim
  - Linear layer: `nn.Linear(768, 512)`

- **Modified Classifier:**
  - **Removed:** FFC and MEC (KAN-based classifiers)
  - **Replaced with:** Simple neural network classifier
  ```python
  self.classifier = nn.Sequential(
      nn.BatchNorm1d(640),
      nn.Dropout(drop_rate),
      nn.Linear(640, 256),
      nn.ReLU(),
      nn.BatchNorm1d(256),
      nn.Dropout(drop_rate),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.BatchNorm1d(128),
      nn.Dropout(drop_rate),
      nn.Linear(128, num_labels)
  )
  ```

- **Updated Forward Pass:**
  - Now returns: `logits` (simplified single output)
  - Format: `(batch_size, num_labels)`
  - Previous version returned 4 outputs: `(ffc_out, mec_out, distance, encode)`

- **Indicator Encoder (DE):**
  - **UNCHANGED** - Still uses original KAN-based architecture
  - Outputs 128-dimensional feature vector

- **Feature Fusion:**
  - Concatenates 512-dim (IE) + 128-dim (DE) = **640-dim fused features**
  - Passed to classifier for final prediction

#### [main.py](main.py)
**Key Changes:**

1. **Parameter Setup (Lines 84-102):**
   - Freezes all TinyViT IE parameters initially: `net.IE.parameters()`
   - Fine-tunes: `ie_proj`, `DE`, `classifier`
   - Removed: `net.IE.base.layerKAN`, `net.IE.base.layer1/2` (no longer exist)

2. **Model Forward Calls:**
   - **Before:** `logits, _, _, _ = net(X, Xf)`
   - **After:** `logits = net(X, Xf)`
   - Updated in 3 locations:
     - Initial training loop (line ~113)
     - Validation loop (line ~141)
     - Hard mining training loop (line ~207)
     - Hard mining validation loop (line ~245)

3. **Hard Sample Mining Updates:**
   - Removed dependency on `distance` output (no longer computed)
   - Hard mining now uses:
     - **Uncertainty:** Still computed from logits via `moi_uncertianty()`
     - **Loss:** Switched from `(distance * y).mean()` to standard cross-entropy
     - Loss: `torch.nn.functional.cross_entropy(logits, y.argmax(dim=1))`

4. **Final Evaluation (Lines 275-330):**
   - Removed MEC-based output switching
   - Now uses logits directly without uncertainty-based thresholding
   - Before: `if uncertain: use MEC_out else use FCC_out`
   - After: `use logits directly`

#### [requirements.txt](requirements.txt)
- **Added:** `timm=0.9.7` (PyTorch Image Models library)

---

## Architecture Comparison

### **Original Architecture:**
```
Image → CNN_Kan → 512-dim features
Indicators → DE (KAN) → 128-dim features
         ↓
    Concatenate (640-dim)
         ↓
    FFC (KAN classifier) → logits
    MEC (KAN expert) → distance/uncertainty
         ↓
    Distance-based loss + MOI loss
```

### **New Architecture:**
```
Image → TinyViT (pretrained) → 768-dim features
              ↓
         Projection layer → 512-dim features
         
Indicators → DE (KAN, unchanged) → 128-dim features
         ↓
    Concatenate (640-dim)
         ↓
    CNN Classifier (MLP) → logits
         ↓
    Cross-entropy loss + MOI loss
```

---

## Key Improvements

1. **Better Visual Features:** TinyViT is pretrained on ImageNet with efficient attention mechanisms
2. **Reduced Parameters:** Simplified classifier (MLP) instead of two KAN networks
3. **Faster Inference:** TinyViT is optimized for speed
4. **Flexibility:** Easier to fine-tune TinyViT layers if needed
5. **Maintainability:** Removed complex MOI distance calculations

---

## Usage Instructions

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python main.py
```

### Optional: Fine-tune TinyViT
To enable TinyViT fine-tuning, modify [main.py](main.py) line ~91:
```python
# Unfreeze IE for fine-tuning
for param in net.IE.parameters():
    opt_parameters.append(param)
    param.requires_grad = True
```

---

## Dimension Tracking

| Component | Input | Output |
|-----------|-------|--------|
| TinyViT | (B, 3, 224, 224) | (B, 768) |
| IE Projection | (B, 768) | (B, 512) |
| DE | (B, num_features) | (B, 128) |
| Concatenate | (B, 512) + (B, 128) | (B, 640) |
| Classifier | (B, 640) | (B, num_labels) |

*B = batch size*

---

## Backward Compatibility Notes

- ✅ Still uses `moi_loss()` for training
- ✅ Still uses `moi_uncertianty()` for hard sample selection
- ✅ Still uses `regularization_loss()` (only from DE now)
- ✅ Hard sample mining pipeline still functional
- ❌ No longer supports accessing: `net.IE.base`, `net.FFC`, `net.MEC`
- ❌ Forward pass now returns 1 output instead of 4

---

## Debugging Tips

1. **Check TinyViT installation:**
   ```python
   import timm
   print(timm.list_models('tiny_vit'))
   ```

2. **Verify feature dimensions:**
   ```python
   # Add to training loop
   f_i = net.IE(X)
   f_p = net.DE(Xf)
   print(f"Image features: {f_i.shape}, Indicator features: {f_p.shape}")
   ```

3. **Monitor gradient flow:**
   ```python
   for name, param in net.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad norm = {param.grad.norm()}")
   ```

---

## References
- TinyViT Paper: [arXiv:2207.10666](https://arxiv.org/abs/2207.10666)
- TIMM Library: [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- Original HM-TDF: [github.com/MLDMXM2017/HM-TDF](https://github.com/MLDMXM2017/HM-TDF)
