# Code Changes Reference - Before & After Comparison

## 1. Model Architecture Changes

### models/Mffkan.py - Imports

**BEFORE:**
```python
from CNNKAN import CNNKan as CNN_Kan
```

**AFTER:**
```python
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Install with: pip install timm")
```

---

### models/Mffkan.py - __init__ Method

**BEFORE:**
```python
class MffKan(nn.Module): 
    def __init__(self, num_labels, num_features, drop_rate):
        super().__init__()
        self.num_features = num_features
        self.kan_linears = []

        self.IE = CNN_Kan()
        self.DE = nn.Sequential(KAN_Linear(num_features, 32, ...),
                                nn.BatchNorm1d(32),
                                nn.Dropout(drop_rate),
                                KAN_Linear(32, 128, ...))

        self.FFC = nn.Sequential(nn.BatchNorm1d(640),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(640, 32, ...),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(32, num_labels, ...))
        
        self.M_orient = nn.Parameter(...)
        self.M_orient_abs = nn.Parameter(...)
        self.expert_num = self.M_orient.shape[-1]
        self.MEC = nn.Sequential(nn.BatchNorm1d(640),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(640, 32, ...),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(32, self.expert_num, ...),
                                  nn.Softsign())
```

**AFTER:**
```python
class MffKan(nn.Module): 
    def __init__(self, num_labels, num_features, drop_rate):
        super().__init__()
        self.num_features = num_features
        self.kan_linears = []
        self.ie_dim = 768  # TinyViT output
        self.de_dim = 128  # DE output
        self.fused_dim = 640  # 512 + 128

        # Image Encoder: TinyViT
        if TIMM_AVAILABLE:
            self.IE = timm.create_model('tiny_vit_5m_224', 
                                       pretrained=True, num_classes=0)
        else:
            # Fallback feature extractor
            self.IE = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.ie_dim = 64

        # IE projection layer
        self.ie_proj = nn.Linear(self.ie_dim, 512)
        self.ie_dim = 512
        self.fused_dim = self.ie_dim + self.de_dim

        # Indicator Encoder (UNCHANGED)
        self.DE = nn.Sequential(KAN_Linear(num_features, 32, ...),
                                nn.BatchNorm1d(32),
                                nn.Dropout(drop_rate),
                                KAN_Linear(32, 128, ...))

        # CNN-based Classifier (replaces FFC + MEC)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.fused_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(drop_rate),
            nn.Linear(128, num_labels)
        )

        # Collect KAN layers (only from DE now)
        for module in self.DE.modules():
            if isinstance(module, KAN_Linear):
                self.kan_linears.append(module)
```

---

### models/Mffkan.py - forward() Method

**BEFORE:**
```python
def forward(self, X, f_p):
    f_i = self.IE(X)
    f_p = self.DE(f_p)
    f_f = torch.cat((f_i, f_p), dim=1)
    ffc_out = self.FFC(f_f)

    encode = self.MEC(f_f)
    distance = (encode.unsqueeze(1) * self.M_orient_abs - self.M_orient).pow(2).mean(2)
    mec_out = - distance
    return ffc_out, mec_out, distance, encode
```

**AFTER:**
```python
def forward(self, X, f_p):
    """
    Forward pass for MFF-KAN with TinyViT Image Encoder.
    
    Args:
        X: Image tensor (batch_size, 3, height, width)
        f_p: Indicator features tensor (batch_size, num_features)
    
    Returns:
        logits: Classification logits (batch_size, num_labels)
    """
    # Image Encoder: TinyViT
    f_i = self.IE(X)  # (batch_size, ie_dim)
    
    # Apply projection layer
    f_i = self.ie_proj(f_i)  # (batch_size, 512)
    
    # Indicator Encoder: DE
    f_p = self.DE(f_p)  # (batch_size, 128)
    
    # Concatenate features
    f_f = torch.cat((f_i, f_p), dim=1)  # (batch_size, 640)
    
    # CNN-based Classifier
    logits = self.classifier(f_f)  # (batch_size, num_labels)
    
    return logits
```

---

## 2. Training Code Changes

### main.py - Parameter Setup

**BEFORE:**
```python
opt_parameters = []
for param in net.IE.base.layerKAN.parameters():
    opt_parameters.append(param)
for param in net.DE.parameters():
    opt_parameters.append(param)
for param in net.FFC.parameters():
    opt_parameters.append(param)

# Freeze layers
for param in net.MEC.parameters():
    param.requires_grad = False
for param in net.IE.base.layer1.parameters():
    param.requires_grad = False
for param in net.IE.base.layer2.parameters():
    param.requires_grad = False
```

**AFTER:**
```python
opt_parameters = []

# Freeze all IE parameters initially
for param in net.IE.parameters():
    param.requires_grad = False

# Fine-tune only the projection layer and other components
for param in net.ie_proj.parameters():
    opt_parameters.append(param)
    param.requires_grad = True

for param in net.DE.parameters():
    opt_parameters.append(param)

for param in net.classifier.parameters():
    opt_parameters.append(param)
```

---

### main.py - Training Loop

**BEFORE:**
```python
for img_tensor, feature_tensor, labels, _ in data_iter(...): 
    X, Xf, y = img_tensor.to(device), feature_tensor.to(device), labels.to(device)
    logits, _, _, _ = net(X, Xf)

    cls_loss = moi_loss(logits, y)
    reg_loss = net.regularization_loss(reg_loss_rate_active, reg_loss_rate_entropy)
    l = cls_loss + reg_loss
```

**AFTER:**
```python
for img_tensor, feature_tensor, labels, _ in data_iter(...): 
    X, Xf, y = img_tensor.to(device), feature_tensor.to(device), labels.to(device)
    logits = net(X, Xf)

    cls_loss = moi_loss(logits, y)
    reg_loss = net.regularization_loss(reg_loss_rate_active, reg_loss_rate_entropy)
    l = cls_loss + reg_loss
```

---

### main.py - Hard Mining Setup

**BEFORE:**
```python
for param in net.MEC.parameters():
    param.requires_grad = True
for param in net.IE.parameters():
    param.requires_grad = False
for param in net.DE.parameters():
    param.requires_grad = False
for param in net.FFC.parameters():
    param.requires_grad = False

hard_opt_parameters = []
for param in net.MEC.parameters():
    hard_opt_parameters.append(param)
```

**AFTER:**
```python
# Fine-tune the classifier
for param in net.classifier.parameters():
    param.requires_grad = True
for param in net.ie_proj.parameters():
    param.requires_grad = True
for param in net.IE.parameters():
    param.requires_grad = False
for param in net.DE.parameters():
    param.requires_grad = False

hard_opt_parameters = []
for param in net.classifier.parameters():
    hard_opt_parameters.append(param)
for param in net.ie_proj.parameters():
    hard_opt_parameters.append(param)
```

---

### main.py - Hard Mining Loss

**BEFORE:**
```python
_, _, distance, _ = net(X, Xf)

cls_loss = (distance * y).mean()
reg_loss = net.regularization_loss(reg_loss_rate_active, reg_loss_rate_entropy)
l = cls_loss + reg_loss
```

**AFTER:**
```python
logits = net(X, Xf)

# Use cross-entropy loss for hard mining training
cls_loss = torch.nn.functional.cross_entropy(logits, y.argmax(dim=1))
reg_loss = net.regularization_loss(reg_loss_rate_active, reg_loss_rate_entropy)
l = cls_loss + reg_loss
```

---

### main.py - Final Evaluation

**BEFORE:**
```python
FCC_out, MEC_out, distance, _ = net(X, Xf)

uncertainty_tensor = moi_uncertianty(FCC_out)
hard_index_val = torch.where(uncertainty_tensor > uncertainty_threshold)[0]
hard_num = hard_index_val.size(0)
if hard_num:
    hard_num_total += hard_num
    FCC_out[hard_index_val] = MEC_out[hard_index_val]

y_hat = FCC_out.detach().to('cpu')
```

**AFTER:**
```python
logits = net(X, Xf)

# Just use logits directly (no MEC-based switching)
y_hat = logits.detach().to('cpu')
```

---

## 3. Dependencies

### requirements.txt

**BEFORE:**
```
numpy=1.23.5
pandas=2.0.3
scikit-learn=1.3.2
scipy=1.10.1
torch=2.2.2
torchvision=0.17.2
pillow=10.2.0
d2l=1.0.3
```

**AFTER:**
```
numpy=1.23.5
pandas=2.0.3
scikit-learn=1.3.2
scipy=1.10.1
torch=2.2.2
torchvision=0.17.2
pillow=10.2.0
d2l=1.0.3
timm=0.9.7
```

---

## 4. Output Format Changes

### Forward Pass Output

**BEFORE:**
```python
ffc_out, mec_out, distance, encode = net(X, Xf)
# Returns 4 tensors for different purposes

# ffc_out: (B, num_labels) - primary predictions
# mec_out: (B, num_labels) - distance-based predictions
# distance: (B, num_labels) - distance to class centers
# encode: (B, expert_num) - expert network outputs
```

**AFTER:**
```python
logits = net(X, Xf)
# Returns 1 tensor

# logits: (B, num_labels) - classification logits
```

---

## 5. Layer Access Changes

### Accessing IE Parameters

**BEFORE:**
```python
# Access TinyViT-specific layers (NO LONGER EXIST)
net.IE.base.layerKAN
net.IE.base.layer1
net.IE.base.layer2

# Access classifier layers
net.FFC[0]  # BatchNorm
net.FFC[2]  # KAN_Linear
net.MEC[0]  # BatchNorm
net.MEC[2]  # KAN_Linear
```

**AFTER:**
```python
# Access TinyViT (vision transformer)
net.IE  # TinyViT model

# Access projection and classifier
net.ie_proj  # Linear projection from 768→512
net.classifier[0]  # BatchNorm
net.classifier[2]  # Linear(640→256)
net.classifier[6]  # Linear(256→128)
net.classifier[10]  # Linear(128→num_labels)
```

---

## 6. Performance Metrics Collection

### Dimension Tracking During Forward Pass

**BEFORE:**
```python
f_i = self.IE(X)              # (B, 512)
f_p = self.DE(f_p)            # (B, 128)
f_f = torch.cat((f_i, f_p))   # (B, 640)
ffc_out = self.FFC(f_f)       # (B, num_labels)
encode = self.MEC(f_f)        # (B, expert_num)
distance = ...                # (B, num_labels)
mec_out = -distance           # (B, num_labels)
```

**AFTER:**
```python
f_i = self.IE(X)              # (B, 768)
f_i = self.ie_proj(f_i)       # (B, 512)
f_p = self.DE(f_p)            # (B, 128)
f_f = torch.cat((f_i, f_p))   # (B, 640)
logits = self.classifier(f_f) # (B, num_labels)
```

---

## Summary of Breaking Changes

❌ **Code that will break if not updated:**

1. `logits, _, _, _ = net(X, Xf)` → Use `logits = net(X, Xf)`
2. `net.IE.base.layer1` → No longer exists
3. `net.FFC` → Replaced with `net.classifier`
4. `net.MEC` → Removed entirely
5. Accessing `distance` directly → Use uncertainty from logits instead

✅ **Code that remains compatible:**

1. `moi_loss()` function still works
2. `moi_uncertianty()` function still works
3. `net.regularization_loss()` still works (now only from DE)
4. `net.DE` architecture unchanged
5. Hard sample mining pipeline still works

---

## Testing Changes

Add this to verify the new architecture:

```python
# Check model structure
print(type(net.IE))      # Should show timm model
print(net.ie_proj)       # Should show Linear layer
print(net.classifier)    # Should show Sequential with MLPs

# Check forward pass
logits = net(dummy_images, dummy_features)
print(logits.shape)  # Should be (batch_size, num_labels)

# Check parameter count
total = sum(p.numel() for p in net.parameters())
trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Total params: {total}, Trainable: {trainable}")
```
