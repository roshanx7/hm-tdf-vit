# @inproceedings{HM-TDF,  
#   title={Hard Sample Mining-Based Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network},  
#   link={https://github.com/MLDMXM2017/HM-TDF}  
# }  

import numpy as np
import torch
import torch.nn as nn
import sys
import os
script_path = os.path.abspath(__file__)
model_path = os.path.dirname(script_path)
sys.path.insert(0, model_path)
from KANLinear import KANLinear as KAN_Linear

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Install with: pip install timm")

class MffKan(nn.Module): 
    def __init__(self, num_labels, num_features, drop_rate):
        super().__init__()
        self.num_features = num_features
        self.kan_linears = []
        self.ie_dim = 768  # TinyViT output dimension
        self.de_dim = 128  # DE output dimension
        self.fused_dim = self.ie_dim + self.de_dim  # 896

        # Image Encoder: TinyViT
        if TIMM_AVAILABLE:
            self.IE = timm.create_model('tiny_vit_5m_224', pretrained=True, num_classes=0)
        else:
            # Fallback: use a simple feature extractor
            print("Warning: Using fallback feature extractor instead of TinyViT")
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
        self.ie_dim = 512  # Update after projection
        self.fused_dim = self.ie_dim + self.de_dim  # 640

        # Indicator Encoder (DE) - UNCHANGED from original
        self.DE = nn.Sequential(KAN_Linear(num_features,        32, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(32,   128, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]))

        # CNN-based Classifier (replaces FFC + MEC)
        # Uses 1D convolution on the flattened fused features
        self.classifier = nn.Sequential(
            # Reshape for 1D conv: (batch_size, 640) → (batch_size, 1, 640)
            nn.Unflatten(1, (1, self.fused_dim)),
            
            # First CNN block: 1 channel → 32 channels
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            
            # Second CNN block: 32 channels → 64 channels
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            
            # Global average pooling: (batch_size, 64, 640) → (batch_size, 64, 1)
            nn.AdaptiveAvgPool1d(1),
            
            # Flatten: (batch_size, 64, 1) → (batch_size, 64)
            nn.Flatten(),
            
            # Final classification layer: 64 → num_labels
            nn.Linear(64, num_labels)
        )

        # Collect KAN layers for regularization (only from DE)
        for module in self.DE.modules():
            if isinstance(module, KAN_Linear):
                self.kan_linears.append(module)

    def forward(self, X, f_p, debug=False):
        """
        Forward pass for MFF-KAN with TinyViT Image Encoder and CNN Classifier.
        
        Args:
            X: Image tensor (batch_size, 3, 224, 224)
            f_p: Indicator features tensor (batch_size, num_features)
            debug: If True, print intermediate tensor shapes (default: False)
        
        Returns:
            logits: Classification logits (batch_size, num_labels)
        """
        # Image Encoder: TinyViT (expects 224x224 input)
        f_i = self.IE(X)  # (batch_size, 768)
        
        if debug:
            print(f"[DEBUG] After IE: f_i.shape = {f_i.shape}")
        
        # Apply projection layer: 768 → 512
        f_i = self.ie_proj(f_i)  # (batch_size, 512)
        
        if debug:
            print(f"[DEBUG] After IE projection: f_i.shape = {f_i.shape}")
        
        # Indicator Encoder: DE (KAN-based, unchanged)
        f_p = self.DE(f_p)  # (batch_size, 128)
        
        if debug:
            print(f"[DEBUG] After DE: f_p.shape = {f_p.shape}")
        
        # Concatenate features: 512 + 128 = 640
        f_f = torch.cat((f_i, f_p), dim=1)  # (batch_size, 640)
        
        if debug:
            print(f"[DEBUG] After concatenation: f_f.shape = {f_f.shape}")
        
        # CNN-based Classifier
        logits = self.classifier(f_f)  # (batch_size, num_labels)
        
        if debug:
            print(f"[DEBUG] After classifier: logits.shape = {logits.shape}")
        
        return logits
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Calculate regularization loss for KAN layers (only from DE)."""
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_linears
        )
    
    def unfreeze_ie_layers(self, unfreeze_ratio=0.5):
        """
        Unfreeze TinyViT layers for fine-tuning (staged fine-tuning strategy).
        
        Args:
            unfreeze_ratio: Fraction of total layers to unfreeze from the end (0.0-1.0).
                           0.0 = keep all frozen, 1.0 = unfreeze all.
                           Default: 0.5 = unfreeze last 50% of layers.
        
        Example:
            # After N epochs, unfreeze last 50% of TinyViT layers
            net.unfreeze_ie_layers(unfreeze_ratio=0.5)
        """
        if not TIMM_AVAILABLE:
            print("[WARNING] TinyViT not available, cannot unfreeze layers")
            return
        
        # Get all named parameters from IE
        ie_params = list(self.IE.named_parameters())
        total_layers = len(ie_params)
        unfreeze_count = max(1, int(total_layers * unfreeze_ratio))
        
        # Unfreeze last N layers
        for i, (name, param) in enumerate(ie_params):
            if i >= (total_layers - unfreeze_count):
                param.requires_grad = True
                print(f"[UNFREEZE] Layer {i}/{total_layers}: {name}")
            else:
                param.requires_grad = False
        
        # Also unfreeze projection layer
        for param in self.ie_proj.parameters():
            param.requires_grad = True
        
        print(f"[INFO] Unfroze {unfreeze_count}/{total_layers} TinyViT layers for fine-tuning")

# define net structure
def get_net(num_features, num_labels, drop_rate):
    net = MffKan(num_labels, num_features, drop_rate)
    return net
