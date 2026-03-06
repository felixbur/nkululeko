import torch
import sys

print("Testing loss functions...")

# Test 1: BCE Loss
print("\n1. Testing BCEWithLogitsLoss:")
from nkululeko.losses.loss_bce import BCEWithLogitsLoss

bce = BCEWithLogitsLoss()
inputs = torch.randn(10)
targets = torch.randint(0, 2, (10,)).float()
loss = bce(inputs, targets)
print(f"   ✓ BCE Loss: {loss.item():.4f}")

# Test 2: Weighted BCE Loss
print("\n2. Testing WeightedBCEWithLogitsLoss:")
from nkululeko.losses.loss_bce import WeightedBCEWithLogitsLoss

weighted_bce = WeightedBCEWithLogitsLoss(pos_weight=2.0)
loss = weighted_bce(inputs, targets)
print(f"   ✓ Weighted BCE Loss (pos_weight=2.0): {loss.item():.4f}")

# Test 3: Focal Loss
print("\n3. Testing FocalLoss:")
from nkululeko.losses.loss_focal import FocalLoss

focal = FocalLoss(alpha=0.25, gamma=2.0)
loss = focal(inputs, targets)
print(f"   ✓ Focal Loss (alpha=0.25, gamma=2.0): {loss.item():.4f}")

# Test 4: Class weight computation
print("\n4. Testing compute_class_weights:")
from nkululeko.losses.loss_bce import compute_class_weights

labels = ["fake", "fake", "fake", "fake", "real", "real"]  # 4:2 ratio
weight = compute_class_weights(labels, ["fake", "real"], mode="balanced")
print(f"   ✓ Class weight (4 fake, 2 real): {weight:.2f}")
assert abs(weight - 2.0) < 0.01, "Weight should be 2.0"

print("\n✓ All loss functions working correctly!")
