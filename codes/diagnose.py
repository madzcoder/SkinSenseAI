"""
SkinSense AI — Diagnostic Script
Run this BEFORE training to pinpoint exactly what is broken.
It tests each component in isolation and prints a clear PASS/FAIL for each.
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("SKINSENSE AI — DIAGNOSTIC")
print("=" * 60)

# ── 1. DEVICE ────────────────────────────────────────────────
print("\n[1] DEVICE")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Using: {device}")
if device.type == "cuda":
    print(f"    GPU:   {torch.cuda.get_device_name(0)}")
    print(f"    VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("    WARNING: No GPU found — training will be very slow")

# ── 2. DATA PIPELINE ─────────────────────────────────────────
print("\n[2] DATA PIPELINE")
try:
    from preprocessing_optimized import main as load_data
    train_loader, val_loader, test_loader, label_encoder = load_data()
    print(f"    PASS — Data loaded")
    print(f"    Train batches : {len(train_loader)}")
    print(f"    Val batches   : {len(val_loader)}")
    print(f"    Test batches  : {len(test_loader)}")
    print(f"    Classes       : {list(label_encoder.classes_)}")
except Exception as e:
    print(f"    FAIL — {e}")
    raise SystemExit("Fix data pipeline before continuing.")

# ── 3. LABEL SANITY ──────────────────────────────────────────
print("\n[3] LABEL SANITY")
all_labels = []
for _, labels in train_loader:
    all_labels.extend(labels.numpy())
all_labels = np.array(all_labels)

unique, counts = np.unique(all_labels, return_counts=True)
print(f"    Unique label values in train_loader : {unique.tolist()}")
print(f"    Counts per label                    : {counts.tolist()}")
print(f"    Expected 7 unique classes (0–6)     : {'PASS' if len(unique) == 7 else 'FAIL — labels are wrong!'}")

if len(unique) != 7:
    print("    >>> Labels are not 0-indexed integers covering all 7 classes.")
    print("    >>> Check that label_encoder.transform() was called correctly in preprocessing.")

# ── 4. SINGLE BATCH FORWARD PASS ────────────────────────────
print("\n[4] SINGLE BATCH FORWARD PASS")
try:
    from torchvision import models

    model = models.efficientnet_b3(
        weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, 7)
    )
    model = model.to(device)

    imgs, labels = next(iter(train_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(imgs)

    print(f"    Input shape   : {imgs.shape}")
    print(f"    Output shape  : {outputs.shape}")
    print(f"    Output sample : {outputs[0].detach().cpu().numpy().round(3)}")
    print(f"    Labels sample : {labels[:8].cpu().numpy()}")
    print(f"    Any NaN in outputs : {torch.isnan(outputs).any().item()}")
    print(f"    Any Inf in outputs : {torch.isinf(outputs).any().item()}")
    print(f"    PASS — Forward pass is clean")
except Exception as e:
    print(f"    FAIL — {e}")

# ── 5. LOSS COMPUTATION ──────────────────────────────────────
print("\n[5] LOSS COMPUTATION")
try:
    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight('balanced', classes=np.arange(7), y=all_labels)
    weights = np.clip(weights, 0.1, 10.0)
    print(f"    Raw class weights : {weights.round(3)}")

    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    model.eval()
    with torch.no_grad():
        outputs_f32 = outputs.float()
        loss = criterion(outputs_f32, labels)

    print(f"    Loss value : {loss.item():.4f}")
    print(f"    Is finite  : {torch.isfinite(loss).item()}")
    print(f"    {'PASS — Loss is clean' if torch.isfinite(loss) else 'FAIL — Loss is NaN/Inf!'}")
except Exception as e:
    print(f"    FAIL — {e}")

# ── 6. ONE TRAINING STEP ─────────────────────────────────────
print("\n[6] ONE TRAINING STEP (float32, no AMP)")
try:
    model.train()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    optimizer.zero_grad()

    outputs = model(imgs)
    outputs = outputs.float()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    preds = outputs.argmax(dim=1)
    acc = (preds == labels).float().mean().item()

    print(f"    Loss after 1 step : {loss.item():.4f}")
    print(f"    Acc  after 1 step : {acc:.4f}")
    print(f"    Is loss finite    : {torch.isfinite(loss).item()}")
    print(f"    {'PASS' if torch.isfinite(loss) and acc >= 0 else 'FAIL'}")
except Exception as e:
    print(f"    FAIL — {e}")

# ── 7. BACKBONE FREEZE CHECK ─────────────────────────────────
print("\n[7] BACKBONE FREEZE CHECK")
trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
frozen    = [(n, p.shape) for n, p in model.named_parameters() if not p.requires_grad]
print(f"    Trainable params : {len(trainable)}")
print(f"    Frozen params    : {len(frozen)}")
if len(trainable) == 0:
    print("    FAIL — ALL parameters are frozen! Model cannot learn anything.")
elif len(frozen) == 0:
    print("    NOTE — Backbone is fully unfrozen (fine-tuning entire model)")
else:
    print("    PASS — Backbone partially frozen, classifier trainable")

# ── SUMMARY ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("Share the full output above with your judge.")
print("=" * 60)