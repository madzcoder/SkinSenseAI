import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)

import torchvision.models as models

from preprocessing_optimized import main as load_data


# ======================================================
# CONFIG
# ======================================================
class Config:

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_NAME = 'mobilenet_v3'

    IMAGE_SIZE = 192

    BATCH_SIZE = 64

    EPOCHS = 25

    LR_HEAD = 1e-3
    LR_FINETUNE = 3e-4

    FREEZE_EPOCHS = 5

    WEIGHT_DECAY = 1e-4

    NUM_CLASSES = 7

    DROPOUT = 0.3

    USE_AMP = True

    GRAD_CLIP = 1.0

    SAVE_DIR = 'checkpoints'


config = Config()

os.makedirs(config.SAVE_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True


# ======================================================
# MODEL
# ======================================================
def build_model():

    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )

    in_features = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.Hardswish(),
        nn.Dropout(config.DROPOUT),
        nn.Linear(1280, config.NUM_CLASSES)
    )

    return model


# ======================================================
# FREEZE / UNFREEZE
# ======================================================
def freeze_backbone(model):

    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):

    for param in model.features.parameters():
        param.requires_grad = True


# ======================================================
# TRAIN
# ======================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler):

    model.train()

    running_loss = 0

    preds_all = []
    labels_all = []

    loop = tqdm(loader, desc='Training')

    for imgs, labels in loop:

        imgs = imgs.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16):

            outputs = model(imgs)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRAD_CLIP
        )

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item() * imgs.size(0)

        preds = outputs.argmax(dim=1)

        preds_all.extend(preds.detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

        loop.set_postfix({
            'loss': f'{loss.item():.4f}'
        })

    epoch_loss = running_loss / len(loader.dataset)

    acc = np.mean(
        np.array(preds_all) == np.array(labels_all)
    )

    f1 = f1_score(
        labels_all,
        preds_all,
        average='macro'
    )

    balanced_acc = balanced_accuracy_score(
        labels_all,
        preds_all
    )

    return epoch_loss, acc, f1, balanced_acc


# ======================================================
# VALIDATION
# ======================================================
def validate(model, loader, criterion):

    model.eval()

    running_loss = 0

    preds_all = []
    labels_all = []

    with torch.no_grad():

        loop = tqdm(loader, desc='Validation')

        for imgs, labels in loop:

            imgs = imgs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16
            ):

                outputs = model(imgs)

                loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    acc = np.mean(
        np.array(preds_all) == np.array(labels_all)
    )

    f1 = f1_score(
        labels_all,
        preds_all,
        average='macro'
    )

    balanced_acc = balanced_accuracy_score(
        labels_all,
        preds_all
    )

    return epoch_loss, acc, f1, balanced_acc


# ======================================================
# TEST TIME AUGMENTATION
# ======================================================
def tta_predict(model, imgs):

    outputs1 = model(imgs)

    outputs2 = model(
        torch.flip(imgs, dims=[3])
    )

    outputs3 = model(
        torch.flip(imgs, dims=[2])
    )

    outputs = (
        outputs1 +
        outputs2 +
        outputs3
    ) / 3

    return outputs


# ======================================================
# MAIN TRAINING
# ======================================================
def train():

    train_loader, val_loader, test_loader, le = load_data()

    model = build_model().to(config.DEVICE)

    # ==========================================
    # TORCH COMPILE
    # ==========================================
    # model = torch.compile(model)

    # ==========================================
    # FREEZE BACKBONE INITIALLY
    # ==========================================
    freeze_backbone(model)

    # ==========================================
    # LOSS
    # ==========================================
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    # ==========================================
    # OPTIMIZER
    # ==========================================
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR_HEAD,
        weight_decay=config.WEIGHT_DECAY
    )

    # ==========================================
    # SCHEDULER
    # ==========================================
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS
    )

    # ==========================================
    # AMP SCALER
    # ==========================================
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_bal_acc': []
    }

    print('\nStarting Training...')

    for epoch in range(config.EPOCHS):

        print(f'\nEpoch {epoch+1}/{config.EPOCHS}')
        print('-' * 50)

        # ======================================
        # UNFREEZE BACKBONE
        # ======================================
        if epoch == config.FREEZE_EPOCHS:

            print('Unfreezing backbone for fine-tuning...')

            unfreeze_backbone(model)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LR_FINETUNE,
                weight_decay=config.WEIGHT_DECAY
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.EPOCHS - epoch
            )

        # ======================================
        # TRAIN
        # ======================================
        train_loss, train_acc, train_f1, train_bal = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        # ======================================
        # VALIDATE
        # ======================================
        val_loss, val_acc, val_f1, val_bal = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step()

        # ======================================
        # STORE HISTORY
        # ======================================
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_bal_acc'].append(val_bal)

        # ======================================
        # PRINT METRICS
        # ======================================
        print(f'Train Loss : {train_loss:.4f}')
        print(f'Train Acc  : {train_acc:.4f}')
        print(f'Train F1   : {train_f1:.4f}')

        print(f'Val Loss   : {val_loss:.4f}')
        print(f'Val Acc    : {val_acc:.4f}')
        print(f'Val F1     : {val_f1:.4f}')
        print(f'Val BalAcc : {val_bal:.4f}')

        current_lr = optimizer.param_groups[0]['lr']

        print(f'Learning Rate : {current_lr:.7f}')

        # ======================================
        # SAVE BEST MODEL
        # ======================================
        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_f1': best_f1
                },
                os.path.join(
                    config.SAVE_DIR,
                    'best_model.pth'
                )
            )

            print('Best model updated.')

    # ==================================================
    # LOAD BEST MODEL
    # ==================================================
    checkpoint = torch.load(
        os.path.join(
            config.SAVE_DIR,
            'best_model.pth'
        )
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    # ==================================================
    # TESTING WITH TTA
    # ==================================================
    print('\nRunning Test-Time Augmentation Evaluation...')

    model.eval()

    preds_all = []
    labels_all = []

    with torch.no_grad():

        for imgs, labels in tqdm(test_loader):

            imgs = imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            with torch.autocast(
                device_type='cuda',
                dtype=torch.float16
            ):

                outputs = tta_predict(model, imgs)

            preds = outputs.argmax(dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    # ==================================================
    # FINAL METRICS
    # ==================================================
    final_acc = np.mean(
        np.array(preds_all) == np.array(labels_all)
    )

    final_f1 = f1_score(
        labels_all,
        preds_all,
        average='macro'
    )

    final_bal_acc = balanced_accuracy_score(
        labels_all,
        preds_all
    )

    print('\nFinal Test Results')
    print('=' * 50)

    print(f'Test Accuracy      : {final_acc:.4f}')
    print(f'Test Macro F1      : {final_f1:.4f}')
    print(f'Test Balanced Acc  : {final_bal_acc:.4f}')

    print('\nClassification Report:\n')

    print(
        classification_report(
            labels_all,
            preds_all,
            target_names=le.classes_
        )
    )

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================
    cm = confusion_matrix(
        labels_all,
        preds_all
    )

    print('\nConfusion Matrix:\n')
    print(cm)

    # ==================================================
    # SAVE HISTORY
    # ==================================================
    with open('training_history.json', 'w') as f:

        json.dump(history, f, indent=4)

    print('\nTraining history saved.')

    return model, history


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == '__main__':

    model, history = train()