import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import torchvision.models as models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Import from optimized preprocessing
from preprocessing_optimized import (
    main as load_data,
    BASE_DIR, BATCH_SIZE as DEFAULT_BATCH_SIZE,
    NUM_WORKERS as DEFAULT_NUM_WORKERS
)


# ==================== CONFIGURATION ====================
class TrainingConfig:
    """Centralized training configuration."""
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True  # Automatic Mixed Precision (2x faster, 40% less memory)
    
    # Training hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-3  # Higher initial LR with warmup
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 7
    
    # Optimization
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    GRADIENT_ACCUMULATION_STEPS = 1  # Increase for larger effective batch size
    
    # Model
    MODEL_NAME = 'mobilenet_v3'  # Options: efficientnet_b0, efficientnet_b3, resnet50
    DROPOUT = 0.3
    
    # Learning rate schedule
    USE_WARMUP = True
    WARMUP_EPOCHS = 3
    SCHEDULER_TYPE = 'cosine'  # Options: 'cosine', 'plateau', 'step'
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    MIN_DELTA = 0.001  # Minimum improvement to count as progress
    
    # Checkpointing
    SAVE_DIR = os.path.join(BASE_DIR, 'checkpoints')
    SAVE_BEST_ONLY = False  # If False, saves checkpoint every epoch
    
    # Logging
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    SAVE_PLOTS = True
    SAVE_METRICS = True


config = TrainingConfig()

# Create directories
os.makedirs(config.SAVE_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)


# ==================== MODEL BUILDER ====================
def build_model(model_name: str = 'mobilenet_v3', num_classes: int = 7, 
                dropout: float = 0.3, pretrained: bool = True):
    """
    Build and return a model with custom classifier.
    
    Supports: efficientnet_b0, efficientnet_b3, resnet50, mobilenet_v3
    """
    print(f"\nBuilding model: {model_name}")
    
    if model_name == 'efficientnet_b0':
        if pretrained:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    elif model_name == 'efficientnet_b3':
        if pretrained:
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    elif model_name == 'resnet50':
        if pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    elif model_name == 'mobilenet_v3':
        if pretrained:
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        else:
            model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


# ==================== LEARNING RATE SCHEDULERS ====================
class WarmupScheduler:
    """Linear warmup followed by main scheduler."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, 
                 scheduler_type='cosine', base_lr=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        
        # Create main scheduler
        if scheduler_type == 'cosine':
            self.main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs
            )
        elif scheduler_type == 'plateau':
            self.main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        elif scheduler_type == 'step':
            self.main_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        self.scheduler_type = scheduler_type
    
    def step(self, val_loss=None):
        """Step the scheduler."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Main scheduler
            if self.scheduler_type == 'plateau' and val_loss is not None:
                self.main_scheduler.step(val_loss)
            else:
                self.main_scheduler.step()
    
    def get_last_lr(self):
        """Get current learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


# ==================== EARLY STOPPING ====================
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        if self.mode == 'max':
            score = val_metric
        else:
            score = -val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(model, train_loader, criterion, optimizer, device, 
                   scaler=None, grad_accum_steps=1):
    """Train for one epoch with optional mixed precision and gradient accumulation."""
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    loop = tqdm(train_loader, desc="Training")
    optimizer.zero_grad()
    
    for batch_idx, (imgs, labels) in enumerate(loop):
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Mixed precision training
        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels) / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels) / grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Metrics
        train_loss += loss.item() * imgs.size(0) * grad_accum_steps
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += imgs.size(0)
        
        # Update progress bar
        loop.set_postfix({
            'loss': loss.item() * grad_accum_steps,
            'acc': f'{train_correct/train_total:.3f}'
        })
    
    avg_loss = train_loss / train_total
    accuracy = train_correct / train_total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += imgs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = val_loss / val_total
    accuracy = val_correct / val_total
    
    return avg_loss, accuracy, all_preds, all_labels


def test_model(model, test_loader, device, label_encoder, save_dir):
    """Comprehensive test evaluation with metrics and visualizations."""
    print("\n" + "="*60)
    print("TEST EVALUATION")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {cm_path}")
    plt.close()
    
    return accuracy, cm


def plot_training_history(history, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    plt.close()


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, history, 
                   filepath, is_best=False):
    """Save a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.main_scheduler.state_dict() if hasattr(scheduler, 'main_scheduler') else None,
        'val_acc': val_acc,
        'history': history
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), 'best_model.pth')
        torch.save(model.state_dict(), best_path)


# ==================== MAIN TRAINING LOOP ====================
def train(config):
    """Main training function."""
    
    # Print configuration
    print("="*60)
    print("SKINSENSE AI - OPTIMIZED TRAINING")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    if config.DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mixed Precision: {config.USE_AMP}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Scheduler: {config.SCHEDULER_TYPE}")
    print(f"Warmup Epochs: {config.WARMUP_EPOCHS if config.USE_WARMUP else 'None'}")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, label_encoder = load_data()
    
    print(f"\nDataset loaded:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Build model
    model = build_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        pretrained=True
    )
    model = model.to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    if config.USE_WARMUP:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=config.WARMUP_EPOCHS,
            total_epochs=config.EPOCHS,
            scheduler_type=config.SCHEDULER_TYPE,
            base_lr=config.LEARNING_RATE
        )
    else:
        if config.SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
        elif config.SCHEDULER_TYPE == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP and config.DEVICE.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.MIN_DELTA,
        mode='max'
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    start_time = datetime.now()
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE,
            scaler=scaler, grad_accum_steps=config.GRADIENT_ACCUMULATION_STEPS
        )
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, config.DEVICE)
        
        # Update scheduler
        if config.USE_WARMUP:
            scheduler.step(val_loss if config.SCHEDULER_TYPE == 'plateau' else None)
        else:
            if config.SCHEDULER_TYPE == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print metrics
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  ✓ New best model! (Val Acc: {best_val_acc:.4f})")
        
        if config.SAVE_BEST_ONLY:
            if is_best:
                checkpoint_path = os.path.join(config.SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, scheduler, epoch+1, val_acc, 
                              history, checkpoint_path, is_best=True)
        else:
            checkpoint_path = os.path.join(config.SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_acc, 
                          history, checkpoint_path, is_best=is_best)
        
        # Early stopping check
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"No improvement for {config.EARLY_STOPPING_PATIENCE} epochs")
            break
    
    # Training complete
    training_time = datetime.now() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {training_time}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training curves
    if config.SAVE_PLOTS:
        plot_training_history(history, config.LOG_DIR)
    
    # Save training history
    if config.SAVE_METRICS:
        history_path = os.path.join(config.LOG_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to: {history_path}")
    
    # Load best model and test
    best_model_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    test_acc, cm = test_model(model, test_loader, config.DEVICE, label_encoder, config.LOG_DIR)
    
    # Save final summary
    summary = {
        'model': config.MODEL_NAME,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'training_time': str(training_time),
        'epochs_trained': len(history['train_loss']),
        'final_lr': float(current_lr)
    }
    
    summary_path = os.path.join(config.LOG_DIR, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*60)
    print("ALL DONE! 🎉")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nCheckpoints saved in: {config.SAVE_DIR}")
    print(f"Logs and plots saved in: {config.LOG_DIR}")
    
    return model, history, test_acc


# ==================== ENTRY POINT ====================
if __name__ == '__main__':
    # Run training
    model, history, test_acc = train(config)
