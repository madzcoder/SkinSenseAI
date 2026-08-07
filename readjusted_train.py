import os
import sys
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import torchvision.models as models
import torchvision.models.quantization as qmodels
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import multiprocessing

# ==================== LOGGING & CONSOLE BEAUTIFICATION ====================
import colorama
from datetime import datetime as dt

class Tee:
    """
    Mirrors a stream (stdout or stderr) to both the terminal and a shared log
    file. Two things this fixes vs. a naive version:
      1. A timestamp is only prepended at the START of each real line, not on
         every individual write() call - print() internally calls write()
         separately for the message and for the trailing '\n', so a naive
         version stamps EVERY fragment and shreds multi-part lines apart.
      2. Warnings/errors get a short, colored one-line pointer on the console
         ("see log for details") while the FULL text still lands in the file.
    """
    def __init__(self, log_file, terminal, colorize=True):
        self.terminal = terminal
        self.log_file = log_file
        self.colorize = colorize and hasattr(terminal, 'isatty') and terminal.isatty()
        self._at_line_start = True
        self.colors = {
            'error': colorama.Fore.RED + colorama.Style.BRIGHT,
            'warning': colorama.Fore.YELLOW,
            'success': colorama.Fore.GREEN + colorama.Style.BRIGHT,
            'info': colorama.Fore.CYAN,
            'reset': colorama.Style.RESET_ALL
        }

    def write(self, message):
        if not message:
            return
        is_new_line = self._at_line_start
        stripped = message.strip()
        lower = stripped.lower()

        # Full, untouched detail always goes to the file - one timestamp per
        # real line, not per internal write() fragment.
        for line in message.splitlines(keepends=True):
            if self._at_line_start and line.strip('\r\n'):
                timestamp = dt.now().strftime('%Y-%m-%d %H:%M:%S')
                self.log_file.write(f"[{timestamp}] {line}")
            else:
                self.log_file.write(line)
            self._at_line_start = line.endswith('\n')
        self.log_file.flush()

        # Console gets a condensed pointer for warnings/errors; everything
        # else (and mid-line fragments, e.g. tqdm bars) passes through as-is.
        if is_new_line and lower.startswith('warning'):
            console_msg = "⚠ Warning - see log for details\n"
        elif is_new_line and (lower.startswith('error') or lower.startswith('traceback')):
            console_msg = "✖ Error - see log for details\n"
        else:
            console_msg = message

        if self.colorize:
            self.terminal.write(self._colorize_message(console_msg))
        else:
            self.terminal.write(console_msg)
        self.terminal.flush()

    def _colorize_message(self, message):
        # Classify by PREFIX, not "contains this word anywhere" - the old
        # substring-anywhere check colored "Successfully cached 800 images
        # (failed: 0)" red because it contains "failed", even at zero
        # failures. Prefix-based checks don't have that false positive.
        lower = message.strip().lower()
        if lower.startswith('⚠') or lower.startswith('warning'):
            return self.colors['warning'] + message + self.colors['reset']
        elif lower.startswith('✖') or lower.startswith('error') or lower.startswith('traceback'):
            return self.colors['error'] + message + self.colors['reset']
        elif lower.startswith('successfully') or lower.startswith('✓') or 'new best model' in lower or lower.startswith('all done'):
            return self.colors['success'] + message + self.colors['reset']
        elif lower.startswith('loading') or lower.startswith('creating') or lower.startswith('building'):
            return self.colors['info'] + message + self.colors['reset']
        else:
            return message

    def flush(self):
        self.log_file.flush()
        self.terminal.flush()

    def isatty(self):
        # tqdm (and other libraries) check this to decide rendering style.
        # Without it, Tee looks like a "dumb" non-interactive stream and tqdm
        # falls back to plain ASCII '#' bars instead of solid Unicode blocks.
        try:
            return self.terminal.isatty()
        except Exception:
            return False

    @property
    def encoding(self):
        # tqdm's unicode-vs-ascii bar decision specifically checks this
        # attribute - Tee not having one at all was the actual cause of the
        # ASCII-style bars, not anything about the bars themselves.
        return getattr(self.terminal, 'encoding', 'utf-8')

    def fileno(self):
        return self.terminal.fileno()

def setup_logging(model_name='model'):
    colorama.init(autoreset=True)
    # Same outputs/logs folder as checkpoints, plots, and training history -
    # one place to look, instead of a separate "logs" folder at the project
    # root plus another one under outputs/.
    logs_dir = os.path.join(OUTPUT_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    # Human-readable: "training_mobilenet_v3_2026-07-25_09-08PM.log" instead
    # of "log_20260725_205443.txt" - sortable, and the model+time are visible
    # at a glance instead of needing to open the file.
    timestamp = dt.now().strftime('%Y-%m-%d_%I-%M%p')
    log_filename = f"training_{model_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)
    # newline='' stops Python's text-mode auto-translation of '\n' -> '\r\n'
    # on Windows, which was doubling up with the timestamps into a mess.
    log_file = open(log_path, 'a', encoding='utf-8', newline='')

    # Redirecting stderr too (not just stdout) is what actually fixes "the
    # console looked longer than the log" - Python's default unhandled
    # exception printer writes to stderr, which a stdout-only redirect never
    # sees. This is also why last run's crash traceback never made it to file.
    sys.stdout = Tee(log_file, sys.__stdout__)
    sys.stderr = Tee(log_file, sys.__stderr__)
    print(f"📝 Logging to: {log_path}")
    print("=" * 60)
    return log_file
# ========================================================================

# ==================== DEBUG CONFIGURATION ====================
DEBUG = os.environ.get('SKINSENSE_DEBUG', 'False').lower() == 'true'

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# ========== EAGER-MODE QUANTIZATION IMPORTS ==========
try:
    from torch.quantization import (
        QuantStub, DeQuantStub,
        prepare_qat, convert,
        get_default_qat_qconfig,
        fuse_modules
    )
except ImportError:
    from torch.ao.quantization import (
        QuantStub, DeQuantStub,
        prepare_qat, convert,
        get_default_qat_qconfig,
        fuse_modules
    )
from torch.ao.quantization import QConfig
from torch.ao.quantization.fake_quantize import FusedMovingAvgObsFakeQuantize, default_fused_per_channel_wt_fake_quant
from torch.ao.quantization.observer import MovingAverageMinMaxObserver

from preprocessing_optimized import (
    main as load_data,
    BASE_DIR,
    OUTPUT_DIR,
    IMAGE_SIZE
)

# ==================== CONFIGURATION ====================
class TrainingConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = True
    # REVERTED: batch 128 was an experiment to use more VRAM headroom, but
    # this run's data shows it didn't deliver - steady-state epoch time was
    # ~29-30s (worse than batch 64's ~25-27s), and best val accuracy came in
    # at 81.38% vs the established 83.87% baseline. No speed win and a real
    # accuracy cost - per the "if it hurts accuracy, leave it" rule, reverting.
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 7
    # Ryzen 5 5600X = 6 cores / 12 threads
    NUM_WORKERS = 8
    GRADIENT_ACCUMULATION_STEPS = 1
    MODEL_NAME = 'mobilenet_v3'  # torchvision's mobilenet_v3_large specifically
    # Back to 0.3 - this was bumped to 0.4 for ResNet50's larger capacity;
    # MobileNetV3 is much smaller and doesn't need the extra pressure.
    DROPOUT = 0.3
    # Standard transfer-learning practice, missing until now: train only the
    # new classifier head for the first few epochs with the pretrained
    # backbone frozen, THEN unfreeze for full fine-tuning. Every run so far
    # fine-tuned the whole network from epoch 1 at a relatively high LR,
    # which risks distorting well-trained ImageNet features before the new
    # head has caught up. Set to 0 to disable and go back to full fine-tuning
    # from the start.
    FREEZE_BACKBONE_EPOCHS = 3
    USE_WARMUP = True
    WARMUP_EPOCHS = 3
    SCHEDULER_TYPE = 'cosine'
    EARLY_STOPPING_PATIENCE = 10
    MIN_DELTA = 0.001
    SAVE_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    # Saving every epoch's checkpoint (False) adds up fast across 30 epochs.
    # Default to only keeping the best one; flip back to False if you want
    # every epoch on disk for later inspection/rollback.
    SAVE_BEST_ONLY = True
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    SAVE_PLOTS = True
    SAVE_METRICS = True
    # QAT (quantization-aware training) and AMP (fp16 mixed precision) don't mix
    # safely in PyTorch's eager-mode quantization - fake-quant ops don't have
    # reliable fp16/CUDA kernels, and running both together risks NaNs or a
    # crash mid-training. We auto-disable AMP below whenever QAT is on.
    USE_QAT = True
    QAT_BACKEND = 'fbgemm'

def initialize_config():
    """
    Everything here used to sit at module level, which meant it ran again
    every time Windows' DataLoader workers (spawn method) re-imported this
    script - once per worker, hence the warnings appearing 8x. Wrapping it
    in a function only called from `if __name__ == "__main__":` means worker
    processes just get the class/function definitions they need and never
    re-run this setup or its prints.
    """
    config = TrainingConfig()
    max_workers = multiprocessing.cpu_count()
    config.NUM_WORKERS = min(config.NUM_WORKERS, max_workers)
    debug_print(f"Using {config.NUM_WORKERS} workers (available: {max_workers})")

    # QAT + fp16 autocast is not reliably supported in eager-mode quantization.
    # Force AMP off whenever QAT is enabled instead of letting it crash mid-run.
    if config.USE_QAT and config.USE_AMP:
        print("Warning: USE_QAT and USE_AMP were both True. Eager-mode QAT doesn't "
              "reliably support fp16 autocast, so disabling AMP for this run.")
        config.USE_AMP = False

    # torchvision only ships quantization-ready ("FloatFunctional" residual
    # add) variants for mobilenet_v3 and resnet50. EfficientNet's blocks use
    # a raw "+=" for their residual connection, which has no QuantizedCPU
    # kernel - that's exactly what crashed last run's quantized test pass.
    if config.USE_QAT and config.MODEL_NAME not in ('mobilenet_v3', 'resnet50'):
        print(f"Warning: QAT was on with MODEL_NAME='{config.MODEL_NAME}', but "
              "torchvision has no quantization-ready variant of that architecture "
              "(only mobilenet_v3 and resnet50 support it here). Disabling QAT "
              "for this run so it doesn't crash on the residual-add op.")
        config.USE_QAT = False

    # Push this script's batch size/worker count into the preprocessing module
    # so the DataLoaders it builds actually use them (previously
    # config.BATCH_SIZE here was never read by preprocessing_optimized.main()).
    import preprocessing_optimized as _prep
    _prep.BATCH_SIZE = config.BATCH_SIZE
    _prep.NUM_WORKERS = config.NUM_WORKERS

    # Fixed 224x224 input size every batch -> let cuDNN pick and cache the
    # fastest conv algorithms instead of re-benchmarking (or using a
    # suboptimal default).
    if config.DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    if config.USE_QAT:
        try:
            if hasattr(torch.backends.quantized, 'supported_engines'):
                supported = torch.backends.quantized.supported_engines
                debug_print(f"Supported quantization engines: {supported}")
            else:
                supported = ['fbgemm', 'qnnpack']
                debug_print("No supported_engines attribute, assuming: fbgemm, qnnpack")

            if config.QAT_BACKEND not in supported:
                print(f"Warning: '{config.QAT_BACKEND}' not supported. Available: {supported}")
                config.QAT_BACKEND = supported[0] if supported else 'fbgemm'
                print(f"Using fallback engine: '{config.QAT_BACKEND}'")
            torch.backends.quantized.engine = config.QAT_BACKEND
            debug_print(f"Quantization engine set to: {torch.backends.quantized.engine}")
        except AttributeError:
            try:
                torch.backends.quantized.engine = config.QAT_BACKEND
            except RuntimeError as e:
                print(f"Failed to set engine '{config.QAT_BACKEND}': {e}")
                config.QAT_BACKEND = 'fbgemm'
                torch.backends.quantized.engine = config.QAT_BACKEND
                print(f"Using fallback engine: '{config.QAT_BACKEND}'")

    return config

# ==================== QUANTIZABLE MODEL WRAPPER ====================
class QuantizableModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.quant = QuantStub()
        self.base_model = base_model
        self.dequant = DeQuantStub()
        debug_print("QuantizableModel wrapper created.")

    def forward(self, x):
        x = self.quant(x)
        x = self.base_model(x)
        x = self.dequant(x)
        return x

# ==================== MODEL BUILDER ====================
def build_model(model_name: str = 'mobilenet_v3', num_classes: int = 7,
                dropout: float = 0.3, pretrained: bool = True,
                quantizable: bool = True):
    print(f"\nBuilding model: {model_name}")
    debug_print(f"Params: num_classes={num_classes}, dropout={dropout}, pretrained={pretrained}, quantizable={quantizable}")

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
        debug_print(f"EfficientNet-B0: in_features={in_features}, replaced classifier.")

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
        debug_print(f"EfficientNet-B3: in_features={in_features}, replaced classifier.")

    elif model_name == 'resnet50':
        if quantizable:
            # Quantizable variant: residual add uses nn.quantized.FloatFunctional
            # instead of a raw "+=", which is required for QAT to actually work.
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = qmodels.resnet50(weights=weights, quantize=False)
        elif pretrained:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        debug_print(f"ResNet50: in_features={in_features}, replaced fc.")

    elif model_name == 'mobilenet_v3':
        if quantizable:
            # Same story: torchvision's quantizable MobileNetV3 swaps the
            # InvertedResidual block's "+=" for FloatFunctional.add(), which
            # is what the QuantizedCPU backend actually supports.
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            model = qmodels.mobilenet_v3_large(weights=weights, quantize=False)
        elif pretrained:
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        else:
            model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        # NOTE: this used to just replace classifier[3] (the final Linear),
        # leaving torchvision's stock classifier[2] = Dropout(0.2) untouched -
        # meaning config.DROPOUT had zero effect for this architecture no
        # matter what you set it to. Now it actually controls regularization
        # strength, same as the other three model options.
        model.classifier[2] = nn.Dropout(p=dropout)
        model.classifier[3] = nn.Linear(in_features, num_classes)
        debug_print(f"MobileNetV3: in_features={in_features}, replaced dropout+Linear layer.")

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # The qmodels.* variants (mobilenet_v3, resnet50) already embed their own
    # QuantStub/DeQuantStub internally - wrapping them again would be redundant.
    # Only wrap architectures that don't have a quantizable variant of their own.
    if quantizable and not hasattr(model, 'fuse_model'):
        model = QuantizableModel(model)
        debug_print("Wrapped model with QuantizableModel.")
    else:
        debug_print("Model not wrapped for quantization (either QAT is off, or the "
                     "quantizable torchvision variant already embeds quant/dequant stubs).")

    return model

# ==================== QAT PREPARATION ====================
def prepare_model_for_qat(model, device, backend='fbgemm'):
    print("\nPreparing model for Quantization-Aware Training (QAT)...")
    debug_print(f"Using backend: {backend}")

    model.to('cpu')
    debug_print("Moved model to CPU for QAT preparation.")

    model.train()
    debug_print("Model set to train mode.")

    if hasattr(model, 'fuse_model'):
        try:
            model.fuse_model(is_qat=True)
            debug_print("Fused conv-bn-relu layers via model.fuse_model(is_qat=True).")
        except TypeError:
            # Older torchvision versions' fuse_model() doesn't take is_qat.
            model.fuse_model()
            debug_print("Fused conv-bn-relu layers via model.fuse_model().")
    else:
        debug_print("Model has no fuse_model() - skipping fusion (expected for "
                     "the QuantizableModel-wrapped fallback path).")

    try:
        if backend == 'onednn':
            # PyTorch's own get_default_qat_qconfig('onednn') leaves the
            # activation observer's reduce_range at False - fbgemm and x86
            # backends both explicitly force reduce_range=True instead.
            # reduce_range=False risks int8 accumulator overflow/saturation
            # specifically on CPUs without Intel VNNI instructions (PyTorch's
            # own PTQ code has a documented warning about exactly this - the
            # QAT path just doesn't surface the same warning, even though the
            # underlying numerical risk is the same). AMD Ryzen CPUs (like
            # your 5600X) don't have VNNI at all, so this is a real, plausible
            # explanation for the ~9-10 point quantized accuracy gap that
            # persisted across multiple BN-freeze/observer timing fixes -
            # those were addressing training instability, not this. Keeping
            # onednn's already-good per-channel weight quantization, just
            # fixing the activation range to the safer setting.
            model.qconfig = QConfig(
                activation=FusedMovingAvgObsFakeQuantize.with_args(
                    observer=MovingAverageMinMaxObserver,
                    quant_min=0, quant_max=255, reduce_range=True
                ),
                weight=default_fused_per_channel_wt_fake_quant
            )
            debug_print("Using onednn qconfig with reduce_range=True "
                         "(safer on non-VNNI CPUs).")
        else:
            model.qconfig = get_default_qat_qconfig(backend)
        debug_print(f"QConfig set: {model.qconfig}")
    except Exception as e:
        print(f"Error setting qconfig: {e}")
        raise

    try:
        model = prepare_qat(model, inplace=False)
        debug_print("prepare_qat completed successfully.")
    except Exception as e:
        print(f"Error during prepare_qat: {e}")
        raise

    model.to(device)
    debug_print(f"Moved model back to {device}.")

    model.train()
    debug_print("Model is in train mode after QAT preparation.")

    return model

# ==================== SCHEDULER, EARLY STOPPING, TRAINING FUNCTIONS ====================
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 scheduler_type='cosine', base_lr=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
        self.scheduler_type = scheduler_type

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
        debug_print(f"WarmupScheduler initialized: warmup_epochs={warmup_epochs}, total_epochs={total_epochs}, type={scheduler_type}")

    def step(self, val_loss=None):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            debug_print(f"Warmup step {self.current_epoch}: lr={lr:.6f}")
        else:
            if self.scheduler_type == 'plateau' and val_loss is not None:
                self.main_scheduler.step(val_loss)
                debug_print(f"Plateau scheduler step with val_loss={val_loss:.4f}")
            else:
                self.main_scheduler.step()
                debug_print(f"Main scheduler step (type={self.scheduler_type})")

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        debug_print(f"EarlyStopping initialized: patience={patience}, min_delta={min_delta}, mode={mode}")

    def __call__(self, val_metric):
        if self.mode == 'max':
            score = val_metric
        else:
            score = -val_metric
        if self.best_score is None:
            self.best_score = score
            debug_print(f"EarlyStopping: first score={score:.4f}")
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            debug_print(f"EarlyStopping: score={score:.4f} < best+delta={self.best_score+self.min_delta:.4f}, counter={self.counter}")
            if self.counter >= self.patience:
                self.early_stop = True
                debug_print("Early stopping triggered!")
        else:
            self.best_score = score
            self.counter = 0
            debug_print(f"EarlyStopping: new best score={score:.4f}")

def train_one_epoch(model, train_loader, criterion, optimizer, device,
                   scaler=None, grad_accum_steps=1):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    loop = tqdm(train_loader, desc="Training")
    optimizer.zero_grad()

    for batch_idx, (imgs, labels) in enumerate(loop):
        imgs, labels = imgs.to(device), labels.to(device)
        debug_print(f"Batch {batch_idx}: images shape={imgs.shape}, labels shape={labels.shape}")

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels) / grad_accum_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                debug_print(f"  Step after accumulation {batch_idx+1}")
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels) / grad_accum_steps
            loss.backward()
            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                debug_print(f"  Step after accumulation {batch_idx+1}")

        train_loss += loss.item() * imgs.size(0) * grad_accum_steps
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += imgs.size(0)

        loop.set_postfix({
            'loss': loss.item() * grad_accum_steps,
            'acc': f'{train_correct/train_total:.3f}'
        })

    avg_loss = train_loss / train_total
    accuracy = train_correct / train_total
    debug_print(f"Epoch finished: avg_loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds, all_labels = [], []

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
    debug_print(f"Validation: avg_loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
    return avg_loss, accuracy, all_preds, all_labels

def test_model(model, test_loader, device, label_encoder, save_dir, model_name='model'):
    print("\n" + "="*60)
    print(f"TEST EVALUATION ({model_name})")
    print("="*60)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    debug_print(f"Test accuracy: {accuracy:.4f}")

    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    debug_print("Confusion matrix:\n", cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_path = os.path.join(save_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    plt.close()

    return accuracy, cm

def plot_training_history(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

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
        debug_print(f"Best model saved to {best_path}")

# ==================== MAIN TRAINING LOOP ====================
def train(config):
    print("="*60)
    print("SKINSENSE AI - OPTIMIZED TRAINING (EAGER-MODE QAT)")
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
    print(f"Quantization-Aware Training: {config.USE_QAT}")
    if config.USE_QAT:
        print(f"Quantization Backend: {config.QAT_BACKEND}")
    print(f"Number of workers: {config.NUM_WORKERS}")
    print("="*60)

    print("\nLoading data...")
    train_loader, val_loader, test_loader, label_encoder = load_data()

    print(f"\nDataset loaded:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    debug_print(f"Batch size: {config.BATCH_SIZE}")
    debug_print(f"Number of classes: {config.NUM_CLASSES}")

    model = build_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        pretrained=True,
        quantizable=config.USE_QAT
    )

    if config.USE_QAT:
        model = prepare_model_for_qat(model, config.DEVICE, backend=config.QAT_BACKEND)
    else:
        model = model.to(config.DEVICE)
        debug_print("Model moved to device (no QAT).")

    if config.FREEZE_BACKBONE_EPOCHS > 0:
        # 'classifier' covers mobilenet_v3's head, 'fc' covers resnet50's -
        # everything else is backbone and gets frozen for the first few epochs.
        head_keywords = ('classifier', 'fc')
        frozen_count = 0
        for name, param in model.named_parameters():
            if not any(k in name for k in head_keywords):
                param.requires_grad = False
                frozen_count += 1
        debug_print(f"Froze {frozen_count} backbone parameter tensors for the first "
                     f"{config.FREEZE_BACKBONE_EPOCHS} epoch(s) - classifier head trains alone first.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    debug_print(f"Total params: {total_params}, Trainable: {trainable_params}")

    # The 'hybrid' balancing strategy in preprocessing only upsamples classes
    # BELOW the median count - it never touches classes above it. Your actual
    # last run's training set was still ~62% 'nv' (4869 of 7849) after
    # "balancing", so the model had every incentive to lean toward predicting
    # the majority class. Class weights push back on that directly by making
    # mistakes on rarer classes cost more in the loss.
    train_counts = train_loader.dataset.df['label'].value_counts().sort_index()
    class_weights = torch.tensor(
        [1.0 / train_counts.get(i, 1) for i in range(config.NUM_CLASSES)],
        dtype=torch.float32
    )
    class_weights = (class_weights / class_weights.sum()) * config.NUM_CLASSES
    class_weights = class_weights.to(config.DEVICE)
    debug_print(f"Class weights: {class_weights.tolist()}")

    # label_smoothing softens the training targets slightly (0.1 instead of a
    # hard 1.0), which discourages the model from becoming overconfident on
    # the training set - a standard, low-risk regularizer against overfitting.
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    debug_print("Optimizer: AdamW")

    if config.USE_WARMUP:
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=config.WARMUP_EPOCHS,
            total_epochs=config.EPOCHS,
            scheduler_type=config.SCHEDULER_TYPE,
            base_lr=config.LEARNING_RATE
        )
        debug_print("Using WarmupScheduler.")
    else:
        if config.SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
        elif config.SCHEDULER_TYPE == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        debug_print(f"Using {config.SCHEDULER_TYPE} scheduler (no warmup).")

    scaler = GradScaler('cuda') if config.USE_AMP and config.DEVICE.type == 'cuda' else None
    if scaler:
        debug_print("Using GradScaler for mixed precision.")
    else:
        debug_print("Not using GradScaler.")

    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.MIN_DELTA,
        mode='max'
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_acc = 0.0
    best_train_acc_seen = 0.0   # tracks the best HEALTHY train_acc, used to detect collapse
    bn_frozen = False
    collapse_recoveries = 0
    MAX_COLLAPSE_RECOVERIES = 3
    start_time = datetime.now()

    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 40)

        if config.FREEZE_BACKBONE_EPOCHS > 0 and epoch == config.FREEZE_BACKBONE_EPOCHS:
            for param in model.parameters():
                param.requires_grad = True
            unfrozen_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Unfroze the backbone ({unfrozen_total:,} trainable params now) - "
                  f"fine-tuning the full network from here on.")

        # Standard QAT recipe: freeze BatchNorm running stats late in training
        # (once weights have mostly converged - NOT while they're still
        # rapidly adapting, which caused a real collapse last run: freezing
        # at epoch//3 while train_acc was still climbing locked in immature
        # BN statistics that didn't match the still-shifting weights, and
        # accuracy collapsed from 81% to 8% within a single epoch and never
        # recovered). 85% through training is a much safer margin.
        if config.USE_QAT and not bn_frozen:
            # Moved from 0.85 to 0.70: the collapse-recovery safety net below
            # is now in place as a backstop, and this run's data shows the
            # 0.85 timing left only a ~2-epoch gap before observer-disable -
            # not enough time for the fake-quant observers to settle on the
            # NEW post-freeze activation distribution before their own values
            # got locked in. That's a plausible contributor to the quantized
            # model still losing ~9.8 points vs float this run. Widening the
            # gap to ~6 epochs gives the observers more time to adapt first.
            if epoch == int(config.EPOCHS * 0.70):
                _freeze_bn_stats = None
                try:
                    import torch.ao.nn.intrinsic.qat as _qat_intrinsic
                    _freeze_bn_stats = _qat_intrinsic.freeze_bn_stats
                except (ImportError, AttributeError):
                    try:
                        import torch.nn.intrinsic.qat as _qat_intrinsic
                        _freeze_bn_stats = _qat_intrinsic.freeze_bn_stats
                    except (ImportError, AttributeError):
                        pass
                if _freeze_bn_stats is not None:
                    model.apply(_freeze_bn_stats)
                    bn_frozen = True
                    debug_print(f"Epoch {epoch+1}: froze BatchNorm running stats.")
                else:
                    debug_print(f"Epoch {epoch+1}: freeze_bn_stats not found in this torch "
                                 f"version - skipping (BN stats will keep updating).")
        if config.USE_QAT:
            if epoch == config.EPOCHS - max(3, config.EPOCHS // 10):
                model.apply(torch.ao.quantization.disable_observer)
                debug_print(f"Epoch {epoch+1}: disabled fake-quant observers "
                             f"for the remaining epochs to let scale/zero-point stabilize.")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE,
            scaler=scaler, grad_accum_steps=config.GRADIENT_ACCUMULATION_STEPS
        )

        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, config.DEVICE)

        # Safety net: if accuracy craters relative to the best healthy epoch
        # so far, something broke (a bad LR step, the BN freeze above, etc.).
        # Just toggling settings back wouldn't undo the damage already done
        # to the WEIGHTS by the bad gradient updates during the collapsed
        # epoch - restoring from the last good checkpoint is what actually
        # recovers, which is why last run never bounced back on its own even
        # after 7 further epochs of training.
        collapsed = (best_train_acc_seen >= 0.3 and train_acc < best_train_acc_seen * 0.5)
        if collapsed and collapse_recoveries < MAX_COLLAPSE_RECOVERIES:
            collapse_recoveries += 1
            print(f"\n⚠ Warning: training collapsed this epoch (train acc {train_acc:.3f} vs "
                  f"{best_train_acc_seen:.3f} before) - restoring the last good checkpoint "
                  f"and continuing. ({collapse_recoveries}/{MAX_COLLAPSE_RECOVERIES} recoveries used)")
            best_model_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, weights_only=True,
                                                   map_location=config.DEVICE))
                # Clear momentum/variance buffers IN PLACE rather than building
                # a new optimizer object - the scheduler holds a reference to
                # this exact optimizer instance, and swapping it for a new one
                # would silently detach the LR schedule from what's actually
                # training (scheduler.step() would then adjust a discarded
                # optimizer's LR while the real one stays frozen).
                optimizer.state.clear()
                if bn_frozen:
                    try:
                        import torch.ao.nn.intrinsic.qat as _qat_intrinsic
                        model.apply(_qat_intrinsic.update_bn_stats)  # un-freeze
                    except (ImportError, AttributeError):
                        pass
                    bn_frozen = False
                model.apply(torch.ao.quantization.enable_observer)
                print("  Restored weights from best_model.pth and reset the optimizer. "
                      "Continuing training.")
            else:
                print("  No checkpoint to restore from yet - continuing as-is (this early, "
                      "that's unusual and worth watching).")
        elif train_acc > best_train_acc_seen:
            best_train_acc_seen = train_acc

        if config.USE_WARMUP:
            scheduler.step(val_loss if config.SCHEDULER_TYPE == 'plateau' else None)
        else:
            if config.SCHEDULER_TYPE == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f"  ✓ New best model! (Val Acc: {best_val_acc:.4f})")
            debug_print(f"Best val_acc updated to {best_val_acc:.4f}")

        if config.SAVE_BEST_ONLY:
            if is_best:
                checkpoint_path = os.path.join(config.SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer, scheduler, epoch+1, val_acc,
                              history, checkpoint_path, is_best=True)
        else:
            checkpoint_path = os.path.join(config.SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch+1, val_acc,
                          history, checkpoint_path, is_best=is_best)

        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break

    training_time = datetime.now() - start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {training_time}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    if config.SAVE_PLOTS:
        plot_training_history(history, config.LOG_DIR)

    if config.SAVE_METRICS:
        history_path = os.path.join(config.LOG_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to: {history_path}")

    # ========== CONVERT TO QUANTIZED MODEL AND EVALUATE ==========
    if config.USE_QAT:
        print("\n" + "="*60)
        print("CONVERTING TO QUANTIZED MODEL")
        print("="*60)

        best_model_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            print(f"Loaded best model from: {best_model_path}")
            debug_print("Best model state dict loaded.")
        else:
            print("Warning: best_model.pth not found. Using current model for conversion.")
            debug_print("best_model.pth not found; using current model.")

        model.eval()
        model.to('cpu')
        debug_print("Model moved to CPU and set to eval mode for conversion.")
        try:
            quantized_model = convert(model, inplace=False)
            debug_print("Conversion to quantized model successful.")
        except Exception as e:
            print(f"Error during conversion: {e}")
            quantized_model = None
            raise

        quantized_path = os.path.join(config.SAVE_DIR, 'quantized_model.pth')
        torch.save(quantized_model.state_dict(), quantized_path)
        print(f"Quantized model saved to: {quantized_path}")

        test_acc_quant, cm_quant = test_model(
            quantized_model,
            test_loader,
            device=torch.device('cpu'),
            label_encoder=label_encoder,
            save_dir=config.LOG_DIR,
            model_name='quantized'
        )

        model.to('cpu')
        test_acc_float, cm_float = test_model(
            model,
            test_loader,
            device=torch.device('cpu'),
            label_encoder=label_encoder,
            save_dir=config.LOG_DIR,
            model_name='float'
        )

        print("\nComparison:")
        print(f"  Float model test accuracy: {test_acc_float:.4f} ({test_acc_float*100:.2f}%)")
        print(f"  Quantized model test accuracy: {test_acc_quant:.4f} ({test_acc_quant*100:.2f}%)")
        print(f"  Accuracy drop: {(test_acc_float - test_acc_quant)*100:.2f} percentage points")
        debug_print(f"Float acc: {test_acc_float}, Quant acc: {test_acc_quant}, drop: {test_acc_float - test_acc_quant}")

    else:
        test_acc_float, cm_float = test_model(
            model,
            test_loader,
            device=config.DEVICE,
            label_encoder=label_encoder,
            save_dir=config.LOG_DIR,
            model_name='float'
        )
        quantized_model = None
        test_acc_quant = None

    summary = {
        'model': config.MODEL_NAME,
        'best_val_acc': float(best_val_acc),
        'test_acc_float': float(test_acc_float),
        'test_acc_quant': float(test_acc_quant) if test_acc_quant is not None else None,
        'training_time': str(training_time),
        'epochs_trained': len(history['train_loss']),
        'final_lr': float(current_lr),
        'qat_enabled': config.USE_QAT,
        'qat_backend': config.QAT_BACKEND if config.USE_QAT else None
    }
    summary_path = os.path.join(config.LOG_DIR, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    debug_print(f"Training summary saved to {summary_path}")

    print("\n" + "="*60)
    print("ALL DONE! 🎉")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Float Test Accuracy: {test_acc_float:.4f} ({test_acc_float*100:.2f}%)")
    if test_acc_quant is not None:
        print(f"  Quantized Test Accuracy: {test_acc_quant:.4f} ({test_acc_quant*100:.2f}%)")
    print(f"\nCheckpoints saved in: {config.SAVE_DIR}")
    print(f"Logs and plots saved in: {config.LOG_DIR}")

    return model, history, test_acc_float, quantized_model

# ==================== ENTRY POINT ====================
if __name__ == '__main__':
    # Set up logging and pretty console first, so everything below - including
    # the config warnings - actually gets captured. TrainingConfig() itself has
    # no side effects (those live in initialize_config()), so it's safe to
    # peek MODEL_NAME here just to name the log file before full init runs.
    _model_name_for_log = TrainingConfig().MODEL_NAME
    log_file = setup_logging(model_name=_model_name_for_log)
    config = initialize_config()

    try:
        model, history, test_acc, quantized_model = train(config)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Flushing logs...")
    except Exception:
        # This is what was missing before: sys.stderr is now redirected too
        # (see setup_logging), so this also prints to console, but we log the
        # full traceback explicitly here as well to be certain it's captured.
        print("\nError: Unhandled exception during training:\n" + traceback.format_exc())
    finally:
        # Ensure the log file is properly closed and streams are restored
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        try:
            log_file.close()
        except Exception:
            pass
        print("Logging session ended.")