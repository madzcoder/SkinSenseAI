import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import gc
from typing import Optional, Tuple, Dict, List
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ==================== LOGGING & CONSOLE BEAUTIFICATION ====================
import colorama
from datetime import datetime

class Tee:
    """Redirect stdout to both console (with colours) and a log file."""
    def __init__(self, log_path, console_output=True, colorize=True):
        self.terminal = sys.__stdout__
        self.log_file = open(log_path, 'a', encoding='utf-8')
        self.console_output = console_output
        self.colorize = colorize and hasattr(self.terminal, 'isatty') and self.terminal.isatty()
        # Colour mappings (using colorama)
        self.colors = {
            'error': colorama.Fore.RED + colorama.Style.BRIGHT,
            'warning': colorama.Fore.YELLOW,
            'success': colorama.Fore.GREEN + colorama.Style.BRIGHT,
            'info': colorama.Fore.CYAN,
            'reset': colorama.Style.RESET_ALL
        }

    def write(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Write to log file with timestamp
        self.log_file.write(f"[{timestamp}] {message}")
        self.log_file.flush()

        if self.console_output:
            if self.colorize:
                colored = self._colorize_message(message)
                self.terminal.write(colored)
            else:
                self.terminal.write(message)
            self.terminal.flush()

    def _colorize_message(self, message):
        lower = message.lower()
        if 'error' in lower or 'exception' in lower or 'failed' in lower:
            return self.colors['error'] + message + self.colors['reset']
        elif 'warning' in lower:
            return self.colors['warning'] + message + self.colors['reset']
        elif 'success' in lower or 'complete' in lower or 'done' in lower:
            return self.colors['success'] + message + self.colors['reset']
        elif 'info' in lower or 'loading' in lower or 'creating' in lower:
            return self.colors['info'] + message + self.colors['reset']
        else:
            return message

    def flush(self):
        self.log_file.flush()
        self.terminal.flush()

    def close(self):
        self.log_file.close()

def setup_logging():
    """Create logs folder, set up Tee, and initialise colorama."""
    colorama.init(autoreset=True)   # ensures reset after each coloured message
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(logs_dir, log_filename)

    # Replace stdout with our Tee
    sys.stdout = Tee(log_path)
    print(f"📝 Logging to: {log_path}")
    print("=" * 60)
# ========================================================================

# ==================== DEBUG CONFIGURATION ====================
DEBUG = os.environ.get('SKINSENSE_DEBUG', 'False').lower() == 'true'

def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# ==================== CONFIGURATION ====================
# EDIT THIS to wherever your "Skin Cancer MNIST HAM10000" folder actually lives,
# e.g. r"C:\Users\rajat\OneDrive\Desktop\Skin Cancer MNIST HAM10000\datasets"
# You can also set it via an environment variable instead of hardcoding it:
#   setx SKINSENSE_DATA_DIR "D:\SkinSense\datasets"   (then restart your terminal)
BASE_DIR = os.environ.get('SKINSENSE_DATA_DIR', r"F:\Skinsense AI")

# Everything the pipeline WRITES (splits, label encoder, checkpoints, logs,
# quantized models) goes here instead of inside the dataset folder - keeps
# your raw data folder untouched and makes it obvious what's an output vs
# what's an input. Sits as a SIBLING of BASE_DIR, e.g.:
#   Skin Cancer MNIST HAM10000/
#     datasets/        <- BASE_DIR (inputs, read-only)
#     outputs/          <- OUTPUT_DIR (everything this pipeline produces)
# Override independently with SKINSENSE_OUTPUT_DIR if you want it elsewhere.
OUTPUT_DIR = os.environ.get('SKINSENSE_OUTPUT_DIR', os.path.join(os.path.dirname(BASE_DIR), 'outputs'))
SPLITS_DIR = os.path.join(OUTPUT_DIR, 'splits')

# NOTE: these must match your actual folder names. The file-listing .txt files
# you generated show your real folders are named "HAM10000_images_part_1" and
# "HAM10000_images_part_2" (with underscores + lowercase "part"), NOT
# "HAM10000_Part1"/"HAM10000_Part2" like the original script had. Edit these
# two lines if your folder names differ from this.
PART1 = os.path.join(BASE_DIR, "HAM10000_images_part_1")
PART2 = os.path.join(BASE_DIR, "HAM10000_images_part_2")
CSV_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")

MEAN = [0.763, 0.546, 0.570]
STD = [0.141, 0.152, 0.169]

IMAGE_SIZE = 224
# RTX 3060 12GB has plenty of headroom at 224x224 for these model sizes.
# 32 was leaving throughput on the table.
BATCH_SIZE = 64
# Ryzen 5 5600X = 6 cores / 12 threads. Leave a couple threads free for the
# main process + GPU feeding rather than pinning all 12.
NUM_WORKERS = 8
PREFETCH_FACTOR = 4  # you have 48GB RAM, buffer more batches ahead of the GPU
RANDOM_SEED = 42

DISEASE_LABELS = {
    'nv':    'Melanocytic Nevi',
    'mel':   'Melanoma',
    'bkl':   'Benign Keratosis',
    'bcc':   'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc':  'Vascular Lesion',
    'df':    'Dermatofibroma'
}

# ==================== HELPER FUNCTIONS ====================
def load_and_prepare_data() -> pd.DataFrame:
    print("Loading metadata...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Total records in CSV: {len(df)}")
    debug_print("First few rows of metadata:", df.head())
    debug_print("Columns:", df.columns.tolist())

    print("\nClass distribution (from CSV):")
    class_counts = df['dx'].value_counts()
    print(class_counts)
    debug_print("Class counts detailed:", class_counts.to_dict())

    print("\nMapping image paths...")
    all_images: Dict[str, str] = {}
    folders = [PART1, PART2]
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found! Skipping.")
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
        debug_print(f"Found {len(files)} images in {folder}")
        for fname in files:
            image_id = fname.replace('.jpg', '')
            all_images[image_id] = os.path.join(folder, fname)

    debug_print(f"Total unique image IDs found: {len(all_images)}")
    df['path'] = df['image_id'].map(all_images)

    missing = df['path'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} images not found in file system. Removing these rows.")
        df = df[df['path'].notna()].reset_index(drop=True)
        debug_print(f"DataFrame after removing missing images: {len(df)} rows")
    else:
        print("All images found.")

    df['dx_full'] = df['dx'].map(DISEASE_LABELS)
    debug_print("Added full disease names column.")

    nan_full = df['dx_full'].isna().sum()
    if nan_full > 0:
        print(f"Warning: {nan_full} rows have unknown dx labels. Check mapping.")
        debug_print("Unknown labels:", df[df['dx_full'].isna()]['dx'].unique())

    return df

def split_by_lesion(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\nSplitting data by lesion_id...")
    unique_lesions = df['lesion_id'].unique()
    print(f"Total unique lesions: {len(unique_lesions)}")
    debug_print("Sample lesion_ids:", unique_lesions[:5])

    lesion_dx = df.groupby('lesion_id')['dx'].nunique()
    if (lesion_dx > 1).any():
        print("Warning: Some lesions have multiple dx labels. This might cause label inconsistency.")
        debug_print("Lesions with multiple dx:", lesion_dx[lesion_dx > 1].index.tolist())

    train_lesions, test_lesions = train_test_split(
        unique_lesions, 
        test_size=test_size, 
        random_state=RANDOM_SEED
    )
    train_lesions, val_lesions = train_test_split(
        train_lesions, 
        test_size=val_size, 
        random_state=RANDOM_SEED
    )

    debug_print(f"Train lesions: {len(train_lesions)}, Val lesions: {len(val_lesions)}, Test lesions: {len(test_lesions)}")

    train_df = df[df['lesion_id'].isin(train_lesions)].reset_index(drop=True)
    val_df = df[df['lesion_id'].isin(val_lesions)].reset_index(drop=True)
    test_df = df[df['lesion_id'].isin(test_lesions)].reset_index(drop=True)

    print(f"Train: {len(train_df)} images | Val: {len(val_df)} images | Test: {len(test_df)} images")
    debug_print("Train class distribution:", train_df['dx'].value_counts().to_dict())
    debug_print("Val class distribution:", val_df['dx'].value_counts().to_dict())
    debug_print("Test class distribution:", test_df['dx'].value_counts().to_dict())

    train_lesions_set = set(train_df['lesion_id'])
    val_lesions_set = set(val_df['lesion_id'])
    test_lesions_set = set(test_df['lesion_id'])
    assert train_lesions_set.isdisjoint(val_lesions_set), "Train and Val share lesions!"
    assert train_lesions_set.isdisjoint(test_lesions_set), "Train and Test share lesions!"
    assert val_lesions_set.isdisjoint(test_lesions_set), "Val and Test share lesions!"
    print("Lesion split verification passed.")

    return train_df, val_df, test_df

def balance_dataset(df: pd.DataFrame, target_col: str = 'dx', strategy: str = 'hybrid') -> pd.DataFrame:
    print(f"\nBalancing dataset using '{strategy}' strategy...")
    print("Before balancing:")
    counts = df[target_col].value_counts()
    print(counts)
    debug_print("Class counts before:", counts.to_dict())

    if strategy == 'hybrid':
        target_count = int(counts.median())
        print(f"Target count (median): {target_count}")
    elif strategy == 'upsample_all':
        target_count = counts.max()
        print(f"Target count (max): {target_count}")
    elif strategy == 'downsample_all':
        target_count = counts.min()
        print(f"Target count (min): {target_count}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    balanced_dfs = []
    for label in df[target_col].unique():
        subset = df[df[target_col] == label]
        n = len(subset)
        debug_print(f"Processing class '{label}': {n} samples")

        if n < target_count:
            upsampled = resample(subset, replace=True, n_samples=target_count, random_state=RANDOM_SEED)
            balanced_dfs.append(upsampled)
            debug_print(f"  Upsampled to {len(upsampled)}")
        elif n > target_count and strategy == 'downsample_all':
            downsampled = resample(subset, replace=False, n_samples=target_count, random_state=RANDOM_SEED)
            balanced_dfs.append(downsampled)
            debug_print(f"  Downsampled to {len(downsampled)}")
        else:
            balanced_dfs.append(subset)
            debug_print(f"  Kept all {len(subset)}")

    balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print("\nAfter balancing:")
    print(balanced_df[target_col].value_counts())
    print(f"Total samples: {len(balanced_df)}")
    debug_print("Class counts after:", balanced_df[target_col].value_counts().to_dict())

    return balanced_df

def create_label_encoder(df: pd.DataFrame, save_path: Optional[str] = None) -> LabelEncoder:
    print("\nCreating label encoder...")
    le = LabelEncoder()
    le.fit(df['dx'])
    print(f"Classes: {list(le.classes_)}")
    debug_print("Label encoder classes mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(le, f)
        print(f"Label encoder saved to: {save_path}")

    return le

def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, base_dir: str):
    print("\nSaving splits to disk...")
    os.makedirs(base_dir, exist_ok=True)
    train_df.to_csv(os.path.join(base_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(base_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(base_dir, 'test_split.csv'), index=False)
    print("Splits saved successfully!")

# ==================== TRANSFORMS ====================
def get_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        # Small translate/scale jitter - lesions aren't always dead-center or
        # a fixed size in the frame, and the model shouldn't rely on that.
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        # Cutout-style regularizer: randomly blanks a small patch post-normalize.
        # Cheap, standard, and directly targets the overfitting seen last run
        # (train acc climbed to 96%+ while val plateaued around 80%).
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08))
    ])

def get_val_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

# ==================== DATASET CLASS ====================
class SkinSenseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None, 
                 cache_images: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}

        if cache_images:
            print(f"Caching {len(self.df)} images in memory...")
            failed = 0
            for idx in tqdm(range(len(self.df)), desc="Caching images", unit="img"):
                row = self.df.iloc[idx]
                try:
                    img = Image.open(row['path']).convert('RGB')
                    self.image_cache[idx] = img
                except Exception as e:
                    failed += 1
                    debug_print(f"Error loading image {row['path']}: {e}")
            print(f"Successfully cached {len(self.image_cache)} images (failed: {failed})")
        else:
            debug_print("Image caching disabled.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.df.iloc[idx]

        if self.cache_images and idx in self.image_cache:
            img = self.image_cache[idx].copy()
            debug_print(f"Loaded from cache: idx={idx}")
        else:
            try:
                img = Image.open(row['path']).convert('RGB')
                debug_print(f"Loaded from disk: idx={idx}")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
                debug_print(f"Using blank image for idx={idx}")

        label = int(row['label'])

        if self.transform:
            img = self.transform(img)

        return img, label

# ==================== MAIN PREPROCESSING PIPELINE ====================
def main():
    print("=" * 60)
    print("SKINSENSE AI - OPTIMIZED PREPROCESSING PIPELINE")
    print("=" * 60)
    debug_print(f"DEBUG mode is ON.")

    df = load_and_prepare_data()
    debug_print(f"DataFrame shape: {df.shape}")
    debug_print("Columns:", df.columns.tolist())

    train_df, val_df, test_df = split_by_lesion(df, test_size=0.2, val_size=0.1)

    train_df = balance_dataset(train_df, strategy='hybrid')

    le = create_label_encoder(
        df, 
        save_path=os.path.join(SPLITS_DIR, 'label_encoder.pkl')
    )

    print("\nEncoding labels...")
    train_df['label'] = le.transform(train_df['dx'])
    val_df['label'] = le.transform(val_df['dx'])
    test_df['label'] = le.transform(test_df['dx'])
    debug_print("Label encoding completed.")
    debug_print("Train label distribution:", train_df['label'].value_counts().to_dict())
    debug_print("Val label distribution:", val_df['label'].value_counts().to_dict())
    debug_print("Test label distribution:", test_df['label'].value_counts().to_dict())

    assert train_df['label'].between(0, len(le.classes_)-1).all(), "Train labels out of range"
    assert val_df['label'].between(0, len(le.classes_)-1).all(), "Val labels out of range"
    assert test_df['label'].between(0, len(le.classes_)-1).all(), "Test labels out of range"

    save_splits(train_df, val_df, test_df, SPLITS_DIR)

    print("\n" + "=" * 60)
    print("CREATING DATASETS")
    print("=" * 60)

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = SkinSenseDataset(train_df, train_transform, cache_images=False)
    val_dataset = SkinSenseDataset(val_df, val_transform, cache_images=True)
    test_dataset = SkinSenseDataset(test_df, val_transform, cache_images=True)

    print("\nCreating DataLoaders...")
    import multiprocessing
    max_workers = multiprocessing.cpu_count()
    actual_workers = min(NUM_WORKERS, max_workers)
    if actual_workers < NUM_WORKERS:
        print(f"Reducing workers from {NUM_WORKERS} to {actual_workers} (available cores: {max_workers})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=True,
        persistent_workers=True if actual_workers > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if actual_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=True,
        persistent_workers=True if actual_workers > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if actual_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=True,
        persistent_workers=True if actual_workers > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if actual_workers > 0 else None
    )

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    print(f"\nDataLoader batches:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")

    print("\nTesting batch loading...")
    try:
        images, labels = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Sample label: {labels[0].item()} -> {le.inverse_transform([labels[0].item()])[0]}")
        print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
    except Exception as e:
        print(f"Error during batch loading test: {e}")
        raise

    gc.collect()

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Use the returned data loaders for training")
    print("2. Or load the saved CSV splits for custom workflows")
    print("3. Load label_encoder.pkl for deployment on Raspberry Pi")

    return train_loader, val_loader, test_loader, le

# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Set up logging and pretty console
    setup_logging()

    try:
        train_loader, val_loader, test_loader, label_encoder = main()

        print("\n" + "=" * 60)
        print("TRAINING LOOP EXAMPLE")
        print("=" * 60)

        print("\nIterating through first 3 batches of training data...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= 3:
                break
            print(f"Batch {batch_idx + 1}: images {images.shape}, labels {labels.shape}")

        print("\nOptimized preprocessing script ready for training! 🚀")
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Flushing logs...")
    finally:
        # Ensure the log file is properly closed
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
        sys.stdout = sys.__stdout__   # restore original
        print("Logging session ended.")