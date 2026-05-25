import os
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
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================
BASE_DIR = r"D:\skinsenseai-project\skinsenseai"
PART1 = os.path.join(BASE_DIR, "HAM10000_images_part_1")
PART2 = os.path.join(BASE_DIR, "HAM10000_images_part_2")
CSV_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")

# Image normalization parameters (computed from HAM10000)
MEAN = [0.763, 0.546, 0.570]
STD = [0.141, 0.152, 0.169]

# Training parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4  # Adjust based on your CPU cores
RANDOM_SEED = 42

# Disease label mapping
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
    """Load metadata and map image paths efficiently."""
    print("Loading metadata...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total records: {len(df)}")
    print("\nClass distribution:")
    print(df['dx'].value_counts())
    
    # Fast path mapping using dictionary
    print("\nMapping image paths...")
    all_images = {}
    for folder in [PART1, PART2]:
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found!")
            continue
        for fname in os.listdir(folder):
            if fname.endswith('.jpg'):
                image_id = fname.replace('.jpg', '')
                all_images[image_id] = os.path.join(folder, fname)
    
    df['path'] = df['image_id'].map(all_images)
    
    # Remove missing images
    missing = df['path'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} images not found, removing from dataset")
        df = df[df['path'].notna()].reset_index(drop=True)
    
    # Add full disease names
    df['dx_full'] = df['dx'].map(DISEASE_LABELS)
    
    return df


def split_by_lesion(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by lesion_id to prevent data leakage.
    Same lesion should not appear in both train and test sets.
    """
    print("\nSplitting data by lesion_id...")
    unique_lesions = df['lesion_id'].unique()
    
    # Split lesions (not images) to prevent leakage
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
    
    train_df = df[df['lesion_id'].isin(train_lesions)].reset_index(drop=True)
    val_df = df[df['lesion_id'].isin(val_lesions)].reset_index(drop=True)
    test_df = df[df['lesion_id'].isin(test_lesions)].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} images | Val: {len(val_df)} images | Test: {len(test_df)} images")
    
    return train_df, val_df, test_df


def balance_dataset(df: pd.DataFrame, target_col: str = 'dx', strategy: str = 'hybrid') -> pd.DataFrame:
    """
    Balance dataset using smart strategy.
    
    Args:
        df: DataFrame to balance
        target_col: Column to balance on
        strategy: 'hybrid' (recommended), 'upsample_all', or 'downsample_all'
    """
    print(f"\nBalancing dataset using '{strategy}' strategy...")
    print("Before balancing:")
    print(df[target_col].value_counts())
    
    counts = df[target_col].value_counts()
    
    if strategy == 'hybrid':
        # Upsample minorities to median, keep majorities
        target_count = int(counts.median())
    elif strategy == 'upsample_all':
        # Upsample all to max (original approach, creates large dataset)
        target_count = counts.max()
    elif strategy == 'downsample_all':
        # Downsample all to min (smallest dataset, no duplicates)
        target_count = counts.min()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    balanced = []
    for label in df[target_col].unique():
        subset = df[df[target_col] == label]
        n = len(subset)
        
        if n < target_count:
            # Upsample with replacement
            upsampled = resample(subset, replace=True, n_samples=target_count, random_state=RANDOM_SEED)
            balanced.append(upsampled)
        elif n > target_count and strategy == 'downsample_all':
            # Downsample without replacement
            downsampled = resample(subset, replace=False, n_samples=target_count, random_state=RANDOM_SEED)
            balanced.append(downsampled)
        else:
            # Keep as is
            balanced.append(subset)
    
    balanced_df = pd.concat(balanced).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print("\nAfter balancing:")
    print(balanced_df[target_col].value_counts())
    print(f"Total samples: {len(balanced_df)}")
    
    return balanced_df


def create_label_encoder(df: pd.DataFrame, save_path: Optional[str] = None) -> LabelEncoder:
    """Create and optionally save label encoder."""
    print("\nCreating label encoder...")
    le = LabelEncoder()
    le.fit(df['dx'])
    
    print(f"Classes: {list(le.classes_)}")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(le, f)
        print(f"Label encoder saved to: {save_path}")
    
    return le


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, base_dir: str):
    """Save split dataframes for reproducibility."""
    print("\nSaving splits to disk...")
    train_df.to_csv(os.path.join(base_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(base_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(base_dir, 'test_split.csv'), index=False)
    print("Splits saved successfully!")


# ==================== TRANSFORMS ====================
def get_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Optimized training augmentations - not too aggressive."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_val_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Validation/test transform - no augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


# ==================== DATASET CLASS ====================
class SkinSenseDataset(Dataset):
    """
    Optimized dataset with optional image caching.
    """
    def __init__(self, df: pd.DataFrame, transform: Optional[transforms.Compose] = None, 
                 cache_images: bool = False):
        """
        Args:
            df: DataFrame with 'path' and 'label' columns
            transform: Transforms to apply
            cache_images: If True, cache images in RAM (good for val/test, risky for large train)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}
        
        if cache_images:
            print(f"Caching {len(self.df)} images in memory...")
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                try:
                    img = Image.open(row['path']).convert('RGB')
                    self.image_cache[idx] = img
                except Exception as e:
                    print(f"Error loading image {row['path']}: {e}")
            print(f"Successfully cached {len(self.image_cache)} images")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple:
        row = self.df.iloc[idx]
        
        # Load image from cache or disk
        if self.cache_images and idx in self.image_cache:
            img = self.image_cache[idx].copy()  # Copy to avoid modifying cache
        else:
            try:
                img = Image.open(row['path']).convert('RGB')
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                # Return a blank image in case of error
                img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        
        label = int(row['label'])
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ==================== MAIN PREPROCESSING PIPELINE ====================
def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("SKINSENSE AI - OPTIMIZED PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_and_prepare_data()
    
    # Step 2: Split by lesion
    train_df, val_df, test_df = split_by_lesion(df, test_size=0.2, val_size=0.1)
    
    # Step 3: Balance training set only (not val/test)
    train_df = balance_dataset(train_df, strategy='hybrid')
    
    # Step 4: Create and save label encoder
    le = create_label_encoder(
        df, 
        save_path=os.path.join(BASE_DIR, 'label_encoder.pkl')
    )
    
    # Step 5: Encode labels
    train_df['label'] = le.transform(train_df['dx'])
    val_df['label'] = le.transform(val_df['dx'])
    test_df['label'] = le.transform(test_df['dx'])
    
    # Step 6: Save splits
    save_splits(train_df, val_df, test_df, BASE_DIR)
    
    # Step 7: Create datasets
    print("\n" + "=" * 60)
    print("CREATING DATASETS")
    print("=" * 60)
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_dataset = SkinSenseDataset(train_df, train_transform, cache_images=False)
    val_dataset = SkinSenseDataset(val_df, val_transform, cache_images=True)  # Cache val set
    test_dataset = SkinSenseDataset(test_df, val_transform, cache_images=True)  # Cache test set
    
    # Step 8: Create data loaders
    print("\nCreating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    # Step 9: Verify everything works
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
    
    # Test loading a batch
    print("\nTesting batch loading...")
    images, labels = next(iter(train_loader))
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Sample label: {labels[0].item()} -> {le.inverse_transform([labels[0].item()])[0]}")
    print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Clean up
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
    # Run the complete pipeline
    train_loader, val_loader, test_loader, label_encoder = main()
    
    # Example: Iterate through one epoch
    print("\n" + "=" * 60)
    print("TRAINING LOOP EXAMPLE")
    print("=" * 60)
    
    print("\nIterating through first 3 batches of training data...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 3:
            break
        print(f"Batch {batch_idx + 1}: images {images.shape}, labels {labels.shape}")
    
    print("\nOptimized preprocessing script ready for training! 🚀")
