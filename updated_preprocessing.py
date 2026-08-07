import os
import gc
import cv2
import random
import pickle
import warnings

import numpy as np
import pandas as pd

from PIL import Image
from typing import Optional

import torch
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

warnings.filterwarnings('ignore')


# ======================================================
# CONFIG
# ======================================================
BASE_DIR = r"F:\Skinsense AI"

PART1 = os.path.join(BASE_DIR, "HAM10000_Part1")
PART2 = os.path.join(BASE_DIR, "HAM10000_Part2")

CSV_PATH = os.path.join(
    BASE_DIR,
    "HAM10000_metadata.csv"
)

IMAGE_SIZE = 192

BATCH_SIZE = 64

NUM_WORKERS = 8

RANDOM_SEED = 42

MEAN = [0.763, 0.546, 0.570]
STD = [0.141, 0.152, 0.169]


# ======================================================
# REPRODUCIBILITY
# ======================================================
def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = True


seed_everything(RANDOM_SEED)


# ======================================================
# LABELS
# ======================================================
DISEASE_LABELS = {
    'nv': 'Melanocytic Nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc': 'Vascular Lesion',
    'df': 'Dermatofibroma'
}


# ======================================================
# LOAD DATA
# ======================================================
def load_and_prepare_data():

    print("\nLoading metadata...")

    df = pd.read_csv(CSV_PATH)

    print(f"Total records: {len(df)}")

    # ==========================================
    # FAST IMAGE PATH MAPPING
    # ==========================================
    all_images = {}

    for folder in [PART1, PART2]:

        if not os.path.exists(folder):
            continue

        for fname in os.listdir(folder):

            if fname.endswith('.jpg'):

                image_id = fname.replace('.jpg', '')

                all_images[image_id] = os.path.join(
                    folder,
                    fname
                )

    df['path'] = df['image_id'].map(all_images)

    # ==========================================
    # REMOVE MISSING
    # ==========================================
    df = df[df['path'].notna()].reset_index(drop=True)

    df['dx_full'] = df['dx'].map(DISEASE_LABELS)

    print("\nClass Distribution:")
    print(df['dx'].value_counts())

    return df


# ======================================================
# SPLIT BY LESION
# ======================================================
def split_by_lesion(
    df,
    test_size=0.2,
    val_size=0.1
):

    print("\nSplitting by lesion_id...")

    lesions = df['lesion_id'].unique()

    train_lesions, test_lesions = train_test_split(
        lesions,
        test_size=test_size,
        random_state=RANDOM_SEED
    )

    train_lesions, val_lesions = train_test_split(
        train_lesions,
        test_size=val_size,
        random_state=RANDOM_SEED
    )

    train_df = df[
        df['lesion_id'].isin(train_lesions)
    ].reset_index(drop=True)

    val_df = df[
        df['lesion_id'].isin(val_lesions)
    ].reset_index(drop=True)

    test_df = df[
        df['lesion_id'].isin(test_lesions)
    ].reset_index(drop=True)

    print(f"Train: {len(train_df)}")
    print(f"Val  : {len(val_df)}")
    print(f"Test : {len(test_df)}")

    return train_df, val_df, test_df


# ======================================================
# LABEL ENCODER
# ======================================================
def create_label_encoder(df):

    le = LabelEncoder()

    le.fit(df['dx'])

    encoder_path = os.path.join(
        BASE_DIR,
        'label_encoder.pkl'
    )

    with open(encoder_path, 'wb') as f:

        pickle.dump(le, f)

    print(f"\nLabel encoder saved: {encoder_path}")

    return le


# ======================================================
# AUGMENTATIONS
# ======================================================
def get_train_transform(image_size=IMAGE_SIZE):

    return transforms.Compose([

        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.RandomRotation(20),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1
            )
        ], p=0.5),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05)
        ),

        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.0)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=MEAN,
            std=STD
        )
    ])


def get_val_transform(image_size=IMAGE_SIZE):

    return transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(image_size),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=MEAN,
            std=STD
        )
    ])


# ======================================================
# DATASET
# ======================================================
class SkinSenseDataset(Dataset):

    def __init__(
        self,
        df,
        transform=None,
        cache=False
    ):

        self.df = df.reset_index(drop=True)

        self.transform = transform

        self.cache = cache

        self.tensor_cache = {}

        if cache:

            print(f"\nCaching {len(self.df)} tensors...")

            for idx in range(len(self.df)):

                img = self.load_image(
                    self.df.iloc[idx]['path']
                )

                if self.transform:
                    img = self.transform(img)

                self.tensor_cache[idx] = img

            print("Caching complete.")

    # ==========================================
    # FAST IMAGE LOADING
    # ==========================================
    def load_image(self, path):

        img = cv2.imread(path)

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        return Image.fromarray(img)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        if self.cache:

            img = self.tensor_cache[idx]

        else:

            img = self.load_image(row['path'])

            if self.transform:
                img = self.transform(img)

        label = int(row['label'])

        return img, label


# ======================================================
# DATALOADERS
# ======================================================
def build_dataloaders(
    train_df,
    val_df,
    test_df
):

    train_transform = get_train_transform()

    val_transform = get_val_transform()

    train_dataset = SkinSenseDataset(
        train_df,
        train_transform
    )

    val_dataset = SkinSenseDataset(
        val_df,
        val_transform,
        cache=True
    )

    test_dataset = SkinSenseDataset(
        test_df,
        val_transform,
        cache=True
    )

    # ==========================================
    # WEIGHTED RANDOM SAMPLER
    # ==========================================
    class_counts = train_df[
        'label'
    ].value_counts().sort_index()

    class_weights = 1.0 / class_counts

    sample_weights = train_df[
        'label'
    ].map(class_weights).values

    sample_weights = torch.DoubleTensor(
        sample_weights
    )

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # ==========================================
    # TRAIN LOADER
    # ==========================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # ==========================================
    # VAL LOADER
    # ==========================================
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # ==========================================
    # TEST LOADER
    # ==========================================
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ======================================================
# MAIN
# ======================================================
def main():

    print("=" * 60)
    print("SKINSENSE OPTIMIZED PREPROCESSING")
    print("=" * 60)

    # ==========================================
    # LOAD DATA
    # ==========================================
    df = load_and_prepare_data()

    # ==========================================
    # SPLIT
    # ==========================================
    train_df, val_df, test_df = split_by_lesion(df)

    # ==========================================
    # LABEL ENCODER
    # ==========================================
    le = create_label_encoder(df)

    train_df['label'] = le.transform(
        train_df['dx']
    )

    val_df['label'] = le.transform(
        val_df['dx']
    )

    test_df['label'] = le.transform(
        test_df['dx']
    )

    # ==========================================
    # DATALOADERS
    # ==========================================
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df,
        val_df,
        test_df
    )

    # ==========================================
    # VERIFY
    # ==========================================
    print("\nVerifying pipeline...")

    images, labels = next(iter(train_loader))

    print(f"Batch Shape : {images.shape}")
    print(f"Labels Shape: {labels.shape}")

    print("\nPipeline Ready.")

    gc.collect()

    return (
        train_loader,
        val_loader,
        test_loader,
        le
    )


# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == '__main__':

    train_loader, val_loader, test_loader, le = main()