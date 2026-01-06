"""
Medical Imaging Data Preparation Pipeline

Downloads, preprocesses, and splits medical imaging datasets for training.
Supports TB Chest X-ray and Diabetic Retinopathy datasets from Kaggle.

Usage:
    python data_prep_medical.py --dataset tb
    python data_prep_medical.py --dataset retinopathy --max-samples 500
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

try:
    import kaggle
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

import zipfile
import shutil

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Configuration for data preparation pipeline."""
    dataset_type: str = 'tb'
    base_dir: Path = field(default_factory=lambda: Path('./medical_imaging_data'))
    img_size: Tuple[int, int] = (224, 224)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_samples_per_class: Optional[int] = None
    random_seed: int = 42
    
    @property
    def raw_data_dir(self) -> Path:
        return self.base_dir / 'raw'
    
    @property
    def processed_data_dir(self) -> Path:
        return self.base_dir / 'processed'
    
    def __post_init__(self):
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Train/val/test ratios must sum to 1.0"


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def create_directories(config: DataConfig) -> None:
    """Create necessary directory structure."""
    config.base_dir.mkdir(exist_ok=True)
    config.raw_data_dir.mkdir(exist_ok=True)
    config.processed_data_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (config.processed_data_dir / split).mkdir(exist_ok=True)
    
    logger.info(f"Created directory structure at {config.base_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_tb_dataset(config: DataConfig) -> bool:
    """Download TB Chest X-ray dataset from Kaggle."""
    logger.info("Downloading TB Chest X-ray dataset from Kaggle...")
    
    if not KAGGLE_AVAILABLE:
        logger.error("Kaggle library not installed. Run: pip install kaggle")
        return False
    
    try:
        kaggle.api.dataset_download_files(
            'tawsifurrahman/tuberculosis-tb-chest-xray-dataset',
            path=config.raw_data_dir,
            unzip=True
        )
        logger.info("✓ TB dataset downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Manual download: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
        return False


def download_retinopathy_dataset(config: DataConfig) -> bool:
    """Download APTOS 2019 Diabetic Retinopathy dataset from Kaggle."""
    logger.info("Downloading APTOS 2019 Diabetic Retinopathy dataset...")
    
    if not KAGGLE_AVAILABLE:
        logger.error("Kaggle library not installed. Run: pip install kaggle")
        return False
    
    try:
        kaggle.api.competition_download_files(
            'aptos2019-blindness-detection',
            path=config.raw_data_dir,
            quiet=False
        )
        
        zip_path = config.raw_data_dir / 'aptos2019-blindness-detection.zip'
        if zip_path.exists():
            logger.info("Extracting dataset archive...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(config.raw_data_dir)
            zip_path.unlink()  # Remove zip after extraction
        
        logger.info("✓ Retinopathy dataset downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Manual download: https://www.kaggle.com/c/aptos2019-blindness-detection/data")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_xray_image(img_path: Path, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Preprocess chest X-ray images.
    
    Applies:
    - Grayscale conversion
    - CLAHE contrast enhancement
    - Resizing and normalization
    """
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            return None
        
        img = cv2.resize(img, target_size)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Convert to RGB (3 channels for model input)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return None


def preprocess_retina_image(img_path: Path, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Preprocess retinal fundus images.
    
    Applies:
    - Gaussian blur for denoising
    - Resizing and normalization
    """
    try:
        img = cv2.imread(str(img_path))
        
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            return None
        
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_tb_dataset(config: DataConfig) -> Optional[pd.DataFrame]:
    """Prepare TB chest X-ray dataset for training."""
    logger.info("Preparing TB dataset...")
    
    tb_dir = config.raw_data_dir / 'TB_Chest_Radiography_Database'
    
    if not tb_dir.exists():
        # Try to find the dataset in alternative locations
        possible_paths = list(config.raw_data_dir.glob('**/Normal'))
        if possible_paths:
            tb_dir = possible_paths[0].parent
            logger.info(f"Found dataset at: {tb_dir}")
        else:
            logger.error("TB dataset not found. Please download manually.")
            return None
    
    normal_dir = tb_dir / 'Normal'
    tb_positive_dir = tb_dir / 'Tuberculosis'
    
    data = []
    
    # Process Normal class
    if normal_dir.exists():
        normal_files = list(normal_dir.glob('*.png')) + list(normal_dir.glob('*.jpg'))
        if config.max_samples_per_class:
            normal_files = normal_files[:config.max_samples_per_class]
        
        for img_path in normal_files:
            data.append({'path': img_path, 'label': 0, 'class': 'Normal'})
        
        logger.info(f"  Found {len(normal_files)} Normal images")
    
    # Process TB Positive class
    if tb_positive_dir.exists():
        tb_files = list(tb_positive_dir.glob('*.png')) + list(tb_positive_dir.glob('*.jpg'))
        if config.max_samples_per_class:
            tb_files = tb_files[:config.max_samples_per_class]
        
        for img_path in tb_files:
            data.append({'path': img_path, 'label': 1, 'class': 'TB'})
        
        logger.info(f"  Found {len(tb_files)} TB Positive images")
    
    if not data:
        logger.error("No images found in dataset directories")
        return None
    
    df = pd.DataFrame(data)
    logger.info(f"✓ Total images: {len(df)}")
    
    return df


def prepare_retinopathy_dataset(config: DataConfig) -> Optional[pd.DataFrame]:
    """Prepare APTOS 2019 Diabetic Retinopathy dataset for training."""
    logger.info("Preparing Diabetic Retinopathy dataset...")
    
    train_csv = config.raw_data_dir / 'train.csv'
    images_dir = config.raw_data_dir / 'train_images'
    
    if not train_csv.exists():
        logger.error("train.csv not found. Please download the dataset manually.")
        return None
    
    df = pd.read_csv(train_csv)
    
    # Convert to binary classification (No DR vs DR present)
    df['binary_label'] = (df['diagnosis'] > 0).astype(int)
    df['path'] = df['id_code'].apply(lambda x: images_dir / f'{x}.png')
    
    # Filter to existing images only
    df = df[df['path'].apply(lambda x: x.exists())]
    
    if config.max_samples_per_class:
        df = df.groupby('binary_label').apply(
            lambda x: x.sample(
                min(len(x), config.max_samples_per_class), 
                random_state=config.random_seed
            )
        ).reset_index(drop=True)
    
    df['label'] = df['binary_label']
    df['class'] = df['binary_label'].apply(lambda x: 'No_DR' if x == 0 else 'DR')
    
    logger.info(f"✓ Total images: {len(df)}")
    logger.info(f"  Class distribution: {df['class'].value_counts().to_dict()}")
    
    return df


# ─────────────────────────────────────────────────────────────────────────────
# DATA SPLITTING & PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def split_and_process_dataset(
    df: pd.DataFrame, 
    config: DataConfig,
    preprocess_fn: Callable
) -> None:
    """Split dataset and preprocess images for each split."""
    logger.info("Splitting and processing dataset...")
    
    # Stratified train/val/test split
    train_df, temp_df = train_test_split(
        df, 
        test_size=(config.val_ratio + config.test_ratio),
        stratify=df['label'], 
        random_state=config.random_seed
    )
    
    relative_test_size = config.test_ratio / (config.val_ratio + config.test_ratio)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=relative_test_size,
        stratify=temp_df['label'], 
        random_state=config.random_seed
    )
    
    logger.info(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    
    for split_name, split_df in splits.items():
        logger.info(f"Processing {split_name} split...")
        
        images = []
        labels = []
        valid_indices = []
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"  {split_name}"):
            img = preprocess_fn(row['path'], config.img_size)
            if img is not None:
                images.append(img)
                labels.append(row['label'])
                valid_indices.append(idx)
        
        split_dir = config.processed_data_dir / split_name
        
        # Save as numpy arrays
        np.save(split_dir / 'images.npy', np.array(images))
        np.save(split_dir / 'labels.npy', np.array(labels))
        
        # Save metadata
        split_df.loc[valid_indices].to_csv(split_dir / 'metadata.csv', index=False)
        
        logger.info(f"  ✓ Saved {len(images)} images to {split_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: DataConfig) -> bool:
    """Execute the complete data preparation pipeline."""
    
    print("\n" + "═" * 60)
    print("  MEDICAL IMAGING DATA PREPARATION PIPELINE")
    print("═" * 60)
    print(f"  Dataset:     {config.dataset_type.upper()}")
    print(f"  Output:      {config.processed_data_dir}")
    print(f"  Image Size:  {config.img_size}")
    print(f"  Max Samples: {config.max_samples_per_class or 'Unlimited'}")
    print("═" * 60 + "\n")
    
    # Create directories
    create_directories(config)
    
    # Download dataset if needed
    if config.dataset_type == 'tb':
        dataset_marker = config.raw_data_dir / 'TB_Chest_Radiography_Database'
        if not dataset_marker.exists():
            if not download_tb_dataset(config):
                logger.warning("Proceeding without download - dataset may already exist")
        df = prepare_tb_dataset(config)
        preprocess_fn = preprocess_xray_image
        
    elif config.dataset_type == 'retinopathy':
        dataset_marker = config.raw_data_dir / 'train.csv'
        if not dataset_marker.exists():
            if not download_retinopathy_dataset(config):
                logger.warning("Proceeding without download - dataset may already exist")
        df = prepare_retinopathy_dataset(config)
        preprocess_fn = preprocess_retina_image
        
    else:
        logger.error(f"Unknown dataset type: {config.dataset_type}")
        return False
    
    if df is None or len(df) == 0:
        logger.error("Failed to prepare dataset. Exiting.")
        return False
    
    # Process and save
    split_and_process_dataset(df, config, preprocess_fn)
    
    print("\n" + "═" * 60)
    print("  ✓ DATA PREPARATION COMPLETE!")
    print(f"  Output: {config.processed_data_dir}")
    print("═" * 60 + "\n")
    
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare medical imaging datasets for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_prep_medical.py --dataset tb
  python data_prep_medical.py --dataset retinopathy --max-samples 500
  python data_prep_medical.py --dataset tb --output ./custom_data
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='tb',
        choices=['tb', 'retinopathy'],
        help='Dataset type to prepare (default: tb)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./medical_imaging_data',
        help='Output directory for processed data (default: ./medical_imaging_data)'
    )
    
    parser.add_argument(
        '--max-samples', '-m',
        type=int,
        default=None,
        help='Maximum samples per class (default: no limit)'
    )
    
    parser.add_argument(
        '--img-size', '-s',
        type=int,
        default=224,
        help='Image size (square) in pixels (default: 224)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    config = DataConfig(
        dataset_type=args.dataset,
        base_dir=Path(args.output),
        img_size=(args.img_size, args.img_size),
        max_samples_per_class=args.max_samples,
        random_seed=args.seed
    )
    
    success = run_pipeline(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
