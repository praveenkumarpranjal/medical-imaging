import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
from tqdm import tqdm
import kaggle
import zipfile


class Config:
    DATASET_TYPE = 'tb'
    BASE_DIR = Path('./medical_imaging_data')
    RAW_DATA_DIR = BASE_DIR / 'raw'
    PROCESSED_DATA_DIR = BASE_DIR / 'processed'
    IMG_SIZE = (224, 224)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    MAX_SAMPLES_PER_CLASS = 1000
    RANDOM_SEED = 42


def create_directories():
    Config.BASE_DIR.mkdir(exist_ok=True)
    Config.RAW_DATA_DIR.mkdir(exist_ok=True)
    Config.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        (Config.PROCESSED_DATA_DIR / split).mkdir(exist_ok=True)


def download_tb_dataset():
    print("Downloading TB Chest X-ray dataset from Kaggle...")
    print("Make sure you have kaggle.json in ~/.kaggle/")
    
    try:
        kaggle.api.dataset_download_files(
            'tawsifurrahman/tuberculosis-tb-chest-xray-dataset',
            path=Config.RAW_DATA_DIR,
            unzip=True
        )
        print("TB dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Manually download from:")
        print("https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
        return False


def download_retinopathy_dataset():
    print("Downloading APTOS 2019 Diabetic Retinopathy dataset from Kaggle...")
    print("Make sure you have kaggle.json in ~/.kaggle/")
    
    try:
        kaggle.api.competition_download_files(
            'aptos2019-blindness-detection',
            path=Config.RAW_DATA_DIR,
            quiet=False
        )
        
        zip_path = Config.RAW_DATA_DIR / 'aptos2019-blindness-detection.zip'
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(Config.RAW_DATA_DIR)
        
        print("Retinopathy dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Manually download from:")
        print("https://www.kaggle.com/c/aptos2019-blindness-detection/data")
        return False


def preprocess_xray_image(img_path, target_size=Config.IMG_SIZE):
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        img = cv2.resize(img, target_size)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def preprocess_retina_image(img_path, target_size=Config.IMG_SIZE):
    try:
        img = cv2.imread(str(img_path))
        
        if img is None:
            return None
        
        img = cv2.resize(img, target_size)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def prepare_tb_dataset():
    print("Preparing TB dataset...")
    
    tb_dir = Config.RAW_DATA_DIR / 'TB_Chest_Radiography_Database'
    
    if not tb_dir.exists():
        possible_paths = list(Config.RAW_DATA_DIR.glob('**/Normal'))
        if possible_paths:
            tb_dir = possible_paths[0].parent
        else:
            print("TB dataset not found. Please download manually.")
            return None
    
    normal_dir = tb_dir / 'Normal'
    tb_positive_dir = tb_dir / 'Tuberculosis'
    
    data = []
    
    if normal_dir.exists():
        normal_files = list(normal_dir.glob('*.png')) + list(normal_dir.glob('*.jpg'))
        if Config.MAX_SAMPLES_PER_CLASS:
            normal_files = normal_files[:Config.MAX_SAMPLES_PER_CLASS]
        for img_path in normal_files:
            data.append({'path': img_path, 'label': 0, 'class': 'Normal'})
    
    if tb_positive_dir.exists():
        tb_files = list(tb_positive_dir.glob('*.png')) + list(tb_positive_dir.glob('*.jpg'))
        if Config.MAX_SAMPLES_PER_CLASS:
            tb_files = tb_files[:Config.MAX_SAMPLES_PER_CLASS]
        for img_path in tb_files:
            data.append({'path': img_path, 'label': 1, 'class': 'TB'})
    
    df = pd.DataFrame(data)
    print(f"Total images found: {len(df)}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    
    return df


def prepare_retinopathy_dataset():
    print("Preparing Diabetic Retinopathy dataset...")
    
    train_csv = Config.RAW_DATA_DIR / 'train.csv'
    images_dir = Config.RAW_DATA_DIR / 'train_images'
    
    if not train_csv.exists():
        print("train.csv not found. Please download the dataset manually.")
        return None
    
    df = pd.read_csv(train_csv)
    
    df['binary_label'] = (df['diagnosis'] > 0).astype(int)
    df['path'] = df['id_code'].apply(lambda x: images_dir / f'{x}.png')
    
    df = df[df['path'].apply(lambda x: x.exists())]
    
    if Config.MAX_SAMPLES_PER_CLASS:
        df = df.groupby('binary_label').apply(
            lambda x: x.sample(min(len(x), Config.MAX_SAMPLES_PER_CLASS), random_state=Config.RANDOM_SEED)
        ).reset_index(drop=True)
    
    df['label'] = df['binary_label']
    df['class'] = df['binary_label'].apply(lambda x: 'No_DR' if x == 0 else 'DR')
    
    print(f"Total images found: {len(df)}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    
    return df


def split_and_process_dataset(df, dataset_type):
    print("\nSplitting and processing dataset...")
    
    train_df, temp_df = train_test_split(
        df, test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
        stratify=df['label'], random_state=Config.RANDOM_SEED
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=Config.TEST_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO),
        stratify=temp_df['label'], random_state=Config.RANDOM_SEED
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    preprocess_fn = preprocess_xray_image if dataset_type == 'tb' else preprocess_retina_image
    
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    
    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} set...")
        
        images = []
        labels = []
        valid_indices = []
        
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df)):
            img = preprocess_fn(row['path'])
            if img is not None:
                images.append(img)
                labels.append(row['label'])
                valid_indices.append(idx)
        
        split_dir = Config.PROCESSED_DATA_DIR / split_name
        np.save(split_dir / 'images.npy', np.array(images))
        np.save(split_dir / 'labels.npy', np.array(labels))
        
        split_df.loc[valid_indices].to_csv(split_dir / 'metadata.csv', index=False)
        
        print(f"Saved {len(images)} images to {split_dir}")


def main():
    print("=" * 60)
    print("Medical Image Data Preparation")
    print(f"Dataset Type: {Config.DATASET_TYPE.upper()}")
    print("=" * 60)
    
    create_directories()
    
    if Config.DATASET_TYPE == 'tb':
        if not (Config.RAW_DATA_DIR / 'TB_Chest_Radiography_Database').exists():
            download_tb_dataset()
        df = prepare_tb_dataset()
    elif Config.DATASET_TYPE == 'retinopathy':
        if not (Config.RAW_DATA_DIR / 'train.csv').exists():
            download_retinopathy_dataset()
        df = prepare_retinopathy_dataset()
    else:
        print("Invalid dataset type. Choose 'tb' or 'retinopathy'")
        return
    
    if df is None or len(df) == 0:
        print("Failed to prepare dataset. Exiting.")
        return
    
    split_and_process_dataset(df, Config.DATASET_TYPE)
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Processed data saved to: {Config.PROCESSED_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
