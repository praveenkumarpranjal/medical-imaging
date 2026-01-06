# 🔬 MedVision AI - Medical Imaging Diagnosis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI-powered medical image analysis for TB and Diabetic Retinopathy screening**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Disclaimer](#%EF%B8%8F-medical-disclaimer)

</div>

---

## ✨ Features

- **🫁 TB Screening** — Detect tuberculosis indicators in chest X-rays
- **👁️ Diabetic Retinopathy** — Screen retinal fundus images for DR
- **🔥 Explainable AI** — Grad-CAM heatmaps show where the model is looking
- **🎨 Modern Interface** — Beautiful, glassmorphic Streamlit web app
- **🚀 Transfer Learning** — ResNet50, EfficientNet, and MobileNet backbones
- **🔒 Privacy First** — Everything runs locally on your machine

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-imaging.git
cd medical-imaging

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- TensorFlow 2.12+
- ~2GB disk space for models
- GPU recommended (but not required)

## 🚀 Quick Start

### 1. Prepare the Data

Download and preprocess the dataset:

```bash
# For TB Chest X-ray dataset
python data_prep_medical.py --dataset tb

# For Diabetic Retinopathy dataset
python data_prep_medical.py --dataset retinopathy

# With options
python data_prep_medical.py --dataset tb --max-samples 500 --output ./my_data
```

> **Note:** You'll need a [Kaggle API key](https://www.kaggle.com/docs/api) in `~/.kaggle/kaggle.json`

### 2. Train the Model

```bash
# Train TB detection model
python train_medical.py --dataset tb --model resnet50

# Train DR detection with EfficientNet
python train_medical.py --dataset retinopathy --model efficientnet

# More options
python train_medical.py --dataset tb --epochs 100 --batch-size 64 --no-finetune
```

**Available models:**
- `resnet50` — Balanced accuracy and speed (default)
- `efficientnet` — Higher accuracy, more parameters
- `efficientnet_b3` — Even higher accuracy
- `mobilenet` — Fastest inference, smaller model
- `custom` — Train from scratch (no transfer learning)

### 3. Launch the App

```bash
streamlit run app_medical.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
medical-imaging/
├── app_medical.py          # Streamlit web application
├── data_prep_medical.py    # Data download & preprocessing
├── model_medical.py        # Model architectures & Grad-CAM
├── train_medical.py        # Training pipeline
├── requirements.txt        # Python dependencies
├── README.md
├── models/                 # Saved model checkpoints
├── results/                # Training results & metrics
└── medical_imaging_data/   # Datasets (created by data_prep)
    ├── raw/                # Downloaded datasets
    └── processed/          # Preprocessed numpy arrays
        ├── train/
        ├── val/
        └── test/
```

## 🏗️ Architecture

### Model Pipeline

```
Input Image (224×224×3)
        ↓
[Data Augmentation]     → Random flip, rotation, zoom, contrast
        ↓
[Pretrained Backbone]   → ResNet50/EfficientNet/MobileNet (ImageNet weights)
        ↓
[Classification Head]   → BatchNorm → Dense(256) → Dense(128) → Sigmoid
        ↓
Output: Probability [0, 1]
```

### Preprocessing

| Dataset | Preprocessing Steps |
|---------|---------------------|
| **TB X-rays** | Grayscale → CLAHE enhancement → Resize → Normalize |
| **Retinal Scans** | Gaussian blur → Color normalization → Resize → Normalize |

### Training Strategy

1. **Phase 1:** Train classification head with frozen backbone (50 epochs)
2. **Phase 2:** Fine-tune last 30 backbone layers with lower LR (20 epochs)

## 📊 Example Results

After training on the TB dataset:

| Metric | Score |
|--------|-------|
| Accuracy | 95.2% |
| AUC-ROC | 0.982 |
| Precision | 94.8% |
| Recall | 95.6% |

> Results may vary based on data quality and training configuration.

## 🔧 CLI Reference

### Data Preparation

```bash
python data_prep_medical.py [OPTIONS]

Options:
  -d, --dataset     Dataset type: 'tb' or 'retinopathy' (default: tb)
  -o, --output      Output directory (default: ./medical_imaging_data)
  -m, --max-samples Maximum samples per class (default: no limit)
  -s, --img-size    Image size in pixels (default: 224)
  --seed            Random seed (default: 42)
```

### Training

```bash
python train_medical.py [OPTIONS]

Options:
  -d, --dataset       Dataset: 'tb' or 'retinopathy' (default: tb)
  -m, --model         Model: resnet50, efficientnet, mobilenet, custom (default: resnet50)
  -e, --epochs        Training epochs (default: 50)
  -b, --batch-size    Batch size (default: 32)
  -lr, --learning-rate Initial learning rate (default: 1e-4)
  --no-finetune       Skip fine-tuning phase
  --data-dir          Path to processed data
```

## 🖥️ Web Interface

The Streamlit app provides:

- **Single Image Analysis** — Upload and analyze individual images
- **Batch Processing** — Analyze multiple images at once
- **Attention Maps** — Grad-CAM visualization of model focus
- **Confidence Scores** — Probability and uncertainty metrics
- **Report Download** — Export analysis reports as text files

## ⚠️ Medical Disclaimer

> **This software is NOT a medical device and should NOT be used for clinical diagnosis.**

- This is an AI screening tool designed to **assist**, not replace, medical professionals
- All results must be verified by qualified healthcare providers
- False positives and false negatives can and will occur
- Use only as part of a comprehensive healthcare workflow
- Not FDA/CE approved for diagnostic use

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [TB Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) — Tawsif Rahman
- [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) — Asia Pacific Tele-Ophthalmology Society
- TensorFlow and Keras teams
- Streamlit team

---

<div align="center">
<sub>Built with ❤️ for accessible healthcare</sub>
</div>
