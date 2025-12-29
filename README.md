# Medical Imaging Diagnosis AI

This project uses deep learning to screen for Tuberculosis (TB) and Diabetic Retinopathy. It's designed to be a helpful tool for healthcare workers, offering a second pair of eyes to flag potential issues.

## key features

- **TB Screening**: Checks chest X-rays for signs of tuberculosis.
- **Eye Disease Detection**: Looks for diabetic retinopathy in retinal scans.
- **Visual Explanations**: Uses heatmaps (Grad-CAM) to show you exactly *where* the model is looking.
- **Simple Interface**: A web app that's easy to use - just upload an image and get results.
- **Privacy First**: Everything runs locally on your machine.

## how to get started

You'll need Python installed. Grab the code and install the dependencies:

```bash
pip install tensorflow numpy pandas opencv-python pillow streamlit kaggle scikit-learn matplotlib seaborn tqdm
```

### 1. get the data

The project needs data to learn. You have two options:
- **Tuberculosis**: Uses the TB Chest X-ray Database.
- **Diabetic Retinopathy**: Uses the APTOS 2019 dataset.

Run the data prep script and it will try to download them for you (you'll need a Kaggle API key):

```bash
python data_prep_medical.py
```

### 2. train the model

Once you have data, teach the AI:

```bash
# For TB detection
python train_medical.py --dataset tb

# For Eye Disease
python train_medical.py --dataset retinopathy
```

This might take a while depending on your computer. It uses transfer learning (ResNetSq/EfficientNet) so it learns pretty fast!

### 3. run the app

Start the web interface:

```bash
streamlit run app_medical.py
```

It should open in your browser automatically.

## a friendly warning

**This is not a doctor.** This is an AI tool designed to assist, not replace, medical professionals. Always verify results with a qualified expert. It can make mistakes (false positives/negatives), so use it as part of a broader screening process.

## under the hood

The system processes images to make them easier for the AI to "see" (like enhancing contrast for X-rays). It then uses powerful pre-trained models that have seen millions of images to detect patterns associated with disease.

Feel free to explore the code - `model_medical.py` has the architecture, and `app_medical.py` runs the interface!
