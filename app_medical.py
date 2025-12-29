import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import io

st.set_page_config(
    page_title="AI Medical Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 16px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .negative {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)


class Config:
    MODELS_DIR = Path('./models')
    RESULTS_DIR = Path('./results')
    IMG_SIZE = (224, 224)
    
    AVAILABLE_MODELS = {
        'TB Detection': {
            'model_path': 'tb_resnet50_best.keras',
            'classes': ['Normal', 'TB Positive'],
            'description': 'Detects tuberculosis from chest X-rays',
            'type': 'tb'
        },
        'Diabetic Retinopathy': {
            'model_path': 'retinopathy_resnet50_best.keras',
            'classes': ['No DR', 'DR Present'],
            'description': 'Detects diabetic retinopathy from retinal images',
            'type': 'retinopathy'
        }
    }


@st.cache_resource
def load_model(model_path):
    try:
        full_path = Config.MODELS_DIR / model_path
        if not full_path.exists():
            keras_files = list(Config.MODELS_DIR.glob('*.keras'))
            if keras_files:
                full_path = keras_files[0]
            else:
                return None
        
        model = tf.keras.models.load_model(full_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_xray_image(image, target_size=Config.IMG_SIZE):
    img = np.array(image)
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img = cv2.resize(img, target_size)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0
    
    return img


def preprocess_retina_image(image, target_size=Config.IMG_SIZE):
    img = np.array(image)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(img, target_size)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype(np.float32) / 255.0
    
    return img


def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None):
    try:
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break
                if isinstance(layer, tf.keras.Model):
                    for sublayer in reversed(layer.layers):
                        if isinstance(sublayer, tf.keras.layers.Conv2D):
                            last_conv_layer_name = sublayer.name
                            break
                    if last_conv_layer_name:
                        break
        
        if last_conv_layer_name is None:
            return None
        
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}")
        return None


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    img_uint8 = np.uint8(255 * img)
    overlayed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap, alpha, 0)
    
    return overlayed


def make_prediction(model, image, model_config):
    if model_config['type'] == 'tb':
        processed_img = preprocess_xray_image(image)
    else:
        processed_img = preprocess_retina_image(image)
    
    img_array = np.expand_dims(processed_img, axis=0)
    
    prediction = model.predict(img_array, verbose=0)[0][0]
    predicted_class = int(prediction > 0.5)
    confidence = prediction if predicted_class == 1 else (1 - prediction)
    
    heatmap = get_gradcam_heatmap(model, img_array)
    if heatmap is not None:
        overlayed = overlay_heatmap(processed_img, heatmap)
    else:
        overlayed = None
    
    return {
        'class': model_config['classes'][predicted_class],
        'class_index': predicted_class,
        'confidence': float(confidence),
        'probability': float(prediction),
        'processed_image': processed_img,
        'heatmap': overlayed
    }


def display_results(result, model_config):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image (Processed)")
        st.image(result['processed_image'], use_container_width=True)
    
    with col2:
        st.subheader("Attention Map (Grad-CAM)")
        if result['heatmap'] is not None:
            st.image(result['heatmap'], use_container_width=True)
            st.caption("Red areas indicate regions the model focused on for its prediction")
        else:
            st.info("Grad-CAM visualization not available")
    
    is_positive = result['class_index'] == 1
    box_class = "positive" if is_positive else "negative"
    
    st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h2 style="margin: 0;">Prediction: {result['class']}</h2>
            <h3 style="margin: 10px 0;">Confidence: {result['confidence']*100:.2f}%</h3>
            <p style="margin: 0;">Probability Score: {result['probability']:.4f}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Confidence Meter")
    fig, ax = plt.subplots(figsize=(10, 2))
    
    colors = ['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF9800', '#FF5722']
    confidence_pct = result['confidence'] * 100
    color_idx = min(int(confidence_pct / 20), 5)
    
    ax.barh([0], [confidence_pct], color=colors[color_idx], height=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Confidence (%)', fontsize=12)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    
    for i in range(0, 101, 20):
        ax.axvline(i, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.warning("""
        ⚠️ **Medical Disclaimer**: This AI model is for screening purposes only and should not be used as a 
        definitive diagnostic tool. All results should be reviewed by qualified medical professionals. 
        This tool is designed to assist healthcare workers in identifying potential cases that require 
        further examination.
    """)


def main():
    st.title("🏥 AI Medical Image Diagnosis")
    st.markdown("*Powered by Deep Learning for Healthcare Access*")
    
    with st.sidebar:
        st.header("Configuration")
        
        selected_model_name = st.selectbox(
            "Select Diagnosis Type",
            options=list(Config.AVAILABLE_MODELS.keys())
        )
        
        model_config = Config.AVAILABLE_MODELS[selected_model_name]
        
        st.info(f"**Model**: {selected_model_name}\n\n{model_config['description']}")
        
        with st.spinner("Loading model..."):
            model = load_model(model_config['model_path'])
        
        if model is None:
            st.error(f"⚠️ Model not found at: {Config.MODELS_DIR / model_config['model_path']}")
            st.info("Please train a model first using train_medical.py")
            st.stop()
        else:
            st.success("✓ Model loaded successfully!")
        
        st.markdown("---")
        
        st.header("About")
        st.markdown("""
            This application uses deep learning to assist in medical diagnosis:
            
            - **TB Detection**: Screens chest X-rays for tuberculosis
            - **Diabetic Retinopathy**: Detects eye disease from retinal scans
            
            **How it works:**
            1. Upload a medical image
            2. AI analyzes the image
            3. Get instant screening results
            4. Review attention maps to see what the AI focused on
        """)
        
        st.markdown("---")
        st.caption("Built with TensorFlow & Streamlit")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Diagnose", "📊 Batch Analysis", "ℹ️ Instructions"])
    
    with tab1:
        st.header("Single Image Diagnosis")
        
        uploaded_file = st.file_uploader(
            "Upload Medical Image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a chest X-ray (for TB) or retinal scan (for DR)"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                st.caption(f"Size: {image.size[0]} x {image.size[1]}")
                st.caption(f"Format: {image.format}")
            
            with col2:
                st.info(f"""
                    **Model**: {selected_model_name}
                    
                    **Classes**: {', '.join(model_config['classes'])}
                    
                    Click 'Analyze Image' to get diagnosis prediction.
                """)
            
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    result = make_prediction(model, image, model_config)
                    
                    st.markdown("---")
                    st.header("Diagnosis Results")
                    display_results(result, model_config)
                    
                    st.markdown("---")
                    report = f"""
MEDICAL IMAGE ANALYSIS REPORT
{'='*50}

Model: {selected_model_name}
Date: {Path(__file__).stat().st_mtime}

RESULTS
-------
Diagnosis: {result['class']}
Confidence: {result['confidence']*100:.2f}%
Probability Score: {result['probability']:.4f}

DISCLAIMER
----------
This is an AI-assisted screening tool. Results should be 
reviewed by qualified medical professionals before making 
any clinical decisions.
                    """
                    
                    st.download_button(
                        label="📄 Download Report",
                        data=report,
                        file_name=f"diagnosis_report_{selected_model_name.replace(' ', '_').lower()}.txt",
                        mime="text/plain"
                    )
    
    with tab2:
        st.header("Batch Image Analysis")
        st.info("Upload multiple images for batch processing")
        
        uploaded_files = st.file_uploader(
            "Upload Medical Images",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple images for batch analysis"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            if st.button("🔍 Analyze All Images", type="primary"):
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {idx+1}/{len(uploaded_files)}...")
                    
                    image = Image.open(uploaded_file)
                    result = make_prediction(model, image, model_config)
                    
                    results.append({
                        'filename': uploaded_file.name,
                        'class': result['class'],
                        'confidence': result['confidence'],
                        'probability': result['probability']
                    })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Analysis complete!")
                
                st.subheader("Batch Results")
                st.dataframe(results, use_container_width=True)
                
                positive_count = sum(1 for r in results if r['class'] == model_config['classes'][1])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    st.metric("Positive Cases", positive_count)
                with col3:
                    st.metric("Average Confidence", f"{np.mean([r['confidence'] for r in results])*100:.1f}%")
    
    with tab3:
        st.header("Instructions & Guidelines")
        
        st.markdown("""
        ### How to Use This Tool
        
        1. **Select Model**: Choose between TB Detection or Diabetic Retinopathy detection
        2. **Upload Image**: Upload a medical image (X-ray or retinal scan)
        3. **Analyze**: Click the analyze button to get AI prediction
        4. **Review Results**: Check the diagnosis, confidence score, and attention map
        5. **Consult Professional**: Always verify results with a qualified healthcare provider
        
        ### Image Requirements
        
        **For TB Detection (Chest X-rays):**
        - Frontal chest X-ray (PA or AP view)
        - Clear, good quality images
        - JPEG, PNG, or TIFF format
        - Minimum resolution: 224x224 pixels
        
        **For Diabetic Retinopathy (Retinal Scans):**
        - Color fundus photographs
        - Well-lit, focused retinal images
        - JPEG, PNG, or TIFF format
        - Minimum resolution: 224x224 pixels
        
        ### Understanding Results
        
        - **Confidence Score**: How certain the model is about its prediction (0-100%)
        - **Attention Map (Grad-CAM)**: Visual explanation showing which parts of the image influenced the prediction
        - **Red areas**: Regions the AI focused on
        
        ### Important Notes
        
        ⚠️ This is a **screening tool**, not a diagnostic tool
        
        ⚠️ Always consult with qualified medical professionals
        
        ⚠️ False positives and negatives are possible
        
        ⚠️ Use as part of comprehensive healthcare workflow
        
        ### Technical Details
        
        - **Model Architecture**: Transfer Learning (ResNet50/EfficientNet)
        - **Training Data**: Public medical imaging datasets
        - **Preprocessing**: CLAHE enhancement for X-rays, normalization for retinal scans
        """)


if __name__ == "__main__":
    main()
