"""
Medical Imaging Diagnosis AI - Streamlit Application
A modern, professional interface for AI-powered medical image analysis.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import io
from typing import Optional, Dict, Any, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedVision AI | Medical Diagnosis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
/* Import modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root variables for theming */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    --danger-gradient: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --dark-bg: #0e1117;
    --card-bg: rgba(17, 25, 40, 0.75);
    --glass-border: rgba(255, 255, 255, 0.125);
    --text-primary: #ffffff;
    --text-secondary: rgba(255, 255, 255, 0.7);
    --accent-blue: #667eea;
    --accent-purple: #764ba2;
}

/* Base styling */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Main container padding */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* Glassmorphism cards */
.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border-radius: 16px;
    border: 1px solid var(--glass-border);
    padding: 24px;
    margin: 16px 0;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
}

/* Hero section */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.2rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-bottom: 2rem;
}

/* Custom buttons */
.stButton>button {
    background: var(--primary-gradient);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
}

.stButton>button:active {
    transform: translateY(0);
}

/* Prediction result boxes */
.result-positive {
    background: linear-gradient(135deg, rgba(235, 51, 73, 0.15) 0%, rgba(244, 92, 67, 0.15) 100%);
    border: 1px solid rgba(235, 51, 73, 0.5);
    border-left: 4px solid #eb3349;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
}

.result-negative {
    background: linear-gradient(135deg, rgba(17, 153, 142, 0.15) 0%, rgba(56, 239, 125, 0.15) 100%);
    border: 1px solid rgba(17, 153, 142, 0.5);
    border-left: 4px solid #11998e;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
}

.result-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.result-confidence {
    font-size: 2.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

/* Metric cards */
.metric-card {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    border-radius: 12px;
    border: 1px solid var(--glass-border);
    padding: 20px;
    text-align: center;
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: scale(1.02);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* File uploader styling */
.stFileUploader {
    border: 2px dashed var(--glass-border);
    border-radius: 16px;
    padding: 20px;
    transition: border-color 0.3s ease;
}

.stFileUploader:hover {
    border-color: var(--accent-blue);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(14, 17, 23, 0.95) 0%, rgba(30, 30, 46, 0.95) 100%);
    border-right: 1px solid var(--glass-border);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: var(--card-bg);
    border-radius: 10px;
    padding: 10px 20px;
    border: 1px solid var(--glass-border);
    color: var(--text-secondary);
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(102, 126, 234, 0.2);
    color: white;
}

.stTabs [aria-selected="true"] {
    background: var(--primary-gradient) !important;
    color: white !important;
    border: none !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: var(--primary-gradient);
}

/* Info/Warning/Error boxes */
.stAlert {
    border-radius: 12px;
    border: none;
}

/* Custom divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--glass-border), transparent);
    margin: 24px 0;
}

/* Pulse animation for loading */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

/* Fade in animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out forwards;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}

.badge-success {
    background: rgba(56, 239, 125, 0.2);
    color: #38ef7d;
}

.badge-warning {
    background: rgba(255, 193, 7, 0.2);
    color: #ffc107;
}

.badge-danger {
    background: rgba(235, 51, 73, 0.2);
    color: #eb3349;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--dark-bg);
}

::-webkit-scrollbar-thumb {
    background: var(--accent-blue);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-purple);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """Application configuration."""
    MODELS_DIR = Path('./models')
    RESULTS_DIR = Path('./results')
    IMG_SIZE = (224, 224)
    
    AVAILABLE_MODELS = {
        'Tuberculosis Detection': {
            'model_path': 'tb_resnet50_best.keras',
            'classes': ['Normal', 'TB Positive'],
            'description': 'AI-powered tuberculosis screening from chest X-ray images',
            'type': 'tb',
            'icon': '🫁',
            'color': '#667eea'
        },
        'Diabetic Retinopathy': {
            'model_path': 'retinopathy_resnet50_best.keras',
            'classes': ['No DR', 'DR Present'],
            'description': 'Early detection of diabetic retinopathy from retinal fundus images',
            'type': 'retinopathy',
            'icon': '👁️',
            'color': '#11998e'
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING & CACHING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_path: str) -> Optional[tf.keras.Model]:
    """Load and cache a trained model."""
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


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_xray_image(image: Image.Image, target_size: Tuple[int, int] = Config.IMG_SIZE) -> np.ndarray:
    """Preprocess chest X-ray images with CLAHE enhancement."""
    img = np.array(image)
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img = cv2.resize(img, target_size)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0
    
    return img


def preprocess_retina_image(image: Image.Image, target_size: Tuple[int, int] = Config.IMG_SIZE) -> np.ndarray:
    """Preprocess retinal fundus images."""
    img = np.array(image)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(img, target_size)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype(np.float32) / 255.0
    
    return img


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def get_gradcam_heatmap(model: tf.keras.Model, img_array: np.ndarray, 
                        last_conv_layer_name: Optional[str] = None) -> Optional[np.ndarray]:
    """Generate Grad-CAM heatmap for model interpretability."""
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
        return None


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, 
                   alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on the original image."""
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Use a custom colormap for better aesthetics
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_MAGMA)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    img_uint8 = np.uint8(255 * img)
    overlayed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap, alpha, 0)
    
    return overlayed


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def make_prediction(model: tf.keras.Model, image: Image.Image, 
                   model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference and generate prediction results."""
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


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def render_hero_section():
    """Render the hero section with title and description."""
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 class="hero-title">🔬 MedVision AI</h1>
            <p class="hero-subtitle">
                Advanced deep learning for medical image diagnosis<br>
                <span style="font-size: 0.9rem; opacity: 0.7;">Powered by ResNet50 & EfficientNet architectures</span>
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_model_selector(selected_model_name: str, model_config: Dict[str, Any]):
    """Render model selection card."""
    st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 2rem;">{model_config['icon']}</span>
                <div>
                    <h3 style="margin: 0; font-weight: 600;">{selected_model_name}</h3>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">{model_config['description']}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_results(result: Dict[str, Any], model_config: Dict[str, Any]):
    """Render prediction results with modern styling."""
    is_positive = result['class_index'] == 1
    result_class = "result-positive" if is_positive else "result-negative"
    status_text = "⚠️ Requires Attention" if is_positive else "✅ Normal"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📷 Processed Image")
        st.image(result['processed_image'], use_container_width=True)
    
    with col2:
        st.markdown("#### 🔥 AI Attention Map")
        if result['heatmap'] is not None:
            st.image(result['heatmap'], use_container_width=True)
            st.caption("Bright areas indicate regions of focus for the AI's prediction")
        else:
            st.info("Attention visualization not available")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Result card
    st.markdown(f"""
        <div class="{result_class} fade-in">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">DIAGNOSIS RESULT</p>
                    <h2 class="result-title">{result['class']}</h2>
                    <span class="badge {'badge-danger' if is_positive else 'badge-success'}">{status_text}</span>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">CONFIDENCE</p>
                    <span class="result-confidence">{result['confidence']*100:.1f}%</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Confidence meter
    st.markdown("#### 📊 Confidence Analysis")
    
    confidence_pct = result['confidence'] * 100
    
    # Create a more visually appealing progress representation
    fig, ax = plt.subplots(figsize=(12, 1.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    # Background bar
    ax.barh([0], [100], color='#1a1a2e', height=0.6, left=0)
    
    # Gradient effect using multiple small bars
    gradient_colors = plt.cm.RdYlGn_r if is_positive else plt.cm.RdYlGn
    for i in range(int(confidence_pct)):
        color = gradient_colors(i / 100)
        ax.barh([0], [1], color=color, height=0.6, left=i)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Disclaimer
    st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #ffc107;">
            <p style="margin: 0; color: #ffc107; font-weight: 600;">⚠️ Important Medical Disclaimer</p>
            <p style="margin: 8px 0 0 0; color: var(--text-secondary); font-size: 0.9rem;">
                This AI tool is designed for screening purposes only and should not replace professional medical diagnosis. 
                All results must be verified by qualified healthcare professionals. False positives and negatives may occur.
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_batch_results(results: list, model_config: Dict[str, Any]):
    """Render batch analysis results."""
    positive_count = sum(1 for r in results if r['class'] == model_config['classes'][1])
    avg_confidence = np.mean([r['confidence'] for r in results]) * 100
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(results)}</div>
                <div class="metric-label">Images Analyzed</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{positive_count}</div>
                <div class="metric-label">Positive Cases</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(results) - positive_count}</div>
                <div class="metric-label">Normal Cases</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_confidence:.1f}%</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results table
    st.dataframe(
        results,
        use_container_width=True,
        column_config={
            "filename": st.column_config.TextColumn("📁 File Name"),
            "class": st.column_config.TextColumn("🏷️ Diagnosis"),
            "confidence": st.column_config.ProgressColumn(
                "📊 Confidence",
                format="%.2f%%",
                min_value=0,
                max_value=1
            ),
            "probability": st.column_config.NumberColumn("📈 Probability", format="%.4f")
        }
    )


def render_sidebar(available_models: Dict[str, Dict]) -> Tuple[str, Dict, Optional[tf.keras.Model]]:
    """Render sidebar with model selection and info."""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                          font-weight: 700; margin: 0;">⚙️ Configuration</h2>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        selected_model_name = st.selectbox(
            "🔬 Select Diagnosis Type",
            options=list(available_models.keys()),
            help="Choose the type of medical analysis to perform"
        )
        
        model_config = available_models[selected_model_name]
        
        st.markdown(f"""
            <div class="glass-card" style="padding: 16px;">
                <p style="margin: 0 0 8px 0;">{model_config['icon']} <strong>{selected_model_name}</strong></p>
                <p style="margin: 0; color: var(--text-secondary); font-size: 0.85rem;">{model_config['description']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Loading model..."):
            model = load_model(model_config['model_path'])
        
        if model is None:
            st.markdown("""
                <div class="glass-card" style="border-left: 4px solid #eb3349; padding: 16px;">
                    <p style="margin: 0; color: #eb3349; font-weight: 600;">⚠️ Model Not Found</p>
                    <p style="margin: 8px 0 0 0; color: var(--text-secondary); font-size: 0.85rem;">
                        Please train a model first using:<br>
                        <code>python train_medical.py</code>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="glass-card" style="border-left: 4px solid #38ef7d; padding: 16px;">
                    <p style="margin: 0; color: #38ef7d; font-weight: 600;">✅ Model Ready</p>
                    <p style="margin: 4px 0 0 0; color: var(--text-secondary); font-size: 0.85rem;">
                        Neural network loaded successfully
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div style="padding: 0 0 1rem 0;">
                <h3 style="color: var(--text-primary); font-size: 1rem; margin-bottom: 12px;">📖 Quick Guide</h3>
                <ol style="color: var(--text-secondary); font-size: 0.85rem; padding-left: 20px; margin: 0;">
                    <li style="margin-bottom: 8px;">Select diagnosis type</li>
                    <li style="margin-bottom: 8px;">Upload medical image</li>
                    <li style="margin-bottom: 8px;">Click analyze</li>
                    <li style="margin-bottom: 8px;">Review AI insights</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <p style="color: var(--text-secondary); font-size: 0.75rem; margin: 0;">
                    Built with TensorFlow & Streamlit<br>
                    <span style="opacity: 0.7;">v2.0 | 2024</span>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        return selected_model_name, model_config, model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    
    # Sidebar
    selected_model_name, model_config, model = render_sidebar(Config.AVAILABLE_MODELS)
    
    # Hero section
    render_hero_section()
    
    if model is None:
        st.warning("Please train a model before using the application. See the sidebar for instructions.")
        return
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Diagnosis", "📊 Batch Analysis", "📚 Documentation"])
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: SINGLE IMAGE DIAGNOSIS
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Upload Medical Image")
        
        render_model_selector(selected_model_name, model_config)
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help=f"Upload a {'chest X-ray' if model_config['type'] == 'tb' else 'retinal fundus'} image"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 📤 Uploaded Image")
                st.image(image, use_container_width=True)
                st.caption(f"📐 {image.size[0]} × {image.size[1]} px  |  📁 {uploaded_file.name}")
            
            with col2:
                st.markdown("#### ℹ️ Analysis Info")
                st.markdown(f"""
                    <div class="glass-card">
                        <p><strong>Model:</strong> {selected_model_name}</p>
                        <p><strong>Detection:</strong> {' vs '.join(model_config['classes'])}</p>
                        <p><strong>Architecture:</strong> ResNet50 with Transfer Learning</p>
                        <p style="margin-bottom: 0;"><strong>Explainability:</strong> Grad-CAM Attention Maps</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔬 AI is analyzing the image..."):
                    result = make_prediction(model, image, model_config)
                
                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                st.markdown("## 📋 Diagnosis Results")
                
                render_results(result, model_config)
                
                # Download report
                st.markdown("<br>", unsafe_allow_html=True)
                report = f"""MEDVISION AI - MEDICAL IMAGE ANALYSIS REPORT
{'═' * 60}

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {selected_model_name}
File: {uploaded_file.name}

RESULTS
{'─' * 60}
Diagnosis: {result['class']}
Confidence: {result['confidence']*100:.2f}%
Probability Score: {result['probability']:.4f}

INTERPRETATION
{'─' * 60}
{'The AI has detected potential indicators that require medical attention.' if result['class_index'] == 1 else 'No significant abnormalities were detected by the AI.'}

DISCLAIMER
{'─' * 60}
This report is generated by an AI screening tool and should NOT be 
used as a definitive medical diagnosis. All findings must be reviewed 
and confirmed by qualified healthcare professionals.

{'═' * 60}
Generated by MedVision AI v2.0
"""
                
                st.download_button(
                    "📥 Download Report",
                    data=report,
                    file_name=f"medvision_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: BATCH ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Batch Image Analysis")
        st.markdown("Upload multiple images for efficient bulk screening")
        
        uploaded_files = st.file_uploader(
            "Upload multiple medical images",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Select multiple files for batch analysis"
        )
        
        if uploaded_files:
            st.markdown(f"""
                <div class="glass-card">
                    <p style="margin: 0;"><strong>📁 {len(uploaded_files)}</strong> images ready for analysis</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Analyze All Images", type="primary", use_container_width=True):
                results = []
                
                progress_bar = st.progress(0)
                status = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status.markdown(f"🔬 Processing **{file.name}** ({idx+1}/{len(uploaded_files)})")
                    
                    image = Image.open(file)
                    result = make_prediction(model, image, model_config)
                    
                    results.append({
                        'filename': file.name,
                        'class': result['class'],
                        'confidence': result['confidence'],
                        'probability': result['probability']
                    })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status.markdown("✅ **Analysis complete!**")
                
                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                st.markdown("## 📊 Batch Results")
                
                render_batch_results(results, model_config)
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: DOCUMENTATION
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 📚 Documentation & Guidelines")
        
        with st.expander("🔍 How to Use This Tool", expanded=True):
            st.markdown("""
                1. **Select Model**: Choose between TB Detection or Diabetic Retinopathy from the sidebar
                2. **Upload Image**: Drag and drop or click to upload a medical image
                3. **Analyze**: Click the analyze button to run AI inference
                4. **Review Results**: Examine the diagnosis, confidence score, and attention map
                5. **Consult Professional**: Always verify results with qualified healthcare providers
            """)
        
        with st.expander("📋 Image Requirements"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    **🫁 TB Detection (Chest X-rays)**
                    - Frontal chest X-ray (PA or AP view)
                    - Clear, high-quality images
                    - JPEG, PNG, or TIFF format
                    - Minimum 224×224 pixels
                """)
            
            with col2:
                st.markdown("""
                    **👁️ Diabetic Retinopathy (Retinal Scans)**
                    - Color fundus photographs
                    - Well-lit, focused images
                    - JPEG, PNG, or TIFF format
                    - Minimum 224×224 pixels
                """)
        
        with st.expander("🧠 Understanding AI Results"):
            st.markdown("""
                **Confidence Score**: The model's certainty about its prediction (0-100%)
                
                **Attention Map (Grad-CAM)**: Visual explanation showing which regions influenced the prediction
                - 🔴 **Bright/warm areas**: High importance regions
                - 🔵 **Dark/cool areas**: Lower importance regions
                
                **Probability Score**: Raw model output before thresholding (0-1)
            """)
        
        with st.expander("⚠️ Important Limitations"):
            st.markdown("""
                - This is a **screening tool**, not a diagnostic tool
                - AI models can produce **false positives** and **false negatives**
                - Results should **always** be verified by qualified medical professionals
                - The tool is designed to **assist**, not replace, clinical judgment
                - Performance may vary with image quality and acquisition conditions
            """)
        
        with st.expander("🔧 Technical Details"):
            st.markdown("""
                **Architecture**: Transfer Learning with ResNet50/EfficientNet
                
                **Training Data**: 
                - TB: TB Chest X-ray Database (Kaggle)
                - DR: APTOS 2019 Blindness Detection (Kaggle)
                
                **Preprocessing**:
                - X-rays: CLAHE contrast enhancement, grayscale conversion
                - Retinal: Gaussian blur denoising, color normalization
                
                **Explainability**: Gradient-weighted Class Activation Mapping (Grad-CAM)
            """)


if __name__ == "__main__":
    main()
