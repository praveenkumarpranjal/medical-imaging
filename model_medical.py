"""
Medical Imaging Model Architectures

Provides CNN architectures for medical image classification using transfer learning.
Supports ResNet50, EfficientNet, and custom CNN architectures with Grad-CAM explainability.

Features:
- Multiple pretrained backbone options
- Data augmentation layers
- Fine-tuning utilities
- Grad-CAM visualization for model interpretability
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, 
    EfficientNetB0, 
    EfficientNetB3,
    MobileNetV3Large
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau, 
    TensorBoard,
    Callback
)


# ─────────────────────────────────────────────────────────────────────────────
# GPU CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def setup_gpu():
    """Configure GPU settings for optimal performance."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🚀 GPU acceleration enabled: {len(physical_devices)} GPU(s)")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("⚠️ Running on CPU - training will be slower")
        return False

GPU_AVAILABLE = setup_gpu()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    img_size: Tuple[int, int] = (224, 224)
    img_channels: int = 3
    num_classes: int = 2
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    model_type: str = 'resnet50'
    freeze_base: bool = True
    dropout_rate: float = 0.5
    l2_regularization: float = 0.01
    model_dir: Path = Path('./models')
    log_dir: Path = Path('./logs')
    early_stop_patience: int = 10
    reduce_lr_patience: int = 5
    
    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (*self.img_size, self.img_channels)


# Default configuration
Config = ModelConfig()


# ─────────────────────────────────────────────────────────────────────────────
# BACKBONE MODELS
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE_REGISTRY = {
    'resnet50': {
        'class': ResNet50,
        'name': 'ResNet50',
        'preprocess': tf.keras.applications.resnet50.preprocess_input
    },
    'efficientnet': {
        'class': EfficientNetB0,
        'name': 'EfficientNetB0',
        'preprocess': tf.keras.applications.efficientnet.preprocess_input
    },
    'efficientnet_b3': {
        'class': EfficientNetB3,
        'name': 'EfficientNetB3',
        'preprocess': tf.keras.applications.efficientnet.preprocess_input
    },
    'mobilenet': {
        'class': MobileNetV3Large,
        'name': 'MobileNetV3Large',
        'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input
    }
}


def get_backbone(
    model_type: str, 
    input_shape: Tuple[int, int, int],
    freeze: bool = True
) -> keras.Model:
    """
    Get a pretrained backbone model.
    
    Args:
        model_type: One of 'resnet50', 'efficientnet', 'efficientnet_b3', 'mobilenet'
        input_shape: Input tensor shape (height, width, channels)
        freeze: Whether to freeze backbone weights
    
    Returns:
        Pretrained backbone model
    """
    if model_type not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    
    backbone_info = BACKBONE_REGISTRY[model_type]
    
    backbone = backbone_info['class'](
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    backbone.trainable = not freeze
    
    print(f"✓ Loaded {backbone_info['name']} backbone")
    print(f"  Layers: {len(backbone.layers)} | Frozen: {freeze}")
    
    return backbone


# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def create_augmentation_layers(
    augmentation_strength: str = 'medium'
) -> List[layers.Layer]:
    """
    Create data augmentation layers.
    
    Args:
        augmentation_strength: 'light', 'medium', or 'strong'
    
    Returns:
        List of Keras augmentation layers
    """
    augmentation_configs = {
        'light': {
            'rotation': 0.05,
            'zoom': 0.05,
            'contrast': 0.05,
            'brightness': 0.05
        },
        'medium': {
            'rotation': 0.1,
            'zoom': 0.1,
            'contrast': 0.1,
            'brightness': 0.1
        },
        'strong': {
            'rotation': 0.2,
            'zoom': 0.2,
            'contrast': 0.2,
            'brightness': 0.2
        }
    }
    
    config = augmentation_configs.get(augmentation_strength, augmentation_configs['medium'])
    
    return [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(config['rotation']),
        layers.RandomZoom(config['zoom']),
        layers.RandomContrast(config['contrast']),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

def create_model(
    model_type: str = 'resnet50',
    num_classes: int = 2,
    freeze_base: bool = True,
    dropout_rate: float = 0.5,
    l2_reg: float = 0.01,
    learning_rate: float = 1e-4,
    use_augmentation: bool = True,
    augmentation_strength: str = 'medium'
) -> keras.Model:
    """
    Create a transfer learning model for medical image classification.
    
    Args:
        model_type: Backbone architecture
        num_classes: Number of output classes
        freeze_base: Whether to freeze backbone weights
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization strength
        learning_rate: Initial learning rate
        use_augmentation: Whether to include data augmentation
        augmentation_strength: 'light', 'medium', or 'strong'
    
    Returns:
        Compiled Keras model
    """
    input_shape = (224, 224, 3)
    inputs = keras.Input(shape=input_shape)
    
    x = inputs
    
    # Add augmentation layers (only active during training)
    if use_augmentation:
        for aug_layer in create_augmentation_layers(augmentation_strength):
            x = aug_layer(x)
    
    # Get pretrained backbone
    backbone = get_backbone(model_type, input_shape, freeze=freeze_base)
    x = backbone(x, training=False)
    
    # Classification head
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        256, 
        activation='relu', 
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        128, 
        activation='relu', 
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    
    # Output layer
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
        metrics = [
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    model = keras.Model(
        inputs=inputs, 
        outputs=outputs, 
        name=f'medical_diagnosis_{model_type}'
    )
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model


def create_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    learning_rate: float = 1e-4
) -> keras.Model:
    """
    Create a custom CNN architecture (no transfer learning).
    
    Useful for smaller datasets or when pretrained weights aren't beneficial.
    
    Args:
        input_shape: Input tensor shape
        num_classes: Number of output classes
        learning_rate: Initial learning rate
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Output
        layers.Dense(
            1 if num_classes == 2 else num_classes,
            activation='sigmoid' if num_classes == 2 else 'softmax'
        )
    ], name='custom_cnn')
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


# ─────────────────────────────────────────────────────────────────────────────
# FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

def unfreeze_model(
    model: keras.Model, 
    num_layers_to_unfreeze: int = 20,
    learning_rate: float = 1e-5
) -> keras.Model:
    """
    Unfreeze layers of a pretrained model for fine-tuning.
    
    Args:
        model: Model with frozen backbone
        num_layers_to_unfreeze: Number of layers from the end to unfreeze
        learning_rate: Learning rate for fine-tuning (typically lower)
    
    Returns:
        Model with partially unfrozen backbone
    """
    # Find the base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break
    
    if base_model is None:
        print("⚠️ No nested model found to unfreeze")
        return model
    
    # Unfreeze specified layers
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"✓ Unfroze {trainable_count} layers for fine-tuning")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=model.loss,
        metrics=model.metrics
    )
    
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class TrainingLogger(Callback):
    """Custom callback for enhanced training logs."""
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        for key, value in logs.items():
            print(f"   {key}: {value:.4f}")


def get_callbacks(
    model_name: str = 'medical_model',
    model_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    early_stop_patience: int = 10,
    reduce_lr_patience: int = 5
) -> List[Callback]:
    """
    Get training callbacks.
    
    Args:
        model_name: Name prefix for saved models
        model_dir: Directory for model checkpoints
        log_dir: Directory for TensorBoard logs
        early_stop_patience: Epochs before early stopping
        reduce_lr_patience: Epochs before learning rate reduction
    
    Returns:
        List of Keras callbacks
    """
    model_dir = model_dir or Config.model_dir
    log_dir = log_dir or Config.log_dir
    
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            filepath=model_dir / f'{model_name}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        EarlyStopping(
            monitor='val_loss',
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ),
        
        TensorBoard(
            log_dir=log_dir / f'{model_name}_{timestamp}',
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        ),
        
        TrainingLogger()
    ]
    
    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def get_gradcam_heatmap(
    model: keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Generate Grad-CAM heatmap for model interpretability.
    
    Grad-CAM (Gradient-weighted Class Activation Mapping) highlights
    regions that were important for the model's prediction.
    
    Args:
        model: Trained Keras model
        img_array: Input image array (batch size 1)
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Class index for which to generate heatmap
    
    Returns:
        Heatmap array normalized to [0, 1]
    """
    # Auto-detect last conv layer if not specified
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
            if isinstance(layer, keras.Model):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, keras.layers.Conv2D):
                        last_conv_layer_name = sublayer.name
                        break
                if last_conv_layer_name:
                    break
    
    if last_conv_layer_name is None:
        print("⚠️ No convolutional layer found for Grad-CAM")
        return None
    
    try:
        # Create gradient model
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"⚠️ Grad-CAM generation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def print_model_summary(model: keras.Model) -> None:
    """Print a formatted model summary."""
    print("\n" + "═" * 60)
    print("  MODEL ARCHITECTURE")
    print("═" * 60)
    model.summary()
    
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print("\n" + "═" * 60)
    print(f"  Total Parameters:         {total_params:,}")
    print(f"  Trainable Parameters:     {trainable_params:,}")
    print(f"  Non-trainable Parameters: {non_trainable_params:,}")
    print("═" * 60 + "\n")


def count_trainable_params(model: keras.Model) -> Tuple[int, int]:
    """Count trainable and non-trainable parameters."""
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    return trainable, non_trainable


# ─────────────────────────────────────────────────────────────────────────────
# MAIN (FOR TESTING)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔬 Testing Model Creation...\n")
    
    # Test different backbones
    for model_type in ['resnet50', 'efficientnet', 'mobilenet']:
        print(f"\n{'─' * 40}")
        print(f"Testing {model_type}...")
        print(f"{'─' * 40}")
        
        model = create_model(
            model_type=model_type,
            num_classes=2,
            freeze_base=True
        )
        
        trainable, non_trainable = count_trainable_params(model)
        print(f"  Trainable: {trainable:,} | Non-trainable: {non_trainable:,}")
        
        # Test inference
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        print(f"  Output shape: {output.shape}")
        print(f"  Sample prediction: {output[0][0]:.4f}")
    
    # Test custom CNN
    print(f"\n{'─' * 40}")
    print("Testing Custom CNN...")
    print(f"{'─' * 40}")
    
    custom_model = create_custom_cnn(num_classes=2)
    custom_model.summary()
    
    print("\n✅ All model tests passed!")
