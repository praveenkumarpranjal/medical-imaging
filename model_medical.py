import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import numpy as np
from pathlib import Path
import datetime

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU acceleration enabled: {physical_devices}")
else:
    print("Running on CPU")


class Config:
    IMG_SIZE = (224, 224)
    IMG_CHANNELS = 3
    NUM_CLASSES = 2
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    MODEL_TYPE = 'resnet50'
    FREEZE_BASE = True
    DROPOUT_RATE = 0.5
    MODEL_DIR = Path('./models')
    LOG_DIR = Path('./logs')
    EARLY_STOP_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5


def create_model(model_type='resnet50', num_classes=2, freeze_base=True):
    input_shape = (*Config.IMG_SIZE, Config.IMG_CHANNELS)
    
    inputs = keras.Input(shape=input_shape)
    
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    
    if model_type == 'resnet50':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        print("Using ResNet50 as base model")
    
    elif model_type == 'efficientnet':
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        print("Using EfficientNetB0 as base model")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if freeze_base:
        base_model.trainable = False
        print(f"Base model frozen: {len(base_model.layers)} layers")
    else:
        base_model.trainable = True
        print(f"Base model trainable: {len(base_model.layers)} layers")
    
    x = base_model(x, training=False)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(Config.DROPOUT_RATE)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(Config.DROPOUT_RATE / 2)(x)
    
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', keras.metrics.AUC(name='auc'), 
                   keras.metrics.Precision(name='precision'),
                   keras.metrics.Recall(name='recall')]
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'medical_diagnosis_{model_type}')
    
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss=loss,
        metrics=metrics
    )
    
    return model


def unfreeze_model(model, num_layers_to_unfreeze=20):
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break
    
    if base_model is None:
        print("No base model found to unfreeze")
        return model
    
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    print(f"Unfroze last {num_layers_to_unfreeze} layers of base model")
    
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE / 10),
        loss=model.loss,
        metrics=model.metrics
    )
    
    return model


def get_callbacks(model_name='medical_model'):
    Config.MODEL_DIR.mkdir(exist_ok=True)
    Config.LOG_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        ModelCheckpoint(
            filepath=Config.MODEL_DIR / f'{model_name}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        EarlyStopping(
            monitor='val_loss',
            patience=Config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        TensorBoard(
            log_dir=Config.LOG_DIR / f'{model_name}_{timestamp}',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    return callbacks


def create_custom_cnn(input_shape=(224, 224, 3), num_classes=2):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(1 if num_classes == 2 else num_classes, 
                     activation='sigmoid' if num_classes == 2 else 'softmax')
    ], name='custom_cnn')
    
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model


def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None, pred_index=None):
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
        print("No convolutional layer found for Grad-CAM")
        return None
    
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def print_model_summary(model):
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    
    total_params = model.count_params()
    print("\n" + "="*60)
    print(f"Total Parameters: {total_params:,}")
    
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Testing model creation...")
    
    model = create_model(model_type='resnet50', num_classes=2, freeze_base=True)
    print_model_summary(model)
    
    dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"Model output shape: {output.shape}")
    print(f"Sample prediction: {output[0]}")
    
    print("\n✓ Model creation successful!")
