import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
)
import json
import argparse
from datetime import datetime

from model_medical import (
    create_model, create_custom_cnn, get_callbacks, 
    unfreeze_model, print_model_summary, Config as ModelConfig
)


class TrainingConfig:
    DATA_DIR = Path('./medical_imaging_data/processed')
    USE_TRANSFER_LEARNING = True
    MODEL_TYPE = 'resnet50'
    FREEZE_BASE = True
    FINE_TUNE = True
    FINE_TUNE_EPOCHS = 20
    DATASET_TYPE = 'tb'
    RESULTS_DIR = Path('./results')
    CLASS_NAMES = {
        'tb': ['Normal', 'TB Positive'],
        'retinopathy': ['No DR', 'DR Present']
    }


def load_data(data_dir):
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    
    try:
        X_train = np.load(data_dir / 'train' / 'images.npy')
        y_train = np.load(data_dir / 'train' / 'labels.npy')
        print(f"✓ Training data loaded: {X_train.shape}")
        
        X_val = np.load(data_dir / 'val' / 'images.npy')
        y_val = np.load(data_dir / 'val' / 'labels.npy')
        print(f"✓ Validation data loaded: {X_val.shape}")
        
        X_test = np.load(data_dir / 'test' / 'images.npy')
        y_test = np.load(data_dir / 'test' / 'labels.npy')
        print(f"✓ Test data loaded: {X_test.shape}")
        
        print(f"\nTraining set class distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    except FileNotFoundError as e:
        print(f"Error: Data files not found in {data_dir}")
        print("Please run data_prep_medical.py first!")
        raise e


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='Train AUC')
        axes[1, 0].plot(history.history['val_auc'], label='Val AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to {save_path}")
    plt.close()


def evaluate_model(model, X_test, y_test, class_names, save_dir):
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    print("Making predictions on test set...")
    y_pred_probs = model.predict(X_test, batch_size=ModelConfig.BATCH_SIZE)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)
    
    with open(save_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved")
    plt.close()
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved (AUC: {roc_auc:.4f})")
    plt.close()
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist()
    }
    
    with open(save_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n" + "="*60)
    print("FINAL METRICS")
    print("="*60)
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric.upper()}: {value:.4f}")
    print("="*60)
    
    return metrics


def visualize_predictions(model, X_test, y_test, class_names, save_dir, num_samples=16):
    indices = np.random.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, ax in zip(indices, axes):
        img = X_test[idx]
        true_label = y_test[idx]
        
        pred_prob = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
        pred_label = int(pred_prob > 0.5)
        
        ax.imshow(img)
        
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(
            f"True: {class_names[true_label]}\n"
            f"Pred: {class_names[pred_label]} ({pred_prob:.2f})",
            color=color, fontsize=10
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_predictions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Sample predictions saved")
    plt.close()


def train_model(args):
    print("\n" + "="*60)
    print("MEDICAL IMAGE DIAGNOSIS - TRAINING")
    print(f"Dataset: {TrainingConfig.DATASET_TYPE.upper()}")
    print(f"Model: {TrainingConfig.MODEL_TYPE.upper() if TrainingConfig.USE_TRANSFER_LEARNING else 'Custom CNN'}")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = TrainingConfig.RESULTS_DIR / f"{TrainingConfig.DATASET_TYPE}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {results_dir}")
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(TrainingConfig.DATA_DIR)
    
    class_names = TrainingConfig.CLASS_NAMES[TrainingConfig.DATASET_TYPE]
    
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    
    if TrainingConfig.USE_TRANSFER_LEARNING:
        model = create_model(
            model_type=TrainingConfig.MODEL_TYPE,
            num_classes=2,
            freeze_base=TrainingConfig.FREEZE_BASE
        )
    else:
        model = create_custom_cnn(num_classes=2)
    
    print_model_summary(model)
    
    callbacks = get_callbacks(
        model_name=f"{TrainingConfig.DATASET_TYPE}_{TrainingConfig.MODEL_TYPE}"
    )
    
    print("\n" + "="*60)
    print("PHASE 1: Initial Training")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        batch_size=ModelConfig.BATCH_SIZE,
        epochs=ModelConfig.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    plot_training_history(history, results_dir / 'training_history_phase1.png')
    
    if TrainingConfig.FINE_TUNE and TrainingConfig.USE_TRANSFER_LEARNING:
        print("\n" + "="*60)
        print("PHASE 2: Fine-Tuning")
        print("="*60)
        
        model = unfreeze_model(model, num_layers_to_unfreeze=30)
        
        history_finetune = model.fit(
            X_train, y_train,
            batch_size=ModelConfig.BATCH_SIZE,
            epochs=TrainingConfig.FINE_TUNE_EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        plot_training_history(history_finetune, results_dir / 'training_history_phase2.png')
    
    best_model_path = ModelConfig.MODEL_DIR / f"{TrainingConfig.DATASET_TYPE}_{TrainingConfig.MODEL_TYPE}_best.keras"
    if best_model_path.exists():
        print(f"\nLoading best model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
    
    metrics = evaluate_model(model, X_test, y_test, class_names, results_dir)
    
    visualize_predictions(model, X_test, y_test, class_names, results_dir)
    
    final_model_path = results_dir / 'final_model.keras'
    model.save(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    config = {
        'dataset_type': TrainingConfig.DATASET_TYPE,
        'model_type': TrainingConfig.MODEL_TYPE if TrainingConfig.USE_TRANSFER_LEARNING else 'custom_cnn',
        'use_transfer_learning': TrainingConfig.USE_TRANSFER_LEARNING,
        'freeze_base': TrainingConfig.FREEZE_BASE,
        'fine_tune': TrainingConfig.FINE_TUNE,
        'batch_size': ModelConfig.BATCH_SIZE,
        'epochs': ModelConfig.EPOCHS,
        'learning_rate': ModelConfig.LEARNING_RATE,
        'class_names': class_names,
        'timestamp': timestamp,
        'final_metrics': metrics
    }
    
    with open(results_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"All results saved to: {results_dir}")
    print("="*60 + "\n")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Train medical image diagnosis model')
    parser.add_argument('--dataset', type=str, default='tb', 
                        choices=['tb', 'retinopathy'],
                        help='Dataset to use for training')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet', 'custom'],
                        help='Model architecture')
    parser.add_argument('--no-finetune', action='store_true',
                        help='Skip fine-tuning phase')
    
    args = parser.parse_args()
    
    TrainingConfig.DATASET_TYPE = args.dataset
    if args.model == 'custom':
        TrainingConfig.USE_TRANSFER_LEARNING = False
    else:
        TrainingConfig.MODEL_TYPE = args.model
    TrainingConfig.FINE_TUNE = not args.no_finetune
    
    model, metrics = train_model(args)
    
    return model, metrics


if __name__ == "__main__":
    main()
