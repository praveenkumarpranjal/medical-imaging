"""
Medical Imaging Diagnosis - Training Pipeline

Trains deep learning models for medical image classification.
Supports TB detection and Diabetic Retinopathy detection.

Usage:
    python train_medical.py --dataset tb --model resnet50
    python train_medical.py --dataset retinopathy --model efficientnet --no-finetune
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

import tensorflow as tf
from tensorflow import keras

from model_medical import (
    create_model, 
    create_custom_cnn, 
    get_callbacks,
    unfreeze_model, 
    print_model_summary, 
    ModelConfig
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    
    # Dataset settings
    dataset_type: str = 'tb'
    data_dir: Path = Path('./medical_imaging_data/processed')
    
    # Model settings
    use_transfer_learning: bool = True
    model_type: str = 'resnet50'
    freeze_base: bool = True
    
    # Training settings
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    
    # Fine-tuning settings
    fine_tune: bool = True
    fine_tune_epochs: int = 20
    fine_tune_learning_rate: float = 1e-5
    fine_tune_layers: int = 30
    
    # Output settings
    results_dir: Path = Path('./results')
    models_dir: Path = Path('./models')
    
    # Class names for each dataset
    class_names: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = {
                'tb': ['Normal', 'TB Positive'],
                'retinopathy': ['No DR', 'DR Present']
            }
    
    def get_class_names(self) -> List[str]:
        return self.class_names.get(self.dataset_type, ['Class 0', 'Class 1'])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to serializable dictionary."""
        d = asdict(self)
        d['data_dir'] = str(self.data_dir)
        d['results_dir'] = str(self.results_dir)
        d['models_dir'] = str(self.models_dir)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Load preprocessed training data.
    
    Args:
        data_dir: Path to processed data directory
    
    Returns:
        Tuple of (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    print("\n" + "═" * 60)
    print("  LOADING DATA")
    print("═" * 60)
    
    try:
        X_train = np.load(data_dir / 'train' / 'images.npy')
        y_train = np.load(data_dir / 'train' / 'labels.npy')
        print(f"  ✓ Training:   {X_train.shape[0]:,} samples")
        
        X_val = np.load(data_dir / 'val' / 'images.npy')
        y_val = np.load(data_dir / 'val' / 'labels.npy')
        print(f"  ✓ Validation: {X_val.shape[0]:,} samples")
        
        X_test = np.load(data_dir / 'test' / 'images.npy')
        y_test = np.load(data_dir / 'test' / 'labels.npy')
        print(f"  ✓ Test:       {X_test.shape[0]:,} samples")
        
        # Class distribution
        print("\n  Class Distribution (Training):")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            pct = count / len(y_train) * 100
            print(f"    Class {int(cls)}: {count:,} samples ({pct:.1f}%)")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    except FileNotFoundError as e:
        print(f"\n  ❌ Error: Data files not found in {data_dir}")
        print("  Please run data_prep_medical.py first!")
        raise e


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def setup_plot_style():
    """Configure matplotlib style for consistent, modern plots."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_training_history(history: keras.callbacks.History, save_path: Path) -> None:
    """
    Generate and save training history plots.
    
    Args:
        history: Keras training history object
        save_path: Where to save the figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Accuracy plot
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2, color='#667eea')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='#38ef7d')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2, color='#667eea')
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='#38ef7d')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC plot (if available)
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='Train', linewidth=2, color='#667eea')
        axes[1, 0].plot(history.history['val_auc'], label='Validation', linewidth=2, color='#38ef7d')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'AUC not tracked', ha='center', va='center',
                       transform=axes[1, 0].transAxes, fontsize=12, alpha=0.5)
        axes[1, 0].set_title('Model AUC')
    
    # Learning rate plot (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='#f093fb')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'LR schedule not tracked', ha='center', va='center',
                       transform=axes[1, 1].transAxes, fontsize=12, alpha=0.5)
        axes[1, 1].set_title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0e1117')
    print(f"  ✓ Training history saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str],
    save_path: Path
) -> None:
    """Generate and save confusion matrix plot."""
    setup_plot_style()
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names, 
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0e1117')
    print(f"  ✓ Confusion matrix saved")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray, 
    y_pred_probs: np.ndarray,
    save_path: Path
) -> float:
    """Generate and save ROC curve plot."""
    setup_plot_style()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='#667eea', lw=2.5, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#666', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.fill_between(fpr, tpr, alpha=0.2, color='#667eea')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0e1117')
    print(f"  ✓ ROC curve saved (AUC: {roc_auc:.4f})")
    plt.close()
    
    return roc_auc


def visualize_predictions(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    save_path: Path,
    num_samples: int = 16
) -> None:
    """Generate grid of sample predictions with confidence."""
    setup_plot_style()
    
    indices = np.random.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)
    
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx, ax in zip(indices, axes):
        img = X_test[idx]
        true_label = int(y_test[idx])
        
        pred_prob = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
        pred_label = int(pred_prob > 0.5)
        
        ax.imshow(img)
        
        is_correct = pred_label == true_label
        color = '#38ef7d' if is_correct else '#eb3349'
        
        ax.set_title(
            f"True: {class_names[true_label]}\n"
            f"Pred: {class_names[pred_label]} ({pred_prob:.2%})",
            color=color, 
            fontsize=10,
            fontweight='bold' if not is_correct else 'normal'
        )
        ax.axis('off')
    
    # Hide unused subplots
    for ax in axes[len(indices):]:
        ax.axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0e1117')
    print(f"  ✓ Sample predictions saved")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    save_dir: Path,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels
        class_names: Names for each class
        save_dir: Directory to save results
        batch_size: Batch size for prediction
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "═" * 60)
    print("  MODEL EVALUATION")
    print("═" * 60)
    
    print("\n  Making predictions on test set...")
    y_pred_probs = model.predict(X_test, batch_size=batch_size, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Classification report
    print("\n  Classification Report:")
    print("  " + "─" * 56)
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    for line in report.split('\n'):
        print(f"  {line}")
    
    with open(save_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    # Generate plots
    print("\n  Generating evaluation plots...")
    plot_confusion_matrix(y_test, y_pred, class_names, save_dir / 'confusion_matrix.png')
    roc_auc = plot_roc_curve(y_test, y_pred_probs, save_dir / 'roc_curve.png')
    visualize_predictions(model, X_test, y_test, class_names, save_dir / 'sample_predictions.png')
    
    # Compile metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'test_samples': int(len(y_test)),
        'positive_samples': int(y_test.sum()),
        'negative_samples': int(len(y_test) - y_test.sum())
    }
    
    with open(save_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print summary
    print("\n" + "═" * 60)
    print("  FINAL METRICS")
    print("═" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print("═" * 60)
    
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_model(config: TrainingConfig) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Execute the complete training pipeline.
    
    Args:
        config: Training configuration
    
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    # Header
    print("\n" + "═" * 60)
    print("  MEDICAL IMAGE DIAGNOSIS - TRAINING PIPELINE")
    print("═" * 60)
    print(f"  Dataset:    {config.dataset_type.upper()}")
    print(f"  Model:      {config.model_type.upper() if config.use_transfer_learning else 'Custom CNN'}")
    print(f"  Epochs:     {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Fine-tune:  {'Yes' if config.fine_tune else 'No'}")
    print("═" * 60)
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model_type if config.use_transfer_learning else 'custom_cnn'
    results_dir = config.results_dir / f"{config.dataset_type}_{model_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Results will be saved to: {results_dir}")
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(config.data_dir)
    
    class_names = config.get_class_names()
    
    # Create model
    print("\n" + "═" * 60)
    print("  CREATING MODEL")
    print("═" * 60)
    
    if config.use_transfer_learning:
        model = create_model(
            model_type=config.model_type,
            num_classes=2,
            freeze_base=config.freeze_base,
            learning_rate=config.learning_rate
        )
    else:
        model = create_custom_cnn(num_classes=2, learning_rate=config.learning_rate)
    
    print_model_summary(model)
    
    # Get callbacks
    callback_model_name = f"{config.dataset_type}_{model_name}"
    callbacks = get_callbacks(
        model_name=callback_model_name,
        model_dir=config.models_dir
    )
    
    # Phase 1: Initial training
    print("\n" + "═" * 60)
    print("  PHASE 1: INITIAL TRAINING")
    print("═" * 60)
    
    history = model.fit(
        X_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    plot_training_history(history, results_dir / 'training_history_phase1.png')
    
    # Phase 2: Fine-tuning (optional)
    if config.fine_tune and config.use_transfer_learning:
        print("\n" + "═" * 60)
        print("  PHASE 2: FINE-TUNING")
        print("═" * 60)
        
        model = unfreeze_model(
            model, 
            num_layers_to_unfreeze=config.fine_tune_layers,
            learning_rate=config.fine_tune_learning_rate
        )
        
        history_finetune = model.fit(
            X_train, y_train,
            batch_size=config.batch_size,
            epochs=config.fine_tune_epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        plot_training_history(history_finetune, results_dir / 'training_history_phase2.png')
    
    # Load best model checkpoint
    best_model_path = config.models_dir / f"{callback_model_name}_best.keras"
    if best_model_path.exists():
        print(f"\n  Loading best model from checkpoint...")
        model = tf.keras.models.load_model(best_model_path)
    
    # Evaluate
    metrics = evaluate_model(
        model, X_test, y_test, class_names, results_dir,
        batch_size=config.batch_size
    )
    
    # Save final model
    final_model_path = results_dir / 'final_model.keras'
    model.save(final_model_path)
    print(f"\n  ✓ Final model saved to {final_model_path}")
    
    # Save training configuration
    config_dict = config.to_dict()
    config_dict['timestamp'] = timestamp
    config_dict['final_metrics'] = metrics
    config_dict['class_names_used'] = class_names
    
    with open(results_dir / 'training_config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Final summary
    print("\n" + "═" * 60)
    print("  ✓ TRAINING COMPLETE!")
    print(f"  Results saved to: {results_dir}")
    print("═" * 60 + "\n")
    
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train medical image diagnosis models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_medical.py --dataset tb
  python train_medical.py --dataset retinopathy --model efficientnet
  python train_medical.py --dataset tb --model custom --epochs 100
  python train_medical.py --dataset tb --no-finetune --batch-size 64
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='tb',
        choices=['tb', 'retinopathy'],
        help='Dataset to train on (default: tb)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='resnet50',
        choices=['resnet50', 'efficientnet', 'efficientnet_b3', 'mobilenet', 'custom'],
        help='Model architecture (default: resnet50)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=1e-4,
        help='Initial learning rate (default: 1e-4)'
    )
    
    parser.add_argument(
        '--no-finetune',
        action='store_true',
        help='Skip fine-tuning phase'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./medical_imaging_data/processed',
        help='Path to processed data directory'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    config = TrainingConfig(
        dataset_type=args.dataset,
        use_transfer_learning=(args.model != 'custom'),
        model_type=args.model if args.model != 'custom' else 'resnet50',
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fine_tune=not args.no_finetune,
        data_dir=Path(args.data_dir)
    )
    
    try:
        model, metrics = train_model(config)
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
