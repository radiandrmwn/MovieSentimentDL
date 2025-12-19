#!/usr/bin/env python
"""
Generate visualizations for IMDb model results
Creates individual plots for each model matching Amazon dataset structure
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import tensorflow as tf

sns.set_style("whitegrid")

def load_results(result_dir):
    """Load results from pickle file"""
    with open(os.path.join(result_dir, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    return results

class AttentionLayer(tf.keras.layers.Layer):
    """Attention mechanism layer"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], input_shape[-1]),
                                initializer='glorot_uniform',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[-1],),
                                initializer='zeros',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

def load_model_and_data(result_dir):
    """Load saved model and tokenizer"""
    import tensorflow as tf
    from tensorflow import keras

    # Load model with custom objects
    model = keras.models.load_model(
        os.path.join(result_dir, 'best_model.keras'),
        custom_objects={'AttentionLayer': AttentionLayer}
    )

    with open(os.path.join(result_dir, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

def plot_individual_model_training(model_name, history, output_path):
    """Create individual training plot for a model"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} - Training History (IMDb Dataset, 15 Epochs)',
                 fontsize=14, fontweight='bold')

    epochs = range(1, len(history['accuracy']) + 1)

    # Plot Accuracy
    ax = axes[0]
    ax.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2.5)
    ax.plot(epochs, history['val_accuracy'], 'r--', label='Validation Accuracy', linewidth=2.5)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Model Accuracy', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.6, 1.0])

    # Add best validation accuracy annotation
    best_val_acc = max(history['val_accuracy'])
    best_epoch = history['val_accuracy'].index(best_val_acc) + 1
    ax.annotate(f'Best Val: {best_val_acc:.2%}\nEpoch {best_epoch}',
                xy=(best_epoch, best_val_acc),
                xytext=(best_epoch + 1, best_val_acc - 0.05),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))

    # Plot Loss
    ax = axes[1]
    ax.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2.5)
    ax.plot(epochs, history['val_loss'], 'r--', label='Validation Loss', linewidth=2.5)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Model Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, output_path):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='black',
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Confusion Matrix (IMDb Test Set)',
                fontsize=14, fontweight='bold')

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Add metrics text
    metrics_text = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return accuracy, precision, recall, f1

def plot_metrics_comparison(all_metrics, output_path):
    """Create comparison plot of all metrics"""
    models = list(all_metrics.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    # Prepare data
    data = {metric: [all_metrics[model][metric] for model in models] for metric in metrics}

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(models))
    width = 0.2

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, data[metric], width, label=metric,
                     color=color, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

    ax.set_xlabel('Models', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('IMDb Dataset - Model Performance Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0.85, 0.92])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_combined_training_history(results_dict, output_path):
    """Plot combined training history for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('IMDb Dataset - All Models Training History (15 Epochs)',
                 fontsize=16, fontweight='bold')

    models = ['LSTM', 'Bi-LSTM', 'LSTM+Attention', 'GRU']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Plot Accuracy
    ax = axes[0, 0]
    for i, (model_name, result) in enumerate(zip(models, results_dict.values())):
        epochs = range(1, len(result['history']['accuracy']) + 1)
        ax.plot(epochs, result['history']['accuracy'],
                label=f'{model_name} (Train)', color=colors[i], linestyle='-', linewidth=2)
        ax.plot(epochs, result['history']['val_accuracy'],
                label=f'{model_name} (Val)', color=colors[i], linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot Loss
    ax = axes[0, 1]
    for i, (model_name, result) in enumerate(zip(models, results_dict.values())):
        epochs = range(1, len(result['history']['loss']) + 1)
        ax.plot(epochs, result['history']['loss'],
                label=f'{model_name} (Train)', color=colors[i], linestyle='-', linewidth=2)
        ax.plot(epochs, result['history']['val_loss'],
                label=f'{model_name} (Val)', color=colors[i], linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Best Validation Accuracy Comparison
    ax = axes[1, 0]
    best_val_accs = [max(r['history']['val_accuracy']) for r in results_dict.values()]
    bars = ax.bar(models, best_val_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Best Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0.85, 0.92])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, best_val_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Test Accuracy Comparison
    ax = axes[1, 1]
    test_accs = [r['test_accuracy'] for r in results_dict.values()]
    bars = ax.bar(models, test_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0.85, 0.92])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, test_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def create_metrics_summary_csv(results_dict, all_metrics, output_path):
    """Create comprehensive CSV summary"""
    models = ['LSTM', 'Bi-LSTM', 'LSTM+Attention', 'GRU']

    summary_data = []
    for model_name, result in zip(models, results_dict.values()):
        metrics = all_metrics[model_name]
        best_val_acc = max(result['history']['val_accuracy'])
        best_val_epoch = result['history']['val_accuracy'].index(best_val_acc) + 1
        final_train_acc = result['history']['accuracy'][-1]
        final_val_acc = result['history']['val_accuracy'][-1]

        summary_data.append({
            'Model': model_name,
            'Best_Val_Accuracy': f"{best_val_acc:.4f}",
            'Best_Val_Epoch': best_val_epoch,
            'Final_Train_Accuracy': f"{final_train_acc:.4f}",
            'Final_Val_Accuracy': f"{final_val_acc:.4f}",
            'Test_Accuracy': f"{metrics['Accuracy']:.4f}",
            'Test_Precision': f"{metrics['Precision']:.4f}",
            'Test_Recall': f"{metrics['Recall']:.4f}",
            'Test_F1_Score': f"{metrics['F1-Score']:.4f}"
        })

    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return df

def main():
    base_dir = "results/deep_learning/imdb"

    # Model directories
    model_dirs = {
        'LSTM': os.path.join(base_dir, '01_lstm'),
        'Bi-LSTM': os.path.join(base_dir, '02_bilstm'),
        'LSTM+Attention': os.path.join(base_dir, '03_lstm_attention'),
        'GRU': os.path.join(base_dir, '04_gru')
    }

    # Load all results
    print("="*60)
    print("Loading results from all models...")
    print("="*60)
    results_dict = {}
    for model_name, path in model_dirs.items():
        print(f"  Loading {model_name}...")
        results_dict[model_name] = load_results(path)

    # Create visualizations directory
    viz_dir = os.path.join(base_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Generate individual training plots for each model
    print("\n" + "="*60)
    print("Generating individual training plots...")
    print("="*60)
    for model_name, result in results_dict.items():
        safe_name = model_name.replace('+', '_')
        output_file = f"{safe_name}_training.png"
        output_path = os.path.join(viz_dir, output_file)
        plot_individual_model_training(model_name, result['history'], output_path)

    # Load test data for confusion matrices
    print("\n" + "="*60)
    print("Loading test data and generating confusion matrices...")
    print("="*60)
    import pandas as pd
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Load test data
    df = pd.read_parquet('data/processed/imdb_reviews.parquet')
    test_df = df[df['split'] == 'test'].copy()
    y_test = test_df['label'].values

    all_metrics = {}

    for model_name, model_dir in model_dirs.items():
        print(f"  Processing {model_name}...")

        # Load model and tokenizer
        model, tokenizer = load_model_and_data(model_dir)

        # Tokenize and pad test data
        sequences = tokenizer.texts_to_sequences(test_df['text'])
        X_test = pad_sequences(sequences, maxlen=200)

        # Get predictions
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

        # Generate confusion matrix
        safe_name = model_name.replace('+', '_')
        cm_path = os.path.join(viz_dir, f"{safe_name}_confusion_matrix.png")
        accuracy, precision, recall, f1 = plot_confusion_matrix(y_test, y_pred, model_name, cm_path)

        all_metrics[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }

    # Generate metrics comparison plot
    print("\n" + "="*60)
    print("Generating metrics comparison plot...")
    print("="*60)
    plot_metrics_comparison(all_metrics, os.path.join(viz_dir, 'metrics_comparison.png'))

    # Generate combined training history
    print("\n" + "="*60)
    print("Generating combined training history plot...")
    print("="*60)
    plot_combined_training_history(results_dict, os.path.join(viz_dir, 'combined_training_history.png'))

    # Create comprehensive metrics summary CSV
    print("\n" + "="*60)
    print("Creating comprehensive metrics summary CSV...")
    print("="*60)
    summary_df = create_metrics_summary_csv(results_dict, all_metrics,
                                           os.path.join(base_dir, 'model_metrics_summary.csv'))

    print("\n" + "="*60)
    print("IMDb Models - Complete Metrics Summary")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)

    print("\n" + "="*60)
    print("All visualizations completed successfully!")
    print("="*60)
    print(f"Results saved in: {base_dir}/")
    print(f"Visualizations saved in: {viz_dir}/")
    print("\nGenerated files:")
    print("  - LSTM_training.png")
    print("  - Bi-LSTM_training.png")
    print("  - LSTM_Attention_training.png")
    print("  - GRU_training.png")
    print("  - LSTM_confusion_matrix.png")
    print("  - Bi-LSTM_confusion_matrix.png")
    print("  - LSTM_Attention_confusion_matrix.png")
    print("  - GRU_confusion_matrix.png")
    print("  - metrics_comparison.png")
    print("  - combined_training_history.png")
    print("  - model_metrics_summary.csv")
    print("="*60)

if __name__ == "__main__":
    main()
