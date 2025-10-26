#!/usr/bin/env python
"""
evaluate_models.py
Calculate detailed metrics (F1, Precision, Recall) from saved models

Usage:
  python evaluate_models.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

# Define custom AttentionLayer for loading LSTM+Attention model
class AttentionLayer(Layer):
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

    def get_config(self):
        return super(AttentionLayer, self).get_config()

def evaluate_model(model_dir, test_data_path='data/deep_learning_data/movies_reviews_200k.parquet'):
    """Evaluate a single model and return detailed metrics"""

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_dir}")
    print(f"{'='*60}")

    # Load test data
    df = pd.read_parquet(test_data_path)

    # Split (same as training: 80-20)
    from sklearn.model_selection import train_test_split
    _, df_test = train_test_split(df, test_size=0.2, random_state=42)

    X_test = df_test['text'].values
    y_test = df_test['label'].values

    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Prepare sequences
    max_length = 200  # Same as training
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

    # Load model (with custom objects for LSTM+Attention)
    model_path = os.path.join(model_dir, 'best_model.keras')
    model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})

    # Make predictions
    y_pred_prob = model.predict(X_test_pad, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print results
    print(f"\nTest Accuracy:  {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def main():
    # Model directories
    models = {
        'LSTM Baseline': 'results/deep_learning/01_lstm',
        'Bi-LSTM': 'results/deep_learning/02_bilstm',
        'LSTM + Attention': 'results/deep_learning/03_lstm_attention',
        'GRU': 'results/deep_learning/04_gru'
    }

    # Evaluate all models
    all_results = {}
    for name, model_dir in models.items():
        if os.path.exists(model_dir):
            try:
                results = evaluate_model(model_dir)
                all_results[name] = results
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        else:
            print(f"\nSkipping {name} - directory not found")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    summary_data = []
    for name, results in all_results.items():
        summary_data.append({
            'Model': name,
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1 Score': f"{results['f1']:.4f}"
        })

    df_summary = pd.DataFrame(summary_data)
    print("\n", df_summary.to_string(index=False))

    # Save summary
    df_summary.to_csv('results/deep_learning/model_metrics_summary.csv', index=False)
    print("\n✓ Summary saved to: results/deep_learning/model_metrics_summary.csv")

    # Generate visualizations
    if all_results:
        generate_visualizations(all_results, df_summary)

def generate_visualizations(all_results, df_summary):
    """Generate comprehensive visualization with metrics"""

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Prepare data
    models = list(all_results.keys())
    metrics_numeric = df_summary.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        metrics_numeric[col] = metrics_numeric[col].astype(float)

    # Create comprehensive figure with better layout
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35, top=0.93, bottom=0.08, left=0.05, right=0.98)

    # 1. Bar chart comparing all metrics (spanning top row)
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(models))
    width = 0.2

    ax1.bar(x - 1.5*width, metrics_numeric['Accuracy'], width, label='Accuracy', color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax1.bar(x - 0.5*width, metrics_numeric['Precision'], width, label='Precision', color='#A23B72', edgecolor='black', linewidth=0.5)
    ax1.bar(x + 0.5*width, metrics_numeric['Recall'], width, label='Recall', color='#F18F01', edgecolor='black', linewidth=0.5)
    ax1.bar(x + 1.5*width, metrics_numeric['F1 Score'], width, label='F1 Score', color='#C73E1D', edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax1.set_title('Model Performance Comparison - All Metrics', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax1.set_ylim([0.90, 0.98])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add value labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.4f', fontsize=8, padding=3)

    # 2-5. Confusion matrices for each model (bottom row)
    for idx, (name, results) in enumerate(all_results.items()):
        ax = fig.add_subplot(gs[1, idx])

        cm = results['confusion_matrix']

        # Normalize for better color scaling
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Neg', 'Pos'],
                   yticklabels=['Neg', 'Pos'],
                   cbar=True, cbar_kws={'label': 'Count', 'shrink': 0.8},
                   linewidths=1, linecolor='gray')

        # Cleaner title with metrics
        title = f'{name}\n'
        title += f'Acc: {results["accuracy"]:.4f} | Prec: {results["precision"]:.4f}\n'
        title += f'Rec: {results["recall"]:.4f} | F1: {results["f1"]:.4f}'
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=10)
        ax.set_ylabel('True', fontsize=9, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=9)

    plt.suptitle('Model Evaluation Results - 200K Movie Reviews Dataset',
                fontsize=17, fontweight='bold', y=0.98)

    # Save figure
    output_path = 'results/deep_learning/model_comparison_with_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    # Create individual metric comparison chart
    fig2, ax = plt.subplots(figsize=(12, 8))

    metrics_df = metrics_numeric.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']]
    metrics_df.plot(kind='bar', ax=ax, width=0.8, colormap='Set2')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Detailed Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Metrics', loc='lower right')
    ax.set_ylim([0.88, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=8)

    plt.tight_layout()
    output_path2 = 'results/deep_learning/metrics_comparison.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics chart saved to: {output_path2}")

    plt.close('all')

if __name__ == '__main__':
    main()
