# Sentiment Analysis on Amazon Reviews using LSTM Networks

## Project Overview

This project reproduces and improves upon "Sentiment Analysis from Movie Reviews Using LSTMs" (Bodapati et al., 2019) using a larger dataset (up to 1.2M Amazon reviews vs 50K IMDB reviews).

**🎯 Achievement:** 93.23% test accuracy on 200K reviews, surpassing the original paper by 4.77%

## Quick Results Summary

| Model | Test Accuracy | Status |
|-------|---------------|---------|
| LSTM (Baseline) | **93.23%** | ✅ Best |
| LSTM + Attention | 93.08% | ✅ |
| GRU | 93.02% | ✅ |
| Bi-LSTM | 92.98% | ✅ |
| Bodapati et al. (2019) | 88.46% | 📄 Reference |

## Dataset

- **Source:** Amazon Customer Reviews (via Kaggle)
- **Size:** 1,200,000 reviews
- **Format:** Binary sentiment (positive/negative)
- **Split:** 80% train, 10% validation, 10% test

## Previous Work (Bodapati et al., 2019)

- **Dataset:** IMDB movie reviews (50K)
- **Method:** LSTM + Word2Vec embeddings
- **Results:** 88.46% accuracy
- **Limitation:** Small dataset size

## Our Approach

### 1. Baseline (Reproduce Previous Work)
- LSTM + Word2Vec on 1.2M reviews
- Same architecture as Bodapati et al.
- Test if larger dataset improves performance

### 2. Proposed Improvements
1. **Bi-directional LSTM (Bi-LSTM)**
   - Reads text forward and backward
   - Captures context from both directions

2. **LSTM + Attention Mechanism**
   - Focuses on important words in reviews
   - Better handles long sequences

3. **GRU (Gated Recurrent Unit)**
   - Simpler architecture than LSTM
   - Faster training, similar performance

## Implementation

### Scripts

1. `src/01_preprocess.py` - Data preprocessing
2. `src/02_lstm_word2vec.py` - Baseline LSTM (Bodapati 2019)
3. `src/03_bilstm.py` - Bi-directional LSTM
4. `src/04_lstm_attention.py` - LSTM with Attention
5. `src/05_gru.py` - GRU model

### Running the Pipeline

```bash
# 1. Preprocess data
python src/01_preprocess.py --input_dir data/raw --output_parquet data/processed/movies_reviews.parquet --drop_neutral

# 2. Train baseline LSTM
python src/02_lstm_word2vec.py --data data/processed/movies_reviews.parquet --output_dir results/lstm_word2vec

# 3. Train improved models
python src/03_bilstm.py --data data/processed/movies_reviews.parquet --output_dir results/bilstm
python src/04_lstm_attention.py --data data/processed/movies_reviews.parquet --output_dir results/lstm_attention
python src/05_gru.py --data data/processed/movies_reviews.parquet --output_dir results/gru
```

## Results

### Deep Learning Project (200K Dataset, 5 Epochs)

All models were trained on a 200K subset of the Amazon reviews dataset for the deep learning project with the following results:

| Model | Train Accuracy | Val Accuracy | Test Accuracy | Training Time |
|-------|----------------|--------------|---------------|---------------|
| **LSTM (Baseline)** | **92.80%** | **93.41%** | **93.23%** ⭐ | ~1 hour |
| LSTM + Attention | 92.67% | 93.40% | 93.08% | ~1 hour |
| GRU | 92.26% | 93.27% | 93.02% | ~1 hour |
| Bi-LSTM | 92.30% | 93.24% | 92.98% | ~1 hour |

**Key Findings:**
- ✅ All models achieved >92.9% test accuracy
- ✅ LSTM baseline performed best (93.23%) despite being the simplest architecture
- ✅ No overfitting observed - training and validation accuracies track closely
- ✅ All models showed consistent improvement across 5 epochs
- ✅ Performance gain of ~4.8% over Bodapati et al. (2019) baseline (88.46%)

**Training Progression:**
- Epoch 1: Models started at ~85-87% training accuracy, ~91% validation accuracy
- Epoch 5: Models converged to ~92-93% on both training and validation sets
- All models showed steady, consistent improvement without overfitting

### Comparison with Bodapati et al. (2019)

| Metric | Bodapati et al. (2019) | Our Implementation |
|--------|------------------------|-------------------|
| Dataset | IMDB (50K) | Amazon Reviews (200K) |
| LSTM Accuracy | 88.46% | 93.23% |
| Improvement | Baseline | **+4.77%** |

### Full Dataset Performance (1.2M Reviews)

Initial testing on full 1.2M dataset with 2 epochs:
- Test Accuracy: **93.98%**
- Training time: ~2 hours per epoch
- Results show further improvement with larger dataset

## Requirements

```bash
pip install tensorflow>=2.13.0 gensim scikit-learn pandas pyarrow tqdm
```

## Paper Structure (IEEE Format)

1. **Introduction**
   - Problem: Sentiment analysis from reviews
   - Motivation: Scale previous work to larger dataset

2. **Related Work**
   - Bodapati et al. (2019): LSTM + Word2Vec (88.46%)
   - Highlight: Good baseline performance
   - Limitation: Small dataset (50K reviews)

3. **Methodology**
   - Dataset: 1.2M Amazon reviews
   - Baseline: LSTM + Word2Vec
   - Improvements: Bi-LSTM, Attention, GRU

4. **Results**
   - Comparison tables
   - Performance gains from larger dataset
   - Model comparison charts

5. **Conclusion**
   - Successfully scaled LSTM sentiment analysis to larger dataset (200K)
   - Achieved 93.23% test accuracy, surpassing Bodapati et al. (2019) by 4.77%
   - LSTM baseline outperformed more complex architectures (Bi-LSTM, Attention, GRU)
   - All models showed excellent generalization with no overfitting
   - Larger datasets (1.2M) show potential for further improvement (93.98%)
   - Future work: Transfer learning, transformer models, multilingual analysis

## Visualizations

Comprehensive visualizations are available in `results/deep_learning/`:
- `model_comparison.png` - 4-panel comparison showing training/validation accuracy, loss curves, and test accuracy
- Individual training curves for each model showing epoch-by-epoch progression

## Project Files

### Data
- `data/deep_learning_data/movies_reviews_200k.parquet` - 200K balanced dataset for deep learning project
- `data/processed/movies_reviews.parquet` - Full 1.2M dataset for data science project

### Results
- `results/deep_learning/01_lstm/` - Baseline LSTM results (93.23% accuracy)
- `results/deep_learning/02_bilstm/` - Bi-LSTM results (92.98% accuracy)
- `results/deep_learning/03_lstm_attention/` - LSTM+Attention results (93.08% accuracy)
- `results/deep_learning/04_gru/` - GRU results (93.02% accuracy)

Each results folder contains:
- `best_model.keras` - Trained model weights
- `results.pkl` - Training history and test accuracy
- `word2vec.model` - Pre-trained Word2Vec embeddings
- `tokenizer.pkl` - Text tokenizer

### Analysis Scripts
- `analyze_results.py` - Compare all model results
- `visualize_results.py` - Generate comparison charts
- `train_all_models_200k.bat` - Automated training script for all 4 models
