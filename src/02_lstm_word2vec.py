#!/usr/bin/env python
"""
02_lstm_word2vec.py
Baseline LSTM + Word2Vec (reproduce Bodapati et al. 2019)

Usage:
  python src/02_lstm_word2vec.py --data data/processed/movies_reviews.parquet --output_dir results/lstm_word2vec
"""
import argparse
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
import pickle

def create_word2vec_embeddings(texts, embedding_dim=100):
    """Create Word2Vec embeddings"""
    print("Training Word2Vec embeddings...")
    sentences = [text.lower().split() for text in texts]
    w2v_model = Word2Vec(sentences, vector_size=embedding_dim, window=5,
                         min_count=2, workers=4, epochs=10)
    return w2v_model

def create_embedding_matrix(tokenizer, w2v_model, embedding_dim=100):
    """Create embedding matrix from Word2Vec model"""
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    return embedding_matrix

def build_lstm_model(vocab_size, embedding_dim, embedding_matrix, max_length, lstm_units=100):
    """Build LSTM model (Bodapati et al. 2019 architecture)"""
    model = Sequential([
        Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                  input_length=max_length, trainable=False),
        LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to processed parquet file")
    ap.add_argument("--output_dir", required=True, help="Directory to save model and results")
    ap.add_argument("--max_length", type=int, default=200, help="Max sequence length")
    ap.add_argument("--embedding_dim", type=int, default=100, help="Word2Vec embedding dimension")
    ap.add_argument("--lstm_units", type=int, default=100, help="LSTM units")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_parquet(args.data)

    # Binary classification: 0=negative, 1=positive (drop neutral if exists)
    df = df[df['label'].isin([0, 1])]

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Tokenization
    print("Tokenizing texts...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['text'])

    X_train = tokenizer.texts_to_sequences(train_df['text'])
    X_val = tokenizer.texts_to_sequences(val_df['text'])
    X_test = tokenizer.texts_to_sequences(test_df['text'])

    X_train = pad_sequences(X_train, maxlen=args.max_length)
    X_val = pad_sequences(X_val, maxlen=args.max_length)
    X_test = pad_sequences(X_test, maxlen=args.max_length)

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Train Word2Vec
    w2v_model = create_word2vec_embeddings(train_df['text'], args.embedding_dim)
    w2v_model.save(os.path.join(args.output_dir, 'word2vec.model'))

    # Create embedding matrix
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = create_embedding_matrix(tokenizer, w2v_model, args.embedding_dim)

    # Build model
    print("Building LSTM model...")
    model = build_lstm_model(vocab_size, args.embedding_dim, embedding_matrix,
                            args.max_length, args.lstm_units)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(os.path.join(args.output_dir, 'best_model.keras'),
                       save_best_only=True, monitor='val_accuracy')
    ]

    # Train
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # Save results
    results = {
        'test_accuracy': accuracy,
        'history': history.history
    }

    with open(os.path.join(args.output_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    with open(os.path.join(args.output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
