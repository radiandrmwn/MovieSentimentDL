# Oct 31 Presentation — Long-Review Sentiment (Movies & TV)

## 1. Problem & Motivation
- Why sentiment on long, user-generated reviews matters
- Business/industry relevance

## 2. Dataset (≥1M)
- Amazon Reviews 2018 — Movies & TV
- Label mapping and splits (time-based)
- Length distribution

## 3. Prior Work (Highlights & Limits)
- fastText, CNN/LSTM, HAN
- BERT/RoBERTa (truncation issue)
- Longformer (long-context)

## 4. Baselines Implemented
- fastText
- RoBERTa-base
- Longformer-base (long reviews)

## 5. Proposed Ideas
- Ordinal-aware head
- Length-aware routing (short→RoBERTa, long→Longformer)
- Optional: domain-adaptive pre-finetuning

## 6. Experimental Setup
- Metrics: macro-F1, AUROC
- Hardware and training budget
- Ablation plan

## 7. Results
- Main table (baseline vs. proposed)
- Stratified by length

## 8. Error Analysis
- Sarcasm, mixed-aspect opinions
- Label noise (stars vs. text)

## 9. Takeaways & Next Steps
- Distillation for deployment
- Multilingual transfer (MARC)
