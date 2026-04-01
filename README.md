# 📈 Stock Price Prediction using FINseqGNN

## 🚀 Overview
This project presents a multimodal deep learning framework that combines:
- Numerical stock price data
- Financial tweet sentiment
- Graph Neural Networks

to predict stock price z-scores and market trends.

---

## 🧠 Features
- Z-score normalization (ACL-2018 standard)
- 8-day temporal modeling
- Sentiment integration (FLAIR)
- LSTM + GNN architecture
- Trading metrics (Sharpe, MRR, CEQ)
- Streamlit deployment

---

## 📊 Models Implemented
- LSTM
- ALSTM
- GCN
- GAT
- Transformer
- FINseqGNN (Proposed)

---

## 🧪 Results
| Model | MSE | MAE | MRR |
|------|-----|-----|-----|
| LSTM | High | High | Low |
| ALSTM | ↓ | ↓ | ↑ |
| GAT | ↓ | ↓ | ↑ |
| FINseqGNN | Lowest | Lowest | Highest |

---

## 🖥️ Run App

```bash
streamlit run app/app.py
