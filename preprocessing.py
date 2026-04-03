import pandas as pd
import numpy as np

def zscore(df):
    for col in df.columns:
        df[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
    return df.dropna()

def create_sequences(df, seq_len=8):
    X = []
    for i in range(len(df) - seq_len):
        X.append(df.iloc[i:i+seq_len].values)
    return np.array(X)

def dummy_sentiment(n):
    # placeholder sentiment
    return np.tanh(np.random.randn(n, 8))
