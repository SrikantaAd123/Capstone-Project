import pandas as pd

def preprocess(df):

    df['adj_factor'] = df['Adj Close'] / df['Close']

    df['adj_open'] = df['Open'] * df['adj_factor']
    df['adj_high'] = df['High'] * df['adj_factor']
    df['adj_low']  = df['Low']  * df['adj_factor']

    for col in ['adj_open','adj_high','adj_low','Adj Close','Volume']:

        mean = df[col].rolling(60).mean()
        std  = df[col].rolling(60).std()

        df[col] = (df[col] - mean) / (std + 1e-8)

    df['target'] = df['Adj Close'].shift(-1)

    return df.dropna()
