import numpy as np

def mse(y, yhat):
    return np.mean((y - yhat) ** 2)

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def mrr(y_true, y_pred):
    ranks = np.argsort(-y_pred)
    best = np.argmax(y_true)
    return 1 / (np.where(ranks == best)[0][0] + 1)

def sharpe(returns):
    return returns.mean() / (returns.std() + 1e-8)

def ceq(returns, gamma=1):
    return returns.mean() - 0.5 * gamma * returns.var()
