import torch
import numpy as np
from model import FINseqGNN

def train_model(Xp, Xs, y, epochs=20):
    model = FINseqGNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for e in range(epochs):
        opt.zero_grad()
        pred = model(Xp, Xs)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        print(f"Epoch {e+1}, Loss {loss.item()}")

    torch.save(model.state_dict(), "../outputs/best_model.pth")
    return model
