import torch

def train(model, loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):

        total_loss = 0

        for xb, yb in loader:

            pred = model(xb)

            loss = torch.nn.functional.mse_loss(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss {total_loss:.4f}")

    torch.save(model, "outputs/best_model.pth")
