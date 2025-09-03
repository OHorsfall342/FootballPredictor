# main.py
import torch
import torch.nn as nn
import torch.optim as optim

class FootballNN(nn.Module):
    def __init__(self, input_size=40, hidden_size=64):
        super(FootballNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 2)  # λ_home, λ_away
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return torch.exp(x)  # keep λ > 0

if __name__ == "__main__":#main allows for direct running with running when imported
    X = torch.randn(10, 40) #x is the dataset, so 10 matches with 40 pieces of data each
    y = torch.tensor([[1,0],[2,1],[0,0],[3,1],[1,2],[2,2],[0,1],[1,3],[2,1],[1,1]], dtype=torch.float) #results of the matches

    model = FootballNN()
    print(model)
    loss_fn = nn.MSELoss()   # Phase 1: regression loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(50):
        pred = model(X)                  # forward
        loss = loss_fn(pred, y)          # compute loss

        optimizer.zero_grad()            # clear old gradients
        loss.backward()                  # backprop
        optimizer.step()                 # update weights

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")