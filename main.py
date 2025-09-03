# main.py
import torch
import torch.nn as nn

class FootballNN(nn.Module):
    def __init__(self, input_size=40, hidden_size=64):
        super(FootballNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 2)  # λ_home, λ_away
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return torch.exp(x)  # keep λ > 0

if __name__ == "__main__":
    model = FootballNN()
    print(model)