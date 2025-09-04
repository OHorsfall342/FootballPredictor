# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime


class FootballNN(nn.Module):
    def __init__(self, input_size=16, hidden_size=64):
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


class FootballTable(string filename):
    this.name = "a"

class FootballTeam():
    
    def __init__(self, teamname):
        self.name = teamname
        self.ppg = 0.0
        self.scores = 0
        self.conceded = 0
        self.form = []
        self.lastmatch = None
        self.matches = 0
        self.points = 0

    def addmatch(self, goalsfor, goalsagainst, date):
        self.scores += goalsfor
        self.concedes += goalsagainst
        self.lastmatch = datetime.strptime(date, "%d-%m-%Y")
        self.matches += 1

        if (goalsfor > goalsagainst):
            self.points += 3
            self.form.append(3)
            if (len(self.form) > 4):
                self.form.pop(0)

        if (goalsfor == goalsagainst):
            self.points += 1
            self.form.append(1)
            if (len(self.form) > 4):
                self.form.pop(0)
        
        else:
            self.form.append(0)
            if (len(self.form) > 4):
                self.form.pop(0)

        self.ppg = self.points / self.matches

    def returndata(self, date):
        form_score = sum(self.form) / (len(self.form) * 3) if self.form else 0.3
        datedifference = datetime.strptime(date, "%d-%m-%Y") - self.lastmatch
        scoredpg = self.scores / self.matches * 3
        concededpg = self.conceded / self.matches * 3 #divide by 3 to keep all values roughly below 1
        ppg = self.points / self.matches * 3

        return [ppg, scoredpg, concededpg, form_score, datedifference]



if __name__ == "__main__":#main allows for direct running with running when imported

    #the data is on the listed order, repeated for away team
    #Home flag, form (% of points from last 12), ppg (divided by 3), squad value, days since last game
    #manager win %, goals scored per game, goals conceded per game and squad age
    X = torch.randn(10, 16) #x is the dataset, so 10 matches with 40 pieces of data each
    y = torch.tensor([[1,0],[2,1],[0,0],[3,1],[1,2],[2,2],[0,1],[1,3],[2,1],[1,1]], dtype=torch.float) #results of the matches

    model = FootballNN()
    print(model)
    loss_fn = nn.MSELoss()   # Phase 1: regression loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(500):
        pred = model(X)                  # forward
        loss = loss_fn(pred, y)          # compute loss

        optimizer.zero_grad()            # clear old gradients
        loss.backward()                  # backprop
        optimizer.step()                 # update weights

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")