# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import csv


class FootballNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=64):#current inputs are ppg, scoredpg, concededpg, form_score, datedifference for each team
        #home and away is apssed in implicitly, as the first team is always home, but a flag can be added if necessary
        super(FootballNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 2)  # score_home, score_away, current just rounded but could use a poisson distribution later
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):#forward pass of the NN
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.out(x)
        return torch.exp(x)  # keep > 0


class FootballTable():
    def __init__ (self, filename):
            self.data = {}  # dictionary: {column_name: [values...]}
            self.teams = [] #a list of objects of tyope footballteam
            self.games = 0
            self.recentloss = [1] * 10#various variable used to store info for the NN
            self.recentresults = [0] * 30
            self.correct = 0
            self.wrong = 0
            with open(filename, newline='', encoding="latin-1") as f:#many football indexes use latin 1 instead of utf-8
                reader = csv.DictReader(f)   # reads rows as dicts
                for row in reader:
                    self.games += 1
                    for key, value in row.items():#get data from the databases
                        if key not in self.data:
                            self.data[key] = []
                        self.data[key].append(value)
            

    def train(self, model):
        teamnames = self.get_unique("HomeTeam")#get a unique list of teamnames
        self.teams = {name: FootballTeam(name) for name in teamnames}#dictionary so that teams can be directly accessed, one for  each teamname
        
        loss_fn = nn.MSELoss()#for measuring loss to use in backprop   
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for i in range (len(teamnames) * 5):#insures 5 matches for each team
            currentdata = self.get_row(i, ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"])
            self.teams[currentdata["HomeTeam"]].addmatch(currentdata["FTHG"], currentdata["FTAG"], currentdata["Date"]) #input the homegoals, away goals and date
            self.teams[currentdata["AwayTeam"]].addmatch(currentdata["FTAG"], currentdata["FTHG"], currentdata["Date"]) #input the homegoals, away goals and date

        for i in range((len(teamnames))*5, (len(teamnames) * (len(teamnames)-1))):#total matches is 20x19 e.g.
            currentdata = self.get_row(i, ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"])
            hometeamdata = self.teams[currentdata["HomeTeam"]].returndata(currentdata["Date"])
            awayteamdata = self.teams[currentdata["AwayTeam"]].returndata(currentdata["Date"])#ppg, scoredpg, concededpg, form_score, datedifference
            if int(currentdata["FTHG"]) > int(currentdata["FTAG"]):
                trueresult = 0
            elif int(currentdata["FTHG"]) == int(currentdata["FTAG"]):
                trueresult = 1
            else:
                trueresult = 2#2 means away win, 1 draw, 0 home win

            

            X = torch.tensor(hometeamdata + awayteamdata, dtype=torch.float32)  # shape [num_features]
            y = torch.tensor([float(currentdata["FTHG"]), float(currentdata["FTAG"])],
                 dtype=torch.float32)  # shape [2]

            # Training loop

            pred = model(X)                  # forward
            loss = loss_fn(pred, y)          # compute loss

            vals = pred.detach().cpu().numpy().flatten()#round the predictions
            home_goals = int(round(vals[0]))
            away_goals = int(round(vals[1]))

            if home_goals > away_goals:#track the predicted and true results
                predresult = 0
            elif home_goals == away_goals:
                predresult = 1
            else:
                predresult = 2#2 means away win, 1 draw, 0 home win

            self.recentresults.pop(0)
            if predresult == trueresult:
                self.recentresults.append(1)
                self.correct += 1
            else:
                self.recentresults.append(0)
                self.wrong += 1

            
            self.recentloss.pop(0)
            self.recentloss.append(loss.item())


            optimizer.zero_grad()            # clear old gradients
            loss.backward()                  # backprop
            optimizer.step()                 # update weights

            if (i+1) % 10 == 0:
                    print(f"match {i+1}, pred %: {sum(self.recentresults)/len(self.recentresults)}, Avg Loss: {sum(self.recentloss)/len(self.recentloss)}")

            self.teams[currentdata["HomeTeam"]].addmatch(currentdata["FTHG"], currentdata["FTAG"], currentdata["Date"]) #input the homegoals, away goals and date
            self.teams[currentdata["AwayTeam"]].addmatch(currentdata["FTAG"], currentdata["FTHG"], currentdata["Date"]) #input the homegoals, away goals and date




    def get_column(self, colname):
        return self.data.get(colname, [])
    
    def get_unique(self, colname):
        return list(set(self.data[colname]))

    def get_row(self, id, fields=None):#table.get_row(1, ["HomeTeam", "AwayTeam", "FTHG", FTAG])
        if fields:
            # Only return selected fields
            return {col: self.data[col][id] for col in fields}
        else:
            # Return full row
            return {col: self.data[col][id] for col in self.data}

class FootballTeam():
    def __init__(self, teamname):
        self.name = teamname#variables needed for returning data to the NN
        self.ppg = 0.0
        self.scores = 0
        self.conceded = 0
        self.form = []
        self.lastmatch = None
        self.matches = 0
        self.points = 0

    def addmatch(self, goalsfor, goalsagainst, date):
        self.scores += int(goalsfor)#add the data from the database into the object
        self.conceded += int(goalsagainst)
        self.lastmatch = self.typedate(date)
        self.matches += 1

        if (goalsfor > goalsagainst):
            self.points += 3
            self.form.append(3)
            if (len(self.form) > 4):
                self.form.pop(0)#remove oldest data frmn formm

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
        datedifference = (self.typedate(date) - self.lastmatch).days
        scoredpg = self.scores / self.matches * 3
        concededpg = self.conceded / self.matches * 3 #divide by 3 to keep all values roughly below 1
        ppg = self.points / self.matches * 3

        return [ppg, scoredpg, concededpg, form_score, datedifference]#return data in  the expected form for the NN
    
    def typedate(self, date):#change the date from a string to a  uniform type date
        date = date.strip()

        
        for fmt in ("%d/%m/%Y", "%d/%m/%y"):
            try:
                return datetime.strptime(date, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unknown date format: {date}")



if __name__ == "__main__":#main allows for direct running with running when imported

    #the data is on the listed order, repeated for away team
    #Home flag, form (% of points from last 12), ppg (divided by 3), squad value, days since last game
    #manager win %, goals scored per game, goals conceded per game and squad age
    X = torch.randn(10, 16) #x is the dataset, so 10 matches with 40 pieces of data each
    y = torch.tensor([[1,0],[2,1],[0,0],[3,1],[1,2],[2,2],[0,1],[1,3],[2,1],[1,1]], dtype=torch.float) #results of the matches

    predavg = []
    #pl24 = FootballTable("pl24.csv")
    model = FootballNN()
    print(model)
    #pl24.train(model)

    for i in range(1, 25):
        p = (26 - i)
        currenttable = FootballTable("databases//E0 (" + str(p) + ").csv")
        currenttable.train(model)
        print("E0 (" + str(p) + ").csv")
        print("\n")
        predavg.append(currenttable.correct / (currenttable.wrong + currenttable.correct))

    print(predavg)

    #loss_fn = nn.MSELoss()   # Phase 1: regression loss
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    #for epoch in range(500):
    #    pred = model(X)                  # forward
    #    loss = loss_fn(pred, y)          # compute loss
#
    #    optimizer.zero_grad()            # clear old gradients
    #    loss.backward()                  # backprop
    #    optimizer.step()                 # update weights
#
    #    if (epoch+1) % 10 == 0:
    #        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")