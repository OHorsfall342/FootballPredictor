# Football Match Predictor


This predicts the results of a football match, currently it uses data such as PPG, recent form, Home, Goals scored and Goals conceded.

This will later be increased to have values such as squad value and manager statistics, etc.

This is then converted via a neural network into predicted goals, which is translated into a result.

The predictor is very much a work in progress, with many thigns still to be added, currently it is jsut a prototype.


## Project Structure


Pytorch is used to create the Neural Network

CSV files with all match data of previous seasons are downloaded from football-data.co.uk

The class Football Team stored key information about recent and historical results about a team.

The class Football Table reads a CSV and creates a Football Team object for each team in the table.

The football table object will then call the train method, which will iterate through every match in that season, training on them.

After each match is predicted and trained on, it is then used to update the database so it is relevant for the next match.

Main just initiates the neural network, and iterates through every database season currently downloaded.
