import os
import random
from xml.parsers.expat import model
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import optuna
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 1. REPRODUCIBILITY
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# 2. DATA PREPARATION
def load():
    data = pd.read_csv('Deep-Learning/1assign/insurance.csv')

    data['sex'] = data['sex'].map({'female': 0, 'male': 1})
    data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
    data = pd.get_dummies(data, drop_first=True).astype(float) #drop_first will give 3 columns instead of 4 for region (delete NE)
    
    X = data.drop('charges', axis=1).values #axis=1 to drop column
    y = data['charges'].values.reshape(-1, 1) #reshape to make it in a matrix form
    
    XtrainVal, Xtest, ytrainVal, ytest = train_test_split(X, y, test_size=0.2, random_state=42) #20% for test, 80% for train and validation
    Xtrain, Xval, ytrain, yval = train_test_split(XtrainVal, ytrainVal, test_size=0.15, random_state=42) #15% of 80% = 12% for val, 68% for train
    
    scalerX = StandardScaler() #standarize to avoid variables dominance
    scalerY = StandardScaler() #standarize to avoid variables dominance
    
    Xtrain = scalerX.fit_transform(Xtrain) 
    Xval = scalerX.transform(Xval)
    Xtest = scalerX.transform(Xtest)
    
    ytrain = scalerY.fit_transform(ytrain)
    yval = scalerY.transform(yval)
    ytest = scalerY.transform(ytest)
    
    return Xtrain, Xval, Xtest, ytrain, yval, ytest, scalerY


# 3. NEURAL NETWORK ARCHITECTURE
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), #input_dim is the number of features (8)
            nn.BatchNorm1d(64), 
            nn.ReLU(), #ReLU if it is negative it will convert it into 0, if positive it will keep the value.
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate), #dropout to avoid overfitting, 0.3357 is the best one
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu') #Kaiming Normal initialization for ReLU activations
                nn.init.zeros_(m.bias) # Initialize biases to zero
        
    def forward(self, x):
        return self.model(x)



# 4. OPTUNA OBJECTIVE FUNCTION
def objective(trial):
    Xtrain, Xval, _, ytrain, yval, _, _ = load()
    
    XtrainTensor = torch.FloatTensor(Xtrain) # Convert to PyTorch tensors
    ytrainTensor = torch.FloatTensor(ytrain) 
    XvalTensor = torch.FloatTensor(Xval)
    yvalTensor = torch.FloatTensor(yval)
    
    lr = trial.suggest_float('lr', 0.0001, 0.1, log=True) 
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5) 
    
    model = NeuralNetwork(XtrainTensor.shape[1], dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() 
    
    #Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad() # Clear gradients
        loss = criterion(model(XtrainTensor), ytrainTensor) # Compare predictions of the model with true values
        loss.backward() # Calculate gradients
        optimizer.step() # Update weights
        
    model.eval() 
    with torch.no_grad(): # It does not save anything
        valLoss = criterion(model(XvalTensor), yvalTensor) #Compare predictions of the model with true values in the validation set
    
    return valLoss.item() # Return the validation loss as the metric to minimize in Optuna



