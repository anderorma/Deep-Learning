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
    data = pd.read_csv('1assign/insurance.csv')

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

# 5. MAIN EXECUTION AND TENSORBOARD
if __name__ == "__main__":
    set_seed(42)
    
    Xtrain, Xval, Xtest, ytrain, yval, ytest, scalerY = load()
    yRealTest = scalerY.inverse_transform(ytest) 
    
    lrModel = LinearRegression()
    lrModel.fit(Xtrain, ytrain)
    lrPredictions = scalerY.inverse_transform(lrModel.predict(Xtest))
    lrMAE = mean_absolute_error(yRealTest, lrPredictions)
    
    print("Starting optimization with Optuna...")
    OptunaStudy = optuna.create_study(direction='minimize') #minimize the validation loss
    OptunaStudy.optimize(objective, n_trials=25) 
    
    bestParameters = OptunaStudy.best_params
    print(f"\nBest parameters found by Optuna: {bestParameters}")
    
    XtrainTensor = torch.FloatTensor(Xtrain)
    ytrainTensor = torch.FloatTensor(ytrain)
    XvalTensor = torch.FloatTensor(Xval)
    yvalTensor = torch.FloatTensor(yval)
    XtestTensor = torch.FloatTensor(Xtest)

    finalModel = NeuralNetwork(XtrainTensor.shape[1], bestParameters['dropout_rate'])
    optimizer = torch.optim.Adam(finalModel.parameters(), lr=bestParameters['lr'])
    criterion = nn.MSELoss()
    
    writer = SummaryWriter('runs/insurance_experiment') #Create a TensorBoard
    
    for epoch in range(200):
        finalModel.train()
        optimizer.zero_grad() # Clear gradients
        trainLoss = criterion(finalModel(XtrainTensor), ytrainTensor) # Compare predictions of the model with true values
        trainLoss.backward() # Calculate gradients
        optimizer.step() # Update weights
        
        finalModel.eval()
        with torch.no_grad(): # It does not save anything
            valLoss = criterion(finalModel(XvalTensor), yvalTensor)  #Compare predictions of the model with true values in the validation set
            
        writer.add_scalars('Learning Curve (MSE)', 
                            {'Train': trainLoss.item(), 'Validation': valLoss.item()}, 
                           epoch)
                           
    writer.close()
    
    finalModel.eval()
    with torch.no_grad():
        predsNorm = finalModel(XtestTensor).numpy() # Get predictions in normalized form
    
    nnPreds = scalerY.inverse_transform(predsNorm) # Convert predictions back to original scale
    nnMAE = mean_absolute_error(yRealTest, nnPreds) # Calculate MAE for the Neural Network predictions

    print(f"\nFINAL RESULTS ON TEST SET")
    print("-----------------------------")
    print(f"Linear Regression MAE: ${lrMAE:,.2f}")
    print(f"Neural Network MAE: ${nnMAE:,.2f}")
    
    if nnMAE < lrMAE:
        absoluteImprovement = lrMAE - nnMAE
        relativeImprovement = (absoluteImprovement / lrMAE) * 100
        
        print("\nCONCLUSION")
        print("-----------------------------")
        print("Neural Network performs better than Linear Regression")
        print(f"Absolute Improvement : Reduced the error by ${absoluteImprovement:,.2f} on average.")
        print(f"Relative Improvement : {relativeImprovement:.2f}% better than Linear Regression.")
    else:
        print("\nLinear Regression performs better than Neural Network")
    
    # 8. RESULTS VISUALIZATION
  
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1
    ax1.scatter(yRealTest, nnPreds, alpha=0.6, color='blue', label='Neural Network Predictions')
    minVal = min(yRealTest.min(), nnPreds.min())
    maxVal = max(yRealTest.max(), nnPreds.max())
    ax1.plot([minVal, maxVal], [minVal, maxVal], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_title('Real vs. Predicted')
    ax1.set_xlabel('Real Price in dollars')
    ax1.set_ylabel('Predicted Price in dollars')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2
    df = pd.read_csv('1assign/insurance.csv')
    smokers = df[df['smoker'] == 'yes']
    nonSmokers = df[df['smoker'] == 'no']
    
    ax2.scatter(smokers['bmi'], smokers['charges'], color='red', alpha=0.6, label='Smokers')
    ax2.scatter(nonSmokers['bmi'], nonSmokers['charges'], color='green', alpha=0.6, label='Non-Smokers')
    
    ax2.set_title('BMI vs Insurance Cost')
    ax2.set_xlabel('BMI (Body Mass Index)')
    ax2.set_ylabel('Cost in dollars')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3
    models_list = ['Linear Regression', 'Neural Network']
    errors = [lrMAE, nnMAE]
    colors = ['gray', 'orange']
    
    bars = ax3.bar(models_list, errors, color=colors, width=0.5)
    ax3.set_title('3. Improvement Comparison (MAE)')
    ax3.set_ylabel('Mean Absolute Error in dollars')
    
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 50, f'${yval:,.0f}', ha='center', va='bottom', fontweight='bold')

    # Show the plots
    plt.tight_layout()
    plt.show()

