import copy
from typing import Type

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from config import N_EPOCHS, GAMMA, SCHEDULER_STEP, LEARNING_RATE, P_DROPOUT
from data_oriented_classes import ModelOutput                   


# Define the model
class ConvolutionalNetwork_wDropout(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork_wDropout, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(), 
            nn.Dropout(P_DROPOUT),                     
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),
            nn.Dropout(P_DROPOUT),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        x = self.out(x)
        return x.squeeze()    # return x for visualization
        

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        x = self.out(x)
        return x.squeeze()    # return x for visualization

    
class DenseNetwork(nn.Module):
    def __init__(self):
        super(DenseNetwork, self).__init__()
        self.Matrix1 = nn.Linear(28**2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.output = nn.Linear(50,10)
        self.R = nn.LeakyReLU()

    def forward(self, x):
        x = x.view(-1,28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.output(x)
        return x.squeeze()
    
class DenseNetwork_wDropout(nn.Module):
    def __init__(self):
        super(DenseNetwork_wDropout, self).__init__()
        self.Matrix1 = nn.Linear(28**2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.output = nn.Linear(50,10)
        self.R = nn.LeakyReLU()
        self.dropout = nn.Dropout(P_DROPOUT)

    def forward(self, x):
        x = x.view(-1,28**2)
        x = self.R(self.Matrix1(x))
        x = self.dropout(x)
        x = self.R(self.Matrix2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x.squeeze()


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module) -> float:
    
    model.eval()
    correct_predictions = 0
    total_samples = 0

    for X_batch, y_batch in data_loader:

        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        y_pred = torch.argmax(y_pred, dim=1)
        y_batch = torch.argmax(y_batch, dim=1)
        
        total_samples += y_batch.size(0)
        correct_predictions += (y_pred == y_batch).sum().item()

    accuracy = correct_predictions/total_samples

    return accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    n_epochs: int = N_EPOCHS,
    weight_decay: float = 0,
) -> ModelOutput:
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay
    )
    optimizer_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=SCHEDULER_STEP, gamma=GAMMA
    )

    best_weights = None
    best_mse = np.inf
    history = {
        "epoch": np.arange(0, n_epochs),
        "train_acc": [],
        "test_acc": [],
        "model_name": model.__class__.__name__,
    }

    for epoch in range(n_epochs):
        # Training
        model.train()
        batch_loss = []
        with tqdm(total=len(train_loader), unit="batch") as bar:
            bar.set_description(f"Epoch {epoch} (LR={optimizer.param_groups[0]['lr']})")

            for X_batch, y_batch in train_loader:
                # Forward pass
                y_pred = model.forward(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()

                batch_loss.append(loss.item())
                bar.set_postfix(bce=np.mean(batch_loss))
                bar.update(1)

        # Evaluate accuracy at the end of each epoch
        mse = evaluate(model, train_loader, loss_fn)
        history["train_acc"].append(mse)
        test_mse = evaluate(model, test_loader, loss_fn)
        history["test_acc"].append(test_mse)

        # Take step toward reducing learning rate
        optimizer_scheduler.step()

        # Check if the current model is the best
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    history = pd.DataFrame(history).melt(id_vars=["epoch", "model_name"])

    # Restore the best model
    model.load_state_dict(best_weights)
    test_mse = evaluate(model, test_loader, loss_fn)

    print("Train Accuracy: %.2f" % best_mse)
    print("Test Accuracy: %.2f" % test_mse)

    return ModelOutput(model, best_mse, best_weights, test_mse, history)
