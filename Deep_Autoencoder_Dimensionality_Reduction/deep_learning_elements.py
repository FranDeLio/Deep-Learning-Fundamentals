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
class Autoencoder(nn.Module):
    def __init__(self):

        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(28**2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 30),
            nn.LeakyReLU(), 
            nn.Linear(30, 12), 
            nn.LeakyReLU(), 
            nn.Linear(12, 2))
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.LeakyReLU(),
            nn.Linear(12, 30),
            nn.LeakyReLU(),
            nn.Linear(30, 64),
            nn.LeakyReLU(), 
            nn.Linear(64, 28**2), 
            nn.LeakyReLU())

    def forward(self, x, decode = True):
        x = x.view(-1,28**2)
        x = self.encoder(x)
        if decode==False: return x.squeeze()
        x = self.decoder(x)
        return x.squeeze()
    
class DeepNetMNIST_wDropout(nn.Module):
    def __init__(self):
        super().__init__()
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
    batch_loss = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, X_batch)
            batch_loss.append(loss.item())

    mse = float(np.mean(batch_loss))
    return mse


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
        "train_mse": [],
        "test_mse": [],
        "model_name": model.__class__.__name__,
    }

    for epoch in range(n_epochs):
        # Training
        model.train()
        batch_loss = []
        with tqdm(total=len(train_loader), unit="batch") as bar:
            bar.set_description(f"Epoch {epoch} (LR={optimizer.param_groups[0]['lr']})")

            for X_batch, _ in train_loader:
                # Forward pass
                y_pred = model.forward(X_batch)
                loss = loss_fn(y_pred, X_batch)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()

                batch_loss.append(loss.item())
                bar.set_postfix(mse=np.mean(batch_loss))
                bar.update(1)

        # Evaluate accuracy at the end of each epoch
        mse = evaluate(model, train_loader, loss_fn)
        history["train_mse"].append(mse)
        test_mse = evaluate(model, test_loader, loss_fn)
        history["test_mse"].append(test_mse)

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

    print("Train MSE: %.2f" % best_mse)
    print("Train RMSE: %.2f" % np.sqrt(best_mse))
    print("Test MSE: %.2f" % test_mse)
    print("Test MSE: %.2f" % np.sqrt(test_mse))

    return ModelOutput(model, best_mse, best_weights, test_mse, history)
