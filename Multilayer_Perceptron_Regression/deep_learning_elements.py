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
class DeepNet_wDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 60)
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(P_DROPOUT)
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(P_DROPOUT)
        self.layer3 = nn.Linear(60, 30)
        self.act3 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(P_DROPOUT)
        self.output = nn.Linear(30, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.act3(self.layer3(x))
        x = self.dropout3(x)
        x = self.output(x)
        return x


class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 60)
        self.act1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(60, 30)
        self.act3 = nn.LeakyReLU()
        self.output = nn.Linear(30, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.output(x)
        return x


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    batch_loss = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
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
