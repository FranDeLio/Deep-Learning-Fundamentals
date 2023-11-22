from config import PLOTS_PATH, N_EPOCHS, BATCH_SIZE, TRAIN_SIZE

from data_oriented_classes import DatasetClass
from deep_learning_elements import train_model, DeepNet, DeepNet_wDropout

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from plotnine import ggplot, geom_line, aes, coord_cartesian, theme, element_text

device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read data
data = fetch_california_housing()
X, y = data.data, data.target

# train-test split for model evaluation
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, train_size=TRAIN_SIZE, shuffle=True
)

# Standardizing data
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
print(X_train.shape)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Instantiate your dataset
train_dataset = DatasetClass(X_train, y_train)
test_dataset = DatasetClass(X_test, y_test)

standard_model = DeepNet()
dropout_model = DeepNet_wDropout()

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error

n_epochs = N_EPOCHS  # number of epochs to run
batch_size = BATCH_SIZE  # size of each batch

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

base_regressor = train_model(standard_model, train_loader, test_loader, loss_fn)

p9 = ggplot(data=base_regressor.epoch_history) + geom_line(
    aes(x="epoch", y="value", color="variable")
)
p9.save(PLOTS_PATH / "standard_regressor_epochs.png", width=6, height=4, dpi=300)

dropout_regressor = train_model(dropout_model, train_loader, test_loader, loss_fn)

p9 = ggplot(data=dropout_regressor.epoch_history) + geom_line(
    aes(x="epoch", y="value", color="variable")
)
p9.save(PLOTS_PATH / "dropout_regressor_epochs.png", width=6, height=4, dpi=300)


all_epoch_history = pd.concat(
    [base_regressor.epoch_history, dropout_regressor.epoch_history]
)
all_epoch_history["mse"] = all_epoch_history.apply(
    lambda row: row["variable"] + "_" + row["model_name"], axis=1
)


p9 = (
    ggplot(data=all_epoch_history)
    + geom_line(aes(x="epoch", y="value", color="mse"))
    + coord_cartesian(xlim=(1, None))
    + theme(legend_position="top", legend_text=element_text(size=5))
)
p9.save(PLOTS_PATH / "all_epochs.png", width=6, height=4, dpi=300)

