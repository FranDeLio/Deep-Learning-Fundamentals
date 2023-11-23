import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from plotnine import ggplot, geom_line, aes, coord_cartesian, theme, element_text

from config import PLOTS_PATH, N_EPOCHS, BATCH_SIZE
from data_oriented_classes import DatasetClass
from deep_learning_elements import train_model, DeepNetMNIST, DeepNetMNIST_wDropout

device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate your dataset
train_dataset = DatasetClass('./MNIST/processed/training.pt')
test_dataset = DatasetClass('./MNIST/processed/test.pt')

standard_model = DeepNetMNIST()
dropout_model = DeepNetMNIST_wDropout()

# loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # mean square error

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

base_regressor = train_model(standard_model, train_loader, test_loader, loss_fn, n_epochs=N_EPOCHS)

p9 = (
    ggplot(data=base_regressor.epoch_history)
    + geom_line(aes(x="epoch", y="value", color="variable"))
    + coord_cartesian(xlim=(0, None))
)
p9.save(PLOTS_PATH / "standard_regressor_epochs_MNIST.png", width=6, height=4, dpi=300)

dropout_regressor = train_model(dropout_model, train_loader, test_loader, loss_fn, n_epochs=N_EPOCHS)

p9 = (
    ggplot(data=dropout_regressor.epoch_history)
    + geom_line(aes(x="epoch", y="value", color="variable"))
    + coord_cartesian(xlim=(0, None))
)
p9.save(PLOTS_PATH / "dropout_regressor_epochs_MNIST.png", width=6, height=4, dpi=300)


all_epoch_history = pd.concat(
    [base_regressor.epoch_history, dropout_regressor.epoch_history]
)
all_epoch_history["mse"] = all_epoch_history.apply(
    lambda row: row["variable"] + "_" + row["model_name"], axis=1
)

p9 = (
    ggplot(data=all_epoch_history)
    + geom_line(aes(x="epoch", y="value", color="mse"))
    + coord_cartesian(xlim=(0, None))
    + theme(legend_position="top", legend_text=element_text(size=5))
)
p9.save(PLOTS_PATH / "all_epochs_MNIST.png", width=6, height=4, dpi=300)


X_test = test_dataset.features
y_test = test_dataset.labels.argmax(axis=1)
y_pred = base_regressor.model(X_test).argmax(axis=1)

print(f"Accuracy: {(y_test==y_pred).float().mean()}")

fig, ax = plt.subplots(10,4,figsize=(10,15))
for i in range(40):
    plt.subplot(10,4,i+1)
    plt.imshow(X_test[i])
    plt.title(f'Predicted Digit: {y_pred[i]}')
fig.tight_layout()
plt.show()
