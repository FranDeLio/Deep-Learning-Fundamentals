import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from plotnine import ggplot, geom_line, geom_point, aes, coord_cartesian, theme, element_text

from config import PLOTS_PATH, N_EPOCHS, BATCH_SIZE
from data_oriented_classes import DatasetClass
from deep_learning_elements import train_model, Autoencoder

device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate your dataset
train_dataset = DatasetClass('./MNIST/processed/training.pt')
test_dataset = DatasetClass('./MNIST/processed/test.pt')

standard_model = Autoencoder()

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
batch_size = BATCH_SIZE  # size of each batch

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

base_regressor = train_model(standard_model, train_loader, test_loader, loss_fn, n_epochs=N_EPOCHS)

p9 = (
    ggplot(data=base_regressor.epoch_history)
    + geom_line(aes(x="epoch", y="value", color="variable"))
    + coord_cartesian(xlim=(0, None))
)
p9.save(PLOTS_PATH / "standard_regressor_epochs_MNIST.png", width=6, height=4, dpi=300)


X_test = test_dataset.features
y_test = test_dataset.labels.argmax(axis=1)
y_pred = base_regressor.model(X_test, decode=False)
df = pd.DataFrame(y_pred.detach().numpy(), columns=('latent_var_1','latent_var_2'))
df['label'] = y_test.detach().numpy().astype(str)


p9 = (ggplot(data=df)+
        geom_point(aes(x='latent_var_1', y='latent_var_2', color='label')))
p9.save(PLOTS_PATH / f"latent_space_MNIST_it{N_EPOCHS}.png", width=6, height=4, dpi=300)
print(p9)