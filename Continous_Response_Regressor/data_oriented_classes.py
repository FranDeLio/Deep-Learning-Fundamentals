from typing import Type
from dataclasses import dataclass
from torch.utils.data import Dataset
import torch.nn as nn


class DatasetClass(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


@dataclass
class ModelOutput:
    """Class for keeping track of an item in inventory."""

    model: Type[nn.Module]
    train_mse: float
    best_weights: float
    test_mse: float
    epoch_history: list[float]