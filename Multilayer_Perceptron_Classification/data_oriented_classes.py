from typing import Type
from dataclasses import dataclass

from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch


class DatasetClass(Dataset):
    def __init__(self, filepath):
        self.features, self.labels = torch.load(filepath)
        self.features = self.features / 255.
        self.labels = F.one_hot(self.labels, num_classes=10).to(float)
    def __len__(self): 
        return self.features.shape[0]
    def __getitem__(self, ix): 
        return self.features[ix], self.labels[ix]

@dataclass
class ModelOutput:
    """Class for keeping track of an item in inventory."""

    model: nn.Module
    train_mse: float
    best_weights: float
    test_mse: float
    epoch_history: list[float]
