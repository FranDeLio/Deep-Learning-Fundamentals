import os
from pathlib import Path


PLOTS_PATH = Path("./plots")
if not os.path.exists(PLOTS_PATH):
    os.makedirs(PLOTS_PATH)

BATCH_SIZE = 10
N_EPOCHS = 50
GAMMA = 0.666
SCHEDULER_STEP = 50
LEARNING_RATE = 0.0001
P_DROPOUT = 0.1
