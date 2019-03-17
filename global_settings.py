import os
import torch

FILENAME = "deu.txt"
DATA_DIR = "data/"
PREPRO_DIR = os.path.join(DATA_DIR, "prepro")
EXPERIMENT_DIR = "experiment/"
SAVE_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
MAX_LENGTH = 10

TRAIN_FILE = "train.pkl"
TEST_FILE = "test.pkl"

LOG_FILE = "last_experiment.txt"


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")