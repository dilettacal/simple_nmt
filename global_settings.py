import os
import torch

FILENAME = "deu.txt"
DATA_DIR = "data/"
PREPRO_DIR = os.path.join(DATA_DIR, "prepro")
EXPERIMENT_DIR = "experiment"
#SAVE_DIR = os.path.join(EXPERIMENT_DIR, "checkpoint")
SAVE_DIR = os.path.join(".",EXPERIMENT_DIR, "checkpoint")
MAX_LENGTH = 10


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")