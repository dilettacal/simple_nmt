import os
import torch

##### FILE AND DIRECTORY SETTINGS #####

FILENAME = "deu.txt"
DATA_DIR = "data/"
PREPRO_DIR = os.path.join(DATA_DIR, "prepro")
DOCU_DIR = "documentation"

EXPERIMENT_DIR = "experiment/"
SAVE_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
MAX_LENGTH = 10

TRAIN_FILE = "train.pkl"
TEST_FILE = "test.pkl"

LOG_FILE = "last_experiment.txt"
SAMPLES_FILE = "translation_examples_for_testing.txt"
TRANSLATIONS_FROM_SAMPLES = "translations_from_sample.txt"


#### CUDA SETTINGS ######

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#### EXPERIMENT (FIXED) SETTINGS #####

LR_DECAY = 0.7 #how much the learning rate should be decayed if valid loss does not improve
MIN_LR = 1e-6
MAX_LR = 0.06
NUM_BAD_VALID_LOSS = 5

VAL_TRAIN_DELTA = 1.5
