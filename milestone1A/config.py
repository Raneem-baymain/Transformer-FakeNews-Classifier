# config.py

# Reproducibility
SEED = 42

# Model hyperparameters
D_MODEL = 64
NUM_HEADS = 8
D_FF = 256
DROPOUT = 0.1

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3

# Data parameters
MAX_VOCAB_SIZE = 10000
MAX_SEQ_LEN = 100

# Dataset
DATASET_NAME = "saurabhshahane/fake-news-classification"

# Device
USE_GPU = True
