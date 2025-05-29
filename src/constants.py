#================== Paths to CSV files ======================
DATA_PATH = './DATA/LFW-a'  # Path to LFW-a dataset
TRAIN_FILE = './DATA/pairsDevTrain.txt'  # Path to train.txt
TEST_FILE = './DATA/pairsDevTest.txt'  # Path to test.txt
#============================================================

# Image and Data Configuration
OPTIMIZED_IMG_SHAPE = (128, 128, 1)  # Optimized from original 250x250
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

#================== Network Architecture ====================
# CNN Layer Configuration
NUM_OF_FILTERS_LAYER1 = 64      # Basic face features
NUM_OF_FILTERS_LAYER2 = 128     # Complex patterns
NUM_OF_FILTERS_LAYER3 = 128     # Complex patterns
NUM_OF_FILTERS_LAYER4 = 256     # Final embedding

KERNAL_SIZE_LAYER1 = (10, 10)   # Broad spatial patterns
KERNAL_SIZE_LAYER2 = (7, 7)     # Medium features
KERNAL_SIZE_LAYER3 = (4, 4)     # Fine details
KERNAL_SIZE_LAYER4 = (4, 4)     # Fine details
POOL_SIZE = (2, 2)

# Regularization Constants
L2_CONV_REG = 2e-4             # L2 regularization for conv layers
L2_DENSE_REG = 1e-3            # L2 regularization for dense layer

#================== Training Configuration =================
# General Training Parameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 6e-5
EARLY_STOPPING_PATIENCE = 5

# Small Batch Test Parameters
SMALL_BATCH_TEST_LEARNING_RATE = 1e-3
SMALL_BATCH_TEST_ITERATIONS = 20
SMALL_BATCH_SUCCESS_THRESHOLD = 0.9
SMALL_BATCH_GOOD_PROGRESS_THRESHOLD = 0.7

#================== N-way One-shot Configuration =================
# Based on the original paper's configuration
N_WAY = 20                     # Number of classes for N-way classification
N_WAY_EPISODES = 1000          # Number of episodes for N-way evaluation
SUPPORT_SET_SIZE = 1           # Number of support examples per class (1 for one-shot)
CLASSIFIER_HIDDEN_SIZE = 512    # Hidden layer size for N-way classifier

# N-way Training Parameters
N_WAY_BATCH_SIZE = 1           # Batch size for N-way training (usually 1 episode at a time)
N_WAY_LEARNING_RATE = 1e-4     # Specific learning rate for N-way training
N_WAY_EPOCHS = 20              # Number of epochs for N-way training
N_WAY_EARLY_STOPPING = 10      # Early stopping patience for N-way training

# N-way Evaluation Thresholds
N_WAY_SUCCESS_THRESHOLD = 0.95  # Accuracy threshold for successful N-way training
N_WAY_PROGRESS_THRESHOLD = 0.85 # Threshold for good progress in N-way training
