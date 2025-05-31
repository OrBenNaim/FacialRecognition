#================== Paths to CSV files ======================
DATA_FOLDER_PATH = './DATA/LFW-a'  # Path to LFW-a dataset
TRAIN_FILE_PATH = './DATA/pairsDevTrain.txt'  # Path to train.txt
TEST_FILE_PATH = './DATA/pairsDevTest.txt'  # Path to test.txt
SAVE_IMG_DIR_PATH = './images'  # Path to save images directory
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
CLASSIFICATION_THRESHOLD = 0.5

# Small Batch Test Parameters
SMALL_BATCH_TEST_LEARNING_RATE = 1e-3
SMALL_BATCH_TEST_ITERATIONS = 20
SMALL_BATCH_SUCCESS_THRESHOLD = 0.9
SMALL_BATCH_GOOD_PROGRESS_THRESHOLD = 0.7

#================== Data Augmentation ====================
