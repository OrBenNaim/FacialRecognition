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
DROPOUT_RATE = 0.3             # Dropout rate for regularization

#================== Training Configuration =================
# General Training Parameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 6e-5
EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP_MAX_NORM = 1.0
EPSILON = 1e-8                 # Numerical stability

# Small Batch Test Parameters
SMALL_BATCH_TEST_LEARNING_RATE = 1e-3
SMALL_BATCH_TEST_ITERATIONS = 20
SMALL_BATCH_TEST_WEIGHT = 2.0
SMALL_BATCH_SUCCESS_THRESHOLD = 0.9
SMALL_BATCH_GOOD_PROGRESS_THRESHOLD = 0.7

# Learning Rate Schedule
LR_SCHEDULE_REDUCTION_FACTOR = 0.5
LR_SCHEDULE_PATIENCE = 3
LR_SCHEDULE_MIN = 1e-7

#================== Data Augmentation ====================
# Augmentation Parameters
AUGMENTATION_ROTATION_RANGE = 20
AUGMENTATION_ZOOM_RANGE = 0.15
AUGMENTATION_WIDTH_SHIFT = 0.2
AUGMENTATION_HEIGHT_SHIFT = 0.2
AUGMENTATION_BRIGHTNESS_MIN = 0.85
AUGMENTATION_BRIGHTNESS_MAX = 1.15