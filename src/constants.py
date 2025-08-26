#================== Paths to CSV files ======================
DATA_FOLDER_PATH = './DATA/LFW-a'  # Path to LFW-a dataset
TRAIN_FILE_PATH = './DATA/pairsDevTrain.txt'  # Path to train.txt
TEST_FILE_PATH = './DATA/pairsDevTest.txt'  # Path to test.txt
SAVE_IMG_DIR_PATH = './images'  # Path to save images directory
#============================================================

#================== Image and Data Configuration ==================
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

# Better architecture parameters
IMPROVED_NUM_OF_FILTERS = {
    'layer1': 64,
    'layer2': 128,
    'layer3': 256,
    'layer4': 512,
}

IMPROVED_KERNEL_SIZES = {
    'layer1': (5, 5),   # Instead of (10, 10)
    'layer2': (5, 5),   # Instead of (7, 7)
    'layer3': (3, 3),   # Instead of (4, 4)
    'layer4': (3, 3),   # Instead of (4, 4)
}
STRIDE = 2

#================== Training Configuration =================
# General Training Parameters
EARLY_STOPPING_PATIENCE = 15
CLASSIFICATION_THRESHOLD = 0.5

#================== Data Augmentation ====================
# Probability thresholds for each augmentation
HORIZONTAL_FLIP_THRESHOLD = 0.5         # 50% chance
BRIGHTNESS_ADJUST_THRESHOLD = 0.3       # 30% chance
GAUSSIAN_NOISE_THRESHOLD = 0.7          # 70% chance
ROTATION_THRESHOLD = 0.6
CONTRAST_THRESHOLD = 0.5

# Brightness adjustment parameters
BRIGHTNESS_MIN_FACTOR = 0.7             # Minimum brightness (80%)
BRIGHTNESS_MAX_FACTOR = 1.4             # Maximum brightness (120%)

# Gaussian noise parameters
NOISE_MEAN = 0                          # Mean of Gaussian noise
NOISE_STD = 0.02                        # Standard deviation of noise

# Pixel value limits
PIXEL_MIN_VALUE = 0                     # Minimum pixel value after clipping
PIXEL_MAX_VALUE = 1                     # Maximum pixel value after clipping
