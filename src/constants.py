#================== Paths to CSV files ======================
DATA_PATH = './DATA/LFW-a'  # Path to LFW-a dataset
TRAIN_FILE = './DATA/pairsDevTrain.txt'  # Path to train.txt
TEST_FILE = './DATA/pairsDevTest.txt'  # Path to test.txt
#============================================================

# Each image was a gray scale image with size of 250x250
# Optimized image size is 128x128 pixels
OPTIMIZED_IMG_SHAPE = (128, 128, 1)

# The relative size of the validation set
VALIDATION_SPLIT = 0.2

# Sets random seed to ensure reproducible and consistent results across runs
RANDOM_SEED = 42

# Number of complete passes through the training data
EPOCHS = 20

# Number of image pairs processed together before updating weights
BATCH_SIZE = 32

# Learning Rate
LEARNING_RATE = 0.00006

# Number of filters for each layer of Base Network (CNN)
NUM_OF_FILTERS_LAYER1 = 64      # Enough to capture basic face features
NUM_OF_FILTERS_LAYER2 = 128     # Double the capacity for complex patterns
NUM_OF_FILTERS_LAYER3 = 128     # Double the capacity for complex patterns
NUM_OF_FILTERS_LAYER4 = 256     # Maximum capacity before final embedding

#
KERNAL_SIZE_LAYER1 = (10, 10)   # Large kernel captures broader spatial patterns
KERNAL_SIZE_LAYER2 = (7, 7)     # Medium kernel for a feature combination
KERNAL_SIZE_LAYER3 = (4, 4)     # Smaller kernel for fine details
KERNAL_SIZE_LAYER4 = (4, 4)     # Smaller kernel for fine details
POOL_SIZE = (2, 2)