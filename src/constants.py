#================== Paths to CSV files ======================
DATA_PATH = './DATA/LFW-a'  # Path to LFW-a dataset
TRAIN_FILE = './DATA/pairsDevTrain.txt'  # Path to train.txt
TEST_FILE = './DATA/pairsDevTest.txt'  # Path to test.txt
#============================================================

# Each image was a gray scale image with size of 250x250
# Optimized image size is 128x128 pixels
OPTIMIZED_IMG_SHAPE= (128, 128, 1)

# The relative size of the validation set
VALIDATION_SPLIT = 0.2

# Sets random seed to ensure reproducible and consistent results across runs
RANDOM_SEED=42
