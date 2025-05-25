from src.constants import OPTIMIZED_IMG_SHAPE
from src.utils import analyze_dataset_distribution

if __name__ == '__main__':
    print("Siamese Network Implementation for One-shot Face Recognition")

    # Analyze dataset first
    train_dist, test_dist = analyze_dataset_distribution()

    # Create and run the model
    siamese_model = SiameseFaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)

    # siamese_model.run_complete_experiment(DATA_PATH, TRAIN_FILE, TEST_FILE)

    print("\nTo run the experiment, update the paths above and uncomment the execution lines!")
