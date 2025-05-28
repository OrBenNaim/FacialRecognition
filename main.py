from src.constants import OPTIMIZED_IMG_SHAPE, DATA_PATH, TRAIN_FILE, TEST_FILE, VALIDATION_SPLIT
from src.face_recognition_model_old_version import SiameseFaceRecognition

if __name__ == '__main__':
    print("Siamese Network Implementation for One-shot Face Recognition")

    # Create a model pipeline
    siamese_model = SiameseFaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)

    # Load and prepare dataset
    siamese_model.load_lfw_dataset(DATA_PATH, TRAIN_FILE,
                                   TEST_FILE, VALIDATION_SPLIT)

    # Analyze dataset first
    siamese_model.analyze_dataset_distribution()

    # Run the complete experiment
    siamese_model.run_complete_experiment()
