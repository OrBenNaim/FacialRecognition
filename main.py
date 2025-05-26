from src.constants import OPTIMIZED_IMG_SHAPE, DATA_PATH, TRAIN_FILE, TEST_FILE, VALIDATION_SPLIT
from src.face_recognition_model import SiameseFaceRecognition

if __name__ == '__main__':
    print("Siamese Network Implementation for One-shot Face Recognition")

    # Create a model pipeline
    siamese_model = SiameseFaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)

    siamese_model.load_lfw_dataset(DATA_PATH, TRAIN_FILE,
                                   TEST_FILE, VALIDATION_SPLIT)

    # Analyze dataset first
    siamese_model.analyze_dataset_distribution()


    # train_person_images, test_person_images = siamese_model.load_lfw_dataset(DATA_PATH, TRAIN_FILE,
    #                                                                          TEST_FILE, VALIDATION_SPLIT)

    # Run the model
    # siamese_model.run_complete_experiment(DATA_PATH, TRAIN_FILE, TEST_FILE)

    print("\nTo run the experiment, update the paths above and uncomment the execution lines!")