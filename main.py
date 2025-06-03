from src.constants import OPTIMIZED_IMG_SHAPE, DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT
from src.face_recognition_model import FaceRecognition

if __name__ == '__main__':
    print("Siamese Network Implementation for One-shot Face Recognition")

    # Run multiple experiments
    experiments = [
        {"lr": 6e-5, "batch_size": 32, "epochs": 50},
        {"lr": 3e-4, "batch_size": 32, "epochs": 50},
    ]

    for exp in experiments:

        # Update constants
        learning_rate= exp["lr"]
        batch_size = exp["batch_size"]
        epochs = exp["epochs"]

        # Create a model pipeline
        model = FaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE, learning_rate=learning_rate,
                                               batch_size=batch_size, epochs=epochs)

        # Load and prepare Train dataset
        model.load_lfw_dataset(DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT)

        # Run experiment
        model.run_complete_experiment(use_improved_arch=True)

        print(f"\n✅ Completed {model.experiment_name}")
