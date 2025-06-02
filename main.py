from src.constants import OPTIMIZED_IMG_SHAPE, DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT, LEARNING_RATE, \
    BATCH_SIZE, EPOCHS
from src.face_recognition_model import SiameseFaceRecognition

if __name__ == '__main__':
    print("Siamese Network Implementation for One-shot Face Recognition")

    # Run multiple experiments
    experiments = [
        {"lr": 6e-5, "batch_size": 32, "epochs": 10},
        {"lr": 3e-5, "batch_size": 64, "epochs": 20},
    ]

    for exp in experiments:

        # Update constants
        LEARNING_RATE = exp["lr"]
        BATCH_SIZE = exp["batch_size"]
        EPOCHS = exp["epochs"]

        # Create a model pipeline
        siamese_model = SiameseFaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE,
                                               experiment_name=f"lr{LEARNING_RATE}_bs{BATCH_SIZE}_epochs{EPOCHS}")

        # Load and prepare Train dataset
        siamese_model.load_lfw_dataset(DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT)

        # Analyze dataset first
        siamese_model.log_dataset_analysis_to_tensorboard()

        # Run experiment
        siamese_model.run_complete_experiment()

        print(f"✅ Completed {siamese_model.experiment_name}")
