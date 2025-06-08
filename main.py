import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.constants import OPTIMIZED_IMG_SHAPE, DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT, TEST_FILE_PATH, \
    CLASSIFICATION_THRESHOLD
from src.face_recognition_model import FaceRecognition

if __name__ == '__main__':
    # print("\nSiamese Network Implementation for One-shot Face Recognition\n")
    #
    # # Create a model pipeline
    # model = FaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)
    #
    # # Load and prepare Train dataset
    # model.load_lfw_dataset(DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT)
    #
    # model.tensorboard_log_dir = os.path.join("tensorboard_logs", "dataset_analysis")
    # os.makedirs(model.tensorboard_log_dir, exist_ok=True)
    # model.writer = SummaryWriter(log_dir=model.tensorboard_log_dir)
    #
    # # Log dataset analysis
    # print("\n📊 Analyzing dataset...")
    # model.log_dataset_analysis_to_tensorboard()

    #run_multiple_experiments(model=model)

    #============ Decided to move on with 'Base with Aug network' ============

    # Run chosen experiment
    # experiment = {"architecture_type": "base", "lr": 6e-5, "batch_size": 32, "epochs": 50}
    #
    # model.reset_exp_attr()
    #
    # architecture_type = experiment["architecture_type"]
    # learning_rate = experiment["lr"]
    # batch_size = experiment["batch_size"]
    # epochs = experiment["epochs"]
    #
    # exp_name = f"lr{learning_rate}_bs{batch_size}_epochs{epochs}_BCE"
    #
    # use_improved_arch = None
    #
    # if architecture_type == "base":
    #     use_improved_arch = False
    #     exp_name += "_base_arch"
    #
    # else:
    #     raise ValueError("Architecture type must be 'base' for this experiment. Please try again.")
    #
    # # Run experiment
    # model.run_complete_experiment(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs,
    #                               use_improved_arch=use_improved_arch, exp_name=exp_name)
    #
    # print(f"\n✅ Completed {model.experiment_name}")

    #========= Run Model on Test Dataset =========
    print("🧪 Starting Test Phase...")


    # Create a new model instance for testing
    test_model = FaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)

    test_model.load_lfw_dataset(DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT)

    # Load test dataset (no validation split needed for test)
    print("📁 Loading test dataset...")
    test_model.load_lfw_dataset(DATA_FOLDER_PATH, TEST_FILE_PATH)

    # Set up TensorBoard logging for test results
    test_model.tensorboard_log_dir = os.path.join("tensorboard_logs", "test_evaluation")
    os.makedirs(test_model.tensorboard_log_dir, exist_ok=True)
    test_model.writer = SummaryWriter(log_dir=test_model.tensorboard_log_dir)

    experiment = {"architecture_type": "base", "lr": 6e-5, "batch_size": 32, "epochs": 50}

    test_model.reset_exp_attr()

    architecture_type = experiment["architecture_type"]
    learning_rate = experiment["lr"]
    batch_size = experiment["batch_size"]
    epochs = experiment["epochs"]

    exp_name = f"lr{learning_rate}_bs{batch_size}_epochs{epochs}_BCE"

    use_improved_arch = None

    if architecture_type == "base":
        use_improved_arch = False
        exp_name += "_base_arch"

    else:
        raise ValueError("Architecture type must be 'base' for this experiment. Please try again.")

    test_model.run_complete_experiment(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs,
                                       use_improved_arch=use_improved_arch, exp_name=exp_name)

    # Log test dataset analysis
    print("📊 Analyzing test dataset...")
    test_model.log_test_dataset_analysis()

    # Load the best trained model
    print("🔄 Loading best trained model...")
    test_model.run_model_on_test_dataset()
