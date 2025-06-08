import os

from torch.utils.tensorboard import SummaryWriter

from src.constants import OPTIMIZED_IMG_SHAPE, DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT
from src.face_recognition_model import FaceRecognition

if __name__ == '__main__':
    print("\nSiamese Network Implementation for One-shot Face Recognition\n")

    # Create a model pipeline
    model = FaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)

    # Load and prepare Train dataset
    model.load_lfw_dataset(DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT)

    model.tensorboard_log_dir = os.path.join("tensorboard_logs", "dataset_analysis")
    os.makedirs(model.tensorboard_log_dir, exist_ok=True)
    model.writer = SummaryWriter(log_dir=model.tensorboard_log_dir)

    # Log dataset analysis
    print("\n📊 Analyzing dataset...")
    model.log_dataset_analysis_to_tensorboard()

    # # Run multiple experiments
    # experiments = [
    #     # Enhanced base network with BatchNorm, Dropout, and smaller kernels
    #     {"architecture_type": "improved", "lr": 6e-5, "batch_size": 32, "epochs": 50},
    #
    #     # Base convolutional network for feature extraction in Siamese architecture
    #     {"architecture_type": "base", "lr": 6e-5, "batch_size": 32, "epochs": 50},
    # ]
    #
    # for exp in experiments:
    #     model.reset_exp_attr()
    #
    #     architecture_type = exp["architecture_type"]
    #     learning_rate = exp["lr"]
    #     batch_size = exp["batch_size"]
    #     epochs = exp["epochs"]
    #
    #     exp_name = f"lr{learning_rate}_bs{batch_size}_epochs{epochs}"
    #
    #     use_improved_arch = None
    #
    #     if architecture_type == "base":
    #         use_improved_arch = False
    #         exp_name += "_base_arch"
    #
    #     elif architecture_type == "improved":
    #         use_improved_arch = True
    #         exp_name += "_improved_arch"    # Enhanced base network with BatchNorm, Dropout, and smaller kernels
    #
    #     elif architecture_type is None:
    #         raise ValueError("Please specify the architecture type")
    #
    #     # Run experiment
    #     model.run_complete_experiment(learning_rate=learning_rate,batch_size=batch_size,epochs=epochs,
    #                                   use_improved_arch=use_improved_arch, exp_name=exp_name)
    #
    #     print(f"\n✅ Completed {model.experiment_name}")

    #============ Decided to move on with 'Enhanced base network' ============

    # Run multiple experiments
    experiments = [
        # Enhanced base network with BatchNorm, Dropout, and smaller kernels
        {"architecture_type": "base", "lr": 6e-5, "batch_size": 32, "epochs": 50},
    ]

    for exp in experiments:
        model.reset_exp_attr()

        architecture_type = exp["architecture_type"]
        learning_rate = exp["lr"]
        batch_size = exp["batch_size"]
        epochs = exp["epochs"]

        exp_name = f"lr{learning_rate}_bs{batch_size}_epochs{epochs}_BCE"

        use_improved_arch = None

        if architecture_type == "base":
            use_improved_arch = False
            exp_name += "_base_arch"

        else:
            raise ValueError("Architecture type must be 'base' for this experiment. Please try again.")

        # Run experiment
        model.run_complete_experiment(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs,
                                      use_improved_arch=use_improved_arch, exp_name=exp_name)

        print(f"\n✅ Completed {model.experiment_name}")
