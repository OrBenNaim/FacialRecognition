import os
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.constants import OPTIMIZED_IMG_SHAPE, DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT
from src.face_recognition_model import FaceRecognition


def run_training_pipeline():
    """
    Complete training pipeline for Siamese Neural Network.

    Steps:
    1. Load and analyze training dataset
    2. Create Siamese network architecture
    3. Train model with enhanced augmentation
    4. Save the best model for testing
    5. Generate training reports
    """

    print("🚀 SIAMESE NETWORK TRAINING PIPELINE")
    print("=" * 60)

    # ========= Step 1: Initialize Model =========
    print("\n📋 Step 1: Initializing model...")
    model = FaceRecognition(input_shape=OPTIMIZED_IMG_SHAPE)
    print(f"✅ Model initialized with input shape: {OPTIMIZED_IMG_SHAPE}")

    # ========= Step 2: Load Training Dataset (with caching) =========
    print(f"\n📁 Step 2: Loading training dataset...")

    # Try to load from cache first
    if model.load_preprocessed_data():
        print(f"✅ Dataset loaded from cache with {VALIDATION_SPLIT * 100:.0f}% validation split")
    else:
        print("🔄 No cache found or cache invalid. Loading and preprocessing data...")
        model.load_lfw_dataset(DATA_FOLDER_PATH, TRAIN_FILE_PATH, VALIDATION_SPLIT)
        print(f"✅ Dataset loaded and preprocessed with {VALIDATION_SPLIT * 100:.0f}% validation split")

        # Save preprocessed data for future use
        print("💾 Caching preprocessed data for future training runs...")
        model.save_preprocessed_data()
        print("✅ Data cached successfully")

    # ========= Step 3: Dataset Analysis =========
    print("\n📊 Step 3: Analyzing dataset...")
    analysis_log_dir = os.path.join("tensorboard_logs", "dataset_analysis")
    os.makedirs(analysis_log_dir, exist_ok=True)

    model.tensorboard_log_dir = analysis_log_dir
    model.writer = SummaryWriter(log_dir=analysis_log_dir)
    model.log_dataset_analysis_to_tensorboard()
    model.writer.close()
    print("✅ Dataset analysis completed")

    # ========= Step 4: Configure Training =========
    print("\n⚙️  Step 4: Configuring training parameters...")

    # Training configuration - your optimized parameters
    training_config = {
        "learning_rate": 6e-5,  # Your optimized LR
        "batch_size": 32,  # Your optimized batch size
        "epochs": 50,  # Max epochs with early stopping
        "architecture": "base"  # Base architecture with enhanced augmentation
    }

    print(f"Learning Rate: {training_config['learning_rate']}")
    print(f"Batch Size: {training_config['batch_size']}")
    print(f"Max Epochs: {training_config['epochs']}")
    print(f"Architecture: {training_config['architecture']} + enhanced augmentation")

    # ========= Step 5: Setup Experiment =========
    print("\n🔧 Step 5: Setting up training experiment...")

    model.reset_exp_attr()

    # Create timestamped experiment name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"siamese_lr{training_config['learning_rate']}_bs{training_config['batch_size']}_enhanced_aug_{timestamp}"

    print(f"✅ Experiment: {exp_name}")

    # ========= Step 6: Train Model =========
    print("\n🎯 Step 6: Training model...")
    print("=" * 50)

    try:
        model.run_complete_experiment(
            learning_rate=training_config["learning_rate"],
            batch_size=training_config["batch_size"],
            epochs=training_config["epochs"],
            use_improved_arch=False,  # Base architecture
            exp_name=exp_name
        )

        print("=" * 50)
        print("✅ Training completed successfully!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

    # ========= Step 7: Verify Model Saved =========
    print(f"\n💾 Step 7: Verifying model save...")
    if os.path.exists("best_model.pth"):
        model_size = os.path.getsize("best_model.pth") / (1024 * 1024)
        print(f"✅ Model saved: best_model.pth ({model_size:.2f} MB)")
    else:
        print("❌ Warning: Model file not found!")

    # ========= Step 8: Training Summary =========
    print(f"\n📈 Step 8: Training Summary")
    print("=" * 40)
    print(f"Experiment: {model.experiment_name}")

    if hasattr(model, 'convergence_results'):
        conv_time = model.convergence_results.get('convergence_time_minutes', 'N/A')
        total_time = model.convergence_results.get('total_training_time_minutes', 'N/A')
        print(f"Convergence time: {conv_time:.2f} minutes")
        print(f"Total training time: {total_time:.2f} minutes")

    # ========= Step 9: Next Steps =========
    print(f"\n🎯 Next Steps:")
    print(f"1. View training: tensorboard --logdir=tensorboard_logs")
    print(f"2. Run test evaluation: python main_testing.py")
    print(f"3. Check validation performance in TensorBoard")

    print(f"\n🏁 TRAINING PIPELINE COMPLETED")
    print("=" * 60)

    return model


def verify_prerequisites():
    """Verify all required files exist before training."""
    print("🔍 Verifying prerequisites...")

    required_files = [DATA_FOLDER_PATH, TRAIN_FILE_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Required file missing: {file_path}")
            return False

    # Create directories
    os.makedirs("tensorboard_logs", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    print("✅ Prerequisites verified")
    return True


if __name__ == '__main__':
    if not verify_prerequisites():
        print("❌ Prerequisites not met. Check your data files.")
        exit(1)

    # Run a training pipeline
    trained_model = run_training_pipeline()