import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from src.constants import (
    SAVED_IMG_DIR_PATH,
    HORIZONTAL_FLIP_THRESHOLD,
    BRIGHTNESS_ADJUST_THRESHOLD,
    GAUSSIAN_NOISE_THRESHOLD,
    BRIGHTNESS_MIN_FACTOR,
    BRIGHTNESS_MAX_FACTOR,
    NOISE_MEAN,
    NOISE_STD,
    PIXEL_MIN_VALUE,
    PIXEL_MAX_VALUE, DATA_PATH_FOLDER, TRAIN_FILE_PATH, TEST_FILE_PATH
)

def plot_distribution_charts(train_val_dist: dict, save_dir: str = SAVED_IMG_DIR_PATH) -> None:
    """
    Create and save distribution plots for the dataset.

    Parameters
    ----------
    train_val_dist : dict
        Distribution of images per person in training/validation set

    save_dir : str
        Directory to save the plots
    """
    # Create a directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Sort dictionaries by key (number of images) to ensure correct order in plots
    # This ensures that the x-axis shows 1, 2, 3, etc. in order
    sorted_train_val = dict(sorted(train_val_dist.items()))

    # ===== Training + Validation Distribution Plot =====
    plt.figure(figsize=(12, 6))  # Create a new figure with a specified size

    # Create x-axis positions starting from 1 instead of 0
    # This aligns the bars with their actual values (1 image, 2 images, etc.)
    x_positions = range(1, len(sorted_train_val) + 1)

    # Create bar plot with specified color
    plt.bar(x_positions,
            list(sorted_train_val.values()),
            color='skyblue')

    # Set plot title and axis labels
    plt.title('Images per Person Distribution (Training + Validation Set)')
    plt.xlabel('Number of Images per Person')
    plt.ylabel('Number of People')

    # Set x-ticks to show the actual number of images
    # This ensures that the x-axis shows "1", "2", "3" etc.
    plt.xticks(x_positions, [str(x) for x in sorted_train_val.keys()])

    # Add value labels on top of each bar
    # i+1 are used because we want positions starting from 1, not 0
    for i, v in enumerate(sorted_train_val.values()):
        plt.text(i + 1, v, str(v), ha='center', va='bottom')

    # Adjust the layout to prevent label cutoff
    plt.tight_layout()
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'train_val_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Distribution plots saved in: {os.path.abspath(save_dir)}")

def move_data_to_appropriate_device(img1, img2, labels, device):
    """
    Moves input tensors to the specified device (CPU/GPU).

    Args:
        img1 (torch.Tensor): First image batch tensor
        img2 (torch.Tensor): Second image batch tensor
        labels (torch.Tensor): Corresponding labels tensor
        device (torch.device): Target device to move the data to

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Triple of tensors (img1, img2, labels)
            moved to the specified device

    Note:
        This helper function is used to ensure consistent device placement
        of input data for model training and evaluation
    """

    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
    return img1, img2, labels

def apply_simple_augmentation(image_array):
    """
    Apply simple data augmentation to face images
    Updated to use constants from constants.py

    Args:
        image_array: numpy array of shape (H, W)

    Returns:
        augmented image array
    """
    # Random horizontal flip
    if np.random.random() > HORIZONTAL_FLIP_THRESHOLD:
        image_array = np.fliplr(image_array)

    # Random brightness adjustment
    if np.random.random() > BRIGHTNESS_ADJUST_THRESHOLD:
        brightness_factor = np.random.uniform(BRIGHTNESS_MIN_FACTOR, BRIGHTNESS_MAX_FACTOR)
        image_array = np.clip(image_array * brightness_factor, PIXEL_MIN_VALUE, PIXEL_MAX_VALUE)

    # Add a small amount of Gaussian noise
    if np.random.random() > GAUSSIAN_NOISE_THRESHOLD:
        noise = np.random.normal(NOISE_MEAN, NOISE_STD, image_array.shape)
        image_array = np.clip(image_array + noise, PIXEL_MIN_VALUE, PIXEL_MAX_VALUE)

    return image_array

def run_multiple_experiments(model):
    # Run multiple experiments
        experiments = [
            # Enhanced base network with BatchNorm, Dropout, and smaller kernels
            {"architecture_type": "improved", "lr": 6e-5, "batch_size": 32, "epochs": 50},

            # Base convolutional network for feature extraction in Siamese architecture
            {"architecture_type": "base", "lr": 6e-5, "batch_size": 32, "epochs": 50},
        ]

        for exp in experiments:
            model.reset_exp_attr()

            architecture_type = exp["architecture_type"]
            learning_rate = exp["lr"]
            batch_size = exp["batch_size"]
            epochs = exp["epochs"]

            exp_name = f"lr{learning_rate}_bs{batch_size}_epochs{epochs}"

            use_improved_arch = None

            if architecture_type == "base":
                use_improved_arch = False
                exp_name += "_base_arch"

            elif architecture_type == "improved":
                use_improved_arch = True
                exp_name += "_improved_arch"    # Enhanced base network with BatchNorm, Dropout, and smaller kernels

            elif architecture_type is None:
                raise ValueError("Please specify the architecture type")

            # Run experiment
            model.run_complete_experiment(learning_rate=learning_rate,batch_size=batch_size,epochs=epochs,
                                          use_improved_arch=use_improved_arch, exp_name=exp_name)

            print(f"\n✅ Completed {model.experiment_name}")

def print_header():
    """Print welcome header."""
    print("🚀" + "=" * 70 + "🚀")
    print("    SIAMESE NEURAL NETWORK - COMPLETE PIPELINE")
    print("    One-Shot Facial Recognition Implementation")
    print("🚀" + "=" * 70 + "🚀")

def print_phase_separator(phase_name):
    """Print phase separator."""
    print("\n" + "🔄" * 20)
    print(f"    {phase_name}")
    print("🔄" * 20 + "\n")

def verify_prerequisites():
    """Verify all prerequisites before starting."""
    print("🔍 Verifying prerequisites...")

    required_files = [
        DATA_PATH_FOLDER,
        TRAIN_FILE_PATH,
        TEST_FILE_PATH
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False

    # Create the necessary directories
    os.makedirs("tensorboard_logs", exist_ok=True)
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    print("✅ All prerequisites satisfied")
    return True

def clear_training_data_cache(cache_file: str) -> None:
    """Clear training data cache."""
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"🗑️  Cleared cache: {cache_file}")
    else:
        print("ℹ️  No cache to clear")

def get_cache_info(cache_file: str) -> dict:
    """Get cache information."""
    if not os.path.exists(cache_file):
        return {}

    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)

        file_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB

        return {
            'exists': True,
            'created_time': cache_data.get('created_time', 'Unknown'),
            'train_count': cache_data.get('train_count', 0),
            'val_count': cache_data.get('val_count', 0),
            'people_count': cache_data.get('people_count', 0),
            'input_shape': cache_data.get('input_shape', 'Unknown'),
            'validation_split': cache_data.get('validation_split', 'Unknown'),
            'file_size_mb': f"{file_size:.1f}"
        }

    except:
        return {'exists': False, 'error': 'Corrupted cache file'}

def generate_final_summary(test_results, total_time):
    """Generate final pipeline summary."""
    print("\n" + "🏆" + "=" * 70 + "🏆")
    print("    COMPLETE PIPELINE SUMMARY")
    print("🏆" + "=" * 70 + "🏆")

    print(f"\n⏱️  TIMING:")
    print(f"   Total Pipeline Time: {total_time / 60:.2f} minutes")

    if test_results:
        print(f"\n📊 FINAL TEST RESULTS:")
        print(f"   Test Accuracy:      {test_results['accuracy']:.4f} ({test_results['accuracy'] * 100:.2f}%)")
        print(f"   True Positive Rate: {test_results['tpr']:.4f} ({test_results['tpr'] * 100:.2f}%)")
        print(f"   True Negative Rate: {test_results['tnr']:.4f} ({test_results['tnr'] * 100:.2f}%)")
        print(f"   F1 Score:          {test_results['f1_score']:.4f}")
        print(f"   AUC Score:         {test_results['auc']:.4f}")

        # Performance assessment
        validation_expected = 0.807  # Your enhanced augmentation result
        performance_drop = validation_expected - test_results['accuracy']

        print(f"\n📈 GENERALIZATION ANALYSIS:")
        print(f"   Expected Validation: {validation_expected:.4f}")
        print(f"   Actual Test:        {test_results['accuracy']:.4f}")
        print(f"   Performance Drop:   {performance_drop:.4f} ({performance_drop * 100:.2f}%)")

        if performance_drop < 0.03:
            assessment = "EXCELLENT ✨"
        elif performance_drop < 0.07:
            assessment = "GOOD ✅"
        elif performance_drop < 0.12:
            assessment = "MODERATE 📊"
        else:
            assessment = "NEEDS IMPROVEMENT ⚠️"

        print(f"   Assessment:         {assessment}")

    print(f"\n📁 OUTPUTS GENERATED:")
    print(f"   Model File:         best_model.pth")
    print(f"   TensorBoard Logs:   tensorboard_logs/")
    print(f"   Test Report:        test_results/final_test_evaluation_report.txt")
    print(f"   Dataset Analysis:   images/")

    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. View training curves:    tensorboard --logdir=tensorboard_logs")
    print(f"   2. Read detailed report:    test_results/final_test_evaluation_report.txt")
    print(f"   3. Analyze misclassifications in TensorBoard")

    print("\n" + "🎉" + "=" * 70 + "🎉")
    print("    SIAMESE NETWORK PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉" + "=" * 70 + "🎉")
