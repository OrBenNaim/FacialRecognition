import os
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_distribution_charts(train_val_dist: dict, test_dist: dict, save_dir: str = "./src/images") -> None:
    """
    Create and save distribution plots for the dataset.

    Parameters
    ----------
    train_val_dist : dict
        Distribution of images per person in training/validation set
    test_dist : dict
        Distribution of images per person in test set
    save_dir : str
        Directory to save the plots
    """
    # Create a directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Sort dictionaries by key (number of images) to ensure correct order in plots
    # This ensures that the x-axis shows 1, 2, 3, etc. in order
    sorted_train_val = dict(sorted(train_val_dist.items()))
    sorted_test = dict(sorted(test_dist.items()))

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

    # ===== Test Distribution Plot =====
    # Same process as above, but for test distribution
    plt.figure(figsize=(12, 6))
    x_positions = range(1, len(sorted_test) + 1)  # Start x-axis from 1
    plt.bar(x_positions,
            list(sorted_test.values()),
            color='lightgreen')

    plt.title('Images per Person Distribution (Test Set)')
    plt.xlabel('Number of Images per Person')
    plt.ylabel('Number of People')

    # Set x-ticks to show the actual number of images
    plt.xticks(x_positions, [str(x) for x in sorted_test.keys()])

    # Add value labels on top of each bar
    for i, v in enumerate(sorted_test.values()):
        plt.text(i + 1, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Distribution plots saved in: {os.path.abspath(save_dir)}")

def analyze_results(history: Dict[str, List[float]]) -> None:
    """
    Analyze and visualize training results as required by the exercise.
    """
    # Create a figure for multiple plots
    plt.figure(figsize=(15, 10))

    # Plot 1: Training and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot 2: Training and Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save plots
    plt.tight_layout()
    plt.savefig('./src/images/training_results.png')