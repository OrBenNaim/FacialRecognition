import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate

from src.constants import (
    SAVE_IMG_DIR_PATH,
    HORIZONTAL_FLIP_THRESHOLD,
    BRIGHTNESS_ADJUST_THRESHOLD,
    GAUSSIAN_NOISE_THRESHOLD,
    BRIGHTNESS_MIN_FACTOR,
    BRIGHTNESS_MAX_FACTOR,
    NOISE_MEAN,
    NOISE_STD,
    PIXEL_MIN_VALUE,
    PIXEL_MAX_VALUE
)

def plot_distribution_charts(train_val_dist: dict, save_dir: str = SAVE_IMG_DIR_PATH) -> None:
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
