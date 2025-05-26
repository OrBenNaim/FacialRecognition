import os
import matplotlib.pyplot as plt

def plot_distribution_charts(train_val_dist: dict, test_dist: dict, save_dir: str = "images") -> None:
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

    # Plot Training + Validation distribution
    plt.figure(figsize=(10, 6))
    plt.bar(list(train_val_dist.keys()), 
            list(train_val_dist.values()),
            color='skyblue')
    plt.title('Images per Person Distribution (Training + Validation Set)')
    plt.xlabel('Number of Images')
    plt.ylabel('Number of People')
    
    # Add value labels on top of each bar
    for i, v in enumerate(train_val_dist.values()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_val_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Test distribution
    plt.figure(figsize=(10, 6))
    plt.bar(list(test_dist.keys()),
            list(test_dist.values()),
            color='lightgreen')
    plt.title('Images per Person Distribution (Test Set)')
    plt.xlabel('Number of Images')
    plt.ylabel('Number of People')
    
    # Add value labels on top of each bar
    for i, v in enumerate(test_dist.values()):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Distribution plots saved in: {os.path.abspath(save_dir)}")