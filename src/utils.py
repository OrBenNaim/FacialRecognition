import os
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

    # Sort the dictionaries by number of images (key)
    sorted_train_val = dict(sorted(train_val_dist.items()))
    sorted_test = dict(sorted(test_dist.items()))

    # Plot Training + Validation distribution
    plt.figure(figsize=(12, 6))
    x_positions = range(1, len(sorted_train_val) + 1)  # Start x-axis from 1
    plt.bar(x_positions, 
            list(sorted_train_val.values()),
            color='skyblue')
    
    plt.title('Images per Person Distribution (Training + Validation Set)')
    plt.xlabel('Number of Images per Person')
    plt.ylabel('Number of People')
    
    # Set x-ticks to show actual number of images
    plt.xticks(x_positions, [str(x) for x in sorted_train_val.keys()])
    
    # Add value labels on top of each bar
    for i, v in enumerate(sorted_train_val.values()):
        plt.text(i + 1, v, str(v), ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_val_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Test distribution
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