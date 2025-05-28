# Standard library imports
import os
import warnings
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict, Counter

# Third-party imports
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local imports
from src.constants import (
    RANDOM_SEED, EPOCHS, BATCH_SIZE,
    NUM_OF_FILTERS_LAYER1, NUM_OF_FILTERS_LAYER2,
    NUM_OF_FILTERS_LAYER3, NUM_OF_FILTERS_LAYER4,
    KERNAL_SIZE_LAYER1, KERNAL_SIZE_LAYER2,
    KERNAL_SIZE_LAYER3, KERNAL_SIZE_LAYER4,
    POOL_SIZE, LEARNING_RATE
)
from src.utils import plot_distribution_charts

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SiameseDataset(Dataset):
    """Custom Dataset for Siamese Network pairs."""

    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a pair of images and label
        img1 = torch.FloatTensor(self.pairs[idx, 0])
        img2 = torch.FloatTensor(self.pairs[idx, 1])
        label = torch.FloatTensor([self.labels[idx]])
        return img1, img2, label


class BaseNetwork(nn.Module):
    """Base convolutional network for feature extraction in Siamese architecture."""

    def __init__(self, input_shape):
        super(BaseNetwork, self).__init__()

        # Calculate the size after convolutions and pooling
        # This is needed to determine the input size for the final dense layer
        self.input_shape = input_shape

        # Layer 1: Conv(64, 10x10) -> ReLU -> MaxPool(2x2)
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[2],  # channels (1 for grayscale)
            out_channels=NUM_OF_FILTERS_LAYER1,
            kernel_size=KERNAL_SIZE_LAYER1,
            padding=0
        )
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_SIZE)

        # Layer 2: Conv(128, 7x7) -> ReLU -> MaxPool(2x2)
        self.conv2 = nn.Conv2d(
            in_channels=NUM_OF_FILTERS_LAYER1,
            out_channels=NUM_OF_FILTERS_LAYER2,
            kernel_size=KERNAL_SIZE_LAYER2,
            padding=0
        )
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_SIZE)

        # Layer 3: Conv(128, 4x4) -> ReLU -> MaxPool(2x2)
        self.conv3 = nn.Conv2d(
            in_channels=NUM_OF_FILTERS_LAYER2,
            out_channels=NUM_OF_FILTERS_LAYER3,
            kernel_size=KERNAL_SIZE_LAYER3,
            padding=0
        )
        self.pool3 = nn.MaxPool2d(kernel_size=POOL_SIZE)

        # Layer 4: Conv(256, 4x4) -> ReLU (no pooling)
        self.conv4 = nn.Conv2d(
            in_channels=NUM_OF_FILTERS_LAYER3,
            out_channels=NUM_OF_FILTERS_LAYER4,
            kernel_size=KERNAL_SIZE_LAYER4,
            padding=0
        )

        # Calculate the flattened size after all convolutions
        self._calculate_flatten_size()

        # Dense layer: 4096 units with sigmoid activation
        self.fc = nn.Linear(self.flatten_size, 4096)

        # Initialize weights
        self._initialize_weights()

    def _calculate_flatten_size(self):
        """Calculate the size of flattened features after convolutions."""
        # Create a dummy input to calculate sizes
        dummy_input = torch.zeros(1, self.input_shape[2], self.input_shape[0], self.input_shape[1])

        # Pass through convolutional layers
        x = self.pool1(F.relu(self.conv1(dummy_input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        # Get the flattened size
        self.flatten_size = x.view(1, -1).size(1)

    def _initialize_weights(self):
        """Initialize weights using Glorot uniform (Xavier uniform in PyTorch)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        # Convolutional layers with ReLU and pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layer with sigmoid activation
        x = torch.sigmoid(self.fc(x))

        return x


class SiameseNetwork(nn.Module):
    """Complete Siamese Network architecture."""

    def __init__(self, input_shape):
        super(SiameseNetwork, self).__init__()

        # Create the shared base network
        self.base_network = BaseNetwork(input_shape)

        # Final prediction layer
        self.prediction = nn.Linear(4096, 1)

        # Initialize the prediction layer
        nn.init.xavier_uniform_(self.prediction.weight)
        nn.init.constant_(self.prediction.bias, 0.5)

    def forward(self, input1, input2):
        """Forward pass for the Siamese network."""
        # Process both inputs through the same network (shared weights)
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)

        # L1 distance between embeddings
        l1_distance = torch.abs(output1 - output2)

        # Final prediction with sigmoid
        prediction = torch.sigmoid(self.prediction(l1_distance))

        return prediction


class SiameseFaceRecognition:
    """
    A Siamese Neural Network implementation for one-shot face recognition tasks using PyTorch.

    This class provides an end-to-end pipeline for:
    - Loading and preprocessing face image datasets
    - Building and training a Siamese neural network
    - Evaluating face recognition performance
    - Analyzing dataset distributions
    - Generating visualizations of results

    The Siamese architecture enables learning face similarity metrics from pairs of images,
    making it suitable for recognizing faces with limited training examples per person.
    """

    def __init__(self, input_shape):
        """
        Initialize the Siamese Network.

        Parameters
        ----------
        input_shape : tuple
            The target shape for input images as (height, width, channels).
            All loaded images will be resized to this shape.

        Note
        -----
            The input_shape parameter defines the size images will be resized to
            during loading, NOT the original size of your image files.
        """
        # Store the target shape for input images (height, width, channels)
        self.input_shape = input_shape

        # Initialize model components
        self.model: Optional[SiameseNetwork] = None
        self.optimizer: Optional[optim.Adam] = None
        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        self.history: Optional[Dict[str, List[float]]] = None

        # Initialize data storage for training/validation set
        self.train_val_person_images: Optional[Dict[str, List[str]]] = None
        self.train_val_image_dict: Optional[Dict[str, np.ndarray]] = None
        self.train_val_dist = None

        # Initialize data storage for a test set
        self.test_person_images: Optional[Dict[str, List[str]]] = None
        self.test_image_dict: Optional[Dict[str, np.ndarray]] = None
        self.test_dist = None

        # Initialize arrays for training data
        self.train_images: Optional[np.ndarray] = None
        self.train_labels: Optional[np.ndarray] = None
        self.train_pairs: Optional[np.ndarray] = None
        self.train_pair_labels: Optional[np.ndarray] = None

        # Initialize arrays for validation data
        self.val_images: Optional[np.ndarray] = None
        self.val_labels: Optional[np.ndarray] = None
        self.val_pairs: Optional[np.ndarray] = None
        self.val_pair_labels: Optional[np.ndarray] = None

        # Initialize arrays for test data
        self.test_images: Optional[np.ndarray] = None
        self.test_labels: Optional[np.ndarray] = None
        self.test_pairs: Optional[np.ndarray] = None
        self.test_pair_labels: Optional[np.ndarray] = None

        # Dictionary storing various statistics about the dataset and training
        self.stats: Dict[str, Any] = {}

        print(f"Siamese Face Recognition initialized with input shape: {input_shape}")

    # Load training and test datasets
    def load_lfw_dataset(self, data_path: str, train_file: str, test_file: str, validation_split: float) -> None:
        """
        Load and preprocess the Labeled Faces in the Wild (LFW) dataset.

        This method handles the complete data pipeline including
        - Loading image pairs from LFW pair files
        - Converting images to grayscale
        - Resizing images to the model's input shape
        - Normalizing pixel values to [0, 1]
        - Creating training/validation/test splits

        Parameters
        ----------
        data_path : str
            Root directory path containing LFW person subdirectories
        train_file : str
            Path to the training pairs file (e.g., pairsDevTrain.txt)
        test_file : str
            Path to the test pairs file (e.g., pairsDevTest.txt)
        validation_split : float
            Fraction of training data to use for validation (0.0 to 1.0)

        Raises
        ------
        FileNotFoundError
            If data_path, train_file, or test_file doesn't exist
        ValueError
            If validation_split is not between 0 and 1

        Notes
        -----
        - Expects LFW pair file format:
          - 3 values per line for the same person pairs: name, img1_num, img2_num
          - 4 values per line for different person pairs: name1, img1_num, name2, img2_num
        - Images are automatically resized to self.input_shape
        - Original image files are not modified
        """
        # === Input Validation ===
        # Ensure all required files and directories exist
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Train file not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        if not 0 <= validation_split <= 1:
            raise ValueError(f"validation_split must be between 0 and 1, got {validation_split}")

        print(f"Image resize target: {self.input_shape[0]}x{self.input_shape[1]}")
        print("Loading LFW-a dataset...")

        # First, load all unique images and create a mapping
        def load_all_images(pairs_file: str, base_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
            """
            Load and preprocess all unique images referenced in a pair's file.

            Parameters
            ----------
            pairs_file : str
                Path to the LFW pairs file to process
            base_path : str
                Root directory containing person subdirectories

            Returns
            -------
            tuple
                (image_dict, person_images) where:
                - image_dict: Dict[str, np.ndarray] mapping image keys to image arrays
                - person_images: Dict[str, List[str]] mapping person names to their image keys

            Notes
            -----
            - Images are converted to grayscale
            - Resized to model's input_shape
            - Pixel values normalized to [0, 1]
            - Skips missing images with a warning
            """
            image_dict = {}  # Maps: image_key -> preprocessed image array
            person_images = defaultdict(list)  # Maps: person_name -> list of their image keys

            with open(pairs_file, 'r') as f:
                lines = f.readlines()

                # Skip the first line which contains the number of pairs
                for i, line in enumerate(tqdm(lines[1:], desc="Loading images")):
                    parts = line.strip().split('\t')

                    # === Handle Same Person Pairs ===
                    if len(parts) == 3:  # Format: person_name, img1_num, img2_num
                        person_name = parts[0]
                        img1_num = int(parts[1])
                        img2_num = int(parts[2])

                        # Process both images from the same person
                        for img_num in [img1_num, img2_num]:
                            img_name = f"{person_name}_{img_num:04d}.jpg"
                            current_img_key = f"{person_name}_{img_num}"

                            # Only process new images (avoid duplicates)
                            if current_img_key not in image_dict:
                                img_path = os.path.join(base_path, person_name, img_name)

                                if os.path.exists(img_path):
                                    # Load and preprocess image:
                                    # 1. Convert to grayscale
                                    # 2. Resize to target dimensions
                                    # 3. Normalize pixel values to [0,1]
                                    img = Image.open(img_path).convert('L')
                                    img = img.resize((self.input_shape[0], self.input_shape[1]))
                                    img_array = np.array(img) / 255.0

                                    # Store preprocessed image and update person's image list
                                    image_dict[current_img_key] = img_array
                                    person_images[person_name].append(current_img_key)
                                else:
                                    raise f"Warning: Image not found: {img_path}"

                    # === Handle Different Person Pairs ===
                    elif len(parts) == 4:  # Format: person1, img1_num, person2, img2_num
                        person1, person2 = parts[0], parts[2]
                        img1_num, img2_num = int(parts[1]), int(parts[3])

                        # Load first person's image
                        img1_name = f"{person1}_{img1_num:04d}.jpg"
                        img1_key = f"{person1}_{img1_num}"

                        if img1_key not in image_dict:
                            img1_path = os.path.join(base_path, person1, img1_name)

                            if os.path.exists(img1_path):
                                # Same preprocessing steps as above
                                img = Image.open(img1_path).convert('L')
                                img = img.resize((self.input_shape[0], self.input_shape[1]))
                                img_array = np.array(img) / 255.0
                                image_dict[img1_key] = img_array
                                person_images[person1].append(img1_key)

                        # Load second person's image
                        img2_name = f"{person2}_{img2_num:04d}.jpg"
                        img2_key = f"{person2}_{img2_num}"

                        if img2_key not in image_dict:
                            img2_path = os.path.join(base_path, person2, img2_name)

                            if os.path.exists(img2_path):
                                img = Image.open(img2_path).convert('L')
                                img = img.resize((self.input_shape[0], self.input_shape[1]))
                                img_array = np.array(img) / 255.0

                                image_dict[img2_key] = img_array
                                person_images[person2].append(img2_key)

            return image_dict, person_images

        # Load pairs from a file
        def load_pairs(pairs_file: str, image_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            """
            Create image pairs and labels from a pair's definition file.

            Parameters
            ----------
            pairs_file : str
                Path to the LFW pairs file
            image_dict : Dict[str, np.ndarray]
                Dictionary mapping image keys to preprocessed image arrays

            Returns
            -------
            tuple
                (pairs, labels) where:
                - pairs: np.ndarray of shapes (n_pairs, 2, height, width, channels)
                - labels: np.ndarray of binary labels (1=same person, 0=different)

            Notes
            -----
            - Only create pairs where both images exist in image_dict
            - Positive pairs (same person) come from 3-value lines
            - Negative pairs (different people) come from 4-value lines
            """
            pairs = []
            labels = []

            with open(pairs_file, 'r') as f:
                lines = f.readlines()

            # Process each pair definition
            for line in lines[1:]:  # Skip the first line
                parts = line.strip().split('\t')

                if len(parts) == 3:  # Same person (positive pair)
                    person_name = parts[0]
                    img1_num = int(parts[1])
                    img2_num = int(parts[2])

                    # Create lookup keys for both images
                    img1_key = f"{person_name}_{img1_num}"
                    img2_key = f"{person_name}_{img2_num}"

                    # Only add a pair if both images were successfully loaded
                    if img1_key in image_dict and img2_key in image_dict:
                        pairs.append([image_dict[img1_key], image_dict[img2_key]])
                        labels.append(1)  # Label 1 indicates the same person

                elif len(parts) == 4:  # Different person (negative pair)
                    person1, person2 = parts[0], parts[2]
                    img1_num, img2_num = int(parts[1]), int(parts[3])

                    # Create lookup keys for images from different people
                    img1_key = f"{person1}_{img1_num}"
                    img2_key = f"{person2}_{img2_num}"

                    # Only create a pair if both images were successfully loaded
                    if img1_key in image_dict and img2_key in image_dict:
                        pairs.append([image_dict[img1_key], image_dict[img2_key]])
                        labels.append(0)  # Label 0 indicates different people

            # Convert lists to numpy arrays for model training
            return np.array(pairs), np.array(labels)

        # === Step 1: Load Training Dataset ===
        print("Loading training images...")
        # Load all unique images from a training file and create mappings
        self.train_val_image_dict, self.train_val_person_images = load_all_images(train_file, data_path)

        # === Step 2: Load Test Dataset ===
        print("\nLoading test images...")
        # Load all unique images from a test file and create mappings
        self.test_image_dict, self.test_person_images = load_all_images(test_file, data_path)

        # === Step 3: Check Dataset Separation ===
        # Identify any overlap between training and test sets (should be minimal)
        common_people = set(self.train_val_person_images.keys()) & set(self.test_person_images.keys())
        print(f"\nCommon people in train_val and test sets: {len(common_people)}")

        # === Step 4: Create Image Pairs ===
        print("\nCreating training + validation pairs...")

        # Create positive and negative pairs from training data
        train_val_pairs, train_val_labels = load_pairs(train_file, self.train_val_image_dict)

        print("\nCreating test pairs...")

        # Create positive and negative pairs from test data
        test_pairs, test_labels = load_pairs(test_file, self.test_image_dict)

        # === Step 5: Split Training Data ===
        print(f"\nSplitting training + validation data (validation split: {validation_split})...")
        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            train_val_pairs, train_val_labels, test_size=validation_split,
            random_state=RANDOM_SEED, stratify=train_val_labels
        )

        # === Step 6: Prepare Individual Images for One-shot Learning ===
        # Extract all unique training images and their labels
        train_images = []
        train_image_labels = []
        for person, img_keys in self.train_val_person_images.items():
            for img_key in img_keys:
                if img_key in self.train_val_image_dict:
                    train_images.append(self.train_val_image_dict[img_key])
                    train_image_labels.append(person)

        # Extract all unique test images and their labels
        test_images = []
        test_image_labels = []
        for person, img_keys in self.test_person_images.items():
            for img_key in img_keys:
                if img_key in self.test_image_dict:
                    test_images.append(self.test_image_dict[img_key])
                    test_image_labels.append(person)

        # === Step 7: Convert Lists to Arrays and Store Data ===
        # Convert image lists to numpy arrays
        self.train_images = np.array(train_images)
        self.test_images = np.array(test_images)

        # Store paired data for model training
        self.train_pairs = train_pairs
        self.train_pair_labels = train_labels
        self.val_pairs = val_pairs
        self.val_pair_labels = val_labels
        self.test_pairs = test_pairs
        self.test_pair_labels = test_labels

        # === Step 8: Final Data Processing ===
        # Add channel dimension for grayscale images if needed
        if len(self.train_images.shape) == 3:  # If images don't have channel dimension
            self.train_images = self.train_images[..., np.newaxis]
            self.test_images = self.test_images[..., np.newaxis]

        # Create sequential labels for individual images (for compatibility)
        self.train_labels = np.arange(len(self.train_images))
        self.test_labels = np.arange(len(self.test_images))

        # Store the validation split of individual images
        val_size = int(len(self.train_images) * validation_split)
        self.val_images = self.train_images[:val_size]
        self.val_labels = self.train_labels[:val_size]

    def analyze_dataset_distribution(self) -> None:
        """
        Perform comprehensive analysis of dataset statistics and distributions.

        This method calculates and stores various dataset metrics including
        - Total number of pairs in train/val/test sets
        - Distribution of positive/negative pairs
        - Number of unique individuals
        - Images per person statistics
        - Train/test set overlap analysis

        The analysis results are stored in the self.stats dictionary and printed
        to provide insights into dataset characteristics for experimental design.

        Notes
        -----
        - Called after dataset loading to validate data quality
        - Help identify potential dataset biases
        - Useful for tuning training parameters
        """
        print("\n=== Detailed Dataset Analysis ===")

        self.train_val_dist = [len(images) for images in self.train_val_person_images.values()]
        self.test_dist = [len(images) for images in self.test_person_images.values()]

        # Calculate and store comprehensive statistics
        self.stats = {
            'total_train_pairs': len(self.train_pairs),
            'total_val_pairs': len(self.val_pairs),

            'positive_train_pairs': np.sum(self.train_pair_labels == 1),
            'negative_train_pairs': np.sum(self.train_pair_labels == 0),

            'positive_val_pairs': np.sum(self.val_pair_labels == 1),
            'negative_val_pairs': np.sum(self.val_pair_labels == 0),

            'unique_train+val_people': len(self.train_val_person_images),
            'unique_train+val_images': len(self.train_val_image_dict),

            # Calculate the distribution of images per person and store in stats
            'train+val_images_per_person_distribution': dict(
                Counter(self.train_val_dist)),

            'average_train+val_images_per_person': round(len(self.train_images) / len(self.train_val_person_images), 3),

            'min_train+val_images_per_person': min(self.train_val_dist),
            'max_train+val_images_per_person': max(self.train_val_dist),

            'train_val_dist_mean': round(np.mean(self.train_val_dist), 3),
            'train_val_dist_median': round(np.median(self.train_val_dist), 3),
            'train_val_dist_std': round(np.std(self.train_val_dist), 3),

            'total_test_pairs': len(self.test_pairs),

            'positive_test_pairs': np.sum(self.test_pair_labels == 1),
            'negative_test_pairs': np.sum(self.test_pair_labels == 0),

            'unique_test_people': len(self.test_person_images),
            'unique_test_images': len(self.test_image_dict),

            'min_test_images_per_person': min(self.test_dist),
            'max_test_images_per_person': max(self.test_dist),

            'average_test_images_per_person': round(len(self.test_images) / len(self.test_person_images), 3),

            'test_dist_mean': round(np.mean(self.test_dist), 3),
            'test_dist_median': round(np.median(self.test_dist), 3),
            'test_dist_std': round(np.std(self.test_dist), 3),

            'test_images_per_person_distribution': dict(
                Counter(self.test_dist)),
        }

        for key, val in self.stats.items():
            print(f"{key}: {val}\n")

        print("=" * 50)

        plot_distribution_charts(Counter(self.train_val_dist), Counter(self.test_dist))

    def create_siamese_network(self) -> SiameseNetwork:
        """
        Create the complete Siamese network architecture using PyTorch.

        Architecture:
        ------------
            Input A -----> Base Network -----> Embedding A
                                                    |
                                                    v
                                              L1 Distance --> Dense(1) --> Sigmoid
                                                    ^
                                                    |
            Input B -----> Base Network -----> Embedding B
                          (shared weights)

        Returns:
        -------
            SiameseNetwork Object - The complete Siamese network model
        """
        # Create the model
        self.model = SiameseNetwork(self.input_shape).to(device)

        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Print model architecture
        print("\nSiamese Network Architecture:")
        print("-" * 50)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("-" * 50)

        return self.model

    def train(self, epochs: int, batch_size: int, pairs_per_epoch: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the Siamese network using PyTorch.

        Args:
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            pairs_per_epoch: Number of pairs to generate per epoch (if not using preloaded)

        Returns:
            Dict containing training history
        """
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50)
        print("Following Karpathy's recipe: starting with simple baseline...")

        # Step 1: Create the model
        if self.model is None:
            self.create_siamese_network()

        print(f"\n✓ Model created successfully!")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Check if we have preloaded pairs
        use_preloaded_pairs = hasattr(self, 'train_pairs') and self.train_pairs is not None

        if use_preloaded_pairs:
            print(f"\nUsing pre-loaded pairs from file:")
            print(f"  Training pairs: {len(self.train_pairs):,}")
            print(f"  Validation pairs: {len(self.val_pairs):,}")
        else:
            raise f"\nThere is a problem with self.train_pairs"

        # Step 2: Check - overfit a single batch
        print("\n" + "-" * 50)
        print("Step 1: Overfitting a single batch to verify model works...")
        print("-" * 50)

        # Prepare a small batch for check
        small_pairs = self.train_pairs[:32]
        small_labels = self.train_pair_labels[:32]

        # Convert to PyTorch tensors
        small_dataset = SiameseDataset(small_pairs, small_labels)
        small_loader = DataLoader(small_dataset, batch_size=32, shuffle=False)

        # Train on single batch
        print("Batch | Loss    | Accuracy | Status")
        print("-" * 40)

        self.model.train()
        for i in range(10):
            for img1, img2, labels in small_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                # Add channel dimension if needed
                if len(img1.shape) == 3:
                    img1 = img1.unsqueeze(1)
                    img2 = img2.unsqueeze(1)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(img1, img2)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                acc = (predictions == labels).float().mean().item()

                status = "Good" if acc > 0.7 else "→ Learning"
                print(f"{i + 1:>5} | {loss.item():>7.4f} | {acc:>8.4f} | {status}")

        if acc < 0.9:
            print("\nWarning: Model may not be learning properly on small batch")
        else:
            print("\nModel successfully overfits small batch - architecture is working!")

        # Step 3: Train on full dataset
        print("\n" + "-" * 50)
        print("Step 2: Training on full dataset...")
        print("-" * 50)

        # Initialize training history
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # Early stopping parameters
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        best_model_state = None

        # Training loop
        print("\nEpoch | Train Loss | Train Acc | Val Loss | Val Acc | Status")
        print("-" * 70)

        for epoch in range(epochs):

            # Prepare data loaders
            indices = np.random.permutation(len(self.train_pairs))  # Shuffle indices
            train_pairs = self.train_pairs[indices]
            train_pair_labels = self.train_pair_labels[indices]
            val_pairs = self.val_pairs
            val_pair_labels = self.val_pair_labels

            # Create data loaders
            train_dataset = SiameseDataset(train_pairs, train_pair_labels)
            val_dataset = SiameseDataset(val_pairs, val_pair_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            train_batches = 0

            for img1, img2, labels in train_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                # Add channel dimension if needed
                if len(img1.shape) == 3:
                    img1 = img1.unsqueeze(1)
                    img2 = img2.unsqueeze(1)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(img1, img2)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Track metrics
                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_acc += (predictions == labels).float().mean().item()
                train_batches += 1

            # Calculate average training metrics
            avg_train_loss = train_loss / train_batches
            avg_train_acc = train_acc / train_batches

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_batches = 0

            with torch.no_grad():
                for img1, img2, labels in val_loader:
                    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                    # Add channel dimension if needed
                    if len(img1.shape) == 3:
                        img1 = img1.unsqueeze(1)
                        img2 = img2.unsqueeze(1)

                    outputs = self.model(img1, img2)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    val_acc += (predictions == labels).float().mean().item()
                    val_batches += 1

            # Calculate average validation metrics
            avg_val_loss = val_loss / val_batches
            avg_val_acc = val_acc / val_batches

            # Update history
            history['loss'].append(avg_train_loss)
            history['accuracy'].append(avg_train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(avg_val_acc)

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the best model state
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, 'best_model.pth')
                status = "✓ Best"

            else:
                patience_counter += 1
                status = f"↓ Wait {patience_counter}/{patience}"

            # Print epoch results
            print(f"{epoch + 1:>5} | {avg_train_loss:>10.4f} | {avg_train_acc:>9.4f} | "
                  f"{avg_val_loss:>8.4f} | {avg_val_acc:>7.4f} | {status}")

            # Early stopping check
            if patience_counter >= patience:
                print(f"\n✓ Early stopping triggered at epoch {epoch + 1}")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                break

        # Load best weights
        print("\n✓ Loading best model weights...")
        self.model.load_state_dict(torch.load('best_model.pth'))

        # Store training history and statistics
        self.history = history
        self.stats['training'] = {
            'epochs_trained': len(history['loss']),
            'final_train_loss': history['loss'][-1],
            'final_train_accuracy': history['accuracy'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_loss': best_val_loss,
            'batch_size': batch_size,
            'used_preloaded_pairs': use_preloaded_pairs,
            'pairs_per_epoch': len(train_pairs) if use_preloaded_pairs else pairs_per_epoch
        }

        # Print training summary
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"✓ Training completed in {len(history['loss'])} epochs")
        print(f"✓ Final training accuracy: {history['accuracy'][-1]:.4f}")
        print(f"✓ Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
        print(f"✓ Best validation loss: {best_val_loss:.4f}")
        print("=" * 50)

        return history

    def evaluate_verification(self) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the model on verification task (same/different person).
        """
        print("\n" + "=" * 50)
        print("Evaluating Verification Performance")
        print("=" * 50)

        # Use test pairs
        test_dataset = SiameseDataset(self.test_pairs, self.test_pair_labels)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for img1, img2, labels in test_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                # Add channel dimension if needed
                if len(img1.shape) == 3:
                    img1 = img1.unsqueeze(1)
                    img2 = img2.unsqueeze(1)

                outputs = self.model(img1, img2)
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        test_labels = np.array(all_labels).flatten()

        # Calculate metrics
        pred_labels = (predictions > 0.5).astype(int).flatten()
        accuracy = np.mean(pred_labels == test_labels)

        # Per-class metrics
        positive_mask = test_labels == 1
        negative_mask = test_labels == 0

        tpr = np.mean(pred_labels[positive_mask] == 1) if np.any(positive_mask) else 0
        tnr = np.mean(pred_labels[negative_mask] == 0) if np.any(negative_mask) else 0

        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"True Positive Rate: {tpr:.4f}")
        print(f"True Negative Rate: {tnr:.4f}")

        return accuracy, predictions, test_labels, self.test_pairs

    def run_complete_experiment(self) -> None:
        """Run the complete experiment pipeline"""
        print("\n" + "=" * 60)
        print("SIAMESE NETWORK FOR ONE-SHOT FACE RECOGNITION")
        print("=" * 60)

        # Create and train the model
        print("\nStep 1: Training Model")
        if self.model is None:
            self.create_siamese_network()

        self.train(EPOCHS, BATCH_SIZE)
