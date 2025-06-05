# Standard library imports
import os
import time
import warnings
from typing import Tuple, Optional, Any, Set  # Type hints for better code documentation
from collections import defaultdict, Counter  # For efficient data structure handling
import matplotlib.pyplot as plt
from typing import Dict, List

# Second-party modules
from numpy import floating, ndarray, dtype
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score

# Third-party imports
import numpy as np  # For numerical operations
import torch  # Primary deep learning framework
import torch.optim as optim  # Optimization algorithms
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Neural network functions
from torch.utils.data import Dataset, DataLoader  # Data handling utilities
from torch.utils.tensorboard import SummaryWriter
from PIL import Image  # Image processing
from tqdm import tqdm  # Progress bar functionality

# Local imports - Constants and utility functions
from src.constants import (
    RANDOM_SEED,  # For reproducibility
    NUM_OF_FILTERS_LAYER1, NUM_OF_FILTERS_LAYER2,  # Network architecture parameters
    NUM_OF_FILTERS_LAYER3, NUM_OF_FILTERS_LAYER4,  # Filter counts for conv layers
    KERNAL_SIZE_LAYER1, KERNAL_SIZE_LAYER2,  # Kernel sizes for conv layers
    KERNAL_SIZE_LAYER3, KERNAL_SIZE_LAYER4,
    POOL_SIZE,  # Pooling layer size
    EARLY_STOPPING_PATIENCE, TRAIN_FILE_PATH,
    TEST_FILE_PATH, CLASSIFICATION_THRESHOLD, IMPROVED_KERNEL_SIZES, STRIDE, IMPROVED_NUM_OF_FILTERS
)
from src.utils import plot_distribution_charts, move_data_to_appropriate_device, \
    apply_simple_augmentation  # Visualization utilities

# Setup for reproducibility
# Setting random seeds for all components to ensure consistent results
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # Ensure deterministic behavior on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Suppress warnings for cleaner output during training
warnings.filterwarnings('ignore')

# Device configuration
# Automatically detects and uses GPU if available, otherwise uses CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SiameseDataset(Dataset):
    """
    Custom Dataset class for handling pairs of images for Siamese Network training.
    Inherits from torch.utils.data.Dataset for PyTorch compatibility.
    """

    def __init__(self, pairs: np.ndarray, labels: np.ndarray, augment: bool):
        """
        Initialize the dataset with image pairs and their corresponding labels.

        Args:
            pairs: numpy array of shape (N, 2, H, W) containing N pairs of images
            labels: numpy array of shape (N, ) containing binary labels
                   (1 for the same person, 0 for different people)
            augment: boolean flag to indicate if augmentation should be applied
        """
        self.pairs = pairs
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        """
        Get the total number of image pairs in the dataset.

        Returns:
            int: Number of pairs in the dataset
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Get a single pair of images and their corresponding label.

        Args:
            idx: Index of the pair to retrieve

        Returns:
            tuple: Contains:
                - img1: First image of the pair as FloatTensor
                - img2: Second image of the pair as FloatTensor
                - label: Binary label as FloatTensor (1 for the same person, 0 for different)
        """
        # Get images and label for the given index
        img1, img2 = self.pairs[idx]
        label = self.labels[idx]

        # Apply simple augmentation if enabled
        if self.augment:
            img1 = apply_simple_augmentation(img1)
            img2 = apply_simple_augmentation(img2)

        # Create a contiguous copy of the arrays before converting to tensors
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)

        # Convert to PyTorch tensors
        img1 = torch.FloatTensor(img1)
        img2 = torch.FloatTensor(img2)
        label = torch.FloatTensor([label])

        return img1, img2, label


class BaseNetwork(nn.Module):
    """
    Base convolutional network for feature extraction in Siamese architecture.

    Architecture:
        - 4 Convolutional layers with increasing filter counts
        - 3 MaxPooling layers
        - 1 Dense layer with sigmoid activation
        - ReLU activations between conv layers

    The network processes input images through a series of convolutional and pooling
    layers to extract meaningful features, which are then passed through a dense layer
    to create a fixed-size feature embedding.
    """

    def __init__(self, input_shape: Tuple[int, int, int]):
        """
        Initialize the network architecture.

        Args:
            input_shape: Tuple of (height, width, channels) for input images
                        Example: (100, 100, 1) for 100x100 grayscale images
        """
        super(BaseNetwork, self).__init__()
        self.input_shape = input_shape

        # Layer 1: First convolutional block
        # Input: (input_shape) -> Output: (NUM_OF_FILTERS_LAYER1 feature maps)
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[2],  # Number of input channels (1 for grayscale)
            out_channels=NUM_OF_FILTERS_LAYER1,  # Number of output feature maps
            kernel_size=KERNAL_SIZE_LAYER1,
            padding=0  # No padding used
        )
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_SIZE)

        # Layer 2: Second convolutional block
        # Increases feature map depth while reducing spatial dimensions
        self.conv2 = nn.Conv2d(
            in_channels=NUM_OF_FILTERS_LAYER1,
            out_channels=NUM_OF_FILTERS_LAYER2,
            kernel_size=KERNAL_SIZE_LAYER2,
            padding=0
        )
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_SIZE)

        # Layer 3: Third convolutional block
        # Further feature extraction with maintained feature map depth
        self.conv3 = nn.Conv2d(
            in_channels=NUM_OF_FILTERS_LAYER2,
            out_channels=NUM_OF_FILTERS_LAYER3,
            kernel_size=KERNAL_SIZE_LAYER3,
            padding=0
        )
        self.pool3 = nn.MaxPool2d(kernel_size=POOL_SIZE)

        # Layer 4: Final convolutional layer
        # Highest level feature extraction without pooling
        self.conv4 = nn.Conv2d(
            in_channels=NUM_OF_FILTERS_LAYER3,
            out_channels=NUM_OF_FILTERS_LAYER4,
            kernel_size=KERNAL_SIZE_LAYER4,
            padding=0
        )

        # Dynamically calculate the size of flattened features
        self._calculate_flatten_size()

        # The final dense layer for creating the feature embedding
        # Output size of 4096 was chosen based on the original architecture
        self.fc = nn.Linear(self.flatten_size, 4096)

        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()

    def _calculate_flatten_size(self):
        """
        Calculate the flattened feature size after all convolutions.
        This is needed to properly size the fully connected layer.
        Use a fake forward pass to compute the dimensions.
        """
        dummy_input = torch.zeros(1, self.input_shape[2], self.input_shape[0], self.input_shape[1])

        # Simulate the forward pass through conv layers
        x = self.pool1(F.relu(self.conv1(dummy_input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        # Calculate flattened size
        self.flatten_size = x.view(1, -1).size(1)

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier/Glorot uniform initialization.
        This helps achieve better convergence by maintaining the appropriate
        variance of activations across layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Feature embedding of shape (batch_size, 4096)
        """
        # Sequential application of conv layers with ReLU activation and pooling
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))

        # Flatten the feature maps
        x = x.view(x.size(0), -1)

        # Final dense layer with sigmoid activation for feature embedding
        x = torch.sigmoid(self.fc(x))

        return x

class ImprovedBaseNetwork(nn.Module):
    """
    Enhanced base network with BatchNorm, Dropout, and smaller kernels
    """

    def __init__(self, input_shape: Tuple[int, int, int]):
        super(ImprovedBaseNetwork, self).__init__()
        self.input_shape = input_shape

        # Layer 1: First convolutional block
        # Input: (input_shape) -> Output: (NUM_OF_FILTERS_LAYER1 feature maps)
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[2],     # Number of input channels (1 for grayscale),
            out_channels=IMPROVED_NUM_OF_FILTERS['layer1'],     # Number of output feature maps
            kernel_size=IMPROVED_KERNEL_SIZES['layer1'],
            padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=STRIDE)

        self.conv2 = nn.Conv2d(
            in_channels=IMPROVED_NUM_OF_FILTERS['layer1'],
            out_channels=IMPROVED_NUM_OF_FILTERS['layer2'],
            kernel_size=IMPROVED_KERNEL_SIZES['layer2'],
            padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=STRIDE)

        self.conv3 = nn.Conv2d(
            in_channels=IMPROVED_NUM_OF_FILTERS['layer2'],
            out_channels=IMPROVED_NUM_OF_FILTERS['layer3'],
            kernel_size=IMPROVED_KERNEL_SIZES['layer3'],
            padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=STRIDE)

        self.conv4 = nn.Conv2d(
            in_channels=IMPROVED_NUM_OF_FILTERS['layer3'],
            out_channels=IMPROVED_NUM_OF_FILTERS['layer4'],
            kernel_size=IMPROVED_KERNEL_SIZES['layer4'],
            padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Calculate flattened size
        self._calculate_flatten_size()

        # FC layers with dropout
        self.fc1 = nn.Linear(self.flatten_size, 4096)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.1)

        self._initialize_weights()

    def _calculate_flatten_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_shape[0], self.input_shape[1])
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = F.relu(self.bn4(self.conv4(x)))
            self.flatten_size = x.view(1, -1).size(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Forward pass with BatchNorm and Dropout
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.sigmoid(self.fc2(x))
        if self.training:  # Only apply dropout during training
            x = self.dropout2(x)

        return x


class ImprovedSiameseNetwork(nn.Module):
    """
    Improved Siamese Network using enhanced base network
    """

    def __init__(self, input_shape: Tuple[int, int, int]):
        super(ImprovedSiameseNetwork, self).__init__()
        self.base_network = ImprovedBaseNetwork(input_shape)
        self.prediction = nn.Linear(4096, 1)

        # Initialize prediction layer
        nn.init.xavier_uniform_(self.prediction.weight)
        nn.init.constant_(self.prediction.bias, 0.5)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        # Get embeddings from both inputs
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)

        # Compute L1 distance and predict similarity
        l1_distance = torch.abs(output1 - output2)
        prediction = torch.sigmoid(self.prediction(l1_distance))

        return prediction


class SiameseNetwork(nn.Module):
    """
    Complete Siamese Network architecture for face similarity comparison.

    The network consists of:
    1. A shared base network that processes both input images
    2. L1 distance calculation between the embeddings
    3. Final prediction layer that outputs similarity score

    The network learns to map face images to a feature space where similar
    faces are close together and different faces are far apart.
    """

    def __init__(self, input_shape: Tuple[int, int, int]):
        """
        Initialize the Siamese Network.

        Args:
            input_shape: Tuple of (height, width, channels) for input images
                        Example: (100, 100, 1) for 100x100 grayscale images
        """
        super(SiameseNetwork, self).__init__()

        # Shared base network for feature extraction
        # Both images will be processed through this same network
        self.base_network = BaseNetwork(input_shape)

        # The Final prediction layer that takes L1 distance and outputs similarity score
        # Input size is 4096 (size of feature embedding from base network)
        # Output size is 1 (single similarity score)
        self.prediction = nn.Linear(4096, 1)

        # Initialize prediction layer weights using Xavier/Glorot uniform
        # Bias initialized to 0.5 to start with balanced predictions
        nn.init.xavier_uniform_(self.prediction.weight)
        nn.init.constant_(self.prediction.bias, 0.5)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing similarity between two input images.

        Args:
            input1: First input image tensor of shape (batch_size, channels, height, width)
            input2: Second input image tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Similarity scores between 0 and 1 for each pair in the batch
                         Shape: (batch_size, 1)
                         - Scores close to 1 indicate similar faces
                         - Scores close to 0 indicate different faces
        """
        # Generate embeddings for both images using a shared base network
        output1 = self.base_network(input1)  # Shape: (batch_size, 4096)
        output2 = self.base_network(input2)  # Shape: (batch_size, 4096)

        # Compute L1 distance between embeddings
        # This measures how different the embeddings are
        l1_distance = torch.abs(output1 - output2)  # Shape: (batch_size, 4096)

        # The Final prediction with sigmoid activation
        # Maps the distance to a similarity score between 0 and 1
        prediction = torch.sigmoid(self.prediction(l1_distance))  # Shape: (batch_size, 1)

        return prediction


class FaceRecognition:
    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        """
        Initialize FaceRecognition class.

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
        self.input_shape: tuple[int, int, int] = input_shape

        self.lr: Optional[float] = None
        self.batch_size: Optional[int] = None
        self.epochs: Optional[int] = None

        self.experiment_name: Optional[str] = None
        self.tensorboard_log_dir: Optional[str] = None
        self.writer: Optional[SummaryWriter] = None

        self.use_improved_arch: Optional[bool] = None

        # Initialize model components
        self.model: Optional[SiameseNetwork] = None
        self.optimizer: Optional[optim.Adam] = None
        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        self.history: Optional[Dict[str, List[float]]] = None

        self.start_time = None
        self.convergence_results = {}

        # Initialize data storage for training + validation set
        self.train_val_person_images: Optional[Dict[str, List[str]]] = None
        self.train_val_image_dict: Optional[Dict[str, np.ndarray]] = None
        self.train_val_distribution = None

        # Initialize arrays for training data
        self.train_people_names: Optional[List[str]] = None
        self.train_pairs: Optional[np.ndarray] = None
        self.train_pair_labels: Optional[np.ndarray] = None

        # Initialize arrays for validation data
        self.val_people_names: Optional[List[str]] = None
        self.val_pairs: Optional[np.ndarray] = None
        self.val_pair_labels: Optional[np.ndarray] = None

        # Initialize arrays for test data
        self.test_person_images: Optional[Dict[str, List[str]]] = None
        self.test_image_dict: Optional[Dict[str, np.ndarray]] = None
        self.test_labels: Optional[np.ndarray] = None
        self.test_pairs: Optional[np.ndarray] = None
        self.test_pair_labels: Optional[np.ndarray] = None

        # Dictionary storing various statistics about the dataset and training
        self.stats: Dict[str, Any] = {}

    def reset_exp_attr(self):
        """
        Resets all experiment-related attributes to their default state (None).

        This method clears and reinitialized:
        - Training parameters (learning rate, batch size, epochs)
        - Experiment identifiers and logging setup
        - Model architecture settings
        - Model components (network, optimizer, loss history)
        - Timing information

        This is typically called before starting a new experiment to ensure
        no settings from previous experiments carry over.

        Attributes reset:
            - lr (float): Learning rate
            - batch_size (int): Batch size for training
            - epochs (int): Number of training epochs
            - experiment_name (str): Name identifier for the experiment
            - tensorboard_log_dir (str): Directory for TensorBoard logs
            - writer (SummaryWriter): TensorBoard writer instance
            - use_improved_arch (bool): Flag for using improved architecture
            - model (SiameseNetwork): Neural network model
            - optimizer (optim.Adam): Adam optimizer instance
            - history (Dict[str, List[float]]): Training history
            - experiment_start_time: Timestamp for experiment start

        Note:
            The criterion (BCELoss) is reset to a new instance rather than None.
        """
        self.lr: Optional[float] = None
        self.batch_size: Optional[int] = None
        self.epochs: Optional[int] = None

        self.experiment_name: Optional[str] = None
        self.tensorboard_log_dir: Optional[str] = None
        self.writer: Optional[SummaryWriter] = None

        self.use_improved_arch: Optional[bool] = None

        # Initialize model components
        self.model: Optional[SiameseNetwork] = None
        self.optimizer: Optional[optim.Adam] = None
        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        self.history: Optional[Dict[str, List[float]]] = None

    # Load training and test datasets
    def load_lfw_dataset(self, data_path_folder: str, dataset_file_path: str, validation_split: float) -> None:
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
        data_path_folder : str
            Root directory path containing LFW person subdirectories
        dataset_file_path : str
            Path to the dataset pairs file (e.g., pairsDevTrain.txt)
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
        # ===== Input Validation =====
        # Ensure all required files and directories exist
        if not os.path.exists(data_path_folder):
            raise FileNotFoundError(f"DATA folder path not found: {data_path_folder}")

        if not os.path.exists(dataset_file_path):
            raise FileNotFoundError(f"Train file not found: {dataset_file_path}")

        if not 0 <= validation_split <= 1:
            raise ValueError(f"validation_split must be between 0 and 1, got {validation_split}")

        print(f"Image resize target: {self.input_shape[0]}x{self.input_shape[1]}")

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

                                    # Store preprocessed image and update a person's image list
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

        def split_people(person_images: Dict[str, List[str]], val_split: float) -> Tuple[List[str], List[str]]:
            """
            Split people into training and validation sets.

            Args:
                person_images: Dictionary mapping person names to their image keys
                val_split: Fraction of people to use for validation

            Returns:
                Tuple of (train_people, val_people) lists
            """
            all_people = list(person_images.keys())
            np.random.shuffle(all_people)

            split_idx = int(len(all_people) * (1 - val_split))
            train_people = all_people[:split_idx]
            val_people = all_people[split_idx:]

            return train_people, val_people

        # Load pairs from a file
        def create_pairs_for_set(pairs_file: str, image_dict: Dict[str, np.ndarray],
                                 allowed_people: Set[str]) -> Tuple[np.ndarray, np.ndarray]:
            """
            Create pairs only using people from the allowed set.

            Args:
                pairs_file: Path to the pair's file
                image_dict: Dictionary of all loaded images
                allowed_people: Set of people allowed in this split

            Returns:
                tuple
                (pairs, labels) where:
                - pairs: np.ndarray of shapes (n_pairs, 2, height, width, channels)
                - labels: np.ndarray of binary labels (1=same person, 0=different)
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
                    if person_name not in allowed_people:
                        continue

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
                    if person1 not in allowed_people or person2 not in allowed_people:
                        continue

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

        # === Step 1: Load Dataset ===
        print("Loading LFW-a dataset...")

        # Load all unique images from a dataset file and create mappings
        temp_image_dict, temp_person_images = load_all_images(dataset_file_path, data_path_folder)

        if dataset_file_path == TRAIN_FILE_PATH:
            self.train_val_person_images = temp_person_images
            self.train_val_image_dict = temp_image_dict

            # ===== Step 2: Split Training Data =====
            print(f"\nSplitting training data into training/validation data (validation split: {validation_split})...")
            self.train_people_names, self.val_people_names = split_people(person_images=temp_person_images,
                                                                          val_split=validation_split)
            # ===== Step 3: Create Image Pairs =====
            print("\nCreating pairs...")

            # Create training pairs from the dataset
            self.train_pairs, self.train_pair_labels = create_pairs_for_set(pairs_file=dataset_file_path,
                                                                            image_dict=temp_image_dict,
                                                                            allowed_people=set(self.train_people_names))
            # Create validation pairs from the dataset
            self.val_pairs, self.val_pair_labels = create_pairs_for_set(pairs_file=dataset_file_path,
                                                                        image_dict=temp_image_dict,
                                                                        allowed_people=set(self.val_people_names))
        elif dataset_file_path == TEST_FILE_PATH:
            self.test_person_images = temp_person_images
            self.test_image_dict = temp_image_dict

            # ===== Step 3: Create Image Pairs =====
            print("\nCreating pairs...")

            # Create testing pairs from the dataset
            self.test_pairs, self.test_pair_labels = (
                create_pairs_for_set(pairs_file=dataset_file_path,
                                     image_dict=temp_image_dict, allowed_people=set(self.test_person_images.keys())))

        else:
            raise ValueError(f"Invalid dataset file path: {dataset_file_path}")

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
            SiameseNetwork Object - The complete Siamese network model
        """
        if self.use_improved_arch:

            # Use the improved architecture
            self.model = ImprovedSiameseNetwork(self.input_shape).to(device)
            print("Using improved architecture with BatchNorm and Dropout\n")

        else:
            # Use your base architecture
            self.model = SiameseNetwork(self.input_shape).to(device)
            print("Using base architecture\n")

            # Create optimizer (keep existing code)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Print model info (keep existing code)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")

        return self.model

    def log_dataset_analysis_to_tensorboard(self) -> None:
        """
        Analyze and log dataset statistics and distributions to TensorBoard.

        This method calculates and logs various dataset metrics including
        - Dataset size statistics (total pairs, unique people, unique images)
        - Class distribution (positive/negative pairs)
        - Images per person distribution
        - Visual distribution plots

        The following metrics are logged to TensorBoard:
        Scalars:
            - Total training and validation pairs
            - Number of positive/negative pairs in train/val sets
            - Number of unique people and images

        Visualizations:
            - Histogram of images per person distribution
            - Bar plot of images per person distribution
            - Text summary of key statistics

        Notes:
            - Requires self.train_val_person_images and related attributes to be populated
            - Logs are saved to the tensorboard_log_dir specified during initialization
            - All visualizations are tagged under the 'Dataset/' namespace in TensorBoard
            - Automatically closes matplotlib figures to prevent memory leaks

        Example TensorBoard tags:
            - Dataset/total_train_pairs
            - Dataset/total_val_pairs
            - Dataset/images_per_person_distribution
            - Dataset/distribution_plot
            - Dataset/Summary
        """

        # Calculate stats (keep existing calculation logic)
        self.train_val_distribution = [len(images) for images in self.train_val_person_images.values()]

        self.stats = {
            'total_train_pairs': len(self.train_pairs),
            'total_val_pairs': len(self.val_pairs),
            'positive_train_pairs': np.sum(self.train_pair_labels == 1),
            'negative_train_pairs': np.sum(self.train_pair_labels == 0),
            'positive_val_pairs': np.sum(self.val_pair_labels == 1),
            'negative_val_pairs': np.sum(self.val_pair_labels == 0),
            'unique_train+val_people': len(self.train_val_person_images),
            'unique_train+val_images': len(self.train_val_image_dict),
        }

        # Log dataset statistics to TensorBoard
        for key, value in self.stats.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Dataset/{key}', value, 0)

        # Log dataset distribution as histogram
        distribution_data = np.array(self.train_val_distribution)
        self.writer.add_histogram('Dataset/images_per_person_distribution', distribution_data, 0)

        # Create and log a distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        distribution_counter = Counter(self.train_val_distribution)
        sorted_dist = dict(sorted(distribution_counter.items()))

        ax.bar(sorted_dist.keys(), sorted_dist.values(), color='skyblue')
        ax.set_title('Images per Person Distribution')
        ax.set_xlabel('Number of Images per Person')
        ax.set_ylabel('Number of People')

        self.writer.add_figure('Dataset/distribution_plot', fig, 0)
        plt.close(fig)

        # Log as text summary
        dataset_summary = f"""
        Dataset Analysis Summary:
        - Total training pairs: {self.stats['total_train_pairs']:,}
        - Total validation pairs: {self.stats['total_val_pairs']:,}
        - Unique people: {self.stats['unique_train+val_people']:,}
        - Unique images: {self.stats['unique_train+val_images']:,}
        - Positive train pairs: {self.stats['positive_train_pairs']:,}
        - Negative train pairs: {self.stats['negative_train_pairs']:,}
        """
        self.writer.add_text('Dataset/Summary', dataset_summary, 0)

        plot_distribution_charts(Counter(self.train_val_distribution))

    def log_model_architecture_to_tensorboard(self) -> None:
        """Log model architecture details to TensorBoard"""

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Log parameter counts
        self.writer.add_scalar('Model/total_parameters', total_params, 0)
        self.writer.add_scalar('Model/trainable_parameters', trainable_params, 0)

        # Log model graph
        try:
            # Create fake input with correct shape (batch_size, channels, height, width)
            sample_input_1 = torch.randn(1, 1, self.input_shape[0], self.input_shape[1]).to(device)
            sample_input_2 = torch.randn(1, 1, self.input_shape[0], self.input_shape[1]).to(device)

            # Wrap inputs in a list instead of passing as separate arguments
            self.writer.add_graph(self.model,[sample_input_1, sample_input_2])  # Pass as a list of tensors

        except Exception as e:
            self.writer.add_text('Model/graph_error', f"Could not log model graph: {e}", 0)

        # Log architecture summary as text
        arch_summary = f"""
        Siamese Network Architecture:
        - Total parameters: {total_params:,}
        - Trainable parameters: {trainable_params:,}
        - Input shape: {self.input_shape}
        - Base network: 4 Conv layers + 1 Dense layer
        - Final layer: L1 distance + prediction layer
        """
        self.writer.add_text('Model/Architecture', arch_summary, 0)

    def log_training_progress_to_tensorboard(self, epoch: int, avg_train_loss: float,
                                             avg_train_acc: float, avg_val_loss: float,
                                             avg_val_acc: float, status: str) -> None:
        """Log training progress to TensorBoard instead of the console"""

        self.writer.add_scalars('Training_Curves/Loss', {
            'Train': avg_train_loss,
            'Val': avg_val_loss
        }, epoch)

        self.writer.add_scalars('Training_Curves/Accuracy', {
            'Train': avg_train_acc,
            'Val': avg_val_acc
        }, epoch)

        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Training/learning_rate', current_lr, epoch)

        # Log training status as text
        progress_text = f"Epoch {epoch + 1}: Loss={avg_train_loss:.4f}, Val_Loss={avg_val_loss:.4f}, Acc={avg_train_acc:.4f}, Val_Acc={avg_val_acc:.4f}, Status={status}"
        self.writer.add_text('Training/Progress', progress_text, epoch)

    def log_final_results_to_tensorboard(self, history: Dict[str, List[float]],
                                         metrics: Dict[str, float]) -> None:
        """Log final experiment results to TensorBoard"""

        # Log final metrics
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f'Final_Results/{metric_name}', metric_value, 0)

        # Log training summary
        final_summary = f"""
        Training Complete!

        Final Results:
        - Training Accuracy: {history['train_accuracy'][-1]:.4f}
        - Validation Accuracy: {history['val_accuracy'][-1]:.4f}
        - Training Loss: {history['train_loss'][-1]:.4f}
        - Validation Loss: {history['val_loss'][-1]:.4f}

        Detailed Metrics:
        - Overall Accuracy: {metrics['accuracy']:.4f}
        - AUC Score: {metrics['auc']:.4f}
        - F1 Score: {metrics['f1_score']:.4f}
        - Average Precision: {metrics['average_precision']:.4f}

        Training completed in {len(history['train_loss'])} epochs.
        """
        self.writer.add_text('Final_Results/Summary', final_summary, 0)

    def find_misclassified_examples(self, pairs: np.ndarray,
                                                  pair_labels: np.ndarray,
                                                  num_examples: int = 10) -> None:
        """
        This function processes image pairs through the Siamese network and identifies cases where
        the model's predictions differ from the true labels. It performs the following tasks:
        1. Finds misclassified pairs in dataset
        2. Creates visualizations of the misclassified pairs
        3. Logs individual misclassified examples to TensorBoard
        4. Generates and logs summary statistics
        5. Saves visualization plots to disk

        Args:
            pairs (np.ndarray): Array of image pairs to evaluate, shape (n_pairs, 2, height, width)
            pair_labels (np.ndarray): Ground truth labels for the pairs (0: different, 1: same), shape (n_pairs, )
            num_examples (int, optional): Maximum number of misclassified examples to log. Default to 5.

        Returns:
            None

        Notes:
            - The function limits analysis to the first 1000 pairs for efficiency
            - Images are logged both to TensorBoard and saved as PNG files
            - For each misclassified pair, both images are shown side by side
            - Summary statistics include the error rate in the examined sample
        """
        dataset_name = 'validation' if pairs is self.val_pairs else 'test'

        self.model.eval()
        misclassified_pairs = []

        with torch.no_grad():
            for i in range(min(len(pairs), 1000)):  # Limit search to the first 1000 pairs
                img1 = torch.from_numpy(pairs[i][0]).float()
                img2 = torch.from_numpy(pairs[i][1]).float()

                if len(img1.shape) == 2:
                    img1 = img1.unsqueeze(0).unsqueeze(0)
                    img2 = img2.unsqueeze(0).unsqueeze(0)
                elif len(img1.shape) == 3:
                    img1 = img1.unsqueeze(0)
                    img2 = img2.unsqueeze(0)

                img1, img2 = img1.to(device), img2.to(device)
                true_label = pair_labels[i]

                output = self.model(img1, img2)
                pred = (output > CLASSIFICATION_THRESHOLD).float().item()

                if pred != true_label:
                    # Create a side-by-side image
                    img1_np = img1.cpu().numpy()[0, 0]
                    img2_np = img2.cpu().numpy()[0, 0]

                    misclassified_pairs.append({
                        'img1': img1_np,
                        'img2': img2_np,
                        'true_label': true_label,
                        'predicted_label': pred,
                        'index': i
                    })

                if len(misclassified_pairs) >= num_examples:
                    break

        # Log misclassified examples
        for idx, example in enumerate(misclassified_pairs):
            combined_img = np.concatenate([example['img1'], example['img2']], axis=1)
            img_tensor = torch.tensor(combined_img).unsqueeze(0)  # Add channel dim

            self.writer.add_image(
                f'{self.experiment_name}/Misclassified_{dataset_name}/Example_{idx + 1}/'
                f'True: {example['true_label']}, Pred: {example['predicted_label']}',
                img_tensor,
                global_step=0,
                dataformats='CHW'  # Explicitly specify the format
            )

        # Log summary text
        misclass_summary = f"""
        Misclassified Examples Analysis ({dataset_name} set):
        - Found {len(misclassified_pairs)} misclassified pairs out of {min(len(pairs), 1000)} examined
        - Error rate in sample: {len(misclassified_pairs) / min(len(pairs), 1000) * 1000:.2f}%
        """
        self.writer.add_text(f'Misclassified_{dataset_name}/Summary', misclass_summary, 0)

    def train(self) -> Dict[str, List[float]]:
        """
        Train the Siamese network using PyTorch.

        Returns:
            Dict containing training history
        """

        # Minimal console output for essential info
        print("\n🚀 Starting Training...")

        self.start_time = time.time()

        if self.model is None:
            self.create_siamese_network()

        # Log model architecture
        self.log_model_architecture_to_tensorboard()

        # Initialize training
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        patience = EARLY_STOPPING_PATIENCE
        best_epoch = 0  # track the best epoch
        convergence_time = 0.0

        print(f"📊 Training for up to {self.epochs} epochs. Monitor progress in TensorBoard!")
        print(f"🔗 TensorBoard: tensorboard --logdir={self.tensorboard_log_dir}")

        for epoch in range(self.epochs):
            # Training phase (existing logic)
            indices = np.random.permutation(len(self.train_pairs))
            train_pairs = self.train_pairs[indices]
            train_pair_labels = self.train_pair_labels[indices]

            train_dataset = SiameseDataset(pairs=train_pairs, labels=train_pair_labels, augment=True)
            val_dataset = SiameseDataset(pairs=self.val_pairs, labels=self.val_pair_labels, augment=False)

            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

            # Training loop (existing logic)
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for img1, img2, labels in train_loader:
                img1, img2, labels = move_data_to_appropriate_device(img1, img2, labels, device)

                if len(img1.shape) == 3:
                    img1 = img1.unsqueeze(1)
                    img2 = img2.unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(img1, img2)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                predictions = (outputs > CLASSIFICATION_THRESHOLD).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_correct / train_total

            # Validation loop (existing logic)
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for img1, img2, labels in val_loader:
                    img1, img2, labels = move_data_to_appropriate_device(img1, img2, labels, device)

                    if len(img1.shape) == 3:
                        img1 = img1.unsqueeze(1)
                        img2 = img2.unsqueeze(1)

                    outputs = self.model(img1, img2)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    predictions = (outputs > CLASSIFICATION_THRESHOLD).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_correct / val_total

            # Update history
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(avg_train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(avg_val_acc)

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                status = "✓ Best"

                # record convergence time
                best_epoch = epoch + 1
                convergence_time = time.time() - self.start_time

            else:
                patience_counter += 1
                status = f"Wait {patience_counter}/{patience}"

            # Log to TensorBoard instead of the console
            self.log_training_progress_to_tensorboard(
                epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, status
            )

            # Early stopping
            if patience_counter >= patience:
                early_stop_text = f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}"
                self.writer.add_text('Training/EarlyStop', early_stop_text, epoch)
                print(f"⏹️  {early_stop_text}")
                break

        # save convergence results
        total_time = time.time() - self.start_time

        # FIXED: Handle case where no improvement was found
        if best_epoch == 0:
            # If no improvement, use the final epoch as a convergence point
            best_epoch = len(history['train_loss'])
            convergence_time = total_time

        self.convergence_results = {
            'convergence_epoch': best_epoch,
            'convergence_time_seconds': convergence_time,
            'total_training_time_seconds': total_time,
            'convergence_time_minutes': convergence_time / 60,
            'total_training_time_minutes': total_time / 60
        }

        # Print simple results
        print(f"\n⏱️  Training completed in {total_time:.1f}s ({total_time / 60:.1f} minutes)")
        print(f"📈 Best model at epoch {best_epoch} (converged in {convergence_time:.1f}s)")

        # Log hyperparameters
        hparam_dict = {
            'lr': self.lr,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'architecture': 'SiameseNetwork',
            'optimizer': 'Adam'
        }

        metric_dict = {
            'final_val_accuracy': history['val_accuracy'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_loss': best_val_loss
        }

        self.writer.add_hparams(hparam_dict, metric_dict)

        print("\n✅ Training completed! Check TensorBoard for detailed results.")
        return history

    def calculate_detailed_metrics(self, pairs: np.ndarray, pair_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate detailed performance metrics as required by the exercise.
        """
        accuracy, predictions, gt_labels = self.__evaluate_verification(pairs, pair_labels)

        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(gt_labels, predictions)
        auc_score = auc(fpr, tpr)

        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(gt_labels, predictions)
        avg_precision = average_precision_score(gt_labels, predictions)

        # Calculate F1 score
        pred_labels = (predictions > CLASSIFICATION_THRESHOLD).astype(int)
        f1 = f1_score(gt_labels, pred_labels)

        metrics = {
            'accuracy': accuracy,
            'auc': auc_score,
            'average_precision': avg_precision,
            'f1_score': f1
        }

        return metrics

    def __evaluate_verification(self, pairs: np.ndarray, pair_labels: np.ndarray) \
            -> tuple[floating[Any], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
        """
        Evaluates the Siamese network's performance on face verification tasks.

        Processes pairs of face images through the model to determine if they belong to the same person
        or different people.
        Calculates and displays various performance metrics, including overall accuracy,
        true positive rate (sensitivity), and true negative rate (specificity).

        Args:
            pairs: np.ndarray
                Array of image pairs to evaluate.
                Shape should be compatible with
                the SiameseDataset format.
            pair_labels: np.ndarray
                Binary labels for each pair (1 for the same person, 0 for different people).

        Returns:
            tuple[float, np.ndarray, np.ndarray]
                A tuple containing:
                - accuracy: Overall accuracy across all pairs
                - predictions: Raw model prediction scores before thresholding
                - gt_labels: Ground truth labels used for evaluation

        Notes:
            - Uses self.batch_size for batch processing
            - Applies CLASSIFICATION_THRESHOLD to convert raw predictions to binary decisions
            - Automatically handles grayscale images by adding channel dimension if necessary
            - Prints detailed evaluation metrics to console
            - Model is set to evaluation mode during inference
            - Gradients are disabled during evaluation for efficiency

        Performance Metrics:
            - Overall Accuracy: Proportion of correctly classified pairs
            - True Positive Rate: Accuracy on same-person pairs (sensitivity)
            - True Negative Rate: Accuracy on different-person pairs (specificity)
        """

        # Print evaluation header
        print("\n" + "=" * 50)
        print("Evaluating Verification Performance")
        print("=" * 50)

        # Create DataLoader for a test set with fixed batch size
        # No shuffling to maintain pair order for analysis
        dataset = SiameseDataset(pairs=pairs, labels=pair_labels, augment=False)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Set model to evaluation mode (affects dropout, batch norm, etc.)
        self.model.eval()

        # Initialize lists to store batch results
        all_predictions = []  # Store model predictions
        all_labels = []  # Store ground truth labels

        # Disable gradient computation for efficiency during inference
        with torch.no_grad():

            # Process each batch of image pairs
            for img1, img2, labels in loader:
                # Move data to appropriate device (CPU/GPU)
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                # Handle grayscale images by adding channel dimension if it's necessary
                # Shape should be [batch_size, channels, height, width]
                if len(img1.shape) == 3:  # If missing channel dimension
                    img1 = img1.unsqueeze(1)
                    img2 = img2.unsqueeze(1)

                # Get model predictions for this batch
                outputs = self.model(img1, img2)

                # Convert predictions and labels to numpy arrays
                # Move to CPU first if they were on GPU
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert lists to numpy arrays for efficient computation
        predictions = np.array(all_predictions)  # Raw prediction scores
        gt_labels = np.array(all_labels).flatten()  # Ground truth labels

        # Convert raw predictions to binary decisions
        # Using 0.5 as a threshold for binary classification
        pred_labels = (predictions > CLASSIFICATION_THRESHOLD).astype(int).flatten()

        # Calculate overall accuracy across all pairs
        accuracy = np.mean(pred_labels == gt_labels)

        # Calculate separate metrics for same-person and different-person pairs
        positive_mask = gt_labels == 1  # Mask for same-person pairs
        negative_mask = gt_labels == 0  # Mask for different-person pairs

        # True Positive Rate (sensitivity): accuracy on same-person pairs
        tpr = np.mean(pred_labels[positive_mask] == 1) if np.any(positive_mask) else 0

        # True Negative Rate (specificity): accuracy on different-person pairs
        tnr = np.mean(pred_labels[negative_mask] == 0) if np.any(negative_mask) else 0

        # Print performance metrics
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"True Positive Rate: {tpr:.4f}")
        print(f"True Negative Rate: {tnr:.4f}")

        # Return all relevant data for further analysis if needed
        return accuracy, predictions, gt_labels

    def analyze_results(self, history: Dict[str, List[float]]) -> None:
        """
        Analyze and visualize training results using TensorBoard.

        Creates and logs training history visualizations showing the model's performance
        over time, including loss and accuracy curves for both training and validation sets.

        Parameters
        ----------
        history : Dict[str, List[float]]
            Dictionary containing training history with the following keys:
            - 'train_loss': List of training loss values per epoch
            - 'val_loss': List of validation loss values per epoch
            - 'train_accuracy': List of training accuracy values per epoch
            - 'val_accuracy': List of validation accuracy values per epoch

        Visualization Details
        -------------------
         Create a single figure with two subplots:
        1. Loss Plot:
           - Training and validation loss curves
           - X-axis: Epochs
           - Y-axis: Loss values

        2. Accuracy Plot:
           - Training and validation accuracy curves
           - X-axis: Epochs
           - Y-axis: Accuracy values (0-1)

        Notes
        -----
        - Visualizations are logged to TensorBoard under the 'Results/training_curves' tag
        - Automatically closes matplotlib figures to prevent memory leaks
        - The figure can be viewed in TensorBoard's Images tab
        """

        # Create plots and log to TensorBoard instead of saving files

        # Loss plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Accuracy plot
        ax2.plot(history['train_accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Add to tensorboard
        self.writer.add_figure('Results/training_curves', fig, 0)

    def run_complete_experiment(self, learning_rate: float, batch_size: int, epochs: int,
                                use_improved_arch: bool, exp_name: str) -> None:
        """Streamlined experiment with TensorBoard-centric logging"""
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_improved_arch = use_improved_arch
        self.experiment_name = exp_name

        self.tensorboard_log_dir = os.path.join("tensorboard_logs", self.experiment_name)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tensorboard_log_dir)

        print(f"\n🔬 Starting experiment: {self.experiment_name}")

        history = self.train()

        # Calculate metrics
        print("\n📈 Calculating metrics...")
        metrics = self.calculate_detailed_metrics(self.val_pairs, self.val_pair_labels)

        # Log final results
        self.log_final_results_to_tensorboard(history, metrics)

        self.analyze_results(history)

        # Log misclassified examples
        print("\n🔍 Analyzing failures...")
        self.find_misclassified_examples(self.val_pairs, self.val_pair_labels)

        # Close writer
        self.writer.close()

        # Minimal final console output
        print("\n✅ Experiment completed!")
        print(f"\n📊 View results: tensorboard --logdir={self.tensorboard_log_dir}\n")

        print("===== Validation results =====")
        print(f"Final Accuracy: {metrics['accuracy']:.4f}")
        print(f"Final Precision: {metrics['average_precision']:.4f}")
        print(f"Final F1 Score: {metrics['f1_score']:.4f}")
