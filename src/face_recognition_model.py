import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import warnings
from typing import Tuple, List, Dict, Optional, Any

from src.constants import RANDOM_SEED

# Suppress TensorFlow warnings for cleaner output
warnings.filterwarnings('ignore')

class SiameseFaceRecognition:
    """
    Complete implementation of Siamese Neural Networks for One-shot Face Recognition.
    This class provides a full pipeline for training and evaluating a Siamese network
    on face recognition tasks, including data loading, model creation, training,
    evaluation, and visualization.
    """

    def __init__(self, input_shape):
        """
        Initialize the Siamese Network
        Args:
            input_shape: Shape of input images

        Note:
            The input_shape parameter defines the size images will be resized to
            during loading, NOT the original size of your image files.
        """
        # Input shape for the images (height, width, channels)
        self.input_shape = input_shape

        # The main Siamese model that computes similarity between image pairs
        self.model: Optional[Model] = None

        # Base network used as a feature extractor in the Siamese architecture
        self.base_network: Optional[Model] = None

        # Training history containing loss and metrics for each epoch
        self.history: Optional[Dict[str, List[float]]] = None

        # Dictionary mapping person names to their image keys (e.g., {'John_Doe': ['John_Doe_0001', 'John_Doe_0002']})
        self.train_person_images: Optional[Dict[str, List[str]]] = None
        self.test_person_images: Optional[Dict[str, List[str]]] = None

        # Dictionary mapping image keys to numpy arrays of image data (e.g., {'John_Doe_0001': array([...])})
        self.train_image_dict: Optional[Dict[str, np.ndarray]] = None
        self.test_image_dict: Optional[Dict[str, np.ndarray]] = None

        # Array of individual training images
        self.train_images: Optional[np.ndarray] = None

        # Labels corresponding to train_images
        self.train_labels: Optional[np.ndarray] = None

        # Array of validation images
        self.val_images: Optional[np.ndarray] = None

        # Labels corresponding to val_images
        self.val_labels: Optional[np.ndarray] = None

        # Array of test images
        self.test_images: Optional[np.ndarray] = None

        # Labels corresponding to test_images
        self.test_labels: Optional[np.ndarray] = None

        # Array of training image pairs used for Siamese network training
        self.train_pairs: Optional[np.ndarray] = None

        # Binary labels for training pairs (1: same person, 0: different person)
        self.train_pair_labels: Optional[np.ndarray] = None

        # Array of validation image pairs
        self.val_pairs: Optional[np.ndarray] = None

        # Binary labels for validation pairs (1: same person, 0: different person)
        self.val_pair_labels: Optional[np.ndarray] = None

        # Array of test image pairs
        self.test_pairs: Optional[np.ndarray] = None

        # Binary labels for test pairs (1: same person, 0: different person)
        self.test_pair_labels: Optional[np.ndarray] = None

        # Dictionary storing various statistics about the dataset and training
        self.stats: Dict[str, Any] = {}

        print(f"Siamese Face Recognition initialized with input shape: {input_shape}")

    def load_lfw_dataset(self, data_path: str, train_file: str, test_file: str, validation_split: float
                         ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Load and preprocess the LFW-a dataset with the specific pair file format.

        This method handles the standard LFW pairs format where:
        - Lines with 3 values indicate same person pairs (positive)
        - Lines with 4 values indicate different person pairs (negative)

        Args:
            data_path: Path to the LFW-a dataset root directory containing person folders
            train_file: Path to pairsDevTrain.txt file containing training pairs
            test_file: Path to pairsDevTest.txt file containing test pairs
            validation_split: Fraction of training data to use for validation (0.0 to 1.0)

        Returns:
            Tuple containing:
                - train_person_images: Dictionary mapping person names to their image keys
                - test_person_images: Dictionary mapping person names to their image keys

        Note:
            Images are automatically resized to self.input_shape during loading.
            Original image files are not modified.
        """
        # Validate inputs
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Train file not found: {train_file}")

        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        if not 0 <= validation_split <= 1:
            raise ValueError(f"validation_split must be between 0 and 1, got {validation_split}")

        print("Loading LFW-a dataset...")
        print(f"Image resize target: {self.input_shape[0]}x{self.input_shape[1]}")

        print("Loading LFW-a dataset...")

        # First, load all unique images and create a mapping
        def load_all_images(
                pairs_file: str,
                base_path: str
        ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
            """
            Load all unique images mentioned in the pair's file.

            Args:
                pairs_file: Path to the pair's file (e.g., pairsDevTrain.txt)
                base_path: Base path to the image dataset

            Returns:
                Tuple of:
                    - image_dict: Dictionary mapping image_key (e.g., 'Al_Pacino_0001') to numpy array
                    - person_images: Dictionary mapping person name to list of image keys
            """
            image_dict = {}  # Stores image_key -> image array
            person_images = defaultdict(list)  # Stores person -> list of image keys

            with open(pairs_file, 'r') as f:
                lines = f.readlines()

                # Skip the first line which contains the number of pairs
                for i, line in enumerate(tqdm(lines[1:], desc="Loading images")):
                    parts = line.strip().split('\t')

                    if len(parts) == 3:  # Same person pair format
                        person_name = parts[0]
                        img1_num = int(parts[1])
                        img2_num = int(parts[2])

                        # Process both images from the same person
                        for img_num in [img1_num, img2_num]:
                            img_name = f"{person_name}_{img_num:04d}.jpg"
                            img_key = f"{person_name}_{img_num}"

                            # Only load if we haven't seen this image before
                            if img_key not in image_dict:
                                img_path = os.path.join(base_path, person_name, img_name)

                                if os.path.exists(img_path):
                                    # Load and preprocess image
                                    img = Image.open(img_path).convert('L')  # Convert to grayscale

                                    # Resize to model's expected input shape
                                    img = img.resize((self.input_shape[0], self.input_shape[1]))

                                    # Normalize pixel values to [0, 1]
                                    img_array = np.array(img) / 255.0

                                    image_dict[img_key] = img_array
                                    person_images[person_name].append(img_key)
                                else:
                                    print(f"Warning: Image not found: {img_path}")

                    elif len(parts) == 4:  # Different person pair
                        person1 = parts[0]
                        img1_num = int(parts[1])
                        person2 = parts[2]
                        img2_num = int(parts[3])

                        # Load first person's image
                        img1_name = f"{person1}_{img1_num:04d}.jpg"
                        img1_key = f"{person1}_{img1_num}"

                        if img1_key not in image_dict:
                            img1_path = os.path.join(base_path, person1, img1_name)

                            if os.path.exists(img1_path):
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
        def load_pairs(pairs_file: str, image_dict: Dict[str, np.ndarray]
                       ) -> Tuple[np.ndarray, np.ndarray]:
            """
            Load pairs and labels from the pair's file.

            Args:
                pairs_file: Path to the pairs file
                image_dict: Dictionary mapping image keys to numpy arrays

            Returns:
                Tuple of:
                    - pairs: Array of shape (n_pairs, 2, height, width, channels)
                    - labels: Binary labels (1 for the same person, 0 for different)
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

                    img1_key = f"{person_name}_{img1_num}"
                    img2_key = f"{person_name}_{img2_num}"

                    # Only add a pair if both images were successfully loaded
                    if img1_key in image_dict and img2_key in image_dict:
                        pairs.append([image_dict[img1_key], image_dict[img2_key]])
                        labels.append(1)  # Same person

                elif len(parts) == 4:  # Different person (negative pair)
                    person1 = parts[0]
                    img1_num = int(parts[1])
                    person2 = parts[2]
                    img2_num = int(parts[3])

                    img1_key = f"{person1}_{img1_num}"
                    img2_key = f"{person2}_{img2_num}"

                    if img1_key in image_dict and img2_key in image_dict:
                        pairs.append([image_dict[img1_key], image_dict[img2_key]])
                        labels.append(0)  # Different person = 0

            return np.array(pairs), np.array(labels)

        # Load all images from a training file
        print("Loading training images...")
        self.train_image_dict, self.train_person_images = load_all_images(train_file, data_path)

        # Load all images from a test file
        print("Loading test images...")
        self.test_image_dict, self.test_person_images = load_all_images(test_file, data_path)

        # Load training pairs
        print("\nCreating training pairs...")
        train_pairs, train_labels = load_pairs(train_file, self.train_image_dict)

        # Load test pairs
        print("\nCreating test pairs...")
        test_pairs, test_labels = load_pairs(test_file, self.test_image_dict)

        # Create validation split from training pairs
        print(f"\nSplitting training data (validation split: {validation_split})...")

        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            train_pairs, train_labels, test_size=validation_split,
            random_state=RANDOM_SEED, stratify=train_labels
        )

        # Extract unique images for compatibility with other methods
        # This allows us to use individual images for one-shot learning evaluation
        train_images = []
        train_image_labels = []
        for person, img_keys in self.train_person_images.items():
            for img_key in img_keys:
                if img_key in self.train_image_dict:
                    train_images.append(self.train_image_dict[img_key])
                    train_image_labels.append(person)

        test_images = []
        test_image_labels = []

        for person, img_keys in self.test_person_images.items():

            for img_key in img_keys:

                if img_key in self.test_image_dict:
                    test_images.append(self.test_image_dict[img_key])
                    test_image_labels.append(person)

        # Convert to numpy arrays
        self.train_images = np.array(train_images)
        self.test_images = np.array(test_images)

        # Store pairs for training
        self.train_pairs = train_pairs
        self.train_pair_labels = train_labels

        self.val_pairs = val_pairs
        self.val_pair_labels = val_labels

        self.test_pairs = test_pairs
        self.test_pair_labels = test_labels

        # Add channel dimension if needed (for grayscale images)
        if len(self.train_images.shape) == 3:
            self.train_images = self.train_images[..., np.newaxis]
            self.test_images = self.test_images[..., np.newaxis]

        # For compatibility, create fake labels for individual images
        self.train_labels = np.arange(len(self.train_images))
        self.test_labels = np.arange(len(self.test_images))

        # Store the validation split of individual images
        val_size = int(len(self.train_images) * validation_split)
        self.val_images = self.train_images[:val_size]
        self.val_labels = self.train_labels[:val_size]

        # Calculate and store comprehensive statistics
        self.stats = {
            'total_train_pairs': len(self.train_pairs),
            'total_val_pairs': len(self.val_pairs),

            'positive_train_pairs': np.sum(self.train_pair_labels == 1),
            'negative_train_pairs': np.sum(self.train_pair_labels == 0),

            'positive_val_pairs': np.sum(self.val_pair_labels == 1),
            'negative_val_pairs': np.sum(self.val_pair_labels == 0),

            'unique_train+val_people': len(self.train_person_images),
            'unique_train+val_images': len(self.train_image_dict),

            'total_test_pairs': len(self.test_pairs),

            'positive_test_pairs': np.sum(self.test_pair_labels == 1),
            'negative_test_pairs': np.sum(self.test_pair_labels == 0),

            'unique_test_people': len(self.test_person_images),
            'unique_test_images': len(self.test_image_dict),

            'image_shape': self.input_shape,
        }

        # Print summary
        print("\n" + "=" * 50)
        print(f"\nDataset loaded successfully!")
        print("=" * 50)

        for key, val in self.stats.items():
            print(f"{key}: {val}")

        print(f"Images resized from original to: {self.input_shape[0]}x{self.input_shape[1]}")
        print("=" * 50)

        return self.train_person_images, self.test_person_images
