import numpy as np
from tensorflow.python.keras import Model
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import warnings
from typing import Tuple, List, Dict, Optional, Any, Counter

from src.constants import RANDOM_SEED
from src.utils import plot_distribution_charts

# Suppress TensorFlow warnings for cleaner output
warnings.filterwarnings('ignore')

class SiameseFaceRecognition:
    """
    A Siamese Neural Network implementation for one-shot face recognition tasks.

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
        # Input shape for the images (height, width, channels)
        self.input_shape = input_shape

        # The main Siamese model that computes similarity between image pairs
        self.model: Optional[Model] = None

        # Base network used as a feature extractor in the Siamese architecture
        self.base_network: Optional[Model] = None

        # Training history containing loss and metrics for each epoch
        self.history: Optional[Dict[str, List[float]]] = None

        # Dictionary mapping person names to their image keys (e.g., {'John_Doe': ['John_Doe_0001', 'John_Doe_0002']})
        self.train_val_person_images: Optional[Dict[str, List[str]]] = None
        self.test_person_images: Optional[Dict[str, List[str]]] = None

        # Dictionary mapping image keys to numpy arrays of image data (e.g., {'John_Doe_0001': array([...])})
        self.train_val_image_dict: Optional[Dict[str, np.ndarray]] = None
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

        self.train_val_dist = None
        self.test_dist = None

        print(f"Siamese Face Recognition initialized with input shape: {input_shape}")

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
        # Validate inputs
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
                            current_img_key = f"{person_name}_{img_num}"

                            # Only load if we haven't seen this image before
                            if current_img_key not in image_dict:
                                img_path = os.path.join(base_path, person_name, img_name)

                                if os.path.exists(img_path):
                                    # Load and preprocess image
                                    img = Image.open(img_path).convert('L')  # Convert to grayscale

                                    # Resize to the model's expected input shape
                                    img = img.resize((self.input_shape[0], self.input_shape[1]))

                                    # Normalize pixel values to [0, 1]
                                    img_array = np.array(img) / 255.0

                                    image_dict[current_img_key] = img_array
                                    person_images[person_name].append(current_img_key)
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
        self.train_val_image_dict, self.train_val_person_images = load_all_images(train_file, data_path)

        # Load all images from a test file
        print("\nLoading test images...")
        self.test_image_dict, self.test_person_images = load_all_images(test_file, data_path)

        # Check for overlap between train_val_person_images and test_person_images
        common_people = set(self.train_val_person_images.keys()) & set(self.test_person_images.keys())
        print(f"\nCommon people in train_val and test sets: {len(common_people)}")

        # Load training pairs
        print("\nCreating training + validation pairs...")
        train_val_pairs, train_val_labels = load_pairs(train_file, self.train_val_image_dict)

        # Load test pairs
        print("\nCreating test pairs...")
        test_pairs, test_labels = load_pairs(test_file, self.test_image_dict)

        # Create validation split from training pairs
        print(f"\nSplitting training + validation data (validation split: {validation_split})...")

        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            train_val_pairs, train_val_labels, test_size=validation_split,
            random_state=RANDOM_SEED, stratify=train_val_labels
        )

        # Extract unique images for compatibility with other methods
        # This allows us to use individual images for one-shot learning evaluation
        train_images = []
        train_image_labels = []
        for person, img_keys in self.train_val_person_images.items():
            for img_key in img_keys:
                if img_key in self.train_val_image_dict:
                    train_images.append(self.train_val_image_dict[img_key])
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

        # Store pairs for validation
        self.val_pairs = val_pairs
        self.val_pair_labels = val_labels

        # Store pairs for test
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
