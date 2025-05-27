import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, layers, regularizers, initializers, optimizers
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm
import warnings
from typing import Tuple, List, Dict, Optional, Any, Counter

from src.constants import RANDOM_SEED, EPOCHS, BATCH_SIZE, \
    NUM_OF_FILTERS_LAYER1, NUM_OF_FILTERS_LAYER2, NUM_OF_FILTERS_LAYER3, NUM_OF_FILTERS_LAYER4, KERNAL_SIZE_LAYER1, \
    KERNAL_SIZE_LAYER2, KERNAL_SIZE_LAYER3, KERNAL_SIZE_LAYER4, POOL_SIZE
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
        # Store the target shape for input images (height, width, channels)
        self.input_shape = input_shape

        # Initialize model components (built in a separate method)
        self.model: Optional[Model] = None  # Main Siamese network that computes similarity between image pairs
        self.base_network: Optional[Model] = None  # Feature extractor
        self.history: Optional[Dict[str, List[float]]] = None # Training history
        # containing loss and metrics for each epoch

        # Initialize data storage for training/validation set
        self.train_val_person_images: Optional[Dict[str, List[str]]] = None  # Person->image keys (e.g.,'John_Doe_0001')
        self.train_val_image_dict: Optional[Dict[str, np.ndarray]] = None  # Image key -> image data (matrix of pixels)
        self.train_val_dist = None  # Distribution of images per person

        # Initialize data storage for a test set
        self.test_person_images: Optional[Dict[str, List[str]]] = None  # Person -> image keys (e.g.,'Alex_Con_0003')
        self.test_image_dict: Optional[Dict[str, np.ndarray]] = None  # Image key -> image data
        self.test_dist = None  # Distribution of images per person

        # Initialize arrays for training data
        self.train_images: Optional[np.ndarray] = None  # Array of individual training images
        self.train_labels: Optional[np.ndarray] = None  # Labels corresponding to train_images
        self.train_pairs: Optional[np.ndarray] = None  # Array of training image pairs used for Siamese network training
        self.train_pair_labels: Optional[np.ndarray] = None  # Array of binary labels for training pairs
        # (1: same person, 0: different person)


        # Initialize arrays for validation data
        self.val_images: Optional[np.ndarray] = None  # Array of validation images
        self.val_labels: Optional[np.ndarray] = None  # Labels corresponding to val_images
        self.val_pairs: Optional[np.ndarray] = None  # Array of validation image pairs
        self.val_pair_labels: Optional[np.ndarray] = None  # Array of binary labels for validation pairs

        # Initialize arrays for test data
        self.test_images: Optional[np.ndarray] = None  # Array of test images
        self.test_labels: Optional[np.ndarray] = None  # Labels corresponding to test_images
        self.test_pairs: Optional[np.ndarray] = None  # Array of test image pairs
        self.test_pair_labels: Optional[np.ndarray] = None  # Array of binary labels for test pairs

        # Dictionary storing various statistics about the dataset and training
        self.stats: Dict[str, Any] = {}

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
            person_images = defaultdict(list)   # Maps: person_name -> list of their image keys

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

    def run_complete_experiment(self) -> None:
        """
        Run the complete experiment pipeline from data loading to report generation.

        2. Train the Siamese network
        3. Evaluate on verification task
        4. Evaluate one-shot learning performance
        5. Generate visualizations
        6. Create a comprehensive report

        Note:
            This is the main entry point for running the complete experiment.
            All intermediate results are saved and can be accessed through
            the instance attributes.
        """
        print("\n" + "=" * 60)
        print("SIAMESE NETWORK FOR ONE-SHOT FACE RECOGNITION")
        print("=" * 60)

        # Step 1: Train model (Assume data is loaded before inside main.py)
        print("\nStep 1: Training Model")
        self.train(EPOCHS, BATCH_SIZE)

    def train(self, epochs: int, batch_size: int, pairs_per_epoch: Optional[int] = None) \
            -> Dict[str, List[float]]:
        """
        Train the Siamese network following Karpathy's best practices.

        This method implements a robust training procedure including
        - Initial sanity check by overfitting a small batch
        - Full dataset training with validation monitoring
        - Early stopping to prevent overfitting
        - Best model checkpointing

        Args:
            epochs: Maximum number of training epochs
            batch_size: Batch size for training (adjust based on GPU memory)
            pairs_per_epoch: Number of pairs to generate per epoch. If None, use preloaded pairs from files

        Returns:
            Dict containing training history:
                - 'loss': Training loss per epoch
                - 'accuracy': Training accuracy per epoch
                - 'val_loss': Validation loss per epoch
                - 'val_accuracy': Validation accuracy per epoch

        Note:
            The method automatically detects whether to use preloaded pairs
            (from pairsDevTrain.txt) or generate pairs dynamically.
        """
        print("\n" + "=" * 50)
        print("Starting Training")
        print("=" * 50)
        print("Following Karpathy's recipe: starting with simple baseline...")

        # Step 1: Create the model
        if self.model is None:
            self.create_siamese_network()

        print(f"\n✓ Model created successfully!")
        print(f"  Total parameters: {self.model.count_params():,}")

        # Check if we have preloaded pairs from files
        use_preloaded_pairs = hasattr(self, 'train_pairs') and self.train_pairs is not None

        if use_preloaded_pairs:
            print(f"\nUsing pre-loaded pairs from file:")
            print(f"  Training pairs: {len(self.train_pairs):,}")
            print(f"  Validation pairs: {len(self.val_pairs):,}")

        else:
            print(f"\nWill generate {pairs_per_epoch or 20000} pairs per epoch dynamically")

        # Step 2: Sanity check - overfit a single batch first
        print("\n" + "-" * 50)
        print("Step 1: Overfitting a single batch to verify model works...")
        print("-" * 50)

        if use_preloaded_pairs:
            # Use the first 32 pairs from preloaded data
            small_pairs = self.train_pairs[:32]
            small_labels = self.train_pair_labels[:32]
        else:
            # Generate pairs dynamically
            small_pairs, small_labels = self.create_pairs(
                self.train_images[:20],
                self.train_labels[:20],
                32
            )

        # Train on a single batch to verify gradient flow
        print("Batch | Loss    | Accuracy | Status")
        print("-" * 40)

        for i in range(10):
            loss, acc = self.model.train_on_batch([small_pairs[:, 0], small_pairs[:, 1]],small_labels)
            status = "Good" if acc > 0.7 else "→ Learning"
            print(f"{i + 1:>5} | {loss:>7.4f} | {acc:>8.4f} | {status}")

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

        # Training loop
        print("\nEpoch | Train Loss | Train Acc | Val Loss | Val Acc | Status")
        print("-" * 70)

        for epoch in range(epochs):
            # Prepare training data for this epoch
            if use_preloaded_pairs:
                # Use preloaded pairs
                train_pairs = self.train_pairs
                train_pair_labels = self.train_pair_labels
                val_pairs = self.val_pairs
                val_pair_labels = self.val_pair_labels

                # Shuffle training data for better generalization
                indices = np.random.permutation(len(train_pairs))
                train_pairs = train_pairs[indices]
                train_pair_labels = train_pair_labels[indices]
            else:
                # Generate training pairs dynamically
                train_pairs, train_pair_labels = self.create_pairs(
                    self.train_images,
                    self.train_labels,
                    pairs_per_epoch or 20000
                )

                # Generate validation pairs
                val_pairs, val_pair_labels = self.create_pairs(
                    self.val_images,
                    self.val_labels,
                    (pairs_per_epoch or 20000) // 5
                )

            # Train for one epoch
            h = self.model.fit(
                [train_pairs[:, 0], train_pairs[:, 1]],
                train_pair_labels,
                batch_size=batch_size,
                epochs=1,
                validation_data=(
                    [val_pairs[:, 0], val_pairs[:, 1]],
                    val_pair_labels
                ),
                verbose=0  # Suppress default output for custom formatting
            )

            # Update history
            train_loss = h.history['loss'][0]
            train_acc = h.history['accuracy'][0]
            val_loss = h.history['val_loss'][0]
            val_acc = h.history['val_accuracy'][0]

            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            # Check for improvement
            #status = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                self.model.save_weights('best_model.weights.h5')
                status = "✓ Best"
            else:
                patience_counter += 1
                status = f"↓ Wait {patience_counter}/{patience}"

            # Print epoch results
            print(f"{epoch + 1:>5} | {train_loss:>10.4f} | {train_acc:>9.4f} | "
                  f"{val_loss:>8.4f} | {val_acc:>7.4f} | {status}")

            # Early stopping check
            if patience_counter >= patience:
                print(f"\n✓ Early stopping triggered at epoch {epoch + 1}")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                break

        # Load best weights
        print("\n✓ Loading best model weights...")
        self.model.load_weights('best_model.weights.h5')

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

    def create_siamese_network(self) -> Model:
        """
        Create the complete Siamese network architecture.

        This method creates the full Siamese network by:
        1. Creating the shared base network
        2. Processing two inputs through the same network
        3. Computing L1 distance between embeddings
        4. Adding the final classification layer

        Returns:
            tf.keras.Model: The complete compiled Siamese network

        Architecture:
            Input A -----> Base Network -----> Embedding A
                                                    |
                                                    v
                                              L1 Distance --> Dense(1) --> Sigmoid
                                                    ^
                                                    |
            Input B -----> Base Network -----> Embedding B
                          (shared weights)
        """
        # Create the base network for feature extraction
        self.base_network = self.create_base_network()

        # Print base network summary for debugging
        print("\nBase Network Architecture:")
        print("-" * 50)
        self.base_network.summary()

        # Define input layers for the two images
        input_a = Input(shape=self.input_shape, name='input_image_a')
        input_b = Input(shape=self.input_shape, name='input_image_b')

        # Process both inputs through the same network (shared weights)
        # This is the key aspect of Siamese networks
        processed_a = self.base_network(input_a)
        processed_b = self.base_network(input_b)

        # L1 distance layer (Manhattan distance)
        # |f(a) - f(b)| where f is the base network
        l1_distance = (layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]),name='l1_distance')
                       ([processed_a, processed_b]))

        # Final prediction layer
        # Single sigmoid unit for binary classification
        prediction = layers.Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer='glorot_uniform',
            bias_initializer=initializers.Constant(0.5),
            name='prediction'
        )(l1_distance)

        # Create the complete model
        self.model = Model(
            inputs=[input_a, input_b],
            outputs=prediction,
            name='siamese_network'
        )

        # Compile with binary cross-entropy (as in the paper)
        # Using Adam optimizer with learning rate from the paper
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.00006),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("\nComplete Siamese Network Architecture:")
        print("-" * 50)
        self.model.summary()
        print("-" * 50)

        return self.model

    def create_base_network(self) -> Model:
        """
        Create the base convolutional network for feature extraction.

        This network is used as the shared component in the Siamese architecture,
        processing both input images through the same set of weights.

        Architecture based on the original paper:
        - 4 convolutional layers with increasing filter sizes
        - MaxPooling after first 3 conv layers
        - Flatten and dense layer for final embedding

        Returns:
            tf.keras.Model: The base network model that outputs 4096-D embeddings

        Note:
            The architecture is optimized for face recognition with:
            - L2 regularization to prevent overfitting
            - Glorot uniform initialization for stable training
            - Sigmoid activation in the final layer for bounded outputs
        """
        # Define input layer
        input_layer = Input(shape=self.input_shape, name='base_input')

        # Layer 1: Conv(64, 10x10) -> ReLU -> MaxPool(2x2)
        # Large kernel size for initial feature detection
        x = layers.Conv2D(filters=NUM_OF_FILTERS_LAYER1, kernel_size=KERNAL_SIZE_LAYER1, activation='relu',
                          kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(2e-4), name='conv1')(input_layer)

        x = layers.MaxPooling2D(pool_size=POOL_SIZE, name='pool1')(x)

        # Layer 2: Conv(128, 7x7) -> ReLU -> MaxPool(2x2)
        # Increasing filters for more complex features
        x = layers.Conv2D(filters=NUM_OF_FILTERS_LAYER2, kernel_size=KERNAL_SIZE_LAYER2, activation='relu', kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(2e-4), name='conv2')(x)

        x = layers.MaxPooling2D(pool_size=POOL_SIZE, name='pool2')(x)

        # Layer 3: Conv(128, 4x4) -> ReLU -> MaxPool(2x2)
        # Maintain filter count but reduce kernel size
        x = layers.Conv2D(filters=NUM_OF_FILTERS_LAYER3, kernel_size=KERNAL_SIZE_LAYER3, activation='relu', kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(2e-4), name='conv3')(x)

        x = layers.MaxPooling2D(pool_size=POOL_SIZE, name='pool3')(x)

        # Layer 4: Conv(256, 4x4) -> ReLU
        # Final convolutional layer without pooling
        x = layers.Conv2D(filters=NUM_OF_FILTERS_LAYER4, kernel_size=KERNAL_SIZE_LAYER4, activation='relu', kernel_initializer='glorot_uniform',
                          kernel_regularizer=regularizers.l2(2e-4), name='conv4')(x)

        # Flatten for dense layer
        x = layers.Flatten(name='flatten')(x)

        # Dense layer: 4096 units with sigmoid activation
        # Sigmoid bounds outputs to [0, 1] for stable distance computation
        x = layers.Dense(units=4096, activation='sigmoid', kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(1e-3), name='embedding')(x)

        # Create and return the model
        model = Model(inputs=input_layer, outputs=x, name='base_network')

        return model
