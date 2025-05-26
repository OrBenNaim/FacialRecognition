from collections import defaultdict
from typing import Tuple

from src.constants import TRAIN_FILE, TEST_FILE


def analyze_dataset_distribution(train_file: str, test_file: str) -> Tuple[set, set]:
    """
    Perform detailed dataset analysis before training.

    This function analyzes the pair's files to understand the dataset structure,
    distribution of positive/negative pairs, and check for train/test overlap.

    Args:
        train_file: Path to pairsDevTrain.txt
        test_file: Path to pairsDevTest.txt

    Returns:
        Tuple containing:
            - train_people: Set of people names in training set
            - test_people: Set of people names in a test set
    """
    print("\n=== Detailed Dataset Analysis ===")

    def analyze_file(file_path, name):
        people_images = defaultdict(int)
        total_images = 0

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split('\t')

            if len(parts) == 3:     # Same person (positive pair)
                person_name = parts[0]
                #num_images = int(parts[2])
                people_images[person_name] += 2  #num_images
                total_images += 2 #num_images

            elif len(parts) == 4:   # Different person (negative pair)
                person_name1 = parts[0]
                people_images[person_name1] += 1

                person_name2 = parts[2]
                people_images[person_name2] += 1

                total_images += 2

        print(f"\n{name} Set Analysis:")
        print(f"- Total people: {len(people_images)}")
        print(f"- Total images: {total_images}")
        print(f"- Average images per person: {total_images / len(people_images):.2f}")
        print(f"- Min images per person: {min(people_images.values())}")
        print(f"- Max images per person: {max(people_images.values())}")

        # Distribution
        distribution = defaultdict(int)
        for count in people_images.values():
            distribution[count] += 1

        print("\nImages per person distribution:")
        for img_count in sorted(distribution.keys()):
            print(f"  {img_count} images: {distribution[img_count]} people")

        return people_images

    train_dist = analyze_file(train_file, "Training")
    test_dist = analyze_file(test_file, "Test")

    # Check for overlap
    train_people = set(train_dist.keys())
    test_people = set(test_dist.keys())
    overlap = train_people.intersection(test_people)

    print(f"\nDataset Split Validation:")
    print(f"- Train/Test overlap: {len(overlap)} people")
    if len(overlap) > 0:
        print("  WARNING: There is overlap between train and test sets!")
        print(f"  Overlapping people: {list(overlap)[:5]}...")
    else:
        print("  ✓ No overlap between train and test sets (good!)")

    return train_dist, test_dist
