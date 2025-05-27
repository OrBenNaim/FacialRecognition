# Facial Recognition Using One-shot Learning
## Implementation Report

## 1. Introduction
This project implements a facial recognition system using Siamese Neural Networks for one-shot learning, based on the paper "Siamese Neural Networks for One-shot Image Recognition." The primary goal is to develop a system capable of determining whether two facial images represent the same person, even when the person was not seen during training.

The implementation uses the Labeled Faces in the Wild (LFW-a) dataset, which presents real-world challenges in face recognition due to its varied lighting conditions, poses, and expressions. Our approach focuses on learning a similarity metric between faces rather than traditional classification, making it suitable for recognizing previously unseen individuals.

## 2. Dataset Analysis

### 2.1 Dataset Overview
- **Dataset**: Labeled Faces in the Wild (LFW-a version)
- **Image Format**: Grayscale images, resized from 250x250 to 128x128
- **Input Shape**: (128, 128, 1)
- **Split Strategy**: Train/Validation/Test with validation split of 20%

### 2.2 Data Distribution
#### Training + Validation Set
- Total images: 2,200
- Total unique persons: 2,132
- Average images per person: 1.615
- Images per person distribution:

##### Distribution Visualization
![Train Val Distribution](./src/images/train_val_dist.png)

   #### Training + Validation Set Analysis
   - Most common case: 1 image per person (62.3% of people)
   - Least common case: 8 images per person (0.14% of people)
   - Dataset imbalance ratio: 8:1 (max:min images per person)

#### Test Set
- Total images: 1000
- Total unique persons: 963
- Average images per person: 1.609

##### Distribution Visualization
![Test Distribution](./src/images/test_dist.png)

   #### Test Set Analysis
   - Most common case: 1 image per person (62.5% of people)
   - Least common case: 7 images per person (0.62% of people)
   - Dataset imbalance ratio: 7:1 (max:min images per person)

##### Key Observations:
1. **Highly Imbalanced Distribution**:
   - In both sets, the majority of people (>60%) have only one image
   - Very few individuals have more than 4 images

2. **Training Challenges**:
   - Limited data per person makes learning person-specific features difficult
   - High imbalance requires careful consideration in training strategy
   - Most validation will be on single-image cases

3. **Distribution Similarity**:
   - Train and test sets show similar patterns
   - Both have heavily skewed distributions towards single images
   - Similar average images per person (1.615 vs 1.609)

### 2.3 Dataset Split Strategy
- **Validation Split**: 20% of training data
- **Data Preprocessing Pipeline**:
  1. Image loading and grayscale conversion
  2. Resizing from the original 250x250 to 128x128
  3. Pixel normalization (0-255 → 0-1 range)
  4. Ensuring a consistent input shape (128, 128, 1)

## 3. Model Architecture

### 3.1 Implementation Details
#### Base Network Structure
- **Input Layer**: 128x128x1 (grayscale images)
- **CNN Architecture**:
  ```
  Layer 1: Conv2D(64, 10x10) -> ReLU -> MaxPool(2x2)
  Layer 2: Conv2D(128, 7x7) -> ReLU -> MaxPool(2x2)
  Layer 3: Conv2D(128, 4x4) -> ReLU -> MaxPool(2x2)
  Layer 4: Conv2D(256, 4x4) -> ReLU
  Flatten
  Dense(4096) with Sigmoid activation
  ```
  
#### Siamese Configuration
- Twin networks with shared weights
- Input: Pairs of face images (128x128x1 each)
- Processing: Parallel feature extraction through identical CNNs
- Distance Metric: L1 (Manhattan) distance between embeddings
- Output Layer: Single sigmoid unit for similarity score (0–1)
- Loss Function: Binary Cross-Entropy

### 3.2 Design Choices
#### Architecture Decisions
1. **Deep CNN Structure**:
   - Progressive increase in filter count (64→128→128→256)
   - Decreasing kernel sizes (10x10→7x7→4x4→4x4)
   - MaxPooling in the first three layers for dimensionality reduction

2. **Regularization Strategy**:
   - L2 regularization on all convolutional layers (2e-4)
   - L2 regularization on dense layer (1e-3)
   - Dropout isn't used (following the original paper design)

3. **Activation Functions**:
   - ReLU for all convolutional layers for non-linearity
   - Sigmoid for final dense layer-to-bound embeddings

4. **Network Capacity**:
   - 4096-dimensional embeddings for rich feature representation
   - ~2.5M trainable parameters

#### Parameter Selection
- **Initialization**: Glorot uniform for stable training
- **Optimizer**: Adam with learning rate 6e-5
- **Batch Size**: Dynamic based on available memory
- **Early Stopping**: 
  - Patience of 5 epochs
  - Monitored on validation loss
  - Best model checkpointing

#### Training Strategy
1. **Initial Validation**: Sanity check by overfitting a small batch
2. **Full Training**:
   - Dynamic pair generation or preloaded pairs
   - Validation monitoring
   - Early stopping to prevent overfitting
