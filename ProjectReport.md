# Facial Recognition Using One-shot Learning


## 1. Introduction
Face recognition technology has become increasingly important in modern applications, from security systems to user authentication. However, traditional face recognition systems often require large amounts of training data for each person they need to recognize, which isn't practical in many real-world scenarios. This project tackles this limitation by implementing a one-shot learning approach using Siamese Neural Networks.

### 1.1 Problem Statement
The challenge is to develop a facial recognition system that can determine whether two facial images represent the same person, even when we have never seen that person during training. This is particularly important because:
- Most real-world applications can't collect multiple images of each person
- New individuals need to be added to the system without retraining
- Traditional deep learning approaches require extensive training data per person

### 1.2 Background Research
This implementation is based on several key works:
1. Koch et al.'s seminal paper "Siamese Neural Networks for One-shot Image Recognition" (2015), which introduced the concept of using Siamese networks for one-shot learning
2. The Labeled Faces in the Wild (LFW) dataset paper by Huang et al., which established benchmark standards for face verification
3. Recent advancements in face recognition architectures, particularly those using contrastive loss functions

### 1.3 Approach Overview
Our solution uses a Siamese Neural Network architecture that:
- Learns to extract meaningful features from face images
- Computes similarity between pairs of faces
- Makes verification decisions based on learned similarities
- Requires only one reference image per person

## 2. Dataset
### 2.1 Dataset Overview
The project uses the Labeled Faces in the Wild (LFW-a) dataset, which contains:
- 13,233 facial images
- 5,749 different individuals
- Images collected from real-world situations
- Varied lighting conditions, poses, and expressions
- Aligned version (LFW-a) for better consistency

Key characteristics:
- Image Resolution: 250x250 pixels
- Format: Grayscale
- Collection Period: Mixed, representing real-world photo conditions
- Annotation: Includes person identities and pair matching information

### 2.2 Dataset Analysis

#### Distribution Statistics
- Training + Validation Set:
  - Total Images: 4,400
  - Unique Individuals: 2,132
  - Average Images per Person: 1.729
  - Maximum Images per Person: 8
  - Minimum Images per Person: 1
    
#### Distribution Visualization
![Train Val Distribution](images/train_val_dist.png)


#### Data Distribution Patterns
1. Person-wise Distribution:
   - 62.3% of people have only 1 image
   - 24.7% have 2 images
   - 8.4% have 3 images
   - 4.6% have 4 or more images

2. Challenging Aspects:
   - Highly imbalanced distribution
   - Limited samples per person
   - Varied image quality and conditions
   - Real-world pose variations

#### Dataset Quality Analysis
1. Image Variations:
   - Lighting: Natural to artificial
   - Poses: Front-facing-to-profile views
   - Expressions: Neutral to expressive
   - Age: Various age ranges
   - Quality: Professional to casual photos

2. Technical Characteristics:
   - Consistent alignment across faces
   - Standardized image size
   - Professional pre-processing
   - Clean annotations

### 2.3 Dataset Organization
The dataset is organized into:
1. Training Set (70%):
   - Used for model training
   - Further split into training and validation
   - Ensures no person overlaps between splits

2. Testing Set (30%):
   - Completely separate individuals
   - Used only for final evaluation
   - Represents real-world scenarios

This organization ensures:
- No data leakage between sets
- Realistic evaluation of one-shot capabilities
- Fair assessment of generalization


## 3. Preprocessing
### 3.1 Data Preparation Pipeline
The preprocessing pipeline is designed to standardize the input data and prepare it for the Siamese network training. The pipeline consists of several key steps:

#### 3.1.1 Image Preprocessing
1. **Size Standardization**
   - Original size: 250x250 pixels
   - Resized to: 128x128 pixels
   - Rationale: Balance between detail preservation and computational efficiency
   - Method: Bilinear interpolation for smooth resizing

2. **Color Processing**
   - Input: Grayscale images
   - Pixel value normalization: Scale from [0-255] to [0-1]
   - Format: Single channel (128, 128, 1)

3. **Quality Enhancement**
   - Contrast normalization
   - Noise reduction while preserving facial features
   - Uniform brightness adjustment

### 3.2 Pair Generation Strategy
A crucial aspect of training a Siamese network is the generation of image pairs. Our approach includes:

#### 3.2.1 Training Pairs Creation
1. **Positive Pairs (Same Person)**
   - Generated from individuals with multiple images
   - Randomized selection within same-person images
   - Balanced sampling for persons with many images

2. **Negative Pairs (Different Persons)**
   - Random selection from different individuals
   - Controlled sampling to maintain class balance
   - Strategy to avoid bias towards specific individuals

    
## 4. Experiments
### 4.1 Base Architecture
Our initial Siamese network architecture serves as the baseline for experiments:

#### 4.1.1 Base Model Configuration
- **CNN Architecture**:
  ```
  Input: (128, 128, 1)
  Layer 1: Conv2D(64, 10x10) -> ReLU -> MaxPool(2x2)
  Layer 2: Conv2D(128, 7x7) -> ReLU -> MaxPool(2x2)
  Layer 3: Conv2D(128, 4x4) -> ReLU -> MaxPool(2x2)
  Layer 4: Conv2D(256, 4x4) -> ReLU
  Flatten
  Dense: 4096 with Sigmoid activation
  ```

Base Training Parameters:
- Learning Rate: 6e-5
- Batch Size: 32
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Epochs: 50

##### Siamese Configuration
- Twin networks with shared weights
- Input: Pairs of face images (128x128x1 each)
- Processing: Parallel feature extraction through identical CNNs
- Distance Metric: L1 (Manhattan) distance between embeddings
- Output Layer: Single sigmoid unit for similarity score (0–1)
- Loss Function: Binary Cross-Entropy

### 4.2 Experimental Trials

#### Experiment 1: Base Architecture with Data Augmentation - Base_Aug
**Motivation**: Investigate if 

**Results**:
- Key Metrics:
  - Accuracy: 0.791
  - F1 Score: 0.841
  - Precision: 0.939

**Analysis**:
- Pros:

- Cons:


#### Experiment 2: Enhanced base network with BatchNorm, Dropout, and smaller kernels—Improved - Enhanced_Base
**Motivation**: Investigate if 

**Changes**:
  ```
  Input: (128, 128, 1)
  Conv1: 64 filters (5x5) + ReLU + MaxPool(2x2)
  Conv2: 128 filters (5x5) + ReLU + MaxPool(2x2)
  Conv3: 256 filters (3x3) + ReLU + MaxPool(2x2)
  Conv4: 512 filters (3x3) + ReLU
  Flatten
  Dense: 4096 with Sigmoid activation
  ```

**Results**:
- Key Metrics:
  - Accuracy: 
  - F1 Score: 
  - Precision: 

**Analysis**:
- Pros:

- Cons:

### 4.3 Comparative Analysis

#### Performance Comparison
| Experiment | Accuracy | F1 Score | AUC   | 
|------------|----------|----------|-------|
| Baseline   | 0.754    | 0.838    | 0.838 | 
| Base_Aug   | 0.791    | 0.865    | 0.798 |
| Enhanced   | 0.84     | 0.912    | 0.566 |




## 5. False Analysis
### 5.1 Error Patterns
- Move content from "4.3 Error Analysis"
- Analysis of misclassified examples
- Visualizations of failure cases

### 5.2 Performance Analysis
- Detailed metrics analysis
- Understanding model weaknesses
- Solutions implemented to address issues

#### Experiment: Hyperparameter Optimization
**Motivation**: Find optimal learning parameters for faster convergence

**Configurations Tested**:
1. Learning Rate Variations:
   ```python
   configurations = [
       {"lr": 6e-5, "batch_size": 32, "epochs": 50},
       {"lr": 3e-5, "batch_size": 64, "epochs": 50},
       {"lr": 1e-4, "batch_size": 32, "epochs": 50}
   ]
   ```

2. Batch Size Impact:
   - Tested: 16, 32, 64, 128
   - Best performing: X
   - Memory usage vs. performance trade-offs

**Results**:
| Configuration | Val Accuracy | Training Time | Convergence |
|--------------|--------------|---------------|-------------|
| Config 1     | X%           | X hours       | Epoch X     |
| Config 2     | X%           | X hours       | Epoch X     |
| Config 3     | X%           | X hours       | Epoch X     |

**Analysis**:
- Optimal configuration found: {...}
- Trade-offs observed
- Impact on model stability


## 6. Takeaways
### 6.1 Conclusions
- Key findings and insights
- Main challenges encountered
- Successful strategies

### 6.2 Future Improvements
- Suggested architectural improvements
- Potential data augmentation techniques
- Training optimization possibilities

### 6.3 Lessons Learned
- Technical insights gained
- Best practices discovered
- What would be done differently

## References
1. Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. ICML deep learning workshop.
