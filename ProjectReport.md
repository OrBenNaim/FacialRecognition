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

#### Experiment 1: Base Architecture with Data Augmentation—Base_with_Aug
**Motivation**: Investigate if 

**Results**:
- Key Metrics:
  - Accuracy: 0.791
  - F1 Score: 0.841
  - Precision: 0.939

- Training Times: 72.8 sec (1.2 minutes)
- Convergence times: 29.5 sec

**Analysis**:
- Pros:
  - Significant improvement in accuracy (0.791) compared to baseline (0.754)
  - High precision (0.939) indicates a very low false positive rate
  - Good F1 Score (0.841) shows balanced performance
  - Data augmentation helped prevent overfitting without architectural changes
  - Maintained good AUC (0.798) suggesting reliable discrimination ability
  - Simple to implement as it only required augmentation pipeline changes

- Cons:
  - Still shows room for improvement in overall accuracy
  - Limited by the base architecture's capacity
  - Data augmentation adds computational overhead during training
  - May not handle extreme variations in face angles/positions
  - Training time increased due to augmentation processing

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
  - Accuracy: 0.84
  - F1 Score: 0.912
  - Precision: 0.86

- Training Times: 134.8 sec (2.2 minutes)
- Convergence times: 22.5 sec

**Analysis**:
- Pros:
  - Best accuracy (0.84) among all experiments
  - Highest F1 Score (0.912) indicating excellent overall performance
  - Architectural improvements provided better feature extraction
  - Better handling of training stability with BatchNorm
  - Reduced risk of overfitting through Dropout layers
  - Smaller kernels captured more fine-grained facial features

- Cons:
  - Lower AUC (0.566) suggests potential issues with the decision boundary
  - More complex architecture requires more computational resources
  - Increased number of parameters to train
  - More hyperparameters to tune (dropout rates, batch norm parameters)
  - May require larger batch sizes for stable batch normalization


### 4.3 Comparative Analysis

#### Performance Comparison
| Experiment    | Accuracy | F1 Score | AUC   | Conv. Time (min) | Training Time (min) |
|---------------|----------|----------|-------|------------------|---------------------|
| Baseline      | 0.754    | 0.838    | 0.838 | 0.36             | 1                   |
| Base_with_Aug | 0.791    | 0.865    | 0.798 | 0.49             | 1.2                 |
| Enhanced_Base | 0.84     | 0.912    | 0.566 | 0.375            | 2.2                 |




## 5. False Analysis
### 5.1 Error Patterns

#### False Positive Analysis
1. **Common Characteristics**:
   - Similar facial features between different individuals
   - Similar lighting and pose conditions
   - Common facial attributes (based on dataset characteristics)

2. **Challenging Scenarios**:
   - Similar facial structures
   - Consistent imaging conditions
   - Similar demographic characteristics

#### False Negative Analysis
1. **Major Contributing Factors**:
   - Variations in pose
   - Lighting differences
   - Expression changes
   - Image quality variations

### 5.2 Performance Analysis

#### Metric Analysis
1. **Enhanced Base Model Performance**:
   - Accuracy: 0.84
     - 4.9% improvement over Base_Aug
     - 8.6% improvement over baseline
   - F1 Score: 0.912
     - Indicates a strong balance of precision and recall
   - AUC: 0.566
     - Lower than other models, suggesting decision boundary issues

2. **Model Strengths**:
   - Improved feature extraction (evidenced by a higher F1 score)
   - Better handling of variations (shown by accuracy improvement)
   - More robust architecture with BatchNorm and Dropout

3. **Model Weaknesses**:
   - Decision boundary optimization is needed (shown by AUC)
   - Potential overfitting risks (complex architecture)
   - Computational overhead with additional layers

#### Performance Improvement Strategies
1. **Architectural Considerations**:
   - Evaluate BatchNorm impact
   - Optimize dropout rates
   - Consider architectural simplification

2. **Training Optimizations**:
   - Decision boundary tuning
   - Learning rate adjustment
   - Batch size optimization


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
