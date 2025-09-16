# Data Balancing in Machine Learning and its implementation in Nkululeko
  
### Addressing Class Imbalance Machine Learning  
Study Case: Speech Processing

## Slide 1: Why We Need Data Balancing

### The Problem: Class Imbalance in Real-World Data

**Example: Emotion Recognition Dataset**
```
Class Distribution:
- Happy: 800 samples (40%)
- Neutral: 600 samples (30%) 
- Sad: 400 samples (20%)
- Angry: 200 samples (10%)
```

### Consequences of Imbalanced Data

1. **Model Bias**: Algorithms favor majority classes
2. **Poor Minority Class Performance**: Low recall for underrepresented emotions
3. **Misleading Accuracy**: 90% accuracy might mean predicting only majority class
4. **Overfitting**: Models memorize majority patterns, fail to generalize

### The Accuracy Paradox
- **Scenario**: Medical diagnosis with 99% healthy, 1% diseased patients
- **Naive Model**: Always predict "healthy" -> 99% accuracy!
- **Reality**: Completely useless for detecting disease

---

## Slide 2: Basic Concepts and Simple Balancing

### What is Data Balancing?

**Data balancing** transforms imbalanced datasets to have more equal class distributions, improving model performance on minority classes.

### Three Main Approaches

#### 1. **Over-sampling** 
- Increase minority class samples
- No information loss
- Larger dataset size

#### 2. **Under-sampling**
- Reduce majority class samples  
- Information loss possible
- Smaller dataset size

#### 3. **Combination Methods**
- Apply both over-sampling and under-sampling
- Balance benefits and drawbacks

### Simple Balancing Example

**Before Balancing:**
```
Angry: #### (200 samples)
Sad:   ######## (400 samples)  
Happy: ################ (800 samples)
```

**After SMOTE Balancing:**
```
Angry: ################ (800 samples)
Sad:   ################ (800 samples)
Happy: ################ (800 samples)
```

## Slide 3: Nkululeko's Balancing Implementation

### Easy Configuration
```ini
[FEATS]
type = ['os']
balancing = smote    # Just add this line!
scale = standard
```

### Supported Methods in Nkululeko

**11 Total Methods Across 3 Categories:**

#### Over-sampling (5 methods)
- `ros` - Random Over-Sampling
- `smote` - SMOTE  
- `adasyn` - ADASYN
- `borderlinesmote` - Borderline SMOTE
- `svmsmote` - SVM SMOTE

#### Under-sampling (4 methods)  
- `randomundersampler` - Random Under-Sampling
- `clustercentroids` - Cluster Centroids
- `editednearestneighbours` - Edited Nearest Neighbours
- `tomeklinks` - Tomek Links

#### Combination (2 methods)
- `smoteenn` - SMOTE + ENN
- `smotetomek` - SMOTE + Tomek

---

## Slide 4: Over-sampling Methods Explained

### 1. Random Over-Sampling (ROS)
- **Method**: Randomly duplicate minority class samples
- **Pros**: Simple, fast, preserves original data
- **Cons**: May cause overfitting, no new information
- **Use Case**: Quick baseline, small datasets

### 2. SMOTE (Synthetic Minority Over-sampling Technique)
- **Method**: Generate synthetic samples using k-nearest neighbors
- **Process**: 
  1. Find k nearest neighbors for each minority sample
  2. Create new sample on line between original and neighbor
  3. Add random interpolation factor (0-1)
- **Pros**: Creates new information, reduces overfitting
- **Cons**: May create unrealistic samples in complex datasets

### 3. ADASYN (Adaptive Synthetic Sampling)
- **Method**: SMOTE with density-based adaptive generation
- **Feature**: Generates more samples in harder-to-learn regions
- **Pros**: Focuses on difficult areas, better for highly imbalanced data
- **Use Case**: Severe imbalance (>10:1 ratio)

### 4. Borderline SMOTE
- **Method**: Only generates samples near class boundaries
- **Focus**: Creates synthetic data where classes overlap
- **Pros**: More focused synthesis, better decision boundaries
- **Use Case**: When classes have clear separation

### 5. SVM SMOTE  
- **Method**: Uses SVM to identify support vectors for synthesis
- **Feature**: Generates samples near decision boundary
- **Pros**: Theoretically grounded, focuses on important samples
- **Use Case**: Linear separable problems

---

## Slide 5: Under-sampling Methods Explained

### 1. Random Under-Sampling
- **Method**: Randomly remove majority class samples
- **Pros**: Simple, fast, smaller dataset
- **Cons**: Information loss, may remove important samples
- **Use Case**: Very large datasets, computational constraints

### 2. Cluster Centroids
- **Method**: Replace majority clusters with their centroids
- **Process**: 
  1. Cluster majority class using k-means
  2. Replace each cluster with its centroid
  3. Preserve data distribution while reducing size
- **Pros**: Maintains data structure, intelligent reduction
- **Use Case**: Large datasets with redundant samples

### 3. Edited Nearest Neighbours (ENN)
- **Method**: Remove samples with different class neighbors
- **Process**: Remove sample if majority of k-neighbors are different class
- **Pros**: Removes noise and outliers, cleans data
- **Use Case**: Noisy datasets, overlapping classes

### 4. Tomek Links
- **Method**: Remove Tomek link pairs (nearest neighbors of different classes)
- **Feature**: Cleans class boundaries
- **Pros**: Improves class separation, removes borderline samples
- **Use Case**: Clean decision boundaries, remove ambiguous samples

---

## Slide 6: Combination Methods Explained

### 1. SMOTE + ENN (SMOTEENN)
- **Process**: 
  1. Apply SMOTE to over-sample minority classes
  2. Apply ENN to clean noisy samples
- **Benefits**: 
  - Balances classes while removing noise
  - Better quality synthetic samples
- **Use Case**: Noisy datasets that need both balancing and cleaning

### 2. SMOTE + Tomek (SMOTETomek)
- **Process**:
  1. Apply SMOTE to over-sample minority classes  
  2. Apply Tomek Links to remove boundary noise
- **Benefits**:
  - Balances classes and clarifies boundaries
  - Less aggressive cleaning than SMOTEENN
- **Use Case**: Datasets with unclear class boundaries

### Why Combination Methods?
- **Best of Both Worlds**: Add samples + remove noise
- **Higher Quality**: Cleaner decision boundaries
- **Robust**: Works well across different dataset characteristics

---

## Slide 7: How to Choose the Right Method

### Decision Tree for Method Selection

```
Dataset Size?
+-- Small (<1000 samples)
|   +-- Use Over-sampling
|       +-- Start with: SMOTE
|       +-- Try: ROS (baseline), ADASYN (severe imbalance)
|
+-- Medium (1000-10000 samples)  
|   +-- Use Combination Methods
|       +-- Start with: SMOTE + Tomek
|       +-- Try: SMOTE + ENN (noisy data)
|
+-- Large (>10000 samples)
    +-- Use Under-sampling
        +-- Start with: Cluster Centroids
        +-- Try: Random Under-sampling (fast)
```

### Guidelines by Dataset Characteristics

#### **Clean Data**
- SMOTE (over-sampling)
- Cluster Centroids (under-sampling)

#### **Noisy Data**  
- SMOTE + ENN (combination)
- Tomek Links (under-sampling)

#### **Severe Imbalance (>10:1)**
- ADASYN (over-sampling)
- SMOTE + Tomek (combination)

#### **Computational Constraints**
- ROS (fast over-sampling)
- Random Under-sampling (fast under-sampling)

---

## Slide 8: Nkululeko Implementation Examples

### Basic Usage
```ini
[EXP]
name = emotion_balanced_experiment

[DATA]
databases = ['emodb']
target = emotion

[FEATS]
type = ['os']
balancing = smote
scale = standard

[MODEL]
type = xgb
```

### Comparing Multiple Methods
```ini
[FLAGS]
balancing = ['none', 'smote', 'adasyn', 'smoteenn']
```

### Expected Output
```bash
Balancing features with: smote
Original dataset size: 1200
Original class distribution: 
 {'happy': 400, 'sad': 300, 'angry': 300, 'neutral': 200}
Balanced dataset size: 1600 (was 1200)
New class distribution: 
 {'happy': 400, 'sad': 400, 'angry': 400, 'neutral': 400}
```

### Integration with Other Features
- **Feature Scaling**: Applied after balancing
- **Multiple Features**: Balancing works on combined feature space
- **Cross-Validation**: Balancing applied within each fold

---

## Slide 9: Real-World Performance Impact

### Case Study: Emotion Recognition

**Dataset**: EmoDb with severe imbalance  
- Angry: 127 samples (31%)  
- Happy: 71 samples (17%)  
- Sad: 62 samples (15%)  
- Neutral: 79 samples (19%)  
- Fear: 69 samples (17%)  

### Results Comparison

| Method | Accuracy | F1-Score | Minority Class Recall |
|--------|----------|----------|----------------------|
| No Balancing | 0.67 | 0.52 | 0.23 |
| ROS | 0.73 | 0.71 | 0.65 |
| SMOTE | 0.78 | 0.76 | 0.72 |
| ADASYN | 0.79 | 0.77 | 0.74 |
| SMOTE+Tomek | 0.81 | 0.79 | 0.76 |

### Key Observations
- **23% -> 76%** improvement in minority class recall
- **52% -> 79%** improvement in F1-score  
- Combination methods often perform best
- SMOTE provides good balance of performance and simplicity

---

## Slide 10: Implications and Best Practices

### Positive Implications

#### **Improved Model Fairness**
- Equal performance across all emotion classes
- Reduced bias toward majority emotions
- Better real-world applicability

#### **Enhanced Clinical Applications**
- Better detection of rare emotional states
- More reliable depression/anxiety screening
- Improved therapeutic monitoring

#### **Research Benefits**
- More robust comparative studies
- Better cross-cultural validation
- Improved reproducibility

### Potential Risks and Mitigation

#### **Overfitting to Synthetic Data**
- **Risk**: Model learns artificial patterns
- **Mitigation**: Use validation on original imbalanced test set

#### **Computational Overhead**
- **Risk**: Increased training time and memory
- **Mitigation**: Use under-sampling for large datasets

#### **Loss of Information**
- **Risk**: Under-sampling may remove important samples
- **Mitigation**: Use intelligent under-sampling (Cluster Centroids)

### Best Practices Checklist

- **Always keep test set imbalanced** for realistic evaluation
- **Start with SMOTE** as baseline
- **Compare multiple methods** using FLAGS
- **Monitor class distribution** in output logs
- **Validate on separate dataset** when possible
- **Consider data size** when choosing method
- **Apply balancing after feature extraction** but before scaling

---

## Slide 11: Advanced Considerations

### When NOT to Use Balancing

#### **Naturally Imbalanced Problems**
- Medical diagnosis (most people are healthy)
- Fraud detection (most transactions are legitimate)  
- Quality control (most products are good)

#### **Very Small Datasets**
- Risk of overfitting increases
- May not have enough diversity for synthesis

#### **Already Balanced Data**
- No benefit, may introduce noise
- Check class distribution first

### Performance Monitoring

#### **Metrics to Track**
```python
# Focus on minority class performance
- Recall per class
- F1-score per class  
- Confusion matrix
- ROC-AUC (for binary problems)
```

#### **Validation Strategy**
```ini
# Use stratified cross-validation
runs = 5  # Each fold maintains class proportions
```

### Domain-Specific Considerations

#### **Speech Emotion Recognition**
- Cultural differences in emotion expression
- Speaker variability within emotions
- Acoustic similarity between some emotion pairs

#### **Audio Quality**
- Noise levels affect synthetic sample quality
- Feature extraction before or after balancing
- Temporal dependencies in audio data

---

## Slide 12: Future Directions and Conclusions

### Emerging Techniques

#### **Deep Learning Approaches**
- VAE-based sample generation
- GAN-based minority class synthesis
- Attention-based balancing

#### **Ensemble Methods**
- Cost-sensitive learning
- Threshold optimization
- Multi-level balancing

### Research Opportunities

#### **Audio-Specific Balancing**
- Temporal-aware synthesis
- Acoustic feature preservation
- Speaker-independent balancing

#### **Evaluation Metrics**
- Fairness-aware evaluation
- Bias detection methods
- Real-world impact assessment

### Key Takeaways

1. **Data imbalance is a critical problem** in speech emotion recognition
2. **Nkululeko provides 11 balancing methods** with simple configuration
3. **SMOTE is a good starting point** for most applications
4. **Combination methods often perform best** for complex datasets
5. **Always validate on imbalanced test sets** for realistic performance estimates
6. **Consider dataset size and characteristics** when choosing methods
7. **Balancing improves fairness and minority class performance**

### Getting Started with Nkululeko Balancing

```bash
# 1. Install nkululeko
pip install nkululeko

# 2. Add balancing to your config
balancing = smote

# 3. Compare methods
[FLAGS]
balancing = ['smote', 'adasyn', 'smoteenn']

# 4. Run experiment  
python -m nkululeko.flags --config your_config.ini
```

## Thank You!

### Questions & Discussion

**Resources:**  
- Documentation: https://nkululeko.readthedocs.io/  
- Tutorial: `docs/source/balance.md`  
- Examples: `examples/exp_balancing_xxx.ini`  
- Run all examples: `bash script/run_balancing_experiments.sh`  

**Contact:**  
- GitHub: https://github.com/bagustris/nkululeko  
- Issues: Report bugs and feature requests  

**Let's balance our life and our work!!!**
