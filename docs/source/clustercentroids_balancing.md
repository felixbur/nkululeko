# ClusterCentroids Balancing Implementation

## Overview

ClusterCentroids is an under-sampling technique that has been implemented in Nkululeko to address class imbalance in datasets. This method reduces the size of the majority class by replacing clusters of majority samples with their centroids.

## How ClusterCentroids Works

1. **Clustering**: The algorithm applies K-means clustering to the majority class samples
2. **Centroid Calculation**: For each cluster, it calculates the centroid (mean of all points in the cluster)
3. **Replacement**: The entire cluster is replaced by its single centroid point
4. **Balancing**: The number of clusters (and thus resulting samples) equals the number of minority class samples

## Benefits

- **Dimensionality Reduction**: Reduces the dataset size while preserving the data distribution
- **Information Preservation**: Centroids represent the essential characteristics of each cluster
- **Computational Efficiency**: Smaller dataset leads to faster training times
- **Noise Reduction**: Clustering can help eliminate outliers and noise in the majority class

## Usage in Nkululeko

Add the following to your INI configuration file:

```ini
[FEATS]
type = ['your_feature_type']
scale = standard
balancing = clustercentroids
```

## Example Configuration

See `data/meld-st/exp_eng_deu_deu_mlp_clustercentroids.ini` for a complete example using ClusterCentroids with emotion2vec features and MLP classifier.

## Technical Details

- **Algorithm**: Uses scikit-learn's KMeans clustering by default
- **Random State**: Set to 42 for reproducible results
- **Sampling Strategy**: Uses 'auto' which targets all classes except the minority
- **Multi-class Support**: Works with both binary and multi-class classification problems

## When to Use ClusterCentroids

- When you have a large majority class that contains redundant samples
- When computational resources are limited and you need to reduce dataset size
- When you want to preserve the general distribution of the majority class while reducing its size
- As an alternative to random under-sampling when you want more representative samples

## Comparison with Other Methods

- **vs ROS (Random Over-Sampling)**: ClusterCentroids reduces dataset size instead of increasing it
- **vs SMOTE**: ClusterCentroids uses real data centroids instead of synthetic samples
- **vs ADASYN**: ClusterCentroids focuses on under-sampling rather than over-sampling
- **vs SMOTEENN**: ClusterCentroids is purely under-sampling, while SMOTEENN combines over and under-sampling

## References

- [Imbalanced-learn ClusterCentroids Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.ClusterCentroids.html)
- [Original Paper on Cluster-based Under-sampling](https://link.springer.com/chapter/10.1007/978-3-540-30115-8_22)
