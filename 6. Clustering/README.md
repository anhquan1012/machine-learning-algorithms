# K-Medians clustering 
K-Medians clustering is similar to K-Means, but instead of using the mean to update cluster centers, it uses the median. This makes it more robust to outliers.

1. Initialize k cluster centroids randomly.
2. Assign each data point to the nearest centroid (using Manhattan distance).
3. Update centroids by computing the median of all assigned points.
4. Repeat until centroids no longer change or max iterations are reached.