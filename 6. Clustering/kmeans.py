# K-means clustering is a technique used to organize data into groups based on their similarity
# Steps:
# 1. Initialize k cluster centroids randomly.
# 2. Assign each data point to the nearest centroid.
# 3. Update centroids by computing the mean of all assigned points.
# 4. Repeat until centroids no longer change or max iterations are reached.


import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol  # Tolerance for centroid movement
        self.centroids = None

    def fit(self, X):
        # Step 1: Initialize centroids randomly from the dataset
        np.random.seed(42)  # For reproducibility
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iters):
            # Step 2: Assign each data point to the nearest centroid
            clusters = self._assign_clusters(X)
            
            # Step 3: Compute new centroids
            new_centroids = self._compute_centroids(X, clusters)
            
            # Step 4: Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        """Assigns each data point to the nearest centroid"""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)  # Compute distance to centroids
        return np.argmin(distances, axis=1)  # Get the index of the closest centroid

    def _compute_centroids(self, X, clusters):
        """Computes new centroids as the mean of assigned points"""
        return np.array([X[clusters == i].mean(axis=0) for i in range(self.k)])

    def predict(self, X):
        """Predicts the closest cluster for each data point"""
        return self._assign_clusters(X)

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    import ipdb;ipdb.set_trace()
    # Train K-Means
    kmeans = KMeans(k=3)
    kmeans.fit(X)

    # Predict cluster assignments
    y_pred = kmeans.predict(X)

    # Plot clusters
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=200, c='red', marker='X', label="Centroids")
    plt.legend()
    plt.show()
