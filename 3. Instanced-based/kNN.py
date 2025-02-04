# K-Nearest Neighbors (KNN) is a supervised learning algorithm used for classification and regression.
# It is a simple way to classify things by looking at what’s nearby
# Steps:
# 1. Storing all training data.
# 2. Calculating distances between a test point and all training points.
# 3. Selecting the k closest points.
# 4. Voting for the most common class (classification) or averaging values (regression).


import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, mode="classification"):
        self.k = k
        self.mode = mode  # "classification" or "regression"
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Stores the training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predicts labels for the given data points."""
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """Finds k-nearest neighbors and returns the predicted label."""
        # Compute distances (Euclidean distance)
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get corresponding labels
        k_neighbors = self.y_train[k_indices]

        if self.mode == "classification":
            # Return the most common label
            return Counter(k_neighbors).most_common(1)[0][0]
        elif self.mode == "regression":
            # Return the average of k-nearest values
            return np.mean(k_neighbors)
        else:
            raise ValueError("Mode should be 'classification' or 'regression'.")

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Generate sample classification data
    X, y = make_classification(
        n_samples=200, 
        n_features=2,  # Total number of features
        n_informative=2,  # Must be <= n_features
        n_redundant=0,  # Should not exceed the remaining feature count
        n_repeated=0,  # Should not exceed the remaining feature count
        n_classes=2, 
        random_state=42
    )
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN classifier
    knn = KNN(k=5, mode="classification")
    knn.fit(X_train, y_train)

    # Predict test labels
    y_pred = knn.predict(X_test)

    # Plot results
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap="coolwarm", alpha=0.6, marker="o", edgecolors="k")
    plt.title("KNN Classification Results")
    plt.show()
