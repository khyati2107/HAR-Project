import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import os

os.makedirs('outputs', exist_ok=True)


class HybridKMeans:
    def __init__(self, n_clusters=6, outlier_threshold=0.9, random_state=42):
        self.n_clusters = n_clusters
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        self.kmeans = None
        self.labels_ = None

    def fit(self, X):
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(X)
        mean_dists = dists.mean(axis=1)
        threshold = np.quantile(mean_dists, self.outlier_threshold)
        self.filtered_X = X[mean_dists <= threshold]
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(self.filtered_X)
        self.labels_ = self.kmeans.predict(X)
        return self

    def predict(self, X):
        return self.kmeans.predict(X)


def calculate_dunn_index(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0

    centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])

    inter_dists = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            d = np.linalg.norm(centroids[i] - centroids[j])
            inter_dists.append(d)

    intra_dists = []
    for i, k in enumerate(unique_labels):
        cluster_points = X[labels == k]
        dists_to_centroid = np.linalg.norm(cluster_points - centroids[i], axis=1)
        intra_dists.append(2 * dists_to_centroid.max())

    if max(intra_dists) == 0:
        return 0

    return min(inter_dists) / max(intra_dists)


def evaluate_clustering(X, labels, method_name="Clustering"):
    print(f"\nEvaluation Metrics for {method_name}:")
    unique_labels = np.unique(labels)

    wcss = sum(
        np.sum((X[labels == k] - X[labels == k].mean(axis=0))**2)
        for k in unique_labels
    )
    print(f"  WCSS (Sum of Squared Error): {wcss:.2f}")

    if len(unique_labels) > 1:
        sil_score = silhouette_score(X, labels, sample_size=2000, random_state=42)
        print(f"  Silhouette Score: {sil_score:.3f}")

        db_score = davies_bouldin_score(X, labels)
        print(f"  Davies-Bouldin Index: {db_score:.3f}")

        dunn_index = calculate_dunn_index(X, labels)
        print(f"  Dunn Index: {dunn_index:.3f}")


def run_clustering(X, n_clusters=6):
    print("Running clustering methods...\n")
    clustering_results = {}

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    evaluate_clustering(X, kmeans_labels, "K-Means")
    clustering_results['K-Means'] = kmeans_labels

    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agglo_labels = agglo.fit_predict(X)
    evaluate_clustering(X, agglo_labels, "Agglomerative Clustering")
    clustering_results['Agglomerative'] = agglo_labels

    dbscan = DBSCAN(eps=5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X)
    evaluate_clustering(X, dbscan_labels, "DBSCAN")
    clustering_results['DBSCAN'] = dbscan_labels

    hybrid = HybridKMeans(n_clusters=n_clusters, outlier_threshold=0.9)
    hybrid.fit(X)
    evaluate_clustering(X, hybrid.labels_, "Hybrid Distance-Filtered KMeans")
    clustering_results['HybridKMeans'] = hybrid.labels_

    # Plot all clustering results
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, labels) in zip(axes.flatten(), clustering_results.items()):
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.5, s=10)
        ax.set_title(f'{name} Clusters')
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
    plt.suptitle('Clustering Results (PCA 2D View)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/clustering_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    return clustering_results