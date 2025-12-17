"""
Clustering Analysis for İzmir Public Transport Data
Academic Implementation - CE-477 Project

Implements:
1. K-Means (Partitioning)
2. Agglomerative Clustering (Hierarchical - Single & Complete Linkage)
3. DBSCAN (Density-based)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import zscore
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 300})


class TransportClustering:
    """Clustering analysis for İzmir public transport data."""

    def __init__(self, output_dir='clustering_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.dbscan_scaler = StandardScaler()
        self.results = {}

    def load_and_prepare_data(self, file_path):
        """Load and prepare transport data with domain-specific features."""
        print("=" * 80)
        print("DATA PREPARATION")
        print("=" * 80)

        df = pd.read_csv(file_path, sep=";")
        df["DATE"] = pd.to_datetime(df["DATE"], format='mixed', dayfirst=True)

        # Remove low-quality institutions
        forbidden = ['Metro Maskematik', 'Özdeniz Maskematik', 'BÖSİM',
                     'İzmir BB - Atik Yönetimi DB', 'Nostaljik Tramvay',
                     'Teleferik', 'İzmir Doğal Yaşam Parkı']
        df = df[~df['INSTITUTION'].isin(forbidden)]

        passenger_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD',
                          'TICKET', 'CHILD', 'PERSONNEL', 'FREE', 'BANK CARD']

        # Daily aggregation
        daily = df.groupby('DATE')[passenger_cols].sum().reset_index()
        daily['TOTAL_PASSENGERS'] = daily[passenger_cols].sum(axis=1)

        # Temporal features
        daily['WEEKDAY'] = daily['DATE'].dt.dayofweek
        daily['MONTH'] = daily['DATE'].dt.month
        daily['IS_WEEKEND'] = (daily['WEEKDAY'] >= 5).astype(int)

        # Proportions (stable features)
        for col in passenger_cols:
            daily[f'{col}_PCT'] = daily[col] / (daily['TOTAL_PASSENGERS'] + 1)

        # Sort for time-series features
        daily = daily.sort_values('DATE').reset_index(drop=True)

        # Rolling statistics
        daily['DEMAND_MA7'] = daily['TOTAL_PASSENGERS'].rolling(7, min_periods=1).mean()
        daily['DEMAND_STD7'] = daily['TOTAL_PASSENGERS'].rolling(7, min_periods=1).std()

        # Transport-specific indicators
        daily['COMMUTER_INTENSITY'] = daily['STUDENT_PCT'] + daily['FULL_FARE_PCT']
        daily['LEISURE_INDEX'] = daily['IS_WEEKEND'] * daily['BANK CARD_PCT'] + daily['SIXTY_YEARS_OLD_PCT']

        daily = daily.dropna()

        print(f"✓ Prepared {len(daily)} daily records")
        print(f"✓ Date range: {daily['DATE'].min().date()} to {daily['DATE'].max().date()}")

        return daily

    def select_features(self, df):
        """Select features for clustering."""
        feature_cols = [
            'TOTAL_PASSENGERS',
            'DEMAND_MA7',
            'STUDENT_PCT',
            'FULL_FARE_PCT',
            'BANK CARD_PCT',
            'SIXTY_YEARS_OLD_PCT',
            'IS_WEEKEND',
            'COMMUTER_INTENSITY',
            'LEISURE_INDEX'
        ]

        self.feature_cols = feature_cols

        X = df[feature_cols].copy()

        # Remove outliers
        z_scores = np.abs(zscore(X))
        X_clean = X[(z_scores < 3).all(axis=1)]
        df_clean = df.loc[X_clean.index].copy()

        print(f"\n✓ Selected {len(feature_cols)} features")
        print(f"✓ Removed {len(X) - len(X_clean)} outliers")

        X_scaled = self.scaler.fit_transform(X_clean)

        return X_scaled, feature_cols, df_clean

    # ==========================================
    # 1. K-MEANS
    # ==========================================

    def apply_kmeans(self, X, features, df):
        """Apply K-Means clustering."""
        print("\n" + "=" * 80)
        print("1. K-MEANS CLUSTERING")
        print("=" * 80)

        # Feature weighting for interpretability (best practice)
        X_kmeans = X.copy()

        """
        weight_map = {
            'TOTAL_PASSENGERS': 1.3,
            'IS_WEEKEND': 1.2,
            'COMMUTER_INTENSITY': 1.1
        }


        for fname, w in weight_map.items():
            if fname in features:
                idx = features.index(fname)
                X_kmeans[:, idx] *= w
        """

        # Find optimal K
        K_range = range(2, 9)
        inertias = []
        silhouettes = []

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = kmeans.fit_predict(X_kmeans)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_kmeans, labels))

        # Elbow plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(K_range, inertias, 'o-', linewidth=2.5, markersize=8, color='#2E86AB')
        ax.set_xlabel('Number of Clusters (k)', fontweight='bold')
        ax.set_ylabel('Inertia', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/kmeans_elbow.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Silhouette plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(K_range, silhouettes, 'o-', linewidth=2.5, markersize=8, color='#A23B72')
        ax.set_xlabel('Number of Clusters (k)', fontweight='bold')
        ax.set_ylabel('Silhouette Score', fontweight='bold')
        ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/kmeans_silhouette.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # MODIFYING OPTIMAL K SELECTION
        # ---------------------------------------------------------
        # Previous logic (auto-select max silhouette):
        # optimal_k = K_range[np.argmax(silhouettes)]

        # New logic (Hardcoded based on Elbow/Domain Analysis):
        optimal_k = 5
        print(f"\n✓ Optimal k selected manually = {optimal_k}")

        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=30)
        labels = kmeans.fit_predict(X_kmeans)

        # Report cluster sizes
        print("\nCluster Sizes:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} days ({count / len(labels) * 100:.1f}%)")

        # Report centroids
        centroids = self.scaler.inverse_transform(kmeans.cluster_centers_)
        centroid_df = pd.DataFrame(centroids, columns=features)
        print("\nCluster Centroids (Selected Features):")
        print(centroid_df[['TOTAL_PASSENGERS', 'STUDENT_PCT', 'COMMUTER_INTENSITY', 'IS_WEEKEND']].round(3))

        # Visualization
        self._visualize_clusters(X_kmeans, labels, "kmeans")

        # Metrics
        sil = silhouette_score(X_kmeans, labels)
        cal = calinski_harabasz_score(X_kmeans, labels)
        dav = davies_bouldin_score(X_kmeans, labels)

        print(f"\nQuality Metrics:")
        print(f"  Silhouette Score: {sil:.3f}")
        print(f"  Calinski-Harabasz: {cal:.1f}")
        print(f"  Davies-Bouldin: {dav:.3f}")

        self.results['kmeans'] = {
            'labels': labels,
            'optimal_k': optimal_k,
            'centroids': centroid_df,
            'silhouette': sil,
            'calinski': cal,
            'davies_bouldin': dav
        }

        # Semantic merge: assign regime names to clusters
        # NOT: k=5 olduğu için buradaki isimler tam oturmayabilir,
        # çıktıya göre güncellenmesi gerekebilir ama kod hata vermez.
        semantic_map = {
            0: 'Cluster 0 (Check Centroids)',
            1: 'Cluster 1 (Check Centroids)',
            2: 'Cluster 2 (Check Centroids)',
            3: 'Cluster 3 (Check Centroids)',
            4: 'Cluster 4 (Check Centroids)',
            5: 'Extra Cluster',
            6: 'Extra Cluster',
            7: 'Extra Cluster'
        }

        df['KMEANS_SEMANTIC'] = pd.Series(labels).map(semantic_map)

        return labels

    # ==========================================
    # 2. HIERARCHICAL
    # ==========================================

    def apply_hierarchical(self, X, features, df):
        """Apply Hierarchical clustering with single and complete linkage."""
        print("\n" + "=" * 80)
        print("2. HIERARCHICAL CLUSTERING")
        print("=" * 80)

        # Sample for dendrogram
        sample_size = min(300, len(X))
        np.random.seed(42)
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_idx]

        # Single linkage dendrogram
        print("\n→ Computing Single Linkage dendrogram...")
        Z_single = linkage(X_sample, method='single')

        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(Z_single, ax=ax, no_labels=True, color_threshold=0, above_threshold_color='#2E86AB')
        ax.set_xlabel('Sample Index', fontweight='bold')
        ax.set_ylabel('Distance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hierarchical_single_linkage.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Complete linkage dendrogram
        print("→ Computing Complete Linkage dendrogram...")
        Z_complete = linkage(X_sample, method='complete')

        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(Z_complete, ax=ax, no_labels=True, color_threshold=0, above_threshold_color='#A23B72')
        ax.set_xlabel('Sample Index', fontweight='bold')
        ax.set_ylabel('Distance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hierarchical_complete_linkage.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Apply complete linkage clustering
        n_clusters = 4
        print(f"\n→ Applying Agglomerative Clustering (Complete Linkage, k={n_clusters})...")

        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        labels = agg.fit_predict(X)

        # Report cluster sizes
        print("\nCluster Sizes:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} days ({count / len(labels) * 100:.1f}%)")

        # Visualization
        self._visualize_clusters(X, labels, "hierarchical")

        # Metrics
        sil = silhouette_score(X, labels)
        cal = calinski_harabasz_score(X, labels)
        dav = davies_bouldin_score(X, labels)

        print(f"\nQuality Metrics:")
        print(f"  Silhouette Score: {sil:.3f}")
        print(f"  Calinski-Harabasz: {cal:.1f}")
        print(f"  Davies-Bouldin: {dav:.3f}")

        self.results['hierarchical'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette': sil,
            'calinski': cal,
            'davies_bouldin': dav
        }

        return labels

    # ==========================================
    # 3. DBSCAN
    # ==========================================

    def apply_dbscan(self, X, features, df):
        """Apply DBSCAN as density-based regime + anomaly detector."""
        print("\n" + "=" * 80)
        print("3. DBSCAN CLUSTERING (DENSITY-BASED)")
        print("=" * 80)

        # Use DBSCAN-specific feature subset (best practice)
        dbscan_features = [
            'TOTAL_PASSENGERS',
            'DEMAND_MA7',
            'COMMUTER_INTENSITY',
            'IS_WEEKEND'
        ]

        X_db = df[dbscan_features].copy()
        X_db_scaled = self.dbscan_scaler.fit_transform(X_db)

        # Parameter selection
        k = int(np.log(len(X_db_scaled)) * 2)
        print(f"→ Computing k-distance graph (k={k})...")

        neighbors = NearestNeighbors(n_neighbors=k)
        distances, _ = neighbors.fit(X_db_scaled).kneighbors(X_db_scaled)
        distances = np.sort(distances[:, k - 1])

        eps = np.percentile(distances, 85)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(distances, linewidth=2)
        ax.axhline(y=eps, color='red', linestyle='--',
                   label=f'Selected ε = {eps:.2f}')
        ax.set_xlabel('Sorted Points')
        ax.set_ylabel(f'{k}-NN Distance')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/dbscan_k_distance.png", dpi=300)
        plt.close()

        print(f"✓ Selected ε={eps:.3f}, min_samples={k}")

        dbscan = DBSCAN(eps=eps, min_samples=k)
        labels = dbscan.fit_predict(X_db_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        print(f"\n✓ Found {n_clusters} clusters")
        print(f"✓ Noise points: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")

        self._visualize_clusters(X_db_scaled, labels, "dbscan", has_noise=True)

        # Explicit anomaly flag
        df['DBSCAN_ANOMALY'] = (labels == -1).astype(int)

        mask = labels != -1
        if mask.sum() > 1 and len(set(labels[mask])) > 1:
            sil = silhouette_score(X_db_scaled[mask], labels[mask])
            cal = calinski_harabasz_score(X_db_scaled[mask], labels[mask])
            dav = davies_bouldin_score(X_db_scaled[mask], labels[mask])
        else:
            sil = cal = dav = None

        self.results['dbscan'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps': eps,
            'min_samples': k,
            'silhouette': sil,
            'calinski': cal,
            'davies_bouldin': dav
        }

        return labels

    # ==========================================
    # VISUALIZATION
    # ==========================================

    def _visualize_clusters(self, X, labels, method, has_noise=False):
        """Visualize clusters using PCA projection."""
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))

        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 10)))

        for lbl in unique_labels:
            mask = labels == lbl
            if has_noise and lbl == -1:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c='lightgray', s=20, alpha=0.4,
                           label='Noise', edgecolors='none')
            else:
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=[colors[lbl]], s=50, alpha=0.7,
                           label=f'Cluster {lbl}',
                           edgecolors='black', linewidths=0.5)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Paper-style plot titles
        if method == "kmeans":
            ax.set_title("K-Means Clustering – Global Demand Regimes", fontweight='bold')
        elif method == "dbscan":
            ax.set_title("DBSCAN Clustering – Density-Based Travel Regimes", fontweight='bold')
        elif method == "hierarchical":
            ax.set_title("Hierarchical Clustering – Baseline Structure", fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{method}_clusters.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ==========================================
    # COMPARISON
    # ==========================================

    def generate_comparison(self):
        """Generate comparison report."""
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        comparison = []
        for method, res in self.results.items():
            comparison.append({
                'Method': method.upper(),
                'Clusters': res.get('optimal_k') or res.get('n_clusters', '-'),
                'Silhouette': res.get('silhouette', 0) if res.get('silhouette') else 0,
                'Calinski-Harabasz': res.get('calinski', 0) if res.get('calinski') else 0,
                'Davies-Bouldin': res.get('davies_bouldin', 0) if res.get('davies_bouldin') else 0
            })

        comp_df = pd.DataFrame(comparison)
        print("\n" + comp_df.to_string(index=False))
        comp_df.to_csv(f"{self.output_dir}/comparison_metrics.csv", index=False)

        # Comparison plot
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = comp_df['Method'].tolist()
        sil_scores = comp_df['Silhouette'].tolist()

        bars = ax.bar(methods, sil_scores,
                      color=['#3498db', '#e74c3c', '#2ecc71'],
                      edgecolor='black', linewidth=1.5, alpha=0.85)

        ax.set_ylabel('Silhouette Score', fontweight='bold')
        ax.axhline(y=0.3, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, sil_scores):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f'{val:.3f}', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/comparison_silhouette.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Results saved to: {self.output_dir}/")


"""
INTERPRETATION SUMMARY:
K-Means clustering captures global and recurring public transport demand regimes.
DBSCAN is employed as a density-based method to detect irregular and anomalous travel patterns.
This complementary use follows best practices in recent urban mobility clustering literature.
"""

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    FILE_PATH = "izmirim-kart-ulasim-istatistikleri-guncel-extended.csv"

    if not os.path.exists(FILE_PATH):
        print(f" ERROR: File not found: {FILE_PATH}")
        exit(1)

    print("=" * 80)
    print("CLUSTERING ANALYSIS - İZMİR PUBLIC TRANSPORT")
    print("=" * 80)

    # Initialize
    analyzer = TransportClustering(output_dir='clustering_results')

    # Prepare data
    df = analyzer.load_and_prepare_data(FILE_PATH)
    X_scaled, features, df_clean = analyzer.select_features(df)

    # Apply clustering algorithms
    analyzer.apply_kmeans(X_scaled, features, df_clean.copy())
    analyzer.apply_hierarchical(X_scaled, features, df_clean.copy())
    analyzer.apply_dbscan(X_scaled, features, df_clean.copy())

    # Generate comparison
    analyzer.generate_comparison()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)