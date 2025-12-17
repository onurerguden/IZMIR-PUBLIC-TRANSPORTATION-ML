"""
Professional Clustering Analysis for İzmir Public Transport Data
Best Practices Implementation - 2025 Standards

Algorithms:
1. K-Means (Partitioning)
2. Agglomerative Clustering (Hierarchical)
3. DBSCAN (Density-based)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10, 'figure.figsize': (14, 8)})


class TransportClusteringAnalysis:
    """
    Professional clustering analysis for public transport data.
    Implements industry best practices for unsupervised learning.
    """

    def __init__(self, output_dir='clustering_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scaler = RobustScaler()  # Better for outliers
        self.results = {}

    def prepare_clustering_data(self, file_path):
        """
        Load and engineer features specifically for clustering analysis.
        Focus: Daily demand patterns, temporal behavior, fare distributions.
        """
        print("=" * 80)
        print("LOADING AND PREPARING DATA FOR CLUSTERING")
        print("=" * 80)

        df = pd.read_csv(file_path, sep=";")
        df["DATE"] = pd.to_datetime(df["DATE"], format='mixed', dayfirst=True)

        # Remove low-quality institutions
        forbidden = ['Metro Maskematik', 'Özdeniz Maskematik', 'BÖSIM',
                     'İzmir BB - Atik Yönetimi DB', 'Nostaljik Tramvay',
                     'Teleferik', 'İzmir Doğal Yaşam Parkı']
        df = df[~df['INSTITUTION'].isin(forbidden)]

        # Aggregate by date to get daily system-wide patterns
        passenger_cols = ['FULL_FARE', 'STUDENT', 'TEACHER', 'SIXTY_YEARS_OLD',
                          'TICKET', 'CHILD', 'PERSONNEL', 'FREE', 'BANK CARD']

        # Daily aggregation
        daily = df.groupby('DATE')[passenger_cols].sum().reset_index()

        # Feature Engineering for Clustering
        daily['TOTAL_PASSENGERS'] = daily[passenger_cols].sum(axis=1)
        daily['WEEKDAY'] = daily['DATE'].dt.dayofweek
        daily['MONTH'] = daily['DATE'].dt.month
        daily['YEAR'] = daily['DATE'].dt.year
        daily['IS_WEEKEND'] = (daily['WEEKDAY'] >= 5).astype(int)
        daily['DAY_OF_YEAR'] = daily['DATE'].dt.dayofyear

        # Proportions (more stable than raw counts)
        for col in passenger_cols:
            daily[f'{col}_RATIO'] = daily[col] / (daily['TOTAL_PASSENGERS'] + 1)

        # Temporal features (cyclical encoding)
        daily['MONTH_SIN'] = np.sin(2 * np.pi * daily['MONTH'] / 12)
        daily['MONTH_COS'] = np.cos(2 * np.pi * daily['MONTH'] / 12)
        daily['DAY_SIN'] = np.sin(2 * np.pi * daily['WEEKDAY'] / 7)
        daily['DAY_COS'] = np.cos(2 * np.pi * daily['WEEKDAY'] / 7)

        # Lag features (rolling patterns)
        daily = daily.sort_values('DATE')
        daily['TOTAL_LAG_1'] = daily['TOTAL_PASSENGERS'].shift(1)
        daily['TOTAL_LAG_7'] = daily['TOTAL_PASSENGERS'].shift(7)
        daily['ROLL_MEAN_7'] = daily['TOTAL_PASSENGERS'].rolling(7, min_periods=1).mean()
        daily['ROLL_STD_7'] = daily['TOTAL_PASSENGERS'].rolling(7, min_periods=1).std()

        # === Literature-driven behavioral proxies ===
        # Morning vs Evening peak proxies (daily-level approximation)
        daily['MORNING_PEAK_PROXY'] = daily['TOTAL_LAG_1'] / (daily['ROLL_MEAN_7'] + 1)
        daily['EVENING_PEAK_PROXY'] = daily['TOTAL_LAG_7'] / (daily['ROLL_MEAN_7'] + 1)

        # Commuter intensity (regular weekday demand)
        daily['COMMUTER_INDEX'] = (
            daily['TOTAL_PASSENGERS'] /
            (daily.groupby('WEEKDAY')['TOTAL_PASSENGERS'].transform('mean') + 1)
        )

        # Visitor / tourist proxy (weekend + bank card usage)
        daily['VISITOR_PROXY'] = daily['BANK CARD_RATIO'] * daily['IS_WEEKEND']

        daily = daily.dropna()

        print(f"✓ Prepared {len(daily)} daily records")
        print(f"✓ Date range: {daily['DATE'].min()} to {daily['DATE'].max()}")

        return daily

    def select_features_for_clustering(self, df):
        """
        Select meaningful features for clustering.
        Strategy: Mix of volume, proportions, and temporal patterns.
        """
        # Core features: Demand volume + Fare type distribution + Temporal + behavioral proxies
        feature_cols = [
            'TOTAL_PASSENGERS',
            'STUDENT_RATIO', 'FULL_FARE_RATIO', 'BANK CARD_RATIO',
            'TEACHER_RATIO', 'SIXTY_YEARS_OLD_RATIO',
            'IS_WEEKEND',
            'MONTH_SIN', 'MONTH_COS', 'DAY_SIN', 'DAY_COS',
            'TOTAL_LAG_1', 'TOTAL_LAG_7', 'ROLL_MEAN_7', 'ROLL_STD_7',
            'MORNING_PEAK_PROXY', 'EVENING_PEAK_PROXY',
            'COMMUTER_INDEX', 'VISITOR_PROXY'
        ]

        X = df[feature_cols].copy()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        print(f"✓ Selected {X.shape[1]} features for clustering")
        print(f"  Features: {', '.join(feature_cols[:5])}...")

        return X_scaled, feature_cols, df

    # ==========================================
    # 1. K-MEANS CLUSTERING (Partitioning)
    # ==========================================

    def apply_kmeans(self, X, feature_names, df, n_clusters_range=(2, 10)):
        """
        Apply K-Means with optimal cluster selection using Elbow + Silhouette.
        """
        print("\n" + "=" * 80)
        print("1. K-MEANS CLUSTERING (Partitioning Algorithm)")
        print("=" * 80)

        # Find optimal K
        inertias = []
        silhouettes = []
        K_range = range(n_clusters_range[0], n_clusters_range[1] + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))

        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
        ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontweight='bold')
        ax1.set_title('Elbow Method for Optimal K', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.plot(K_range, silhouettes, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (K)', fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontweight='bold')
        ax2.set_title('Silhouette Analysis', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/01_kmeans_optimal_k.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Choose K with best silhouette
        optimal_k = K_range[np.argmax(silhouettes)]
        print(f"✓ Optimal K selected: {optimal_k} (Silhouette: {max(silhouettes):.3f})")

        # Final K-Means with optimal K
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
        labels = kmeans_final.fit_predict(X)

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print("\n📊 CLUSTER SIZES:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} days ({count / len(labels) * 100:.1f}%)")

        # Centroids (in original space)
        centroids_scaled = kmeans_final.cluster_centers_
        centroids = self.scaler.inverse_transform(centroids_scaled)

        print("\n🎯 CLUSTER CENTROIDS (Selected Features):")
        centroid_df = pd.DataFrame(centroids, columns=feature_names)
        key_features = ['TOTAL_PASSENGERS', 'STUDENT_RATIO', 'FULL_FARE_RATIO', 'IS_WEEKEND']
        print(centroid_df[key_features].round(2))

        # Visualization: PCA projection
        self._visualize_clusters(X, labels, "K-Means", optimal_k, df)

        # Evaluation metrics
        sil = silhouette_score(X, labels)
        cal = calinski_harabasz_score(X, labels)
        dav = davies_bouldin_score(X, labels)

        print(f"\n📈 CLUSTERING QUALITY METRICS:")
        print(f"  Silhouette Score: {sil:.3f} (higher is better, range: -1 to 1)")
        print(f"  Calinski-Harabasz: {cal:.1f} (higher is better)")
        print(f"  Davies-Bouldin: {dav:.3f} (lower is better)")

        self.results['kmeans'] = {
            'labels': labels,
            'optimal_k': optimal_k,
            'centroids': centroids,
            'silhouette': sil,
            'calinski': cal,
            'davies_bouldin': dav
        }

        return labels

    # ==========================================
    # 2. HIERARCHICAL CLUSTERING
    # ==========================================

    def apply_hierarchical(self, X, feature_names, df, n_clusters=4):
        """
        Apply Agglomerative Clustering with Single and Complete linkage.
        """
        print("\n" + "=" * 80)
        print("2. HIERARCHICAL CLUSTERING (Agglomerative)")
        print("=" * 80)

        # Sample data for dendrogram (too large otherwise)
        sample_size = min(500, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_indices]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        for idx, (linkage_type, ax) in enumerate(zip(['single', 'complete'], axes)):
            print(f"\n→ Computing {linkage_type.upper()} linkage...")

            # Compute linkage
            Z = linkage(X_sample, method=linkage_type)

            # Draw dendrogram
            dendrogram(Z, ax=ax, no_labels=True, color_threshold=0)
            ax.set_title(f'Dendrogram - {linkage_type.capitalize()} Linkage',
                         fontweight='bold', fontsize=14)
            ax.set_xlabel('Sample Index', fontweight='bold')
            ax.set_ylabel('Distance', fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/02_hierarchical_dendrograms.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Dendrograms saved (sampled {sample_size} points for clarity)")

        # Apply complete linkage on full dataset
        print(f"\n→ Applying Agglomerative Clustering (Complete Linkage, {n_clusters} clusters)...")
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        labels = agg.fit_predict(X)

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print("\n📊 CLUSTER SIZES:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} days ({count / len(labels) * 100:.1f}%)")

        # Visualization
        self._visualize_clusters(X, labels, "Hierarchical (Complete)", n_clusters, df)

        # Metrics
        sil = silhouette_score(X, labels)
        cal = calinski_harabasz_score(X, labels)
        dav = davies_bouldin_score(X, labels)

        print(f"\n📈 CLUSTERING QUALITY METRICS:")
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
    # 3. DBSCAN (Density-based)
    # ==========================================

    def apply_dbscan(self, X, feature_names, df):
        """
        Apply DBSCAN - Classic density-based clustering.
        Uses K-distance graph to find optimal eps parameter.
        """
        print("\n" + "=" * 80)
        print("3. DBSCAN CLUSTERING (Density-based Algorithm)")
        print("=" * 80)

        # Step 1: Find optimal eps using k-distance graph
        print("→ Computing optimal eps parameter using K-distance graph...")
        # MinPts best practice for large-scale Smart Card Data
        k = max(10, int(np.log(len(X))))
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)

        # Sort distances
        distances = np.sort(distances[:, k - 1], axis=0)

        # Plot K-distance graph
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(distances, linewidth=2, color='#2E86AB')
        ax.set_xlabel('Data Points (sorted by distance)', fontweight='bold')
        ax.set_ylabel(f'{k}-NN Distance', fontweight='bold')
        ax.set_title('K-Distance Graph for Optimal Eps Selection', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Conservative eps selection to avoid micro-clusters
        optimal_eps = np.percentile(distances, 95)
        ax.axhline(y=optimal_eps, color='red', linestyle='--', linewidth=2,
                   label=f'Suggested eps = {optimal_eps:.2f}')
        ax.legend(fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/03_dbscan_eps_selection.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Optimal eps selected: {optimal_eps:.3f}")

        # Step 2: Apply DBSCAN with optimal parameters
        print(f"→ Running DBSCAN (eps={optimal_eps:.3f}, min_samples={k})...")
        dbscan = DBSCAN(
            eps=optimal_eps,
            min_samples=k,
            metric='euclidean',
            algorithm='ball_tree'
        )
        labels = dbscan.fit_predict(X)

        # Cluster statistics
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        print(f"\n✓ DBSCAN found {n_clusters} clusters")
        print(f"✓ Noise points: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print("\n📊 CLUSTER SIZES:")
        for cluster_id, count in zip(unique, counts):
            if cluster_id == -1:
                print(f"  Noise: {count} days ({count / len(labels) * 100:.1f}%)")
            else:
                print(f"  Cluster {cluster_id}: {count} days ({count / len(labels) * 100:.1f}%)")

        # Cluster characteristics (centroids for non-noise)
        if n_clusters > 0:
            print("\n🎯 CLUSTER CHARACTERISTICS (Mean Values):")
            df_analysis = df.copy()
            df_analysis['cluster'] = labels

            for cluster_id in sorted(set(labels)):
                if cluster_id != -1:
                    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]
                    avg_passengers = cluster_data['TOTAL_PASSENGERS'].mean()
                    avg_student_ratio = cluster_data['STUDENT'].sum() / cluster_data['TOTAL_PASSENGERS'].sum()
                    weekend_pct = cluster_data['IS_WEEKEND'].mean() * 100

                    print(f"  Cluster {cluster_id}:")
                    print(f"    - Avg Passengers: {avg_passengers:,.0f}")
                    print(f"    - Student Ratio: {avg_student_ratio:.1%}")
                    print(f"    - Weekend Days: {weekend_pct:.1f}%")

        # Visualization
        self._visualize_clusters(X, labels, "DBSCAN", n_clusters, df, noise_label=-1)

        # Metrics (excluding noise)
        mask = labels != -1
        if mask.sum() > 0 and len(set(labels[mask])) > 1:
            sil = silhouette_score(X[mask], labels[mask])
            cal = calinski_harabasz_score(X[mask], labels[mask])
            dav = davies_bouldin_score(X[mask], labels[mask])

            print(f"\n📈 CLUSTERING QUALITY METRICS (excluding noise):")
            print(f"  Silhouette Score: {sil:.3f} (higher is better, range: -1 to 1)")
            print(f"  Calinski-Harabasz: {cal:.1f} (higher is better)")
            print(f"  Davies-Bouldin: {dav:.3f} (lower is better)")
        else:
            sil = cal = dav = None
            print("\n⚠ Not enough clusters for quality metrics")

        self.results['dbscan'] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'optimal_eps': optimal_eps,
            'min_samples': k,
            'silhouette': sil,
            'calinski': cal,
            'davies_bouldin': dav
        }

        return labels

    # ==========================================
    # VISUALIZATION HELPERS
    # ==========================================

    def _visualize_clusters(self, X, labels, method_name, n_clusters, df, noise_label=None):
        """
        Visualize clusters using PCA projection + temporal distribution.
        """
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: PCA Scatter
        unique_labels = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == noise_label:
                # Noise points in gray
                mask = labels == label
                axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                                c='gray', s=20, alpha=0.3, label='Noise')
            else:
                mask = labels == label
                axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                                c=[color], s=50, alpha=0.7,
                                edgecolors='k', linewidths=0.5,
                                label=f'Cluster {label}')

        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontweight='bold')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontweight='bold')
        axes[0].set_title(f'{method_name} - PCA Projection', fontweight='bold', fontsize=12)
        axes[0].legend(loc='best', frameon=True)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Temporal Distribution
        df_temp = df.copy()
        df_temp['Cluster'] = labels
        df_temp['Month'] = df_temp['DATE'].dt.to_period('M')

        cluster_monthly = df_temp.groupby(['Month', 'Cluster']).size().reset_index(name='Count')
        cluster_monthly['Month'] = cluster_monthly['Month'].astype(str)

        pivot = cluster_monthly.pivot(index='Month', columns='Cluster', values='Count').fillna(0)

        pivot.plot(kind='bar', stacked=True, ax=axes[1], colormap='tab10', width=0.8)
        axes[1].set_title(f'{method_name} - Temporal Distribution', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Month', fontweight='bold')
        axes[1].set_ylabel('Number of Days', fontweight='bold')
        axes[1].legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(f"{self.output_dir}/03_{safe_name}_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualization saved: {safe_name}")

    # ==========================================
    # COMPARISON & INSIGHTS
    # ==========================================

    def generate_comparison_report(self):
        """
        Generate comprehensive comparison of all clustering methods.
        """
        print("\n" + "=" * 80)
        print("FINAL COMPARISON REPORT")
        print("=" * 80)

        comparison = []
        for method, res in self.results.items():
            comparison.append({
                'Method': method.upper(),
                'Clusters': res.get('optimal_k') or res.get('n_clusters', 'N/A'),
                'Silhouette': res.get('silhouette', 'N/A'),
                'Calinski-Harabasz': res.get('calinski', 'N/A'),
                'Davies-Bouldin': res.get('davies_bouldin', 'N/A')
            })

        comp_df = pd.DataFrame(comparison)
        print("\n📊 METRICS COMPARISON:")
        print(comp_df.to_string(index=False))

        # Save to CSV
        comp_df.to_csv(f"{self.output_dir}/clustering_comparison.csv", index=False)
        print(f"\n✓ Comparison saved to: {self.output_dir}/clustering_comparison.csv")

        # Visual comparison
        fig, ax = plt.subplots(figsize=(12, 6))

        methods = [r['Method'] for r in comparison]
        silhouettes = [r['Silhouette'] if r['Silhouette'] != 'N/A' else 0 for r in comparison]

        bars = ax.bar(methods, silhouettes, color=['#3498db', '#e74c3c', '#2ecc71'],
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.set_ylabel('Silhouette Score', fontweight='bold', fontsize=12)
        ax.set_title('Clustering Quality Comparison', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, silhouettes):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f'{val:.3f}', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/04_final_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("\n✓ All clustering analyses completed successfully!")
        print(f"✓ Results saved to: {self.output_dir}/")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    FILE_PATH = "izmirim-kart-ulasim-istatistikleri-guncel-extended.csv"

    if not os.path.exists(FILE_PATH):
        print(f"❌ ERROR: File '{FILE_PATH}' not found!")
        exit(1)

    # Initialize analyzer
    analyzer = TransportClusteringAnalysis(output_dir='clustering_results')

    # Prepare data
    df_daily = analyzer.prepare_clustering_data(FILE_PATH)
    X_scaled, feature_names, df_daily = analyzer.select_features_for_clustering(df_daily)

    # Apply all three clustering methods
    analyzer.apply_kmeans(X_scaled, feature_names, df_daily, n_clusters_range=(2, 8))
    analyzer.apply_hierarchical(X_scaled, feature_names, df_daily, n_clusters=4)
    analyzer.apply_dbscan(X_scaled, feature_names, df_daily)

    # Generate final comparison
    analyzer.generate_comparison_report()

    print("\n" + "=" * 80)
    print("✅ CLUSTERING ANALYSIS COMPLETE")
    print("=" * 80)