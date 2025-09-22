
#Clustering Module for Customer Segmentation



import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import joblib
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerClustering:
   
    #Class to handle customer segmentation using clustering algorithms
   
    
    def __init__(self, random_state=42):
       
        #Initialize the clustering module
        
        
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.silhouette_scores = {}
        self.elbow_values = {}
        
    def prepare_data(self, rfm_data, scale_features=True):
        
        #Prepare RFM data for clustering
        
        
        try:
            logger.info("Preparing data for clustering...")
            
            # Select RFM features
            if isinstance(rfm_data, pd.DataFrame):
                X = rfm_data[['Recency', 'Frequency', 'Monetary']].values
            else:
                X = rfm_data
            
            logger.info(f"Data shape: {X.shape}")
            
            # Scale features if requested
            if scale_features:
                X_scaled = self.scaler.fit_transform(X)
                logger.info("Features scaled using StandardScaler")
                return X_scaled
            else:
                return X
                
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def find_optimal_clusters(self, X, max_clusters=10):
        
        #Find optimal number of clusters using elbow method and silhouette scores
        
       
        try:
            logger.info("Finding optimal number of clusters...")
            
            wcss = []  # Within-cluster sum of squares
            silhouette_scores = []
            calinski_scores = []
            davies_scores = []
            
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                logger.info(f"Testing k = {k}...")
                
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                kmeans.fit(X)
                
                # Calculate metrics
                wcss.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X, kmeans.labels_))
                calinski_scores.append(calinski_harabasz_score(X, kmeans.labels_))
                davies_scores.append(davies_bouldin_score(X, kmeans.labels_))
            
            # Store results
            self.elbow_values = {
                'k_values': list(k_range),
                'wcss': wcss,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'davies_scores': davies_scores
            }
            
            # Find optimal k (using silhouette score)
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            logger.info(f"Optimal number of clusters: {optimal_k}")
            logger.info(f"Best silhouette score: {max(silhouette_scores):.4f}")
            
            return {
                'optimal_k': optimal_k,
                'silhouette_scores': silhouette_scores,
                'wcss': wcss,
                'k_range': list(k_range)
            }
            
        except Exception as e:
            logger.error(f"Error finding optimal clusters: {str(e)}")
            raise
    
    def perform_clustering(self, X, n_clusters=3):
        
        #Perform K-means clustering
        
        
        try:
            logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
            
            self.kmeans_model = KMeans(
                n_clusters=n_clusters, 
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            
            self.cluster_labels = self.kmeans_model.fit_predict(X)
            self.cluster_centers = self.kmeans_model.cluster_centers_
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X, self.cluster_labels)
            self.silhouette_scores[n_clusters] = silhouette_avg
            
            logger.info(f"Clustering completed. Silhouette score: {silhouette_avg:.4f}")
            logger.info(f"Cluster distribution:\n{pd.Series(self.cluster_labels).value_counts()}")
            
            return self.cluster_labels
            
        except Exception as e:
            logger.error(f"Error performing clustering: {str(e)}")
            raise
    
    def analyze_clusters(self, rfm_data, cluster_labels):
       
        #Analyze cluster characteristics
        
        
        try:
            logger.info("Analyzing cluster characteristics...")
            
            # Add cluster labels to data
            clustered_data = rfm_data.copy()
            clustered_data['Cluster'] = cluster_labels
            
            # Calculate cluster statistics
            cluster_stats = clustered_data.groupby('Cluster').agg({
                'Recency': ['mean', 'std', 'min', 'max'],
                'Frequency': ['mean', 'std', 'min', 'max'],
                'Monetary': ['mean', 'std', 'min', 'max']
            }).round(2)
            
            # Calculate cluster sizes
            cluster_sizes = clustered_data['Cluster'].value_counts().to_frame('Size')
            cluster_sizes['Percentage'] = (cluster_sizes['Size'] / len(clustered_data) * 100).round(2)
            
            logger.info("Cluster statistics:")
            logger.info(f"\n{cluster_stats}")
            logger.info(f"\nCluster sizes:\n{cluster_sizes}")
            
            return clustered_data, cluster_stats, cluster_sizes
            
        except Exception as e:
            logger.error(f"Error analyzing clusters: {str(e)}")
            raise
    
    
    #Plot elbow method results
    def plot_elbow_method(self, output_path=None):
       
        
        
       
        try:
            if not self.elbow_values:
                raise ValueError("Run find_optimal_clusters first")
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Elbow method plot
            ax1.plot(self.elbow_values['k_values'], self.elbow_values['wcss'], 'bo-')
            ax1.set_xlabel('Number of clusters')
            ax1.set_ylabel('WCSS')
            ax1.set_title('Elbow Method')
            ax1.grid(True)
            
            # Silhouette scores plot
            ax2.plot(self.elbow_values['k_values'], self.elbow_values['silhouette_scores'], 'ro-')
            ax2.set_xlabel('Number of clusters')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Scores')
            ax2.grid(True)
            
            # Calinski-Harabasz scores plot
            ax3.plot(self.elbow_values['k_values'], self.elbow_values['calinski_scores'], 'go-')
            ax3.set_xlabel('Number of clusters')
            ax3.set_ylabel('Calinski-Harabasz Score')
            ax3.set_title('Calinski-Harabasz Scores')
            ax3.grid(True)
            
            # Davies-Bouldin scores plot (lower is better)
            ax4.plot(self.elbow_values['k_values'], self.elbow_values['davies_scores'], 'mo-')
            ax4.set_xlabel('Number of clusters')
            ax4.set_ylabel('Davies-Bouldin Score')
            ax4.set_title('Davies-Bouldin Scores (Lower is Better)')
            ax4.grid(True)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                logger.info(f"Elbow method plot saved to: {output_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting elbow method: {str(e)}")
            raise
    
    #Plot cluster distribution and characteristics
    def plot_cluster_distribution(self, clustered_data, output_path=None):
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Cluster size distribution
            cluster_counts = clustered_data['Cluster'].value_counts()
            axes[0, 0].pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Cluster Distribution')
            
            # RFM means by cluster
            cluster_means = clustered_data.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
            cluster_means.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('RFM Means by Cluster')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Pairplot of clusters (using PCA for visualization if >3 features)
            if clustered_data.shape[1] > 3:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(clustered_data[['Recency', 'Frequency', 'Monetary']])
                scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=clustered_data['Cluster'], cmap='viridis')
                axes[1, 0].set_xlabel('PCA Component 1')
                axes[1, 0].set_ylabel('PCA Component 2')
                axes[1, 0].set_title('Cluster Visualization (PCA)')
                axes[1, 0].legend(*scatter.legend_elements(), title="Clusters")
            else:
                scatter = axes[1, 0].scatter(
                    clustered_data['Recency'], 
                    clustered_data['Frequency'], 
                    c=clustered_data['Cluster'], 
                    cmap='viridis'
                )
                axes[1, 0].set_xlabel('Recency')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Cluster Visualization')
                axes[1, 0].legend(*scatter.legend_elements(), title="Clusters")
            
            # Boxplot of Monetary by cluster
            sns.boxplot(x='Cluster', y='Monetary', data=clustered_data, ax=axes[1, 1])
            axes[1, 1].set_title('Monetary Distribution by Cluster')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                logger.info(f"Cluster distribution plot saved to: {output_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting cluster distribution: {str(e)}")
            raise
    
    #Save clustering results to files
    def save_clustering_results(self, clustered_data, output_dir="clustering_results"):
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save clustered data
            clustered_data_path = os.path.join(output_dir, "clustered_customers.csv")
            clustered_data.to_csv(clustered_data_path, index=False)
            logger.info(f"Clustered data saved to: {clustered_data_path}")
            
            # Save cluster statistics
            cluster_stats = clustered_data.groupby('Cluster').agg({
                'Recency': ['mean', 'std', 'min', 'max'],
                'Frequency': ['mean', 'std', 'min', 'max'],
                'Monetary': ['mean', 'std', 'min', 'max']
            }).round(2)
            
            stats_path = os.path.join(output_dir, "cluster_statistics.csv")
            cluster_stats.to_csv(stats_path)
            logger.info(f"Cluster statistics saved to: {stats_path}")
            
            # Save model
            if self.kmeans_model:
                model_path = os.path.join(output_dir, "kmeans_model.pkl")
                joblib.dump(self.kmeans_model, model_path)
                logger.info(f"K-means model saved to: {model_path}")
            
            # Save scaler
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")
            
            # Save silhouette scores
            scores_path = os.path.join(output_dir, "clustering_scores.json")
            with open(scores_path, 'w') as f:
                import json
                json.dump(self.silhouette_scores, f, indent=2)
            logger.info(f"Clustering scores saved to: {scores_path}")
            
        except Exception as e:
            logger.error(f"Error saving clustering results: {str(e)}")
            raise

# Example usage
def main():
    
    
    # Create sample RFM data
    np.random.seed(42)
    n_samples = 1000
    
    rfm_data = pd.DataFrame({
        'Recency': np.random.gamma(2, 20, n_samples),
        'Frequency': np.random.poisson(3, n_samples),
        'Monetary': np.random.lognormal(3, 1, n_samples)
    })
    
    # Initialize clustering
    clustering = CustomerClustering(random_state=42)
    
    # Prepare data
    X_scaled = clustering.prepare_data(rfm_data, scale_features=True)
    
    # Find optimal clusters
    optimal_info = clustering.find_optimal_clusters(X_scaled, max_clusters=8)
    
    # Plot elbow method
    clustering.plot_elbow_method("elbow_method.png")
    
    # Perform clustering with optimal k
    cluster_labels = clustering.perform_clustering(X_scaled, n_clusters=optimal_info['optimal_k'])
    
    # Analyze clusters
    clustered_data, cluster_stats, cluster_sizes = clustering.analyze_clusters(rfm_data, cluster_labels)
    
    # Plot cluster distribution
    clustering.plot_cluster_distribution(clustered_data, "cluster_distribution.png")
    
    # Save results
    clustering.save_clustering_results(clustered_data)
    
    return clustering, clustered_data

if __name__ == "__main__":
    clustering, clustered_data = main()
    print("Clustering completed successfully!")
    print(f"Cluster distribution:\n{clustered_data['Cluster'].value_counts()}")
