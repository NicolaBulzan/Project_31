import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# FEATURES_FOR_CLUSTERING = ["age_v", "sex_v", "greutate", "inaltime"] # Can be passed as arg
# OTHER_EXPECTED_NUMERIC_FEATURES = ["age_v", "greutate", "inaltime"] # Can be passed as arg
K_RANGE_CONFIG = range(1, 11)
# K_OPTIMAL_CONFIG = 3 # Optimal K can be determined or passed

def load_data_for_clustering(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Clustering: Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        msg = f"Clustering Error: File '{filepath}' not found."
        print(msg)
        raise FileNotFoundError(msg) # Re-raise for GUI to catch
    except Exception as e:
        msg = f"Clustering Error: Could not read file '{filepath}'. Reason: {e}"
        print(msg)
        raise Exception(msg)


def check_features_exist_for_clustering(df, features):
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        msg = f"Clustering Error: The following features are not in the CSV: {missing}.\nAvailable columns: {df.columns.tolist()}"
        print(msg)
        raise ValueError(msg)
    print("Clustering: All required features are present.")
    return True

def preprocess_data_for_clustering(input_df, features_to_cluster, other_numeric_features):
    df = input_df.copy()
    
    # Ensure features_to_cluster are present before trying to select them
    check_features_exist_for_clustering(df, features_to_cluster)
    X_cluster = df[features_to_cluster].copy()

    sex_v_col = 'sex_v'
    if sex_v_col in X_cluster.columns:
        if X_cluster[sex_v_col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X_cluster[sex_v_col]):
            print(f"Clustering: Encoding '{sex_v_col}' column...")
            le = LabelEncoder()
            X_cluster[sex_v_col] = le.fit_transform(X_cluster[sex_v_col].astype(str))
            df[sex_v_col] = X_cluster[sex_v_col].copy() 
            print(f"Clustering: '{sex_v_col}' column is now numeric.")

    for col in other_numeric_features:
        if col in X_cluster.columns: # Check if it's one of the selected features
            if not pd.api.types.is_numeric_dtype(X_cluster[col]):
                print(f"Clustering Warning: Column '{col}' (dtype: {X_cluster[col].dtype}) is not numeric. Attempting conversion...")
                try:
                    X_cluster[col] = pd.to_numeric(X_cluster[col], errors='coerce') # Coerce errors to NaN
                    df[col] = X_cluster[col].copy()
                    print(f"Clustering: Successfully converted '{col}' to numeric.")
                except ValueError: # Should be caught by coerce now
                    msg = f"Clustering Error: Could not convert column '{col}' to numeric. Please clean the data."
                    print(msg)
                    raise ValueError(msg)
    
    # Handle potential NaNs from coercion or original data before scaling
    if X_cluster.isnull().values.any():
        print(f"Clustering Warning: NaN values found in features for clustering. Imputing with mean for columns: {X_cluster.columns[X_cluster.isnull().any()].tolist()}")
        for col in X_cluster.columns[X_cluster.isnull().any()]:
            if pd.api.types.is_numeric_dtype(X_cluster[col]): # Only impute numeric columns
                 X_cluster[col] = X_cluster[col].fillna(X_cluster[col].mean())
        if X_cluster.isnull().values.any(): # Check again
            raise ValueError("Clustering Error: Unhandled NaN values remain after attempting mean imputation.")


    return df, X_cluster

def scale_data_for_clustering(X_cluster_processed):
    print("Clustering: Scaling features...")
    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(X_cluster_processed)
    X_scaled_df = pd.DataFrame(X_scaled_values, columns=X_cluster_processed.columns, index=X_cluster_processed.index)
    return X_scaled_df, scaler

def get_elbow_inertia_for_clustering(X_scaled_df, k_values_range):
    print("Clustering: Determining optimal number of clusters using the Elbow Method...")
    inertia = []
    for k_val in k_values_range:
        kmeans_elbow = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
        kmeans_elbow.fit(X_scaled_df)
        inertia.append(kmeans_elbow.inertia_)
    return inertia

def plot_elbow_curve_for_clustering(k_values_range, inertia_values, output_dir="plots", filename="clustering_elbow_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values_range, inertia_values, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k (Clustering)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_values_range)
    plt.grid(True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Clustering: Elbow plot saved as '{filepath}'.")
    plt.close()
    return filepath

def apply_kmeans_for_clustering(X_scaled_df, n_clusters):
    print(f"\nClustering: Applying K-Means with k={n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled_df)
    return kmeans, labels

def analyze_clusters_summary_for_clustering(df_with_labels, features_for_clustering):
    summary_str = f"\nFirst 10 data points with their assigned cluster labels:\n{df_with_labels[['cluster_label'] + features_for_clustering].head(10).to_string()}"
    summary_str += "\n\nMean feature values per cluster (on original scales, with sex_v numerically encoded):\n"
    cluster_summary = df_with_labels.groupby('cluster_label')[features_for_clustering].mean()
    summary_str += cluster_summary.to_string()
    summary_str += "\n\nCluster sizes:\n"
    summary_str += df_with_labels['cluster_label'].value_counts().sort_index().to_string()
    print(summary_str) # For console logging during dev
    return summary_str, cluster_summary


def plot_cluster_visualization_for_clustering(df_with_labels, k_optimal, x_feature='greutate', y_feature='inaltime', output_dir="plots", filename_template="clustering_scatter_plot_k{}.png"):
    plt.figure(figsize=(10, 7))
    
    # Ensure x_feature and y_feature are in df_with_labels
    if x_feature not in df_with_labels.columns or y_feature not in df_with_labels.columns:
        print(f"Clustering Plot Warning: Features '{x_feature}' or '{y_feature}' not found. Using first two numeric features if available.")
        numeric_cols = df_with_labels.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            x_feature = numeric_cols[0]
            y_feature = numeric_cols[1]
            print(f"Clustering Plot: Using '{x_feature}' and '{y_feature}' for visualization.")
        else:
            print("Clustering Plot Error: Not enough numeric features for scatter plot.")
            return None # Cannot generate plot

    sns.scatterplot(data=df_with_labels, x=x_feature, y=y_feature, hue='cluster_label', palette='viridis', s=60, alpha=0.8, legend="full")
    plt.title(f'Data Points Clustered by K-Means (k={k_optimal})')
    plt.xlabel(x_feature.capitalize())
    plt.ylabel(y_feature.capitalize())
    plt.legend(title='Cluster Label')
    plt.grid(True)
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = filename_template.format(k_optimal)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Clustering: Cluster visualization saved as '{filepath}'")
    plt.close()
    return filepath

def calculate_silhouette_score_for_clustering(X_scaled_df, labels, n_clusters):
    if n_clusters > 1 and len(np.unique(labels)) > 1: # Silhouette score requires at least 2 labels
        from sklearn.metrics import silhouette_score
        score = silhouette_score(X_scaled_df, labels)
        print(f"\nClustering: Silhouette Score for k={n_clusters}: {score:.3f}")
        return score
    else:
        print(f"\nClustering: Silhouette Score is not applicable for k={n_clusters} or not enough distinct labels.")
        return None

def run_clustering_pipeline(filepath, 
                            features_to_cluster=["age_v", "sex_v", "greutate", "inaltime"], 
                            other_numeric_features=["age_v", "greutate", "inaltime"], 
                            k_optimal=3, 
                            k_range=range(1, 11)):
    """
    Main function to run K-Means clustering.
    Returns:
        elbow_plot_path (str): Path to elbow plot image.
        scatter_plot_path (str): Path to cluster scatter plot image.
        summary_text (str): Text summary of cluster analysis.
        silhouette_val (float or None): Silhouette score.
        error (str or None): Error message if any.
    """
    output_plot_dir = "plots" # Define where plots will be saved

    try:
        df_loaded = load_data_for_clustering(filepath)
        df_processed, X_cluster_processed = preprocess_data_for_clustering(df_loaded, features_to_cluster, other_numeric_features)
        
        if X_cluster_processed.empty:
            return None, None, "No data available for clustering after preprocessing.", None, "No data for clustering."

        X_scaled_df, _ = scale_data_for_clustering(X_cluster_processed)
        
        inertia_values = get_elbow_inertia_for_clustering(X_scaled_df, k_range)
        elbow_plot_path = plot_elbow_curve_for_clustering(k_range, inertia_values, output_dir=output_plot_dir)
        
        if k_optimal <= 0 or k_optimal > len(X_scaled_df): # Basic check for k_optimal
            k_optimal = min(max(1, k_optimal), len(X_scaled_df)-1 if len(X_scaled_df) > 1 else 1) # Adjust k if out of bounds
            print(f"Clustering: Adjusted k_optimal to {k_optimal} based on data size.")
            if k_optimal == 0 and len(X_scaled_df) > 0: k_optimal = 1 # Ensure k is at least 1 if data exists

        if k_optimal == 0: # If no data or k could not be set
             return elbow_plot_path, None, "Not enough data points for clustering.", None, "Not enough data for clustering."


        _, labels = apply_kmeans_for_clustering(X_scaled_df, k_optimal)
        df_processed['cluster_label'] = labels
        
        summary_text, _ = analyze_clusters_summary_for_clustering(df_processed, features_to_cluster)
        
        # Determine features for scatter plot visualization
        vis_x_feature, vis_y_feature = 'greutate', 'inaltime'
        if 'greutate' not in features_to_cluster or 'inaltime' not in features_to_cluster:
            # Fallback if standard features are not in the clustering set
            numeric_cols_in_cluster_features = [f for f in features_to_cluster if pd.api.types.is_numeric_dtype(X_cluster_processed[f])]
            if len(numeric_cols_in_cluster_features) >= 2:
                vis_x_feature = numeric_cols_in_cluster_features[0]
                vis_y_feature = numeric_cols_in_cluster_features[1]
            elif len(numeric_cols_in_cluster_features) == 1:
                 vis_x_feature = numeric_cols_in_cluster_features[0]
                 # Need a second feature; could use index or a constant if desperate, but better to signal issue
                 print("Clustering Warning: Only one numeric feature for scatter plot. Visualization might be limited.")
                 # For now, we'll let plot_cluster_visualization_for_clustering handle it or error out.


        scatter_plot_path = plot_cluster_visualization_for_clustering(df_processed, k_optimal, 
                                                                      x_feature=vis_x_feature, y_feature=vis_y_feature, 
                                                                      output_dir=output_plot_dir)
        
        silhouette_val = calculate_silhouette_score_for_clustering(X_scaled_df, labels, k_optimal)
        
        return elbow_plot_path, scatter_plot_path, summary_text, silhouette_val, None
    except (FileNotFoundError, ValueError) as e:
        print(f"Clustering pipeline error: {e}")
        return None, None, None, None, str(e)
    except Exception as e:
        print(f"An unexpected error occurred in clustering pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"An unexpected error in clustering: {e}"

if __name__ == "__main__":
    # Example usage (standalone testing)
    # Assumes a 'doctor31_cazuri_cleaned.csv' or similar file exists
    # The anomaly detection script should generate this.
    test_cleaned_csv = 'doctor31_cazuri_cleaned.csv' 
    if os.path.exists(test_cleaned_csv):
        print(f"\n--- Running Clustering Pipeline on: {test_cleaned_csv} ---")
        # You might want to adjust features_to_cluster and k_optimal for your specific test data
        elbow_p, scatter_p, summary_s, sil_score, error_c = run_clustering_pipeline(
            filepath=test_cleaned_csv,
            features_to_cluster=["age_v", "sex_v", "greutate", "inaltime"], # Example features
            other_numeric_features=["age_v", "greutate", "inaltime"],
            k_optimal=3 
        )
        if error_c:
            print(f"Clustering Error: {error_c}")
        else:
            print("\nClustering Results:")
            print(f"  Elbow Plot: {elbow_p}")
            print(f"  Scatter Plot: {scatter_p}")
            print(f"  Silhouette Score: {sil_score}")
            print("\n  Summary:")
            print(summary_s)
    else:
        print(f"Test file '{test_cleaned_csv}' not found. Skipping standalone clustering test.")
        print("Please run the anomaly detection script first to generate a cleaned CSV, or provide a valid path.")

