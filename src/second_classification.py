import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DEFAULT_FILEPATH = "doctor31_cazuri_cleaned.csv"
FEATURES_FOR_CLUSTERING = ["age_v", "sex_v", "greutate", "inaltime"]
OTHER_EXPECTED_NUMERIC_FEATURES = ["age_v", "greutate", "inaltime"]
K_RANGE_CONFIG = range(1, 11)
K_OPTIMAL_CONFIG = 3

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        msg = f"Error: '{filepath}' not found. Make sure the file is in the same directory."
        print(msg)
        raise FileNotFoundError(msg)

def check_features_exist(df, features):
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        msg = f"Error: The following features are not in the CSV: {missing}.\nAvailable columns: {df.columns.tolist()}"
        print(msg)
        raise ValueError(msg)
    print("All required features are present.")
    return True

def preprocess_data(input_df, features_to_cluster, other_numeric_features):
    df = input_df.copy()
    X_cluster = df[features_to_cluster].copy()

    sex_v_col = 'sex_v'
    if sex_v_col in X_cluster.columns:
        if X_cluster[sex_v_col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X_cluster[sex_v_col]):
            if X_cluster[sex_v_col].dtype == 'object':
                print(f"Encoding '{sex_v_col}' column (was object type)...")
                le = LabelEncoder()
                X_cluster[sex_v_col] = le.fit_transform(X_cluster[sex_v_col].astype(str))
            elif not pd.api.types.is_numeric_dtype(X_cluster[sex_v_col]):
                print(f"Warning: '{sex_v_col}' (dtype: {X_cluster[sex_v_col].dtype}) is not object but not strictly numeric. Attempting conversion.")
                try:
                    X_cluster[sex_v_col] = pd.to_numeric(X_cluster[sex_v_col])
                except ValueError:
                    msg = f"Error: Could not convert '{sex_v_col}' (dtype: {X_cluster[sex_v_col].dtype}) to numeric."
                    print(msg)
                    raise ValueError(msg)
            
            # --- CORRECTED LINE for updating df['sex_v'] ---
            df[sex_v_col] = X_cluster[sex_v_col].copy() 
            # --- End Correction ---
            print(f"'{sex_v_col}' column is now numeric.")

    for col in other_numeric_features:
        if col in X_cluster.columns:
            if not pd.api.types.is_numeric_dtype(X_cluster[col]):
                print(f"Warning: Column '{col}' (dtype: {X_cluster[col].dtype}) is not numeric. Attempting conversion...")
                try:
                    X_cluster[col] = pd.to_numeric(X_cluster[col])
                    # --- CORRECTED LINE for updating other numeric features in df ---
                    df[col] = X_cluster[col].copy()
                    # --- End Correction ---
                    print(f"Successfully converted '{col}' to numeric.")
                except ValueError:
                    msg = f"Error: Could not convert column '{col}' to numeric. Please clean the data."
                    print(msg)
                    raise ValueError(msg)
    return df, X_cluster

def scale_data(X_cluster_processed):
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(X_cluster_processed)
    X_scaled_df = pd.DataFrame(X_scaled_values, columns=X_cluster_processed.columns, index=X_cluster_processed.index)
    return X_scaled_df, scaler

def get_elbow_inertia(X_scaled_df, k_values_range):
    print("Determining optimal number of clusters using the Elbow Method...")
    inertia = []
    for k_val in k_values_range:
        kmeans_elbow = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
        kmeans_elbow.fit(X_scaled_df)
        inertia.append(kmeans_elbow.inertia_)
    return inertia

def plot_elbow_curve(k_values_range, inertia_values, filename="elbow_plot_unsupervised.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values_range, inertia_values, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.xticks(k_values_range)
    plt.grid(True)
    plt.savefig(filename)
    print(f"Elbow plot saved as '{filename}'.")
    plt.close()

def apply_kmeans(X_scaled_df, n_clusters):
    print(f"\nApplying K-Means with k={n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled_df)
    return kmeans, labels

def analyze_clusters_summary(df_with_labels, features_for_clustering):
    print(f"\nFirst 10 data points with their assigned cluster labels:\n{df_with_labels[['cluster_label'] + features_for_clustering].head(10)}")
    print("\nMean feature values per cluster (on original scales, with sex_v numerically encoded):")
    cluster_summary = df_with_labels.groupby('cluster_label')[features_for_clustering].mean()
    print(cluster_summary)
    print("\nCluster sizes:")
    print(df_with_labels['cluster_label'].value_counts().sort_index())
    return cluster_summary

def plot_cluster_visualization(df_with_labels, k_optimal, x_feature='greutate', y_feature='inaltime', filename_template="clusters_visualization_k{}.png"):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_with_labels, x=x_feature, y=y_feature, hue='cluster_label', palette='viridis', s=60, alpha=0.8, legend="full")
    plt.title(f'Data Points Clustered by K-Means (k={k_optimal})')
    plt.xlabel(x_feature.capitalize())
    plt.ylabel(y_feature.capitalize())
    plt.legend(title='Cluster Label')
    plt.grid(True)
    plt.tight_layout()
    filename = filename_template.format(k_optimal)
    plt.savefig(filename)
    print(f"Cluster visualization saved as '{filename}'")
    plt.close()

def calculate_silhouette_score(X_scaled_df, labels, n_clusters):
    if n_clusters > 1:
        from sklearn.metrics import silhouette_score
        score = silhouette_score(X_scaled_df, labels)
        print(f"\nSilhouette Score for k={n_clusters}: {score:.3f} (Higher is better, range -1 to 1)")
        return score
    else:
        print("\nSilhouette Score is not applicable for k=1.")
        return None

def run_pipeline(filepath=DEFAULT_FILEPATH, features=FEATURES_FOR_CLUSTERING,
                 other_numeric=OTHER_EXPECTED_NUMERIC_FEATURES,
                 k_range=K_RANGE_CONFIG, k_optimal=K_OPTIMAL_CONFIG):
    df_loaded = load_data(filepath)
    check_features_exist(df_loaded, features)
    df_processed, X_cluster_processed = preprocess_data(df_loaded, features, other_numeric)
    X_scaled_df, _ = scale_data(X_cluster_processed)
    inertia_values = get_elbow_inertia(X_scaled_df, k_range)
    plot_elbow_curve(k_range, inertia_values)
    _, labels = apply_kmeans(X_scaled_df, k_optimal)
    df_processed['cluster_label'] = labels
    analyze_clusters_summary(df_processed, features)
    plot_cluster_visualization(df_processed, k_optimal)
    calculate_silhouette_score(X_scaled_df, labels, k_optimal)
    
if __name__ == "__main__":
    run_pipeline()