import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_cluster_names(cluster_summary_df):
    """
    Analyzes cluster characteristics and generates descriptive names.
    """
    names = {}
    if cluster_summary_df.empty:
        return {}
        
    overall_mean_age = cluster_summary_df['age_v'].mean()
    overall_mean_weight = cluster_summary_df['greutate'].mean()
    
    for i, cluster in cluster_summary_df.iterrows():
        age_desc = "Elderly" if cluster['age_v'] > overall_mean_age * 1.15 else "Younger" if cluster['age_v'] < overall_mean_age * 0.85 else "Middle-Aged"
        weight_desc = "Heavier Build" if cluster['greutate'] > overall_mean_weight * 1.1 else "Lighter Build" if cluster['greutate'] < overall_mean_weight * 0.9 else "Average Build"
        
        sex_desc = ""
        if 'sex_v' in cluster_summary_df.columns:
            if cluster['sex_v'] > 0.8: sex_desc = " (Predominantly Male)"
            elif cluster['sex_v'] < 0.2: sex_desc = " (Predominantly Female)"
            elif 0.6 <= cluster['sex_v'] <= 0.8: sex_desc = " (Mostly Male)"
            elif 0.2 <= cluster['sex_v'] <= 0.4: sex_desc = " (Mostly Female)"

        names[i] = f"{age_desc}, {weight_desc}{sex_desc}"
    return names

# --- Load and Preprocess Functions ---
def load_data_for_clustering(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        raise Exception(f"Clustering Error: Could not read file '{filepath}'. Reason: {e}")

def check_features_exist_for_clustering(df, features):
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Clustering Error: Missing features {missing}. Available: {df.columns.tolist()}")
    return True

def preprocess_data_for_clustering(input_df, features_to_cluster, other_numeric_features):
    df = input_df.copy()
    check_features_exist_for_clustering(df, features_to_cluster)
    X_cluster = df[features_to_cluster].copy()
    if 'sex_v' in X_cluster.columns and (X_cluster['sex_v'].dtype == 'object' or pd.api.types.is_categorical_dtype(X_cluster['sex_v'])):
        le = LabelEncoder()
        X_cluster['sex_v'] = le.fit_transform(X_cluster['sex_v'].astype(str))
        df['sex_v'] = X_cluster['sex_v'].copy() 
    for col in other_numeric_features:
        if col in X_cluster.columns:
            if not pd.api.types.is_numeric_dtype(X_cluster[col]):
                X_cluster[col] = pd.to_numeric(X_cluster[col], errors='coerce')
                df[col] = X_cluster[col].copy()
    if X_cluster.isnull().values.any():
        for col in X_cluster.columns[X_cluster.isnull().any()]:
            if pd.api.types.is_numeric_dtype(X_cluster[col]):
                 X_cluster[col] = X_cluster[col].fillna(X_cluster[col].mean())
    if X_cluster.isnull().values.any():
        raise ValueError("Clustering Error: Unhandled NaN values remain.")
    return df, X_cluster

def scale_data_for_clustering(X_cluster_processed):
    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(X_cluster_processed)
    X_scaled_df = pd.DataFrame(X_scaled_values, columns=X_cluster_processed.columns, index=X_cluster_processed.index)
    return X_scaled_df, scaler

# --- Plotting and Analysis Functions ---
def apply_kmeans_for_clustering(X_scaled_df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled_df)
    return kmeans, labels

def format_sex_label(value):
    if value > 0.8: return "Male"
    if value < 0.2: return "Female"
    if 0.6 <= value <= 0.8: return "Mostly Male"
    if 0.2 <= value <= 0.4: return "Mostly Female"
    return "Mixed"

def analyze_clusters_summary_for_clustering(df_with_labels, features_for_clustering, cohort_names):
    summary_str = "Cohort Characteristics:\n\n"
    cluster_summary = df_with_labels.groupby('cluster_label')[features_for_clustering].mean()
    cluster_sizes = df_with_labels['cluster_label'].value_counts().sort_index()

    for i, summary in cluster_summary.iterrows():
        name = cohort_names.get(i, f"Cluster {i}")
        size = cluster_sizes.get(i, 0)
        # CORRECTED: Removed asterisks from the name line
        summary_str += f"{name} ({size} patients)\n"
        
        feature_map = {
            'age_v': ('Age', 'years'),
            'sex_v': ('Sex', ''),
            'greutate': ('Weight', 'kg'),
            'inaltime': ('Height', 'cm')
        }

        for feature, value in summary.items():
            label, unit = feature_map.get(feature, (feature.capitalize(), ''))
            if feature == 'sex_v':
                formatted_value = format_sex_label(value)
                summary_str += f"  - Average {label}: {formatted_value}\n"
            else:
                summary_str += f"  - Average {label}: {value:.1f} {unit}\n"
        summary_str += "\n"
    
    return summary_str, cluster_summary


def plot_cluster_visualization_for_clustering(df_with_labels, k_optimal, cohort_names, x_feature='greutate', y_feature='inaltime', output_dir="plots"):
    df_with_labels['cohort_name'] = df_with_labels['cluster_label'].map(cohort_names)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_with_labels, x=x_feature, y=y_feature, hue='cohort_name', palette='viridis', s=60, alpha=0.8, legend="full")
    plt.title('Patient Population by Weight and Height')
    plt.xlabel(x_feature.replace('_', ' ').capitalize())
    plt.ylabel(y_feature.replace('_', ' ').capitalize())
    plt.legend(title='Discovered Cohorts')
    plt.grid(True)
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"clustering_scatter_plot_k{k_optimal}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

def run_clustering_pipeline(filepath, 
                            features_to_cluster=["age_v", "sex_v", "greutate", "inaltime"], 
                            other_numeric_features=["age_v", "greutate", "inaltime"], 
                            k_optimal=3):
    try:
        df_loaded = load_data_for_clustering(filepath)
        df_processed, X_cluster_processed = preprocess_data_for_clustering(df_loaded, features_to_cluster, other_numeric_features)
        
        if len(df_processed) < k_optimal:
             return None, None, None, None, f"Not enough data for {k_optimal} clusters. Only {len(df_processed)} rows available."

        X_scaled_df, _ = scale_data_for_clustering(X_cluster_processed)
        
        _, labels = apply_kmeans_for_clustering(X_scaled_df, k_optimal)
        df_processed['cluster_label'] = labels
        
        cluster_summary_df = df_processed.groupby('cluster_label')[features_to_cluster].mean()
        cohort_names = generate_cluster_names(cluster_summary_df)
        summary_text, _ = analyze_clusters_summary_for_clustering(df_processed, features_to_cluster, cohort_names)
        
        df_processed['cohort_name'] = df_processed['cluster_label'].map(cohort_names)
        
        scatter_plot_path = plot_cluster_visualization_for_clustering(df_processed, k_optimal, cohort_names)
        
        return scatter_plot_path, summary_text, df_processed, cohort_names, None
    except (FileNotFoundError, ValueError) as e:
        return None, None, None, None, str(e)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"An unexpected error in clustering: {e}"