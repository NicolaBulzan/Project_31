import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix_image(cm, classes, filename="confusion_matrix.png"):
    """Plots the confusion matrix and saves it as an image."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix for IMC Prediction')
    plt.ylabel('Actual IMC Category')
    plt.xlabel('Predicted IMC Category')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure 'sex_v' and 'IMC' columns exist
    if 'sex_v' not in df.columns:
        raise ValueError("Column 'sex_v' not found in the CSV.")
    if 'IMC' not in df.columns:
        raise ValueError("Column 'IMC' not found in the CSV.")
        
    df["sex_v"] = LabelEncoder().fit_transform(df["sex_v"])
    label_encoder = LabelEncoder()
    df["IMC_encoded"] = label_encoder.fit_transform(df["IMC"])
    
    # Features for prediction
    features = ["age_v", "sex_v", "greutate", "inaltime"]
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in CSV: {', '.join(missing_features)}")
        
    X = df[features]
    y = df["IMC_encoded"]
    return X, y, label_encoder

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42, n_estimators=100) # Added n_estimators
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    return report, matrix, label_encoder.classes_

def run_imc_classification(csv_path):
    """
    Main function to run IMC classification.
    Returns:
        report (str): Classification report.
        matrix_image_path (str): Path to the saved confusion matrix image.
        error (str or None): Error message if any, None otherwise.
    """
    try:
        X, y, label_encoder = load_and_prepare_data(csv_path)
        if X.empty or y.empty:
            return None, None, "No data available for training after preparation."
            
        model, X_test, y_test = train_model(X, y)
        report, matrix, classes = evaluate_model(model, X_test, y_test, label_encoder)
        
        # Ensure the 'plots' directory exists
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        matrix_image_path = os.path.join(output_dir, "imc_confusion_matrix.png")

        plot_confusion_matrix_image(matrix, classes, matrix_image_path)
        
        return report, matrix_image_path, None
    except ValueError as ve:
        print(f"ValueError in IMC classification: {ve}")
        return None, None, str(ve)
    except FileNotFoundError:
        return None, None, f"File not found: {csv_path}"
    except Exception as e:
        print(f"An unexpected error occurred in IMC classification: {e}")
        import traceback
        traceback.print_exc()
        return None, None, f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    # Example usage:
    # Replace 'your_data.csv' with the actual path to your data file
    # This part is for standalone testing of this script
    test_csv_path = 'doctor31_cazuri.csv' # Or use a specific cleaned/anomalous file
    if os.path.exists(test_csv_path):
        print(f"--- Running IMC Classification on: {test_csv_path} ---")
        report_str, cm_img_path, error_msg = run_imc_classification(test_csv_path)
        if error_msg:
            print(f"Error: {error_msg}")
        else:
            print("\nClassification Report:")
            print(report_str)
            print(f"\nConfusion Matrix image saved to: {cm_img_path}")
    else:
        print(f"Test file '{test_csv_path}' not found. Skipping standalone IMC classification test.")

