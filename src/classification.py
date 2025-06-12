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
    if 'sex_v' not in df.columns:
        raise ValueError("Column 'sex_v' not found in the CSV.")
    if 'IMC' not in df.columns:
        raise ValueError("Column 'IMC' not found in the CSV.")
        
    df["sex_v"] = LabelEncoder().fit_transform(df["sex_v"])
    label_encoder = LabelEncoder()
    
    df["IMC"].fillna("Unknown", inplace=True)
    df["IMC_encoded"] = label_encoder.fit_transform(df["IMC"])
    
    features = ["age_v", "sex_v", "greutate", "inaltime"]
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in CSV: {', '.join(missing_features)}")
    
    for feature in features:
        if df[feature].isnull().any():
            df[feature].fillna(df[feature].median(), inplace=True)
            
    X = df[features]
    y = df["IMC_encoded"]
    return X, y, label_encoder

def train_model(X, y):
    # Stratify can fail if a class has only one member
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        print("Stratify failed, performing normal split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    
    # Ensure all possible classes are in the report, even if not in y_test/y_pred
    known_labels_encoded = list(range(len(label_encoder.classes_)))
    present_labels = sorted(list(set(y_test) | set(y_pred)))
    target_names = label_encoder.inverse_transform(present_labels)

    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=target_names,
        zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=known_labels_encoded)
    return report, matrix, label_encoder.classes_

def run_imc_classification(csv_path):
    """
    Main function to run classification.
    Returns a consistent tuple of 7 values: (report, matrix, error_msg, model, X_test, y_test, label_encoder)
    On error, object values will be None.
    """
    try:
        X, y, label_encoder = load_and_prepare_data(csv_path)
        if X.empty or y.empty:
            error_msg = "No data available for training after preparation."
            return None, None, error_msg, None, None, None, None
            
        model, X_test, y_test = train_model(X, y)
        report, matrix, classes = evaluate_model(model, X_test, y_test, label_encoder)
        
        return report, matrix, None, model, X_test, y_test, label_encoder

    except Exception as e:
        print(f"An error occurred in IMC classification: {e}")
        import traceback
        traceback.print_exc()
        return None, None, f"An unexpected error occurred: {e}", None, None, None, None