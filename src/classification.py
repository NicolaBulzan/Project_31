import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df["sex_v"] = LabelEncoder().fit_transform(df["sex_v"])
    label_encoder = LabelEncoder()
    df["IMC_encoded"] = label_encoder.fit_transform(df["IMC"])
    X = df[["age_v", "sex_v", "greutate", "inaltime"]]
    y = df["IMC_encoded"]
    return X, y, label_encoder

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    matrix = confusion_matrix(y_test, y_pred)
    return report, matrix