import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Load dataset
df = pd.read_csv("doctor31_cazuri_cleaned.csv")

# Features and target (without imcINdex)
features = ["age_v", "sex_v", "greutate", "inaltime"]
target = "IMC"

# Encode sex and target
df["sex_v"] = LabelEncoder().fit_transform(df["sex_v"])
label_encoder = LabelEncoder()
df["IMC_encoded"] = label_encoder.fit_transform(df[target])

# Prepare training data
X = df[features]
y = df["IMC_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")

# Visualize one decision tree from the forest
estimator = model.estimators_[0]
dot_data = export_graphviz(
    estimator,
    out_file=None,
    feature_names=features,
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)
print("Decision tree saved as decision_tree.png")
