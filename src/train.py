import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gdown

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# === Ensure necessary output folders exist ===
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# === Download dataset from Google Drive if not present ===
data_path = "data/transactions.csv"
gdrive_file_id = "196ZYPE-buPq9CUF4XDVLUhgBmpLoLUfA"  # Your actual file ID

if not os.path.exists(data_path):
    print(f"{data_path} not found. Downloading from Google Drive...")
    os.makedirs("data", exist_ok=True)
    url = f"https://drive.google.com/uc?id={gdrive_file_id}"
    gdown.download(url, data_path, quiet=False)

# === Load the dataset ===
print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# === Separate features and labels ===
X = df.drop(['Class'], axis=1)
y = df['Class']

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Logistic Regression model ===
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# === Make predictions ===
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# === Print Classification Report ===
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === Confusion Matrix (Seaborn Heatmap version) ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# === (Optional) Scikit-learn Confusion Matrix Display ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix (SKLearn Display)")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_alt.png")
plt.close()

# === ROC Curve ===
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_curve.png")
plt.show()

# === Save model and scaler ===
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model and scaler saved successfully.")
print("📊 Confusion matrix and ROC curve saved in 'outputs/' folder.")

