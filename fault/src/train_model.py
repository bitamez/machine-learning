
# ==========================================
# PUMP CONDITION CLASSIFICATION
# RANDOM FOREST MODEL
# ==========================================

import pandas as pd
import joblib

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = Path("fault/data/combined_sensor_data.csv")

df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("PUMP CONDITION CLASSIFICATION")
print("=" * 55)

print("\nOriginal dataset shape:", df.shape)

# ==========================================
# 2. PREPARE DATASET
# ==========================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort records chronologically
df = df.sort_values("timestamp").reset_index(drop=True)

# Remove rows without target labels
df = df.dropna(subset=["machine_status"])

# Define target classes
classes = ["FAULT", "WARNING", "NORMAL"]

print("\nClass distribution:")
print(df["machine_status"].value_counts().reindex(classes))

# ==========================================
# 3. SELECT FEATURES AND TARGET
# ==========================================

# Timestamp is excluded from model features
X = df.drop(columns=["timestamp", "machine_status"])

# Target variable
y = df["machine_status"]

print("\nNumber of sensor features:", X.shape[1])

# ==========================================
# 4. CHRONOLOGICAL TRAIN/TEST SPLIT
# ==========================================

# 50% training and 50% testing
split_index = int(len(df) * 0.50)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n" + "=" * 55)
print("TRAINING AND TESTING DATA")
print("=" * 55)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().reindex(classes, fill_value=0))

print("\nTesting class distribution:")
print(y_test.value_counts().reindex(classes, fill_value=0))

print("\nTraining period:")
print(df.iloc[0]["timestamp"], "to", df.iloc[split_index - 1]["timestamp"])

print("\nTesting period:")
print(df.iloc[split_index]["timestamp"], "to", df.iloc[-1]["timestamp"])

# ==========================================
# 5. BUILD RANDOM FOREST PIPELINE
# ==========================================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
])

# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\n" + "=" * 55)
print("MODEL TRAINING")
print("=" * 55)

print("\nTraining Random Forest classifier...")

model.fit(X_train, y_train)

print("Model training completed successfully!")

# ==========================================
# 7. PREDICT TEST DATA
# ==========================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

# ==========================================
# 8. MODEL EVALUATION
# ==========================================

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n1. Accuracy:")
print(f"{accuracy:.4f} ({accuracy * 100:.2f}%)")

# Balanced accuracy
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\n2. Balanced Accuracy:")
print(f"{balanced_accuracy:.4f} ({balanced_accuracy * 100:.2f}%)")

# Classification report
print("\n3. Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=classes,
        zero_division=0
    )
)

# Confusion matrix
print("\n4. Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {c}" for c in classes],
    columns=[f"Predicted {c}" for c in classes]
)

print(cm_df)

# ==========================================
# 9. PREDICTED CLASS DISTRIBUTION
# ==========================================

print("\n5. Predicted Class Distribution:")

print(
    pd.Series(y_pred)
    .value_counts()
    .reindex(classes, fill_value=0)
)

# ==========================================
# 10. SAVE MODEL AND RESULTS
# ==========================================

output_dir = Path("fault/models")
output_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    output_dir / "pump_condition_model.pkl"
)

print("\nModel saved:")
print(output_dir / "pump_condition_model.pkl")

# Save confusion matrix
results_dir = Path("fault/outputs")
results_dir.mkdir(parents=True, exist_ok=True)

cm_df.to_csv(
    results_dir / "confusion_matrix.csv"
)

# Save predictions for further analysis
predictions = pd.DataFrame({
    "actual": y_test.reset_index(drop=True),
    "predicted": y_pred
})

predictions.to_csv(
    results_dir / "model_predictions.csv",
    index=False
)

print("\nEvaluation files saved:")
print(results_dir / "confusion_matrix.csv")
print(results_dir / "model_predictions.csv")

print("\n" + "=" * 55)
print("MODEL TRAINING AND EVALUATION FINISHED")
print("=" * 55)
