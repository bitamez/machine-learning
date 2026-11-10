
"""
Practical Engineering Case Study:
Supervised Machine Learning Classification

Application:
Student Academic Performance and Failure-Risk Prediction

Algorithms:
1. Support Vector Machines (SVM)
2. Naive Bayes (Gaussian Naive Bayes)
3. Decision Tree
4. Random Forest
5. AdaBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ============================================================
# PART 1. CREATE DATASET (30 RECORDS)
# ============================================================

data = [
    # ID, Study_Hours, Attendance_Percent, Assignment_Score, Exam_Score, Failure_Risk

    [1,  8, 95, 85, 88, 0],
    [2,  7, 90, 80, 82, 0],
    [3,  9, 98, 92, 95, 0],
    [4,  6, 85, 75, 78, 0],
    [5, 10, 96, 90, 91, 0],
    [6,  8, 88, 82, 85, 0],
    [7,  7, 92, 78, 80, 0],
    [8,  9, 94, 88, 90, 0],
    [9,  6, 86, 72, 76, 0],
    [10, 8, 91, 84, 87, 0],
    [11, 7, 89, 79, 81, 0],
    [12, 9, 97, 91, 93, 0],
    [13, 6, 84, 74, 77, 0],
    [14, 8, 93, 86, 89, 0],
    [15, 7, 90, 81, 83, 0],

    [16, 2, 45, 35, 30, 1],
    [17, 1, 38, 28, 25, 1],
    [18, 3, 50, 40, 35, 1],
    [19, 2, 42, 32, 29, 1],
    [20, 1, 30, 25, 20, 1],
    [21, 3, 48, 38, 33, 1],
    [22, 2, 35, 30, 24, 1],
    [23, 4, 55, 45, 40, 1],
    [24, 1, 32, 22, 18, 1],
    [25, 3, 47, 36, 31, 1],
    [26, 2, 40, 29, 26, 1],
    [27, 1, 28, 20, 15, 1],
    [28, 4, 52, 42, 38, 1],
    [29, 2, 37, 27, 23, 1],
    [30, 3, 49, 39, 34, 1],
]

columns = [
    "Student_ID",
    "Study_Hours",
    "Attendance_Percent",
    "Assignment_Score",
    "Exam_Score",
    "Failure_Risk"
]

df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 STUDENT RECORDS")
print(df.to_string(index=False))

print("\nClass distribution:")
print(
    df["Failure_Risk"]
    .value_counts()
    .rename(index={0: "Normal", 1: "Failure risk"})
)

# Separate predictors (X) and target (y).

feature_cols = [
    "Study_Hours",
    "Attendance_Percent",
    "Assignment_Score",
    "Exam_Score"
]

X = df[feature_cols]
y = df["Failure_Risk"]

# Initial data visualization.

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Exam_Score",
    hue="Failure_Risk",
    style="Failure_Risk",
    s=100
)

plt.title("Student Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend(title="0 = Normal, 1 = Failure risk")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.heatmap(
    df[feature_cols + ["Failure_Risk"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Student Performance")
plt.tight_layout()
plt.show()


# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

# Stratification preserves the class ratio in both subsets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")


# ============================================================
# PART 3. DEFINE FIVE INDEPENDENT CLASSIFIERS
# ============================================================

# 1. Support Vector Machine (SVM)

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ))
])


# 2. Gaussian Naive Bayes

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])


# 3. Decision Tree

decision_tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)


# 4. Random Forest

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)


# 5. AdaBoost

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)


models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}


# ============================================================
# PART 4. TRAIN, PREDICT, EVALUATE, AND VISUALIZE
# ============================================================

results = []
predictions = {}

for model_name, model in models.items():

    print("\n" + "=" * 65)
    print(model_name.upper())
    print("=" * 65)

    # Train model.

    model.fit(X_train, y_train)

    # Predict unseen test records.

    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    # Calculate evaluation metrics.

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Failure risk"],
            zero_division=0
        )
    )

    # Confusion matrix visualization.

    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Normal", "Failure risk"],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )

    ax.set_title(f"{model_name}: Confusion Matrix")

    plt.tight_layout()
    plt.show()


# ============================================================
# PART 5. COMPARE THE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")

print("\nMODEL COMPARISON")

print(results_df.round(3))

results_df.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.05)
)

plt.title("Comparison of Five Supervised Learning Algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# ============================================================
# PART 6. INTERPRET TREE-BASED FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():

    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} Feature Importance:")

    print(importance.round(3))

    plt.figure(figsize=(7, 4))

    plt.barh(
        importance.index,
        importance.values
    )

    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Student Performance Feature")

    plt.tight_layout()
    plt.show()


# Visualize the learned Decision Tree.

plt.figure(figsize=(16, 8))

plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Normal", "Failure risk"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree Learned from Student Records")

plt.tight_layout()
plt.show()


# ============================================================
# PART 7. TEST A NEW STUDENT OBSERVATION
# ============================================================

# Example values are illustrative.
# They are not a real academic diagnosis.

new_student = pd.DataFrame([{
    "Study_Hours": 4,
    "Attendance_Percent": 55,
    "Assignment_Score": 45,
    "Exam_Score": 38
}])

print("\nNEW STUDENT OBSERVATION")

print(new_student)

for name, model in models.items():

    predicted_class = int(
        model.predict(new_student)[0]
    )

    label = (
        "Failure risk"
        if predicted_class == 1
        else "Normal"
    )

    print(f"{name:15s} prediction: {label}")


print("\nEND OF PRACTICAL CASE STUDY")
