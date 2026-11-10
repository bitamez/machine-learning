
# ============================================================
# PRACTICAL ENGINEERING CASE STUDY
# SUPERVISED MACHINE LEARNING CLASSIFICATION
#
# Algorithms:
# 1. Support Vector Machine (SVM)
# 2. Gaussian Naive Bayes
# 3. Decision Tree
# 4. Random Forest
# 5. AdaBoost
#
# Dataset: Industrial Pump Failure Prediction
# Total Records: 30
# ============================================================

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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# PART 1. CREATE DATASET (30 RECORDS)
# ============================================================

# ID, Pressure_Bar, Flow_Rate_L_min, Motor_Current_A,
# Pump_Age_Years, Failure_Status
#
# 0 = Normal
# 1 = Failure Risk

data = [

    # Normal operating condition
    [1,  5.2, 120, 8.1,  1, 0],
    [2,  5.5, 115, 8.4,  2, 0],
    [3,  5.1, 125, 7.9,  1, 0],
    [4,  5.8, 118, 8.6,  3, 0],
    [5,  5.4, 122, 8.2,  2, 0],
    [6,  5.6, 110, 8.5,  3, 0],
    [7,  5.0, 128, 7.8,  1, 0],
    [8,  5.7, 116, 8.3,  2, 0],
    [9,  5.3, 124, 8.0,  2, 0],
    [10, 5.9, 112, 8.7,  3, 0],
    [11, 5.2, 126, 8.1,  1, 0],
    [12, 5.5, 119, 8.4,  2, 0],
    [13, 5.1, 121, 7.9,  1, 0],
    [14, 5.8, 114, 8.6,  3, 0],
    [15, 5.4, 123, 8.2,  2, 0],

    # Failure-risk operating condition
    [16, 9.2,  72, 14.5, 8, 1],
    [17, 9.8,  65, 15.2, 9, 1],
    [18, 10.5, 58, 16.1, 10, 1],
    [19, 8.9,  78, 13.8, 7, 1],
    [20, 11.2, 52, 17.0, 12, 1],
    [21, 9.5,  68, 14.9, 9, 1],
    [22, 11.8, 45, 18.2, 13, 1],
    [23, 8.6,  82, 13.4, 7, 1],
    [24, 10.1, 61, 15.8, 11, 1],
    [25, 9.7,  64, 15.1, 10, 1],
    [26, 12.1, 42, 18.8, 14, 1],
    [27, 9.0,  75, 14.1, 8, 1],
    [28, 10.8, 55, 16.5, 12, 1],
    [29, 9.4,  70, 14.7, 9, 1],
    [30, 11.5, 48, 17.5, 13, 1]
]


columns = [
    "Pump_ID",
    "Pressure_Bar",
    "Flow_Rate_L_min",
    "Motor_Current_A",
    "Pump_Age_Years",
    "Failure_Status"
]

df = pd.DataFrame(data, columns=columns)


# Display dataset
print("\nINDUSTRIAL PUMP DATASET: 30 RECORDS")
print("=" * 65)
print(df.to_string(index=False))


# Class distribution
print("\nCLASS DISTRIBUTION")
print("=" * 65)

print(
    df["Failure_Status"]
    .value_counts()
    .rename(index={
        0: "Normal",
        1: "Failure Risk"
    })
)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

feature_cols = [
    "Pressure_Bar",
    "Flow_Rate_L_min",
    "Motor_Current_A",
    "Pump_Age_Years"
]

X = df[feature_cols]
y = df["Failure_Status"]


# ============================================================
# INITIAL DATA VISUALIZATION
# ============================================================

sns.set_theme(style="whitegrid")


# Scatter plot
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Pressure_Bar",
    y="Flow_Rate_L_min",
    hue="Failure_Status",
    style="Failure_Status",
    s=100
)

plt.title("Pump Pressure vs Flow Rate")
plt.xlabel("Pressure (Bar)")
plt.ylabel("Flow Rate (L/min)")

plt.legend(
    title="0 = Normal, 1 = Failure Risk"
)

plt.tight_layout()
plt.show()


# Correlation heatmap
plt.figure(figsize=(9, 6))

sns.heatmap(
    df[feature_cols + ["Failure_Status"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Pump Engineering Variables")

plt.tight_layout()
plt.show()


# ============================================================
# PART 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTRAIN-TEST SPLIT")
print("=" * 65)

print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")


# ============================================================
# PART 3. DEFINE FIVE CLASSIFICATION ALGORITHMS
# ============================================================


# 1. SUPPORT VECTOR MACHINE

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )
    )
])


# 2. GAUSSIAN NAIVE BAYES

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])


# 3. DECISION TREE

decision_tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)


# 4. RANDOM FOREST

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)


# 5. ADABOOST

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)


# Store all models

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}


# ============================================================
# PART 4. TRAIN, PREDICT AND EVALUATE MODELS
# ============================================================

results = []
predictions = {}


for model_name, model in models.items():

    print("\n" + "=" * 65)
    print(model_name.upper())
    print("=" * 65)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    predictions[model_name] = y_pred

    # Evaluation metrics
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

    # Store results
    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # Display metrics
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")

    # Classification report
    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Normal",
                "Failure Risk"
            ],
            zero_division=0
        )
    )

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=[
            "Normal",
            "Failure Risk"
        ],
        cmap="Blues",
        colorbar=False,
        ax=ax
    )

    ax.set_title(
        f"{model_name}: Confusion Matrix"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# PART 5. COMPARE FIVE CLASSIFICATION ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 65)

print(results_df.round(3))


# Performance comparison graph
results_df.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.05)
)

plt.title(
    "Performance Comparison of Five Classification Algorithms"
)

plt.ylabel("Performance Score")
plt.xlabel("Machine Learning Algorithm")

plt.xticks(rotation=20)

plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# ============================================================
# PART 6. FEATURE IMPORTANCE ANALYSIS
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

    print(f"\n{name.upper()} FEATURE IMPORTANCE")
    print("=" * 65)

    print(importance.round(3))

    # Feature importance graph
    plt.figure(figsize=(8, 5))

    plt.barh(
        importance.index,
        importance.values
    )

    plt.title(
        f"{name}: Feature Importance"
    )

    plt.xlabel("Importance Score")
    plt.ylabel("Engineering Feature")

    plt.tight_layout()
    plt.show()


# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================

plt.figure(figsize=(17, 9))

plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=[
        "Normal",
        "Failure Risk"
    ],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title(
    "Decision Tree for Industrial Pump Failure Prediction"
)

plt.tight_layout()
plt.show()


# ============================================================
# PART 7. TEST A NEW ENGINEERING OBSERVATION
# ============================================================

# New pump observation
# These values are illustrative and not a real diagnosis.

new_pump = pd.DataFrame([{
    "Pressure_Bar": 10.0,
    "Flow_Rate_L_min": 62,
    "Motor_Current_A": 15.5,
    "Pump_Age_Years": 10
}])


print("\nNEW INDUSTRIAL PUMP OBSERVATION")
print("=" * 65)

print(new_pump.to_string(index=False))


# Predict using all five algorithms

print("\nPREDICTION RESULTS")
print("=" * 65)

for name, model in models.items():

    predicted_class = int(
        model.predict(new_pump)[0]
    )

    label = (
        "Failure Risk"
        if predicted_class == 1
        else "Normal"
    )

    print(
        f"{name:15s} prediction: {label}"
    )


# ============================================================
# PART 8. FINAL SUMMARY
# ============================================================

print("\nFINAL MODEL SUMMARY")
print("=" * 65)

best_model = results_df["F1-score"].idxmax()

print(
    f"Best model based on test F1-score: {best_model}"
)

print(
    f"Best F1-score: {results_df.loc[best_model, 'F1-score']:.3f}"
)

print("\nEND OF PRACTICAL ENGINEERING CASE STUDY")
