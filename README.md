# Pump Condition Classification System

## 1. Project Overview

The **Pump Condition Classification System** is a machine-learning project designed to analyze industrial pump sensor data and classify pump operating conditions into three categories:

- **NORMAL** – The pump is operating under normal conditions.
- **WARNING** – The pump is in a recovering or warning condition according to the project's agreed label mapping.
- **FAULT** – The pump has experienced a failure condition.

The project uses sensor measurements, data analysis, preprocessing techniques, and machine-learning classification to investigate patterns associated with different pump operating conditions.

The main objective is to develop a multiclass classification model that can distinguish between NORMAL, WARNING, and FAULT conditions.

## 2. Project Objectives

1. Analyze industrial pump sensor data.
2. Understand sensor distributions and statistical characteristics.
3. Identify missing values and data-quality issues.
4. Investigate sensor patterns associated with different pump conditions.
5. Prepare a combined dataset containing three condition classes.
6. Develop a machine-learning classification model.
7. Evaluate the model using appropriate classification metrics.
8. Document the results and limitations of the classification system.

## 3. Team Members

| No. | Student Name | Student ID | Responsibility |
|---|---|---|---|
| 1 | Meseret Mezgebu | 1349/16 | FAULT Analysis |
| 2 | Hawlet Yimam | 1001 | WARNING Analysis |
| 3 | Ekram Yasin | 0638 | NORMAL Analysis |

Each team member worked on a separate pump-condition analysis. Their contributions were integrated into one project repository and combined classification workflow.

## 4. Dataset Information

The project uses the publicly available Pump Sensor Data dataset.

**Dataset source:**  
https://www.kaggle.com/datasets/nphantawee/pump-sensor-data

The original dataset contains:

- 220,320 observations.
- Timestamp information.
- 52 sensor columns, including an entirely empty sensor column.
- Original machine condition labels.
- Missing sensor measurements.

### Original Class Distribution

| Original Class | Number of Records |
|---|---:|
| NORMAL | 205,836 |
| RECOVERING | 14,477 |
| BROKEN | 7 |
| **Total** | **220,320** |

## 5. Class Label Mapping

The team agreed on the following mapping for the classification project:

| Original Dataset Label | Project Label | Description |
|---|---|---|
| NORMAL | NORMAL | Normal pump operation |
| RECOVERING | WARNING | Recovering/warning condition |
| BROKEN | FAULT | Pump failure condition |

This mapping is the agreed project interpretation and should not be confused with an independently verified diagnosis of pump health.

### Final Class Distribution

| Class | Records |
|---|---:|
| NORMAL | 205,836 |
| WARNING | 14,477 |
| FAULT | 7 |
| **Total** | **220,320** |

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Git
- GitHub

### Machine-Learning Algorithm

The combined classification experiment uses:

**Random Forest Classifier**

The model uses:
- 200 decision trees.
- Balanced class weights.
- Random state of 42.
- Median imputation for missing sensor measurements.
- A Scikit-learn Pipeline for preprocessing and model training.

## 7. Project Structure

```text
machine-learning/
│
├── README.md
│
├── fault/
│   ├── README.md
│   ├── data/
│   │   ├── sensor.csv
│   │   └── combined_sensor_data.csv
│   │
│   ├── src/
│   │   ├── fault_preprocessing.py
│   │   └── train_model.py
│   │
│   ├── notebooks/
│   │   └── fault_analysis.ipynb
│   │
│   ├── models/
│   │   └── pump_condition_model.pkl
│   │
│   └── outputs/
│       ├── fault_records.csv
│       ├── fault_sensor_frequency.csv
│       ├── confusion_matrix.csv
│       └── model_predictions.csv
│
├── warning/
│   ├── README.md
│   ├── src/
│   ├── notebooks/
│   └── outputs/
│
└── normal/
    ├── README.md
    ├── src/
    ├── notebooks/
    └── outputs/
```

*Note: Some generated files and the original dataset are stored locally and are not necessarily included in the GitHub repository.*

## 8. Data Preprocessing

The following preprocessing operations were performed for the combined dataset:

1. Load the original sensor dataset.
2. Remove the unnecessary `Unnamed: 0` index column.
3. Remove the entirely empty `sensor_15` column.
4. Convert original machine-condition labels into the agreed project labels.
5. Preserve the timestamp column for chronological ordering.
6. Inspect missing sensor values.
7. Separate sensor features from the target variable.
8. Handle missing sensor values using median imputation within the training pipeline.

The final combined dataset has:

- 220,320 rows.
- 53 columns.
- 51 sensor features.
- 1 timestamp column.
- 1 target column (`machine_status`).

## 9. Exploratory Data Analysis

Each team member performed an individual analysis of one pump condition.

### FAULT Analysis

Responsible member: Meseret Mezgebu.

The FAULT analysis investigates observations associated with the BROKEN condition in the original dataset.

The analysis includes:
- Fault record identification.
- Fault sensor measurements.
- Sensor statistics.
- Fault-related data exploration.
- Fault output generation.

### WARNING Analysis

Responsible member: Hawlet Yimam.

The WARNING analysis investigates observations associated with the RECOVERING condition in the original dataset.

The analysis includes:
- Warning record identification.
- Sensor statistics.
- Warning and normal comparisons.
- Sensor correlation analysis.
- Sensor trends and daily counts.

### NORMAL Analysis

Responsible member: Ekram Yasin.

The NORMAL analysis investigates normal pump operating observations.

The analysis includes:
- Normal sensor measurements.
- Descriptive statistics.
- Sensor comparisons.
- Sensor correlation analysis.
- Sensor frequency analysis.

## 10. Model Development

A Random Forest classifier was developed for the combined three-class classification task.

### Input Features

The model uses the 51 available sensor columns as input features.

The timestamp is excluded from the model input.

### Target Variable

The target variable is:

`machine_status`

It contains three possible classes:

- FAULT
- WARNING
- NORMAL

### Train-Test Splitting

The dataset is chronologically ordered and divided into two parts.

| Dataset | Percentage | Records |
|---|---:|---:|
| Training | 50% | 110,160 |
| Testing | 50% | 110,160 |
| Total | 100% | 220,320 |

The training period is:

2018-04-01 00:00:00 to 2018-06-16 11:59:00

The testing period is:

2018-06-16 12:00:00 to 2018-08-31 23:59:00

A chronological split was used to preserve the temporal order of the observations.

### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Number of estimators | 200 |
| Class weights | Balanced |
| Random state | 42 |
| Missing value strategy | Median imputation |
| Parallel processing | All available cores |

## 11. Model Evaluation

The trained model was evaluated using accuracy, balanced accuracy, precision, recall, F1-score, and a confusion matrix.

### Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 93.54% |
| Balanced Accuracy | 39.21% |

### Classification Results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| FAULT | 0.00 | 0.00 | 0.00 | 3 |
| WARNING | 0.93 | 0.18 | 0.30 | 8,506 |
| NORMAL | 0.94 | 1.00 | 0.97 | 101,651 |

### Confusion Matrix

| Actual / Predicted | FAULT | WARNING | NORMAL |
|---|---:|---:|---:|
| FAULT | 0 | 1 | 2 |
| WARNING | 0 | 1,511 | 6,995 |
| NORMAL | 0 | 121 | 101,530 |

The results show that the model performs well when identifying NORMAL observations but struggles to identify WARNING and FAULT observations.

## 12. Limitations

The most significant limitation is the severe class imbalance in the original dataset.

Only seven FAULT observations are available in the entire dataset.

In the chronological 50/50 experiment:
- Four FAULT records were used for training.
- Three FAULT records were used for testing.
- The model correctly identified zero of the three test FAULT records.

The model also has a low WARNING recall.

Although the overall accuracy is 93.54%, this value does not demonstrate reliable detection of all three pump conditions.

The limited number of fault observations prevents a reliable assessment of real-world FAULT detection performance.

## 13. Future Improvements

Possible future improvements include:

1. Collecting additional genuine FAULT observations.
2. Investigating sensor availability and missing-value patterns.
3. Evaluating additional classification algorithms.
4. Exploring suitable imbalance-handling techniques without fabricating fault observations.
5. Performing time-aware validation.
6. Improving WARNING and FAULT recall.
7. Testing the model on additional independent pump sensor datasets.
8. Developing a monitoring interface for pump-condition predictions.

## 14. How to Run the Project

### Requirements

Install Python and the required libraries:

```bash
py -m pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Dataset Setup

Obtain the original dataset from Kaggle.

Place the dataset at:

```text
fault/data/sensor.csv
```

Prepare the combined dataset according to the preprocessing and label-mapping procedure described above.

The training script expects:

```text
fault/data/combined_sensor_data.csv
```

### Train the Model

From the project root directory, execute:

```bash
py fault/src/train_model.py
```

### Generated Results

The training script saves:

```text
fault/models/pump_condition_model.pkl
fault/outputs/confusion_matrix.csv
fault/outputs/model_predictions.csv
```

## 15. GitHub Repository

Project repository:

https://github.com/bitamez/machine-learning

## 16. Conclusion

The Pump Condition Classification System integrates individual FAULT, WARNING, and NORMAL analyses into a combined multiclass machine-learning experiment.

The project demonstrates sensor-data exploration, preprocessing, feature preparation, Random Forest training, and model evaluation.

The experimental model achieved 93.54% accuracy and 39.21% balanced accuracy. However, the very small number of FAULT observations limits its ability to identify failure conditions reliably.

The project provides a foundation for further research and development in industrial pump condition monitoring and predictive maintenance.