# FAULT Condition Analysis

## 1. Overview

This directory contains the FAULT-condition analysis component of the Pump Condition Classification System.

**Responsible team member:** Meseret Mezgebu  
**Student ID:** 1349/16

The purpose of this component is to investigate pump sensor observations associated with the BROKEN condition in the original dataset.

## 2. Dataset

The original dataset contains seven BROKEN observations.

These observations are mapped to the project label:

`BROKEN → FAULT`

The fault analysis uses the original pump sensor measurements and their associated timestamps.

## 3. Analysis Objectives

- Identify FAULT observations.
- Examine sensor values associated with FAULT conditions.
- Analyze sensor statistics.
- Investigate missing sensor measurements.
- Produce fault-related analysis outputs.
- Contribute FAULT observations to the combined classification dataset.

## 4. Directory Structure

```text
fault/
│
├── README.md
│
├── data/
│   ├── sensor.csv
│   └── combined_sensor_data.csv
│
├── src/
│   ├── fault_preprocessing.py
│   └── train_model.py
│
├── notebooks/
│   └── fault_analysis.ipynb
│
├── models/
│   └── pump_condition_model.pkl
│
└── outputs/
    ├── fault_records.csv
    ├── fault_sensor_frequency.csv
    ├── confusion_matrix.csv
    └── model_predictions.csv
```

## 5. Preprocessing

The fault observations are identified using the original `machine_status` column.

The agreed project mapping is:

- BROKEN → FAULT

The combined classification workflow also handles missing sensor values using median imputation fitted on training data.

## 6. Classification Contribution

The FAULT class is included in the combined Random Forest classification model.

The original dataset contains only seven FAULT observations, which creates a severe class imbalance.

In the chronological 50/50 experiment:

- Training FAULT observations: 4
- Testing FAULT observations: 3
- Correctly classified test FAULT observations: 0

Therefore, the current model has not demonstrated reliable FAULT detection.

## 7. Outputs

The fault analysis and combined model outputs are stored in the `outputs/` directory.

These include fault-related records, sensor frequency information, the confusion matrix, and model predictions.

## 8. Conclusion

This component contributes the FAULT class to the combined pump-condition classification project. The limited number of original failure observations remains a major challenge for reliable fault classification.