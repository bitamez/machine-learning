# FAULT Analysis – Pump Condition Classification

## 1. Overview

This folder contains the exploratory data analysis and preprocessing work for the FAULT class in the Pump Condition Classification project.

The project uses pump sensor measurements to study machine operating conditions and support classification into three classes:

- FAULT
- WARNING
- NORMAL

## 2. Dataset

The analysis uses the shared `sensor.csv` dataset.

The original dataset contains:
- 220,320 observations
- 52 sensor columns
- One timestamp column
- One machine status column
- Original statuses: NORMAL, RECOVERING, and BROKEN

The original dataset is kept unchanged.

## 3. FAULT Data

The original dataset contains only 7 observations with the status BROKEN.

For the FAULT contribution, these BROKEN observations are extracted and relabeled as FAULT.

**Important:** The seven recorded BROKEN observations are not sufficient by themselves to train a reliable FAULT classification model.

## 4. Exploratory Data Analysis

The following analyses were performed:

1. Dataset structure and data type inspection.
2. Missing-value analysis.
3. Duplicate-record inspection.
4. Machine-status distribution analysis.
5. Inspection of the seven BROKEN observations.
6. Sensor distribution visualizations.
7. Comparison of FAULT and NORMAL sensor averages.
8. Analysis of sensor readings during the 30 minutes before each recorded failure.
9. Comparison of pre-FAULT sensor averages with overall NORMAL averages.
10. Identification of sensors frequently appearing among the top 10 largest differences.

## 5. Main Findings

The sensors most frequently appearing among the top 10 differences across the seven events were:

| Sensor | Number of events |
|---|---:|
| sensor_01 | 5 |
| sensor_04 | 4 |
| sensor_02 | 4 |
| sensor_00 | 4 |
| sensor_38 | 3 |
| sensor_46 | 3 |
| sensor_36 | 3 |

These results indicate differences in sensor readings before some recorded failures. They do not establish that these sensors reliably predict failures.

## 6. Preprocessing

The preprocessing code is located in:

`src/fault_preprocessing.py`

It includes functions to:

- Inspect the dataset.
- Extract records originally labeled BROKEN.
- Relabel the extracted records as FAULT.
- Remove the unnecessary index column, if present.
- Remove sensor columns that are completely empty within the extracted FAULT records.

Missing sensor readings are not automatically replaced with zero or other values in this preprocessing step.

## 7. Limitations

- Only seven BROKEN observations are available.
- Failure examples are extremely limited.
- Different failure events show different sensor patterns.
- Sensor differences from NORMAL averages do not prove predictive relationships.
- Additional failure data and appropriate validation are needed for reliable model development.
- WARNING labeling must follow the team's agreed definition; RECOVERING should not automatically be treated as WARNING.

## 8. Team Integration

The FAULT contribution will be combined with the WARNING and NORMAL contributions prepared by the other team members.

The final project should use one consistent feature set, agreed labeling rules, and one combined multiclass classification model.

## 9. Project Structure

```text
fault/
├── data/
├── notebooks/
│   └── fault_analysis.ipynb
├── src/
│   └── fault_preprocessing.py
├── outputs/
└── README.md
```