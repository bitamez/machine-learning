# WARNING Module – Machine Learning Project

## 1. Project Overview

This module focuses on preprocessing, extracting, and analyzing sensor data related to the WARNING condition of an industrial machine.

The purpose is to identify patterns in sensor readings that are associated with the WARNING state and provide useful information for the team's integrated machine condition monitoring system.

**Scope:** This module handles WARNING data analysis only. Model training and final classification will be performed by the team after integrating all modules.

## 2. Dataset

The original dataset is `sensor.csv`.

It contains:
- Timestamp information.
- Sensor readings (`sensor_00` to `sensor_51`).
- Machine status labels.

The original machine status values are:
- `NORMAL`
- `RECOVERING`
- `BROKEN`

For this module, `RECOVERING` is treated as `WARNING`. This is a project-level label mapping and should be confirmed with the project supervisor or domain expert.

## 3. WARNING Data Summary

After preprocessing:

- Total WARNING records: **14,477**
- Empty sensor column removed: `sensor_15`
- Remaining missing sensor values: **0**
- Output dataset: `outputs/warning_data.csv`

## 4. Project Structure

```text
warning/
│
├── data/
│   └── sensor.csv
│
├── notebooks/
│   └── warning-analysis.ipynb
│
├── outputs/
│   ├── warning_data.csv
│   ├── warning_daily_counts.csv
│   ├── warning_sensor_statistics.csv
│   ├── warning_sensor_correlation.csv
│   ├── top_warning_sensors.csv
│   ├── warning_sensor_trends.csv
│   ├── warning_normal_comparison.csv
│   └── top_warning_normal_sensors.csv
│
├── src/
│   └── preprocessing.py
│
└── README.md
```

## 5. Preprocessing

The preprocessing script performs the following operations:

1. Loads the original sensor dataset.
2. Removes unnecessary columns.
3. Converts timestamps into datetime format.
4. Sorts records by timestamp.
5. Converts sensor readings to numeric values.
6. Removes sensor columns that contain only missing values.
7. Fills remaining missing sensor values using the median.
8. Selects records with `RECOVERING` status.
9. Assigns the label `WARNING`.
10. Saves the cleaned dataset as `outputs/warning_data.csv`.

## 6. Data Analysis

The Jupyter Notebook includes:

- Dataset overview and validation.
- WARNING record distribution.
- Sensor statistical analysis.
- Sensor mean and variability analysis.
- Sensor correlation heatmap.
- Daily WARNING record trends.
- Hourly trends for selected sensors.
- Comparison between WARNING and NORMAL sensor readings.
- Identification of sensors with high variability.

The WARNING vs NORMAL comparison is exploratory and does not establish causation.

## 7. Generated Outputs

| File | Description |
|---|---|
| `warning_data.csv` | Cleaned WARNING records |
| `warning_daily_counts.csv` | Daily WARNING record counts |
| `warning_sensor_statistics.csv` | Statistical summary of sensors |
| `warning_sensor_correlation.csv` | Sensor correlation matrix |
| `top_warning_sensors.csv` | Sensors with highest variability |
| `warning_sensor_trends.csv` | Hourly trends of selected sensors |
| `warning_normal_comparison.csv` | WARNING and NORMAL comparison |
| `top_warning_normal_sensors.csv` | Top sensors from the comparison |

## 8. Requirements

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

## 9. How to Run

Open a terminal in the `warning` project directory.

Run preprocessing:

```bash
python src/preprocessing.py
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/warning-analysis.ipynb
```

Run the notebook cells in order.

## 10. Model Training and Integration

Model training is not included in this individual WARNING module.

After merging the WARNING, NORMAL, and FAULT modules, the project team will integrate the datasets and perform model training and evaluation using the combined data.

## 11. Conclusion

The WARNING module prepares and analyzes sensor records associated with the machine's WARNING condition. The generated datasets and analysis results will support the team's final integrated machine condition monitoring system.