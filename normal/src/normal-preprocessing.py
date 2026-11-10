
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "sensor.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("Loading sensor dataset...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Please put sensor.csv inside the data folder."
    )

df = pd.read_csv(DATA_FILE)

print("\nOriginal dataset shape:", df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ==========================================
# 3. IDENTIFY STATUS COLUMN
# ==========================================

possible_status_columns = [
    "machine_status",
    "status",
    "condition",
    "label",
    "target"
]

status_column = None

for column in df.columns:
    if column.lower().strip() in possible_status_columns:
        status_column = column
        break

if status_column is None:
    raise ValueError(
        "Could not find the machine status column. "
        "Expected one of: machine_status, status, "
        "condition, label, target."
    )

print("\nStatus column:", status_column)

print("\nStatus values:")
print(df[status_column].value_counts(dropna=False))


# ==========================================
# 4. SELECT NORMAL RECORDS
# ==========================================

normal_df = df[
    df[status_column]
    .astype(str)
    .str.strip()
    .str.upper() == "NORMAL"
].copy()

print("\nNormal records shape:", normal_df.shape)

if normal_df.empty:
    raise ValueError(
        "No NORMAL records found. Check the status values "
        "in your CSV file."
    )


# ==========================================
# 5. REMOVE DUPLICATES
# ==========================================

before_duplicates = len(normal_df)

normal_df.drop_duplicates(inplace=True)

after_duplicates = len(normal_df)

print("\nDuplicate rows removed:",
      before_duplicates - after_duplicates)


# ==========================================
# 6. IDENTIFY SENSOR COLUMNS
# ==========================================

sensor_columns = [
    column for column in normal_df.columns
    if column.lower().startswith("sensor_")
]

# If sensor-prefixed columns are not available,
# use all numeric columns except the status column.

if not sensor_columns:
    sensor_columns = [
        column for column in normal_df.select_dtypes(
            include=np.number
        ).columns
        if column != status_column
    ]

if not sensor_columns:
    raise ValueError("No numeric sensor columns found.")

print("\nSensor columns:")
print(sensor_columns)


# ==========================================
# 7. CLEAN SENSOR VALUES
# ==========================================

for column in sensor_columns:

    normal_df[column] = pd.to_numeric(
        normal_df[column],
        errors="coerce"
    )

# Replace infinite values with missing values.

normal_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# Remove sensor columns that contain no valid values.

empty_columns = [
    column for column in sensor_columns
    if normal_df[column].isna().all()
]

if empty_columns:
    print("\nRemoving empty sensor columns:", empty_columns)

    normal_df.drop(
        columns=empty_columns,
        inplace=True
    )

    sensor_columns = [
        column for column in sensor_columns
        if column not in empty_columns
    ]

if not sensor_columns:
    raise ValueError("No sensor columns contain valid numeric data.")

# Fill missing values using each sensor's median.

for column in sensor_columns:

    median_value = normal_df[column].median()

    normal_df[column] = normal_df[column].fillna(
        median_value
    )


# ==========================================
# 8. SAVE CLEANED NORMAL DATA
# ==========================================

normal_output = OUTPUT_DIR / "normal_records.csv"

normal_df.to_csv(
    normal_output,
    index=False
)

print("\nCleaned Normal data saved:", normal_output)


# ==========================================
# 9. SENSOR FREQUENCY AND STATISTICS
# ==========================================

sensor_summary = []

for column in sensor_columns:

    values = normal_df[column]

    mode_values = values.mode()

    mode_value = (
        mode_values.iloc[0]
        if not mode_values.empty
        else np.nan
    )

    mode_frequency = (
        (values == mode_value).sum()
        if pd.notna(mode_value)
        else 0
    )

    sensor_summary.append({
        "Sensor": column,
        "Count": values.count(),
        "Unique_Values": values.nunique(),
        "Mode": mode_value,
        "Mode_Frequency": mode_frequency,
        "Mean": values.mean(),
        "Median": values.median(),
        "Standard_Deviation": values.std(),
        "Minimum": values.min(),
        "Maximum": values.max()
    })

summary_df = pd.DataFrame(sensor_summary)

summary_output = OUTPUT_DIR / "normal_sensor_frequency.csv"

summary_df.to_csv(
    summary_output,
    index=False
)

print("\nSensor summary saved:", summary_output)

print("\nSensor statistics:")
print(summary_df.to_string(index=False))


# ==========================================
# 10. SENSOR MEAN COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=summary_df,
    x="Sensor",
    y="Mean"
)

plt.title("Mean Sensor Values - Normal Condition")
plt.xlabel("Sensor")
plt.ylabel("Mean Value")

plt.xticks(rotation=45)
plt.tight_layout()

comparison_output = (
    OUTPUT_DIR / "normal_sensor_comparison.png"
)

plt.savefig(
    comparison_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison graph saved:", comparison_output)


# ==========================================
# 11. SENSOR DISTRIBUTION GRAPHS
# ==========================================

number_of_sensors = min(len(sensor_columns), 12)

fig, axes = plt.subplots(
    number_of_sensors,
    1,
    figsize=(12, 4 * number_of_sensors)
)

if number_of_sensors == 1:
    axes = [axes]

for index, column in enumerate(sensor_columns[:12]):

    sns.histplot(
        normal_df[column],
        kde=True,
        ax=axes[index]
    )

    axes[index].set_title(
        f"Distribution of {column} - Normal Condition"
    )

    axes[index].set_xlabel(column)
    axes[index].set_ylabel("Frequency")

plt.tight_layout()

graphs_output = OUTPUT_DIR / "normal_analysis_graphs.png"

plt.savefig(
    graphs_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDistribution graphs saved:", graphs_output)


# ==========================================
# 12. SENSOR CORRELATION GRAPH
# ==========================================

if len(sensor_columns) > 1:

    plt.figure(figsize=(12, 9))

    correlation = normal_df[sensor_columns].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Sensor Correlation - Normal Condition")

    plt.tight_layout()

    correlation_output = (
        OUTPUT_DIR / "normal_sensor_correlation.png"
    )

    plt.savefig(
        correlation_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nCorrelation graph saved:", correlation_output)


# ==========================================
# 13. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 50)
print("NORMAL DATA PREPROCESSING COMPLETED")
print("=" * 50)

print("Original records:", len(df))
print("Normal records:", len(normal_df))
print("Number of sensors:", len(sensor_columns))

print("\nGenerated files:")

for file in OUTPUT_DIR.iterdir():
    if file.is_file():
        print("-", file.name)

print("\nAll Normal preprocessing tasks completed successfully!")
