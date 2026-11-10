
import pandas as pd
from pathlib import Path

# =====================================
# PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "sensor.csv"

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_PATH = OUTPUT_DIR / "warning_data.csv"


# =====================================
# 1. LOAD DATA
# =====================================

def load_data():

    print("Loading machine sensor dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("Original dataset shape:", df.shape)

    return df


# =====================================
# 2. CLEAN DATA
# =====================================

def clean_data(df):

    print("\nCleaning dataset...")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "machine_status"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    sensor_columns = [
        col for col in df.columns
        if col.startswith("sensor_")
    ]

    for col in sensor_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove completely empty sensors
    empty_columns = [
        col for col in sensor_columns
        if df[col].isna().all()
    ]

    if empty_columns:
        df = df.drop(columns=empty_columns)

    sensor_columns = [
        col for col in df.columns
        if col.startswith("sensor_")
    ]

    # Fill missing values using median
    for col in sensor_columns:

        median_value = df[col].median()

        if pd.notna(median_value):
            df[col] = df[col].fillna(median_value)

    print("Data cleaning completed.")

    return df


# =====================================
# 3. FILTER WARNING STATUS ONLY
# =====================================

def filter_warning_data(df):

    print("\nFiltering WARNING records...")

    warning_df = df[
        df["machine_status"] == "RECOVERING"
    ].copy()

    warning_df["target"] = "WARNING"

    print(
        "WARNING records:",
        len(warning_df)
    )

    return warning_df


# =====================================
# 4. SAVE WARNING DATA
# =====================================

def save_data(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nWARNING dataset saved successfully!")

    print("Output:", OUTPUT_PATH)


# =====================================
# 5. DISPLAY SUMMARY
# =====================================

def display_summary(df):

    print("\n========== WARNING DATA SUMMARY ==========")

    print("Dataset shape:", df.shape)

    print("\nStatus distribution:")

    print(df["target"].value_counts())

    print(
        "\nMissing values:",
        df.isna().sum().sum()
    )

    print("\nFirst five records:")

    print(df.head())


# =====================================
# 6. MAIN
# =====================================

def main():

    print("====================================")
    print("MACHINE LEARNING PROJECT")
    print("WARNING STATUS PREPROCESSING")
    print("====================================")

    df = load_data()

    df = clean_data(df)

    warning_df = filter_warning_data(df)

    if warning_df.empty:
        raise ValueError(
            "No WARNING records found in dataset."
        )

    save_data(warning_df)

    display_summary(warning_df)

    print("\nWARNING preprocessing completed!")


if __name__ == "__main__":
    main()
