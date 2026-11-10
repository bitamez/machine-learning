
import pandas as pd


def inspect_dataset(df):
    """
    Inspect the original pump sensor dataset.
    """
    print("Dataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

    return df


def filter_fault_records(df, label_column="machine_status"):
    """
    Select original BROKEN records and label them as FAULT.
    """

    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found.")

    fault_df = df[
        df[label_column].astype(str).str.upper() == "BROKEN"
    ].copy()

    # Convert the original BROKEN label to the team's FAULT label
    fault_df["machine_status"] = "FAULT"

    return fault_df


def preprocess_fault_data(df):
    """
    Prepare FAULT records for further analysis.
    """

    fault_df = filter_fault_records(df)

    # Remove unnecessary index column if present
    if "Unnamed: 0" in fault_df.columns:
        fault_df = fault_df.drop(columns=["Unnamed: 0"])

    # Remove completely empty sensor columns
    empty_columns = [
        col for col in fault_df.columns
        if col.startswith("sensor_") and fault_df[col].isna().all()
    ]

    fault_df = fault_df.drop(columns=empty_columns)

    print("FAULT dataset shape:", fault_df.shape)
    print("FAULT records:", len(fault_df))
    print("Removed completely empty columns:", empty_columns)

    return fault_df
