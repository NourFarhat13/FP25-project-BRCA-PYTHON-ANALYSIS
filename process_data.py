# process_data.py -- Load raw BRCA data, clean it, and save the processed dataset.


import pandas as pd
from pathlib import Path



# Loading


def load_csv(filepath):
    """Load a CSV file with error handling.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {filepath}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Check the path.")
        raise



# Cleaning


def standardize_columns(df):
    """Lowercase column names and replace spaces with underscores.

    
    (e.g. 'ER status' becomes 'er_status').
    """
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def remove_empty_rows(df):
    """Drop rows where patient_id is missing.

    """
    rows_before = len(df)
    df = df.dropna(subset=["patient_id"])
    removed = rows_before - len(df)
    if removed > 0:
        print(f"  Removed {removed} empty rows (missing patient_id)")
    return df


def handle_missing_values(df):
    """Handle remaining missing values with explicit, documented decisions.

    Strategy
    --------
    - patient_status : Drop rows where this is missing
    - date_of_last_visit : Keep as NaN.  These patients lack follow-up data;
      we note this in the report but do not impute a date.
    """
    # Report what is missing before we act
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        print("  Missing values before handling:")
        for col, count in cols_with_missing.items():
            print(f"    {col}: {count}")

    # Drop rows without a survival status (target variable)
    rows_before = len(df)
    df = df.dropna(subset=["patient_status"])
    dropped = rows_before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing patient_status")

    # date_of_last_visit: intentionally kept as NaN (documented decision)
    return df


def validate_tumour_stage(df):
    """Check that tumour_stage only contains expected values (I, II, III).

    Unexpected values are printed as warnings but not removed
    """
    expected = {"I", "II", "III"}
    if "tumour_stage" in df.columns:
        actual = set(df["tumour_stage"].dropna().unique())
        unexpected = actual - expected
        if unexpected:
            print(f"  Warning: unexpected tumour_stage values: {unexpected}")
        else:
            print(f"  tumour_stage values OK: {sorted(actual)}")
    return df


def clean_data(df):
    """Run the full cleaning pipeline on the raw BRCA DataFrame."""
    print("\n--- Cleaning pipeline ---")
    df = standardize_columns(df)
    df = remove_empty_rows(df)
    df = handle_missing_values(df)
    df = validate_tumour_stage(df)
    print(f"--- Cleaning complete: {df.shape[0]} rows, {df.shape[1]} columns ---\n")
    return df



# Main


if __name__ == "__main__":
    # Use relative path so the project works on any machine
    raw_path = Path("raw_data") / "BRCA.csv"

    df = load_csv(raw_path)
    df = clean_data(df)

    output_path = Path("data_full.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}: {df.shape[0]} rows, {df.shape[1]} columns")
