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
        return df
    except FileNotFoundError:
        raise



# Cleaning


def standardize_columns(df):
    """Lowercase column names and replace spaces with underscores (e.g. 'ER status' -> 'er_status')."""

    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def remove_empty_rows(df):
    """Drop rows where patient_id is missing.

    """
    df = df.dropna(subset=["patient_id"])
    return df



def handle_missing_values(df):
    """Handle remaining missing values with explicit decisions.

    Strategy
    --------
    - patient_status : Drop rows where this is missing
    - date_of_last_visit : Keep as NaN.  These patients lack follow-up data;
      we note this in the report but do not impute a date.
    """
    df = df.dropna(subset=["patient_status"])
    return df
    
    

def validate_tumour_stage(df):
    """Check that tumour_stage only contains expected values (I, II, III).

    """
    expected = {"I", "II", "III"}
    if "tumour_stage" in df.columns:
        actual = set(df["tumour_stage"].dropna().unique())
        unexpected = actual - expected
        if unexpected:
            raise ValueError(f"Unexpected tumour_stage values: {unexpected}")
    return df


def clean_data(df):
    """Run the full cleaning pipeline on the raw BRCA DataFrame."""
    
    df = standardize_columns(df)
    df = remove_empty_rows(df)
    df = handle_missing_values(df)
    df = validate_tumour_stage(df)
    return df



# Main


if __name__ == "__main__":
    # Use relative path 
    raw_path = Path("raw_data") / "BRCA.csv"

    df = load_csv(raw_path)
    df = clean_data(df)

    output_path = Path("data_full.csv")
    df.to_csv(output_path, index=False)
    
