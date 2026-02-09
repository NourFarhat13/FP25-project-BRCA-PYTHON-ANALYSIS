# functions.py -- Reusable analysis and visualization functions.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans



# Validation

def validate_dataframe(df, required_cols):
    """Check that *df* is a DataFrame and contains all *required_cols*.

    Parameters
    ----------
    df : pd.DataFrame
        The data to validate.
    required_cols : list of str
        Column names that must be present.

    Raises
    ------
    TypeError
        If *df* is not a DataFrame.
    ValueError
        If any column in *required_cols* is missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame, got {type(df).__name__}")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")



# Descriptive statistics


def helper(series):
    """Compute basic descriptive statistics for a numeric Series.

    Parameters
    ----------
    series : pd.Series
        A numeric column.

    Returns
    -------
    dict
        Keys: count, mean, median, std, min, max.
    """
    return {
        "count": int(series.count()),
        "mean": round(series.mean(), 4),
        "median": round(series.median(), 4),
        "std": round(series.std(), 4),
        "min": round(series.min(), 4),
        "max": round(series.max(), 4),
    }


def analyze(df, group_col="patient_status"):
    """Grouped descriptive statistics for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset (must contain *group_col*).
    group_col : str
        Column to group by (default: 'patient_status').

    Returns
    -------
    pd.DataFrame
        Multi-index DataFrame with group values as rows and
        (column, statistic) pairs as columns.
    """
    validate_dataframe(df, [group_col])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Build results row by row using a loop 
    rows = []
    for group_name, group_df in df.groupby(group_col):
        row = {"group": group_name}
        for col in numeric_cols:
            stats = helper(group_df[col])
            for stat_name, stat_value in stats.items():
                row[f"{col}_{stat_name}"] = stat_value
        rows.append(row)

    result = pd.DataFrame(rows).set_index("group")
    return result


# Visualisations


def visualize_survival_by(df, category_col, target_col="patient_status"):
    """Stacked percentage bar chart showing survival proportions by category.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *category_col* and *target_col*.
    category_col : str
        The categorical column to group by (e.g. 'tumour_stage').
    target_col : str
        The survival column (default: 'patient_status').
    """
    validate_dataframe(df, [category_col, target_col])

    # Count occurrences, then convert to percentages per category
    counts = df.groupby([category_col, target_col]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    ax = percentages.plot(kind="bar", stacked=True, figsize=(8, 5))
    ax.set_title(f"Survival Proportion by {category_col.replace('_', ' ').title()}")
    ax.set_xlabel(category_col.replace("_", " ").title())
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Status")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def visualize_proteins(df, group_col="patient_status"):
    """Side-by-side boxplots comparing protein levels by survival status.

    The four protein columns (protein1-4) are melted into long format
    so they can be shown in a single faceted figure.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain protein1..protein4 and *group_col*.
    group_col : str
        Column to colour by (default: 'patient_status').
    """
    protein_cols = ["protein1", "protein2", "protein3", "protein4"]
    validate_dataframe(df, protein_cols + [group_col])

    # Reshape to long format for seaborn
    melted = df.melt(
        id_vars=[group_col],
        value_vars=protein_cols,
        var_name="protein",
        value_name="expression",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=melted, x="protein", y="expression", hue=group_col, ax=ax)
    ax.set_title("Protein Expression Levels by Patient Status")
    ax.set_xlabel("Protein")
    ax.set_ylabel("Expression Level")
    plt.tight_layout()
    plt.show()


def visualize_age(df, group_col="patient_status"):
    """Overlapping histograms of age distribution by survival status.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'age' and *group_col*.
    group_col : str
        Column to split by (default: 'patient_status').
    """
    validate_dataframe(df, ["age", group_col])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Use density=True so both groups are normalised to the same scale.
    # Without this, the larger group (Alive, n=255) dwarfs the smaller
    # group (Dead, n=66) and the shapes become impossible to compare.
    for status, group_df in df.groupby(group_col):
        ax.hist(
            group_df["age"].dropna(),
            bins=15,
            alpha=0.5,
            label=status,
            density=True,
        )

    ax.set_title("Age Distribution by Patient Status")
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    ax.legend(title="Status")
    plt.tight_layout()
    plt.show()


# Clustering


def run_kmeans(df_scaled, n_clusters=2, random_state=42):
    """Run K-Means clustering and return cluster labels.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        Scaled numeric features.
    n_clusters : int
        Number of clusters to find (default: 2).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        (cluster_labels as array, fitted KMeans model)
    """
    if not isinstance(df_scaled, pd.DataFrame):
        raise TypeError("Expected a DataFrame of scaled features")
    if len(df_scaled) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} rows, got {len(df_scaled)}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    return labels, kmeans
