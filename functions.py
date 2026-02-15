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


def describe_series(series):
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
    return df.groupby(group_col).describe().round(4)


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

    
    percentages = pd.crosstab(df[category_col], df[target_col], normalize="index") * 100

    # Enforce stage order I → II → III for tumour_stage
    if category_col == "tumour_stage":
        order = ["I", "II", "III"]
        percentages = percentages.reindex([s for s in order if s in percentages.index]).dropna(how="all")

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
    unique_groups = sorted(df[group_col].unique())
    alphas_hist = (0.7, 0.4)
    alphas_kde = (0.35, 0.2)

    # Use fixed colors for patient_status; otherwise a palette by index (no None).
    if group_col == "patient_status":
        palette = {"Alive": "steelblue", "Dead": "coral"}
    else:
        default_colors = list(sns.color_palette("husl", n_colors=max(len(unique_groups), 2)))
        palette = {g: default_colors[i] for i, g in enumerate(unique_groups)}

    for i, group in enumerate(unique_groups):
        subset = df[df[group_col] == group]
        alpha_h = alphas_hist[i] if i < len(alphas_hist) else 0.5
        alpha_k = alphas_kde[i] if i < len(alphas_kde) else 0.25
        color = palette.get(group)
        sns.histplot(
            data=subset, x="age", stat="density", bins=15,
            alpha=alpha_h, ax=ax, label=group, color=color,
        )
        sns.kdeplot(
            data=subset, x="age", ax=ax, color=color, fill=True,
            alpha=alpha_k, linewidth=2, legend=False,
        )

    ax.set_title(f"Age Distribution by {group_col.replace('_', ' ').title()}")
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    ax.legend(title=group_col.replace("_", " ").title())
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
