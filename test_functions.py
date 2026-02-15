# test_functions.py -- Unit tests for functions.py


import unittest
import pandas as pd
from functions import validate_dataframe, describe_series, analyze



# Quick stand-alone assertions

_test_series = pd.Series([1, 2, 3, 4, 5])
_result = describe_series(_test_series)
assert _result["mean"] == 3.0, f"Expected mean 3.0, got {_result['mean']}"
assert _result["count"] == 5, f"Expected count 5, got {_result['count']}"


# Formal test class


class TestDescribeSeries(unittest.TestCase):
    """Tests for the describe_series() function."""

    def test_mean_simple(self):
        """describe_series() should compute the correct mean."""
        result = describe_series(pd.Series([10, 20, 30]))
        self.assertEqual(result["mean"], 20.0)

    def test_median_even(self):
        """describe_series() should compute the correct median for even-length series."""
        result = describe_series(pd.Series([1, 2, 3, 4]))
        self.assertEqual(result["median"], 2.5)

    def test_count_ignores_nan(self):
        """describe_series() count should exclude NaN values."""
        result = describe_series(pd.Series([1, 2, None, 4]))
        self.assertEqual(result["count"], 3)

    def test_single_value(self):
        """describe_series() should work for a single-element series (std will be NaN)."""
        result = describe_series(pd.Series([42]))
        self.assertEqual(result["mean"], 42.0)
        self.assertEqual(result["count"], 1)


class TestValidateDataframe(unittest.TestCase):
    """Tests for the validate_dataframe() function."""

    def test_valid_dataframe(self):
        """Should not raise when all required columns are present."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        # No exception means success
        validate_dataframe(df, ["a", "b"])

    def test_missing_column(self):
        """Should raise ValueError when a required column is missing."""
        df = pd.DataFrame({"a": [1]})
        with self.assertRaises(ValueError):
            validate_dataframe(df, ["a", "b"])

    def test_not_a_dataframe(self):
        """Should raise TypeError when input is not a DataFrame."""
        with self.assertRaises(TypeError):
            validate_dataframe("not a dataframe", ["a"])


class TestAnalyze(unittest.TestCase):
    """Tests for the analyze() function."""

    def setUp(self):
        """Shared DataFrame: Alive mean age 55, Dead mean age 75."""
        self.df = pd.DataFrame({
            "patient_status": ["Alive", "Alive", "Dead", "Dead"],
            "age": [50, 60, 70, 80],
        })

    def test_returns_correct_groups(self):
        """analyze() should return one row per group."""
        result = analyze(self.df, group_col="patient_status")
        self.assertEqual(len(result), 2)

    def test_correct_group_mean(self):
        """analyze() should compute correct grouped means (describe() uses MultiIndex columns)."""
        result = analyze(self.df, group_col="patient_status")
        # describe() columns are (column_name, statistic), e.g. ("age", "mean")
        self.assertEqual(result.loc["Alive", ("age", "mean")], 55.0)
        self.assertEqual(result.loc["Dead", ("age", "mean")], 75.0)

    def test_missing_group_col(self):
        """analyze() should raise ValueError if group column is missing."""
        df = pd.DataFrame({"age": [1, 2, 3]})
        with self.assertRaises(ValueError):
            analyze(df, group_col="nonexistent")


if __name__ == "__main__":
    unittest.main()
