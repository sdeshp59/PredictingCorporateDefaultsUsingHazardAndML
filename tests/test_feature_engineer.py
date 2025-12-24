import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import feature_engineer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from feature_engineer import FeatureEngineer


class TestFeatureEngineer:

    @pytest.fixture
    def feature_engineer(self):
        """Fixture to create a FeatureEngineer instance for each test"""
        return FeatureEngineer()

    @pytest.fixture
    def sample_df(self):
        """Fixture to create sample DataFrame for testing"""
        return pd.DataFrame({
            'PERMNO': [10001, 10001, 10002, 10002],
            'year': [1980, 1981, 1980, 1981],
            'PRC': [10.0, 12.0, 15.0, 18.0],
            'SHROUT': [1000, 1200, 1500, 1800],
            'RET': [0.05, 0.10, 0.08, 0.12],
            'lct': [500, 600, 700, 800],
            'dltt': [1000, 1200, 1400, 1600],
            'act': [800, 900, 1000, 1100],
            'ceq': [2000, 2200, 2400, 2600],
            'lt': [1500, 1700, 1900, 2100],
            'ni': [200, 220, 240, 260],
            'at': [5000, 5500, 6000, 6500],
            'sale': [3000, 3300, 3600, 3900],
            'gp': [1200, 1320, 1440, 1560],
            'CUSIP6': ['123456', '123456', '789012', '789012'],
            'bankruptcy': [0, 0, 1, 0]
        })

    @pytest.fixture
    def minimal_df(self):
        """Fixture for minimal DataFrame without optional columns"""
        return pd.DataFrame({
            'PERMNO': [10001],
            'year': [1980],
            'PRC': [10.0],
            'lct': [500],
            'dltt': [1000],
            'act': [800],
            'ceq': [2000],
            'lt': [1500],
            'ni': [200],
            'at': [5000],
            'sale': [3000],
            'gp': [1200],
            'CUSIP6': ['123456'],
            'bankruptcy': [0]
        })

    # Tests for derive_features method
    def test_derive_features_current_ratio(self, feature_engineer, sample_df):
        """Test that current_ratio is calculated correctly"""
        result = feature_engineer.derive_features(sample_df.copy())

        assert 'current_ratio' in result.columns
        expected = sample_df['act'] / sample_df['lct']
        pd.testing.assert_series_equal(result['current_ratio'], expected, check_names=False)

    def test_derive_features_debt_to_equity(self, feature_engineer, sample_df):
        """Test that debt_to_equity is calculated correctly"""
        result = feature_engineer.derive_features(sample_df.copy())

        assert 'debt_to_equity' in result.columns
        expected = sample_df['lt'] / sample_df['ceq']
        pd.testing.assert_series_equal(result['debt_to_equity'], expected, check_names=False)

    def test_derive_features_roe(self, feature_engineer, sample_df):
        """Test that ROE (Return on Equity) is calculated correctly"""
        result = feature_engineer.derive_features(sample_df.copy())

        assert 'roe' in result.columns
        expected = sample_df['ni'] / sample_df['ceq']
        pd.testing.assert_series_equal(result['roe'], expected, check_names=False)

    def test_derive_features_roa(self, feature_engineer, sample_df):
        """Test that ROA (Return on Assets) is calculated correctly"""
        result = feature_engineer.derive_features(sample_df.copy())

        assert 'roa' in result.columns
        expected = sample_df['ni'] / sample_df['at']
        pd.testing.assert_series_equal(result['roa'], expected, check_names=False)

    def test_derive_features_gross_margin(self, feature_engineer, sample_df):
        """Test that gross_margin is calculated correctly"""
        result = feature_engineer.derive_features(sample_df.copy())

        assert 'gross_margin' in result.columns
        expected = sample_df['gp'] / sample_df['sale']
        pd.testing.assert_series_equal(result['gross_margin'], expected, check_names=False)

    def test_derive_features_asset_turnover(self, feature_engineer, sample_df):
        """Test that asset_turnover is calculated correctly"""
        result = feature_engineer.derive_features(sample_df.copy())

        assert 'asset_turnover' in result.columns
        expected = sample_df['sale'] / sample_df['at']
        pd.testing.assert_series_equal(result['asset_turnover'], expected, check_names=False)

    def test_derive_features_handles_zero_denominator(self, feature_engineer):
        """Test that derive_features handles zero denominators correctly"""
        df = pd.DataFrame({
            'lct': [0],
            'ceq': [0],
            'at': [0],
            'sale': [0],
            'act': [100],
            'lt': [100],
            'ni': [100],
            'gp': [100]
        })

        result = feature_engineer.derive_features(df)

        assert result['current_ratio'].iloc[0] == 0
        assert result['debt_to_equity'].iloc[0] == 0
        assert result['roe'].iloc[0] == 0
        assert result['roa'].iloc[0] == 0
        assert result['gross_margin'].iloc[0] == 0
        assert result['asset_turnover'].iloc[0] == 0

    # Tests for distance_to_default method
    def test_distance_to_default_with_shrout(self, feature_engineer, sample_df):
        """Test distance_to_default calculation with SHROUT column"""
        result = feature_engineer.distance_to_default(sample_df.copy())

        assert 'mve' in result.columns
        assert 'debt_bs' in result.columns
        assert 'va' in result.columns
        assert 'sigma_e' in result.columns
        assert 'sigma_a' in result.columns
        assert 'dd' in result.columns
        assert 'dd_flag_nonfinite' in result.columns

    def test_distance_to_default_with_me(self, feature_engineer):
        """Test distance_to_default calculation with ME column instead of SHROUT"""
        df = pd.DataFrame({
            'PERMNO': [10001, 10001],
            'year': [1980, 1981],
            'PRC': [10.0, 12.0],
            'ME': [10000, 14400],
            'RET': [0.05, 0.10],
            'lct': [500, 600],
            'dltt': [1000, 1200]
        })

        result = feature_engineer.distance_to_default(df)

        assert 'mve' in result.columns
        pd.testing.assert_series_equal(result['mve'], df['ME'], check_names=False)

    def test_distance_to_default_without_shrout_or_me(self, feature_engineer):
        """Test distance_to_default when neither SHROUT nor ME is present"""
        df = pd.DataFrame({
            'PERMNO': [10001],
            'year': [1980],
            'PRC': [10.0],
            'RET': [0.05],
            'lct': [500],
            'dltt': [1000]
        })

        result = feature_engineer.distance_to_default(df)

        assert 'mve' in result.columns
        assert result['mve'].iloc[0] == abs(df['PRC'].iloc[0])

    def test_distance_to_default_handles_missing_lct_dltt_values(self, feature_engineer):
        """Test distance_to_default when lct and dltt have NaN values"""
        df = pd.DataFrame({
            'PRC': [10.0],
            'lct': [np.nan],
            'dltt': [np.nan]
        })

        result = feature_engineer.distance_to_default(df)

        assert 'debt_bs' in result.columns
        assert result['debt_bs'].iloc[0] > 0  # Should be eps since fillna(0) results in 0, then replaced with eps

    def test_distance_to_default_sigma_calculation(self, feature_engineer):
        """Test that sigma_e is calculated with rolling window"""
        df = pd.DataFrame({
            'PERMNO': [10001] * 15,
            'year': list(range(1980, 1995)),
            'PRC': [10.0] * 15,
            'SHROUT': [1000] * 15,
            'RET': [0.05, 0.10, -0.03, 0.08, 0.12, 0.02, -0.01, 0.07,
                   0.04, 0.09, 0.06, -0.02, 0.11, 0.03, 0.08],
            'lct': [500] * 15,
            'dltt': [1000] * 15
        })

        result = feature_engineer.distance_to_default(df)

        assert 'sigma_e' in result.columns
        assert not result['sigma_e'].isna().all()
        assert (result['sigma_e'] >= 0).all()

    def test_distance_to_default_dd_flag_nonfinite(self, feature_engineer):
        """Test that dd_flag_nonfinite correctly identifies non-finite values"""
        df = pd.DataFrame({
            'PRC': [10.0, 0.0],
            'SHROUT': [1000, 0],
            'lct': [500, 0],
            'dltt': [1000, 0]
        })

        result = feature_engineer.distance_to_default(df)

        assert 'dd_flag_nonfinite' in result.columns
        assert result['dd_flag_nonfinite'].dtype == int

    def test_distance_to_default_clipping(self, feature_engineer):
        """Test that distance to default values are clipped to 1st and 99th percentiles"""
        # Create data with outliers
        df = pd.DataFrame({
            'PRC': [10.0] * 100 + [1000.0, 0.01],
            'SHROUT': [1000] * 102,
            'lct': [500] * 102,
            'dltt': [1000] * 102
        })

        result = feature_engineer.distance_to_default(df)

        assert 'dd' in result.columns
        # Check that extreme values are clipped
        assert result['dd'].max() <= result['dd'].quantile(0.99) or len(result['dd'].unique()) == 1

    # Tests for sanitize_columns method
    def test_sanitize_columns_removes_special_characters(self, feature_engineer):
        """Test that sanitize_columns removes special characters from column names"""
        df = pd.DataFrame({
            'normal_column': [1],
            'column-with-dash': [2],
            'column.with.dot': [3],
            'column with space': [4],
            'column@with#special!': [5]
        })

        result = feature_engineer.sanitize_columns(df)

        expected_columns = [
            'normal_column',
            'column_with_dash',
            'column_with_dot',
            'column_with_space',
            'column_with_special_'
        ]

        assert list(result.columns) == expected_columns

    def test_sanitize_columns_preserves_data(self, feature_engineer):
        """Test that sanitize_columns preserves the actual data"""
        df = pd.DataFrame({
            'col-1': [1, 2, 3],
            'col.2': [4, 5, 6]
        })

        result = feature_engineer.sanitize_columns(df)

        assert result['col_1'].tolist() == [1, 2, 3]
        assert result['col_2'].tolist() == [4, 5, 6]

    def test_sanitize_columns_creates_copy(self, feature_engineer):
        """Test that sanitize_columns creates a copy and doesn't modify original"""
        df = pd.DataFrame({
            'col-1': [1, 2, 3]
        })

        result = feature_engineer.sanitize_columns(df)

        assert 'col-1' in df.columns
        assert 'col_1' in result.columns

    # Tests for split_data method
    def test_split_data_creates_correct_splits(self, feature_engineer, sample_df):
        """Test that split_data creates train/test splits based on years"""
        # Create data spanning both train and test periods
        df = pd.DataFrame({
            'PERMNO': [10001] * 10,
            'CUSIP6': ['123456'] * 10,
            'year': [1980, 1985, 1990, 1991, 2000, 2010, 2020, 1964, 1975, 2015],
            'bankruptcy': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'feature1': list(range(10)),
            'feature2': list(range(10, 20))
        })

        X_train, X_test, Y_train, Y_test = feature_engineer.split_data(df)

        # Check that splits are not empty
        assert len(X_train) > 0
        assert len(X_test) > 0

        # Check that year ranges are correct
        assert (X_train['year'] >= 1964).all()
        assert (X_train['year'] <= 1990).all()
        assert (X_test['year'] >= 1991).all()
        assert (X_test['year'] <= 2020).all()

    def test_split_data_drops_correct_columns(self, feature_engineer):
        """Test that split_data drops CUSIP6, PERMNO, and bankruptcy columns from X"""
        df = pd.DataFrame({
            'PERMNO': [10001, 10002],
            'CUSIP6': ['123456', '789012'],
            'year': [1980, 1995],
            'bankruptcy': [0, 1],
            'feature1': [100, 200]
        })

        X_train, X_test, Y_train, Y_test = feature_engineer.split_data(df)

        # Check that dropped columns are not in X
        assert 'CUSIP6' not in X_train.columns
        assert 'PERMNO' not in X_train.columns
        assert 'bankruptcy' not in X_train.columns

        # Check that year is still present
        assert 'year' in X_train.columns

    def test_split_data_y_contains_bankruptcy(self, feature_engineer):
        """Test that Y contains bankruptcy labels"""
        df = pd.DataFrame({
            'PERMNO': [10001, 10002],
            'CUSIP6': ['123456', '789012'],
            'year': [1980, 1995],
            'bankruptcy': [0, 1],
            'feature1': [100, 200]
        })

        X_train, X_test, Y_train, Y_test = feature_engineer.split_data(df)

        assert Y_train.name == 'bankruptcy'
        assert set(Y_train.unique()).issubset({0, 1})

    def test_split_data_raises_error_without_year(self, feature_engineer):
        """Test that split_data raises error when year column is missing"""
        df = pd.DataFrame({
            'PERMNO': [10001],
            'CUSIP6': ['123456'],
            'bankruptcy': [0],
            'feature1': [100]
        })

        with pytest.raises(ValueError, match="Provide a `year` Series or include a 'year' column in X"):
            feature_engineer.split_data(df)

    # Tests for run method
    def test_run_executes_full_pipeline(self, feature_engineer):
        """Test that run method executes the full feature engineering pipeline"""
        df = pd.DataFrame({
            'PERMNO': [10001] * 5 + [10002] * 5,
            'CUSIP6': ['123456'] * 5 + ['789012'] * 5,
            'year': [1980, 1985, 1990, 1995, 2000] * 2,
            'PRC': [10.0] * 10,
            'SHROUT': [1000] * 10,
            'RET': [0.05, 0.10, -0.03, 0.08, 0.12] * 2,
            'lct': [500] * 10,
            'dltt': [1000] * 10,
            'act': [800] * 10,
            'ceq': [2000] * 10,
            'lt': [1500] * 10,
            'ni': [200] * 10,
            'at': [5000] * 10,
            'sale': [3000] * 10,
            'gp': [1200] * 10,
            'bankruptcy': [0, 1, 0, 1, 0] * 2
        })

        X_train, X_test, Y_train, Y_test = feature_engineer.run(df)

        # Check that all outputs are present
        assert X_train is not None
        assert X_test is not None
        assert Y_train is not None
        assert Y_test is not None

        # Check that derived features are present
        assert 'current_ratio' in X_train.columns
        assert 'debt_to_equity' in X_train.columns
        assert 'roe' in X_train.columns

        # Check that distance to default features are present
        assert 'dd' in X_train.columns
        assert 'mve' in X_train.columns

        # Check that columns are sanitized (no special characters)
        for col in X_train.columns:
            assert not any(char in col for char in ['-', '.', ' ', '@', '#', '!'])

    def test_run_with_minimal_data(self, feature_engineer, minimal_df):
        """Test that run works with minimal required columns"""
        # Add more years to satisfy train/test split
        df_list = []
        for year in range(1980, 2000):
            df_temp = minimal_df.copy()
            df_temp['year'] = year
            df_list.append(df_temp)

        df = pd.concat(df_list, ignore_index=True)

        X_train, X_test, Y_train, Y_test = feature_engineer.run(df)

        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_distance_to_default_handles_nan_ret(self, feature_engineer):
        """Test that distance_to_default handles NaN RET values correctly"""
        df = pd.DataFrame({
            'PERMNO': [10001] * 5,
            'year': list(range(1980, 1985)),
            'PRC': [10.0] * 5,
            'SHROUT': [1000] * 5,
            'RET': [0.05, np.nan, 0.10, np.nan, 0.08],
            'lct': [500] * 5,
            'dltt': [1000] * 5
        })

        result = feature_engineer.distance_to_default(df)

        assert 'sigma_e' in result.columns
        # sigma_e should be filled with median or 0.10 default
        assert not result['sigma_e'].isna().all()

    def test_distance_to_default_negative_prc(self, feature_engineer):
        """Test that distance_to_default handles negative PRC (bid-ask average)"""
        df = pd.DataFrame({
            'PRC': [-10.0, 12.0],
            'SHROUT': [1000, 1200],
            'lct': [500, 600],
            'dltt': [1000, 1200]
        })

        result = feature_engineer.distance_to_default(df)

        # PRC should be used as absolute value
        assert result['mve'].iloc[0] == 10.0 * 1000 * 1000

    def test_derive_features_preserves_original_columns(self, feature_engineer, sample_df):
        """Test that derive_features preserves original columns"""
        original_cols = set(sample_df.columns)
        result = feature_engineer.derive_features(sample_df.copy())

        # All original columns should still be present
        assert original_cols.issubset(set(result.columns))
