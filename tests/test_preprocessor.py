import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os

# Add parent directory to path to import preprocessor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessor import PreProcessor


class TestPreProcessor:

    @pytest.fixture
    def preprocessor(self):
        """Fixture to create a PreProcessor instance for each test"""
        return PreProcessor()

    def test_init_bankruptcy_config(self, preprocessor):
        """Test that bankruptcy configuration is properly initialized"""
        expected_config = {
            'PERMNO': 'Int64',
            'B_date': 'string'
        }
        assert preprocessor.bankruptcy_config == expected_config

    def test_init_crsp_config(self, preprocessor):
        """Test that CRSP configuration is properly initialized"""
        assert preprocessor.crsp_config['DATE'] == 'string'
        assert preprocessor.crsp_config['PERMNO'] == 'Int64'
        assert preprocessor.crsp_config['PRC'] == 'float64'
        assert len(preprocessor.crsp_config) == 17

    def test_clean_numeric_with_valid_number(self, preprocessor):
        """Test clean_numeric with a valid numeric string"""
        result = preprocessor.clean_numeric('123.45')
        assert result == 123.45

    def test_clean_numeric_with_nan(self, preprocessor):
        """Test clean_numeric with NaN input"""
        result = preprocessor.clean_numeric(np.nan)
        assert pd.isna(result)

    def test_clean_numeric_with_none(self, preprocessor):
        """Test clean_numeric with None input"""
        result = preprocessor.clean_numeric(None)
        assert pd.isna(result)

    def test_clean_numeric_with_special_characters(self, preprocessor):
        """Test clean_numeric with special characters that should be removed"""
        result = preprocessor.clean_numeric('$1,234.56')
        assert result == 1234.56

    def test_clean_numeric_with_negative_number(self, preprocessor):
        """Test clean_numeric with negative number"""
        result = preprocessor.clean_numeric('-456.78')
        assert result == -456.78

    def test_clean_numeric_with_scientific_notation(self, preprocessor):
        """Test clean_numeric with scientific notation"""
        result = preprocessor.clean_numeric('1.23e-4')
        assert result == 1.23e-4

    def test_clean_numeric_with_invalid_string(self, preprocessor):
        """Test clean_numeric with completely invalid string"""
        result = preprocessor.clean_numeric('abc')
        assert pd.isna(result)

    def test_clean_numeric_with_mixed_content(self, preprocessor):
        """Test clean_numeric with mixed valid and invalid characters"""
        result = preprocessor.clean_numeric('A123B456C')
        assert result == 123456.0

    def test_clean_numeric_with_positive_sign(self, preprocessor):
        """Test clean_numeric with explicit positive sign"""
        result = preprocessor.clean_numeric('+789.12')
        assert result == 789.12

    @patch('preprocessor.pd.read_csv')
    def test_read_data_integration(self, mock_read_csv, preprocessor):
        """Test the read_data method with mocked CSV files"""
        # Create mock bankruptcy data
        bankruptcy_data = pd.DataFrame({
            'PERMNO': [10001, 10002],
            'B_date': pd.to_datetime(['2020-01-15', '2021-03-20'])
        })

        # Create mock compustat data
        compustat_data = pd.DataFrame({
            'datadate': ['2019-12-31', '2020-12-31'],
            'cusip': ['12345678', '87654321'],
            'sale': [1000, 2000],
            'gp': [500, 1000],
            'ebit': [200, 400],
            'ni': [100, 200],
            'xint': [10, 20],
            'at': [5000, 6000],
            'lt': [3000, 3500],
            'ceq': [2000, 2500],
            'che': [500, 600],
            'invt': [300, 400],
            'rect': [200, 250],
            'dltt': [1500, 1800],
            'act': [1000, 1200],
            'lct': [800, 900],
            'oancf': [150, 200],
            'capx': [100, 120],
            'fincf': [50, 60],
            'ivncf': [30, 40],
            'dv': [20, 25]
        })

        # Create mock CRSP data
        crsp_data = pd.DataFrame({
            'DATE': ['20191231', '20201231'],
            'COMNAM': ['Company A', 'Company B'],
            'PERMNO': [10001, 10002],
            'PERMCO': [20001, 20002],
            'SHRCD': [10, 10],
            'SICCD': ['1234', '5678'],
            'CUSIP': ['123456789', '876543210'],
            'BIDLO': [10.5, 20.5],
            'ASKHI': [11.5, 21.5],
            'PRC': [11.0, 21.0],
            'VOL': [1000000.0, 2000000.0],
            'BID': [10.8, 20.8],
            'ASK': [11.2, 21.2],
            'SHROUT': [10000000, 20000000],
            'RET': ['0.05', '0.10'],
            'RETX': ['0.04', '0.09'],
            'vwretd': [0.03, 0.06]
        })

        # Configure mock to return different dataframes for each call
        mock_read_csv.side_effect = [bankruptcy_data, compustat_data, crsp_data]

        # Call read_data
        result = preprocessor.read_data()

        # Verify read_csv was called 3 times
        assert mock_read_csv.call_count == 3

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Verify bankruptcy column exists and is binary
        assert 'bankruptcy' in result.columns
        assert result['bankruptcy'].dtype == int
        assert set(result['bankruptcy'].unique()).issubset({0, 1})

    @patch('preprocessor.pd.read_csv')
    def test_read_data_creates_cusip6(self, mock_read_csv, preprocessor):
        """Test that read_data creates CUSIP6 fields correctly"""
        # Create minimal mock data
        bankruptcy_data = pd.DataFrame({
            'PERMNO': [10001],
            'B_date': pd.to_datetime(['2020-01-15'])
        })

        compustat_data = pd.DataFrame({
            'datadate': ['2019-12-31'],
            'cusip': ['123456789'],
            'sale': [1000], 'gp': [500], 'ebit': [200], 'ni': [100],
            'xint': [10], 'at': [5000], 'lt': [3000], 'ceq': [2000],
            'che': [500], 'invt': [300], 'rect': [200], 'dltt': [1500],
            'act': [1000], 'lct': [800], 'oancf': [150], 'capx': [100],
            'fincf': [50], 'ivncf': [30], 'dv': [20]
        })

        crsp_data = pd.DataFrame({
            'DATE': ['20191231'],
            'COMNAM': ['Company A'], 'PERMNO': [10001], 'PERMCO': [20001],
            'SHRCD': [10], 'SICCD': ['1234'], 'CUSIP': ['123456789'],
            'BIDLO': [10.5], 'ASKHI': [11.5], 'PRC': [11.0],
            'VOL': [1000000.0], 'BID': [10.8], 'ASK': [11.2],
            'SHROUT': [10000000], 'RET': ['0.05'], 'RETX': ['0.04'],
            'vwretd': [0.03]
        })

        mock_read_csv.side_effect = [bankruptcy_data, compustat_data, crsp_data]

        result = preprocessor.read_data()

        # Verify CUSIP6 column exists
        assert 'CUSIP6' in result.columns

    def test_clean_numeric_preserves_zero(self, preprocessor):
        """Test that clean_numeric correctly handles zero"""
        result = preprocessor.clean_numeric('0')
        assert result == 0.0

        result = preprocessor.clean_numeric('0.0')
        assert result == 0.0

    def test_clean_numeric_handles_whitespace(self, preprocessor):
        """Test that clean_numeric handles strings with whitespace"""
        result = preprocessor.clean_numeric('  123.45  ')
        assert result == 123.45
