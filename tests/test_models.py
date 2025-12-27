"""
Test script for ModelPipeline class.
This demonstrates how to use the ModelPipeline with the existing preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from preprocessor import PreProcessor
from feature_engineer import FeatureEngineer
from models import ModelPipeline


def test_model_pipeline():
    """Test the ModelPipeline class with sample data."""

    print("="*80)
    print("Testing ModelPipeline Class")
    print("="*80)

    # Step 1: Load and preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    preprocessor = PreProcessor()
    df = preprocessor.read_data()
    print(f"  Loaded {len(df)} rows")

    # Step 2: Feature engineering
    print("\n[Step 2] Running feature engineering...")
    engineer = FeatureEngineer()
    df = engineer.derive_features(df)
    df = engineer.distance_to_default(df)

    # For survival random forest, we need event and duration columns
    # Add them if not present
    if 'PERMNO' in df.columns:
        # Find first bankruptcy and create features for time till first bankruptcy
        first_bk = (
            df.loc[df["bankruptcy"] == 1, ["PERMNO", "year"]]
            .groupby("PERMNO", as_index=False)["year"].min()
            .rename(columns={"year": "bk_first_year"})
        )
        df = df.merge(first_bk, on="PERMNO", how="left")

        last_year_firm = df.groupby("PERMNO", as_index=False)["year"].max().rename(columns={"year":"firm_last_year"})
        df = df.merge(last_year_firm, on="PERMNO", how="left")
        has_event = df["bk_first_year"].notna()
        event_mask = has_event & (df["bk_first_year"] >= df["year"])

        dur_to_event = np.where(has_event, df["bk_first_year"] - df["year"] + 1, np.nan)
        dur_to_censor = df["firm_last_year"] - df["year"] + 1

        df["event"] = event_mask.astype(bool)
        df["duration"] = np.where(event_mask, dur_to_event, dur_to_censor).astype(float)
        df.loc[df["duration"] < 1, "duration"] = 1.0

    df = engineer.sanitize_columns(df)

    # Prepare X and y
    X = df.drop(['CUSIP6', 'PERMNO', 'bankruptcy'], axis=1, errors='ignore')
    y = df['bankruptcy']

    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(X)}")
    print(f"  Bankruptcy rate: {y.mean():.4f}")

    # Step 3: Initialize ModelPipeline with proper splits
    print("\n[Step 3] Initializing ModelPipeline...")
    print("  Train: 1964-1990")
    print("  Validation: 1991-2000 (used for hyperparameter tuning)")
    print("  Test: 2001-2020 (used for final evaluation)")

    pipeline = ModelPipeline(
        train_start=1964,
        train_end=1990,
        val_start=1991,
        val_end=2000,
        test_start=2001,
        test_end=2020,
        random_state=42
    )

    # Step 4: Prepare data splits
    print("\n[Step 4] Preparing data splits...")
    pipeline.prepare_data(X, y)

    # Step 5: Fit individual models (for testing)
    print("\n[Step 5] Testing individual models...")

    # Test Logistic Regression (plain mode only for quick test)
    print("\n  Testing Logistic Regression (plain)...")
    pipeline.fit_logistic_regression(mode='plain')

    # Test KNN
    print("\n  Testing KNN...")
    pipeline.fit_knn(k_values=[3, 5, 7])

    # Test Random Forest
    print("\n  Testing Random Forest...")
    pipeline.fit_random_forest(
        n_estimators_list=[50, 100],
        max_depth_list=[3, 5]
    )

    # Test XGBoost
    print("\n  Testing XGBoost...")
    pipeline.fit_xgboost(n_estimators_list=[50, 100])

    # Test LightGBM
    print("\n  Testing LightGBM...")
    pipeline.fit_lightgbm(max_depth_list=[3, 5])

    # Step 6: View results
    print("\n[Step 6] Model Results Summary:")
    print("="*80)
    results = pipeline.get_results()
    print(results.to_string(index=False))

    # Step 7: Plot ROC curves
    print("\n[Step 7] Plotting ROC curves...")
    aurocs = pipeline.plot_roc_curve()

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)

    return pipeline, results


def test_all_models():
    """Test fitting all models at once."""

    print("="*80)
    print("Testing fit_all_models() method")
    print("="*80)

    # Load and prepare data
    print("\n[Step 1] Loading and preprocessing data...")
    preprocessor = PreProcessor()
    df = preprocessor.read_data()

    print("\n[Step 2] Running feature engineering...")
    engineer = FeatureEngineer()
    df = engineer.derive_features(df)
    df = engineer.distance_to_default(df)

    # Add survival features
    if 'PERMNO' in df.columns:
        first_bk = (
            df.loc[df["bankruptcy"] == 1, ["PERMNO", "year"]]
            .groupby("PERMNO", as_index=False)["year"].min()
            .rename(columns={"year": "bk_first_year"})
        )
        df = df.merge(first_bk, on="PERMNO", how="left")

        last_year_firm = df.groupby("PERMNO", as_index=False)["year"].max().rename(columns={"year":"firm_last_year"})
        df = df.merge(last_year_firm, on="PERMNO", how="left")
        has_event = df["bk_first_year"].notna()
        event_mask = has_event & (df["bk_first_year"] >= df["year"])

        dur_to_event = np.where(has_event, df["bk_first_year"] - df["year"] + 1, np.nan)
        dur_to_censor = df["firm_last_year"] - df["year"] + 1

        df["event"] = event_mask.astype(bool)
        df["duration"] = np.where(event_mask, dur_to_event, dur_to_censor).astype(float)
        df.loc[df["duration"] < 1, "duration"] = 1.0

    df = engineer.sanitize_columns(df)

    X = df.drop(['CUSIP6', 'PERMNO', 'bankruptcy'], axis=1, errors='ignore')
    y = df['bankruptcy']

    # Initialize and prepare pipeline
    pipeline = ModelPipeline()
    pipeline.prepare_data(X, y)

    # Fit all models
    pipeline.fit_all_models()

    # View results
    print("\n" + "="*80)
    print("Final Results Summary:")
    print("="*80)
    results = pipeline.get_results()
    print(results.to_string(index=False))

    # Plot all ROC curves
    print("\nPlotting all ROC curves...")
    pipeline.plot_roc_curve()

    return pipeline, results


if __name__ == "__main__":
    # Run individual model tests
    # pipeline, results = test_model_pipeline()

    # Or run all models at once
    pipeline, results = test_all_models()
