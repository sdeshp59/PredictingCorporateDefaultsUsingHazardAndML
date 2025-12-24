import numpy as np
import pandas as pd
import re

class FeatureEngineer():    
    def distance_to_default(self, df):
        if "SHROUT" in df.columns:
            df["mve"] = df["PRC"].abs() * df["SHROUT"].fillna(0) * 1000
        elif "ME" in df.columns:
            df["mve"] = df["ME"]
        else:
            df["mve"] = df["PRC"].abs()

        df["debt_bs"] = df.get("lct", 0).fillna(0) + 0.5 * df.get("dltt", 0).fillna(0)
        eps = 1e-12
        df["debt_bs"] = df["debt_bs"].where(df["debt_bs"] > 0, eps)


        df["va"] = df["mve"].fillna(0) + df["debt_bs"].fillna(0)
        df["va"] = df["va"].where(df["va"] > 0, eps)

        if "PERMNO" in df.columns and "RET" in df.columns:
            df = df.sort_values(["PERMNO", "year"])  
            df["sigma_e"] = (
                df.groupby("PERMNO")["RET"]
                .transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(12, min_periods=3).std())
            )
        else:
            df["sigma_e"] = np.nan

        df["sigma_e"] = df["sigma_e"].replace([np.inf, -np.inf], np.nan)
        df["sigma_e"] = df["sigma_e"].fillna(df["sigma_e"].median())  
        df["sigma_e"] = df["sigma_e"].fillna(0.10)                    
        df["sigma_e"] = df["sigma_e"].clip(lower=1e-6)

        leverage_scale = df["mve"] / (df["mve"] + df["debt_bs"])
        leverage_scale = leverage_scale.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
        df["sigma_a"] = (df["sigma_e"] * leverage_scale).clip(lower=1e-6)

        T = 1.0
        mu = 0.0

        log_va_debt = np.log(np.maximum(df["va"] / df["debt_bs"], eps))

        df["dd_raw"] = (log_va_debt + (mu - 0.5 * df["sigma_a"]**2) * T) / (df["sigma_a"] * np.sqrt(T))

        df["dd_flag_nonfinite"] = (~np.isfinite(df["dd_raw"])).astype(int)
        df["dd"] = df["dd_raw"].replace([np.inf, -np.inf], np.nan)

        q1, q99 = df["dd"].quantile([0.01, 0.99])
        df["dd"] = df["dd"].clip(lower=q1, upper=q99)

        df["dd"] = df["dd"].fillna(0.0)
        return df
    
    def derive_features(self, df):
        df["current_ratio"] = np.where(df["lct"] != 0, df["act"] / df["lct"], 0)
        df["debt_to_equity"] = np.where(df["ceq"] != 0, df["lt"] / df["ceq"], 0)
        df["roe"] = np.where(df["ceq"] != 0, df["ni"] / df["ceq"], 0)
        df["roa"] = np.where(df["at"] != 0, df["ni"] / df["at"], 0)
        df["gross_margin"] = np.where(df["sale"] != 0, df["gp"] / df["sale"], 0)
        df["asset_turnover"] = np.where(df["at"] != 0, df["sale"] / df["at"], 0)
        return df
    
    def sanitize_columns(self, df):
        df = df.copy()
        df.columns = [
            re.sub(r"[^\w_]", "_", str(c)) 
            for c in df.columns
        ]
        return df
    
    def split_data(self, df):
        def extract_year_series(X):
            if "year" in X.columns:
                return X["year"].astype(int)
            raise ValueError("Provide a `year` Series or include a 'year' column in X.")
        X = df.drop(['CUSIP6', 'PERMNO', 'bankruptcy'], axis=1)
        Y = df['bankruptcy']
        year = extract_year_series(X)
        train_mask = (year >= 1964) & (year <= 1990)
        test_mask  = (year >= 1991) & (year <= 2020)

        years_test = np.arange(1991, 2021)

        X_train, Y_train = X.loc[train_mask, :], Y.loc[train_mask]
        X_test, Y_test = X.loc[test_mask, :],  Y.loc[test_mask]
        return X_train, X_test, Y_train, Y_test
    
    def run(self, df):
        df = self.derive_features(df)
        df = self.distance_to_default(df)
        df = self.sanitize_columns(df)
        X_train, X_test, Y_train, Y_test = self.split_data(df)
        return X_train, X_test, Y_train, Y_test 