import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, List, Tuple, Optional
import gc


class ModelPipeline:
    def __init__(self, X_train, X_val, X_test, Y_train, Y_val, Y_test, random_state: int = 42):
        self.random_state = random_state
    
        self.models = {}
        self.results = pd.DataFrame(columns=['name', 'auc', 'ks', 'misclassification_rate'])
        self.predictions = {}
    
        self.permno_train = X_train['PERMNO']
        self.permno_val = X_val['PERMNO']
        self.permno_test = X_test['PERMNO']
        
        self.year_train = X_train['year']
        self.year_val = X_val['year']
        self.year_test = X_test['year']
        
        feature_cols = [col for col in X_train.columns if col not in ['PERMNO', 'year']]
        self.X_train = X_train[feature_cols]
        self.X_val = X_val[feature_cols]
        self.X_test = X_test[feature_cols]
        
        self.y_train = Y_train
        self.y_val = Y_val
        self.y_test = Y_test
    
        self.train_start = 1964
        self.train_end = 1990
        self.val_start = 1991
        self.val_end = 2000
        self.test_start = 2001
        self.test_end = 2020
    
        full_permno = pd.concat([self.permno_train, self.permno_val, self.permno_test], axis=0, copy=False)
        full_year = pd.concat([self.year_train, self.year_val, self.year_test], axis=0, copy=False)
        full_features = pd.concat([self.X_train, self.X_val, self.X_test], axis=0, copy=False)
        
        self.full_X = full_features.copy()
        self.full_X['PERMNO'] = full_permno.values
        self.full_X['year'] = full_year.values
        
        self.full_y = pd.concat([Y_train, Y_val, Y_test], axis=0, copy=False)
        self.year = self.full_X['year'].astype('int16')

    def _make_logit_pipeline(self, class_weight: Optional[str] = None, max_iter: int = 1000
                            ,solver: str = "lbfgs") -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("logit", LogisticRegression(
                penalty=None,
                class_weight=class_weight,
                max_iter=max_iter,
                solver=solver, # type: ignore
                random_state=self.random_state
            ))
        ])

    def _get_feature_columns(self):
        """Get feature columns excluding identifiers"""
        return [col for col in self.full_X.columns if col not in ['PERMNO', 'year', 'event', 'duration']] # type: ignore
    
    def fit_logistic_regression(self, mode: str = 'plain',
                                window_length: Optional[int] = None) -> None:
        
        if mode == 'plain':
            pipe = self._make_logit_pipeline()
            pipe.fit(self.X_train, self.y_train)

            y_pred = pipe.predict_proba(self.X_test)[:, 1]
            self.models['Logistic: Plain OOS'] = pipe
            self.predictions['Logistic: Plain OOS'] = y_pred
            self._summarize_model('Logistic: Plain OOS', self.y_test, y_pred)

        elif mode == 'rolling':
            self._fit_rolling_window_logit()

        elif mode == 'fixed':
            if window_length is None:
                window_length = self.train_end - self.train_start + 1
            self._fit_fixed_window_logit(window_length)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'plain', 'rolling', or 'fixed'")

    def _fit_rolling_window_logit(self) -> None:
        years_test = np.arange(self.test_start, self.test_end + 1)
    
        n_test = (self.year >= self.test_start).sum()
        all_proba = np.zeros(n_test)
        all_y = np.zeros(n_test)
        idx = 0
        
        feature_cols = self._get_feature_columns()
    
        for t in years_test:
            tr_mask_t = (self.year >= self.train_start) & (self.year <= t - 1)
            te_mask_t = (self.year == t)
    
            n_t = te_mask_t.sum()
            if n_t == 0:
                continue
    
            pipe = self._make_logit_pipeline()
            pipe.fit(self.full_X.loc[tr_mask_t, feature_cols], self.full_y.loc[tr_mask_t])
    
            p_t = pipe.predict_proba(self.full_X.loc[te_mask_t, feature_cols])[:, 1]
    
            all_proba[idx:idx+n_t] = p_t
            all_y[idx:idx+n_t] = self.full_y.loc[te_mask_t].values
            idx += n_t
    
            del pipe
            gc.collect()
    
        self.predictions['Logistic: Rolling Window'] = all_proba[:idx]
        self._summarize_model('Logistic: Rolling Window', all_y[:idx], all_proba[:idx])

    def _fit_fixed_window_logit(self, window_length: int) -> None:
        years_test = np.arange(self.test_start, self.test_end + 1)
    
        n_test = (self.year >= self.test_start).sum()
        all_proba = np.zeros(n_test)
        all_y = np.zeros(n_test)
        idx = 0
        
        feature_cols = self._get_feature_columns()
    
        for t in years_test:
            tr_end = t - 1
            tr_start = tr_end - window_length + 1
            tr_mask_t = (self.year >= tr_start) & (self.year <= tr_end)
            te_mask_t = (self.year == t)
    
            n_t = te_mask_t.sum()
            if tr_mask_t.sum() == 0 or n_t == 0:
                continue
    
            pipe = self._make_logit_pipeline()
            pipe.fit(self.full_X.loc[tr_mask_t, feature_cols], self.full_y.loc[tr_mask_t])
    
            p_t = pipe.predict_proba(self.full_X.loc[te_mask_t, feature_cols])[:, 1]
    
            all_proba[idx:idx+n_t] = p_t
            all_y[idx:idx+n_t] = self.full_y.loc[te_mask_t].values
            idx += n_t
    
            del pipe
            gc.collect()
    
        self.predictions['Logistic: Fixed Window'] = all_proba[:idx]
        self._summarize_model('Logistic: Fixed Window', all_y[:idx], all_proba[:idx])

    def fit_lasso_logistic(self, mode: str = 'plain', alphas: Optional[np.ndarray] = None, 
                       window_length: Optional[int] = None) -> None:
        if alphas is None:
            alphas = np.logspace(-4, 1, 40)
    
        combined_mask = (self.year >= self.train_start) & (self.year <= self.val_end)
        
        # Get only feature columns for LASSO selection
        feature_cols = self._get_feature_columns()
        X_combined = self.full_X.loc[combined_mask, feature_cols]
        y_combined = self.full_y.loc[combined_mask]
    
        lasso_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(alphas=alphas, cv=10, random_state=self.random_state, n_jobs=-1))
        ])
    
        lasso_pipe.fit(X_combined, y_combined.astype(float))
        coef = lasso_pipe.named_steps["lasso"].coef_
        selected_mask = (coef != 0)
        selected_features = list(X_combined.columns[selected_mask])
    
        if len(selected_features) == 0:
            k = min(10, len(coef))
            topk_idx = np.argsort(np.abs(coef))[-k:]
            selected_features = list(X_combined.columns[topk_idx])
        else:
            pass
    
        self.selected_features_lasso = selected_features
    
        if mode == 'plain':
            pipe = self._make_logit_pipeline()
            pipe.fit(self.X_train.loc[:, selected_features], self.y_train)
    
            y_pred = pipe.predict_proba(self.X_test.loc[:, selected_features])[:, 1]
            self.models['Post-LASSO Logistic: Plain OOS'] = pipe
            self.predictions['Post-LASSO Logistic: Plain OOS'] = y_pred
            self._summarize_model('Post-LASSO Logistic: Plain OOS', self.y_test, y_pred)
    
        elif mode == 'rolling':
            self._fit_rolling_window_lasso(selected_features)
    
        elif mode == 'fixed':
            if window_length is None:
                window_length = self.train_end - self.train_start + 1
            self._fit_fixed_window_lasso(selected_features, window_length)

    def _fit_rolling_window_lasso(self, selected_features: List[str]) -> None:
        years_test = np.arange(self.test_start, self.test_end + 1)
    
        n_test = (self.year >= self.test_start).sum()
        all_proba = np.zeros(n_test)
        all_y = np.zeros(n_test)
        idx = 0
    
        for t in years_test:
            tr_mask_t = (self.year >= self.train_start) & (self.year <= t - 1)
            te_mask_t = (self.year == t)
    
            n_t = te_mask_t.sum()
            if n_t == 0:
                continue
    
            pipe = self._make_logit_pipeline()
            pipe.fit(self.full_X.loc[tr_mask_t, selected_features], self.full_y.loc[tr_mask_t])
    
            p_t = pipe.predict_proba(self.full_X.loc[te_mask_t, selected_features])[:, 1]
    
            all_proba[idx:idx+n_t] = p_t
            all_y[idx:idx+n_t] = self.full_y.loc[te_mask_t].values
            idx += n_t
    
            del pipe
            gc.collect()
    
        self.predictions['Post-LASSO Logistic: Rolling Window'] = all_proba[:idx]
        self._summarize_model('Post-LASSO Logistic: Rolling Window', all_y[:idx], all_proba[:idx])

    def _fit_fixed_window_lasso(self, selected_features: List[str], window_length: int) -> None:
        years_test = np.arange(self.test_start, self.test_end + 1)
    
        n_test = (self.year >= self.test_start).sum()
        all_proba = np.zeros(n_test)
        all_y = np.zeros(n_test)
        idx = 0
    
        for t in years_test:
            tr_end = t - 1
            tr_start = tr_end - window_length + 1
            tr_mask_t = (self.year >= tr_start) & (self.year <= tr_end)
            te_mask_t = (self.year == t)
    
            n_t = te_mask_t.sum()
            if tr_mask_t.sum() == 0 or n_t == 0:
                continue
    
            pipe = self._make_logit_pipeline()
            pipe.fit(self.full_X.loc[tr_mask_t, selected_features], self.full_y.loc[tr_mask_t])
    
            p_t = pipe.predict_proba(self.full_X.loc[te_mask_t, selected_features])[:, 1]
    
            all_proba[idx:idx+n_t] = p_t
            all_y[idx:idx+n_t] = self.full_y.loc[te_mask_t].values
            idx += n_t
    
            del pipe
            gc.collect()
    
        self.predictions['Post-LASSO Logistic: Fixed Window'] = all_proba[:idx]
        self._summarize_model('Post-LASSO Logistic: Fixed Window', all_y[:idx], all_proba[:idx])

    def fit_ridge_logistic(self, mode: str = 'plain', alphas: Optional[np.ndarray] = None, 
                       window_length: Optional[int] = None) -> None:
    
        if alphas is None:
            alphas = np.logspace(-4, 1, 40)
    
        # Get only feature columns (exclude PERMNO and year)
        feature_cols = self._get_feature_columns()
        selected_features = feature_cols
        self.selected_features_ridge = selected_features
    
        if mode == 'plain':
            pipe = self._make_logit_pipeline()
            pipe.fit(self.X_train.loc[:, selected_features], self.y_train)
    
            y_pred = pipe.predict_proba(self.X_test.loc[:, selected_features])[:, 1]
            self.models['Post-Ridge Logistic: Plain OOS'] = pipe
            self.predictions['Post-Ridge Logistic: Plain OOS'] = y_pred
            self._summarize_model('Post-Ridge Logistic: Plain OOS', self.y_test, y_pred)
    
        elif mode == 'rolling':
            self._fit_rolling_window_ridge(selected_features)
    
        elif mode == 'fixed':
            if window_length is None:
                window_length = self.train_end - self.train_start + 1
            self._fit_fixed_window_ridge(selected_features, window_length)

    def _fit_rolling_window_ridge(self, selected_features: List[str]) -> None:
        years_test = np.arange(self.test_start, self.test_end + 1)
    
        n_test = (self.year >= self.test_start).sum()
        all_proba = np.zeros(n_test)
        all_y = np.zeros(n_test)
        idx = 0
    
        for t in years_test:
            tr_mask_t = (self.year >= self.train_start) & (self.year <= t - 1)
            te_mask_t = (self.year == t)
    
            n_t = te_mask_t.sum()
            if n_t == 0:
                continue
    
            pipe = self._make_logit_pipeline()
            pipe.fit(self.full_X.loc[tr_mask_t, selected_features], self.full_y.loc[tr_mask_t])
    
            p_t = pipe.predict_proba(self.full_X.loc[te_mask_t, selected_features])[:, 1]
    
            all_proba[idx:idx+n_t] = p_t
            all_y[idx:idx+n_t] = self.full_y.loc[te_mask_t].values
            idx += n_t
    
            del pipe
            gc.collect()
    
        self.predictions['Post-Ridge Logistic: Rolling Window'] = all_proba[:idx]
        self._summarize_model('Post-Ridge Logistic: Rolling Window', all_y[:idx], all_proba[:idx])

    def _fit_fixed_window_ridge(self, selected_features: List[str], window_length: int) -> None:
        years_test = np.arange(self.test_start, self.test_end + 1)
    
        n_test = (self.year >= self.test_start).sum()
        all_proba = np.zeros(n_test)
        all_y = np.zeros(n_test)
        idx = 0
    
        for t in years_test:
            tr_end = t - 1
            tr_start = tr_end - window_length + 1
            tr_mask_t = (self.year >= tr_start) & (self.year <= tr_end)
            te_mask_t = (self.year == t)
    
            n_t = te_mask_t.sum()
            if tr_mask_t.sum() == 0 or n_t == 0:
                continue
    
            pipe = self._make_logit_pipeline()
            pipe.fit(self.full_X.loc[tr_mask_t, selected_features], self.full_y.loc[tr_mask_t])
    
            p_t = pipe.predict_proba(self.full_X.loc[te_mask_t, selected_features])[:, 1]
    
            all_proba[idx:idx+n_t] = p_t
            all_y[idx:idx+n_t] = self.full_y.loc[te_mask_t].values
            idx += n_t
    
            del pipe
            gc.collect()
    
        self.predictions['Post-Ridge Logistic: Fixed Window'] = all_proba[:idx]
        self._summarize_model('Post-Ridge Logistic: Fixed Window', all_y[:idx], all_proba[:idx])

    def fit_knn(self, k_values: Optional[List[int]] = None) -> None:
        if k_values is None:
            k_values = list(range(1, 10))
    
        best_k = None
        best_mis_rate = float('inf')
    
        for k in k_values:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k))
            ])
            pipe.fit(self.X_train, self.y_train)
            y_pred = pipe.predict(self.X_val)
            mis_rate = (y_pred != self.y_val).mean()
    
            if mis_rate < best_mis_rate:
                best_mis_rate = mis_rate
                best_k = k
    
        combined_mask = (self.year >= self.train_start) & (self.year <= self.val_end)
        feature_cols = self._get_feature_columns()
        X_combined = self.full_X.loc[combined_mask, feature_cols]
        y_combined = self.full_y.loc[combined_mask]
    
        best_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=best_k)) # type: ignore
        ])
        best_pipe.fit(X_combined, y_combined)
    
        y_pred_proba = best_pipe.predict_proba(self.X_test)[:, 1]
    
        model_name = f'KNN (K={best_k})'
        self.models[model_name] = best_pipe
        self.predictions[model_name] = y_pred_proba
        self._summarize_model(model_name, self.y_test, y_pred_proba)

    def fit_random_forest(self, n_estimators_list: Optional[List[int]] = None, 
                      max_depth_list: Optional[List] = None) -> None:
        if n_estimators_list is None:
            n_estimators_list = [50, 100, 200, 300]
        if max_depth_list is None:
            max_depth_list = [3, 5, 10, None]
    
        best_params = None
        best_mis_rate = float('inf')
    
        for n in n_estimators_list:
            for d in max_depth_list:
                rf = RandomForestClassifier(
                    n_estimators=n,
                    max_depth=d,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                rf.fit(self.X_train, self.y_train)
                y_pred = rf.predict(self.X_val)
                mis_rate = (y_pred != self.y_val).mean()
    
                if mis_rate < best_mis_rate:
                    best_mis_rate = mis_rate
                    best_params = {'n_estimators': n, 'max_depth': d}
    
        combined_mask = (self.year >= self.train_start) & (self.year <= self.val_end)
        feature_cols = self._get_feature_columns()
        X_combined = self.full_X.loc[combined_mask, feature_cols]
        y_combined = self.full_y.loc[combined_mask]
    
        best_rf = RandomForestClassifier(
            n_estimators=best_params['n_estimators'], # type: ignore
            max_depth=best_params['max_depth'], # type: ignore
            random_state=self.random_state,
            n_jobs=-1
        )
        best_rf.fit(X_combined, y_combined)
    
        y_pred_proba = best_rf.predict_proba(self.X_test)[:, 1]
    
        model_name = f"Random Forest (n={best_params['n_estimators']}, depth={best_params['max_depth']})" # type: ignore
        self.models[model_name] = best_rf
        self.predictions[model_name] = y_pred_proba
        self._summarize_model(model_name, self.y_test, y_pred_proba)

    def fit_survival_random_forest(self, n_estimators_list: Optional[List[int]] = None) -> None:
        self._prepare_survival_data()
    
        if n_estimators_list is None:
            n_estimators_list = [50, 75, 100]
    
        y_surv_train = Surv.from_arrays(
            event=self.event_train,
            time=self.duration_train
        )
    
        best_n = None
        best_mis_rate = float('inf')
    
        for n in n_estimators_list:
            rsf = RandomSurvivalForest(
                n_estimators=n,
                max_features="sqrt",
                min_samples_split=10,
                min_samples_leaf=10,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            rsf.fit(self.X_train, y_surv_train)
            y_score = rsf.predict(self.X_val).astype(float)
    
            _, mis = self._best_cutoff_misrate_survival(self.y_val.values, y_score)
    
            if mis < best_mis_rate:
                best_mis_rate = mis
                best_n = n
        
        combined_mask = (self.year >= self.train_start) & (self.year <= self.val_end)
        feature_cols = self._get_feature_columns()
        X_combined = self.full_X.loc[combined_mask, feature_cols]
        event_combined = self.full_event.loc[combined_mask]
        duration_combined = self.full_duration.loc[combined_mask]
    
        y_surv_combined = Surv.from_arrays(
            event=event_combined,
            time=duration_combined
        )
    
        best_rsf = RandomSurvivalForest(
            n_estimators=best_n, # type: ignore
            max_features="sqrt",
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        best_rsf.fit(X_combined, y_surv_combined)
    
        y_score_test = best_rsf.predict(self.X_test).astype(float)
    
        model_name = f'Survival RF (n={best_n})'
        self.models[model_name] = best_rsf
        self.predictions[model_name] = y_score_test
        self._summarize_model(model_name, self.y_test, y_score_test, score_scale='score')

    def _prepare_survival_data(self) -> None:
        if 'event' not in self.full_X.columns or 'duration' not in self.full_X.columns:
            df = pd.DataFrame({
                'PERMNO': self.full_X['PERMNO'],
                'year': self.year,
                'bankruptcy': self.full_y
            })

            first_bk = (
                df.loc[df["bankruptcy"] == 1, ["PERMNO", "year"]]
                  .groupby("PERMNO", as_index=False)["year"].min()
                  .rename(columns={"year": "bk_first_year"})
            ) # type: ignore
            df = df.merge(first_bk, on="PERMNO", how="left")

            last_year_firm = df.groupby("PERMNO", as_index=False)["year"].max().rename(columns={"year":"firm_last_year"}) # type: ignore
            df = df.merge(last_year_firm, on="PERMNO", how="left")

            has_event = df["bk_first_year"].notna()
            event_mask = has_event & (df["bk_first_year"] >= df["year"])

            dur_to_event  = np.where(has_event, df["bk_first_year"] - df["year"] + 1, np.nan)
            dur_to_censor = df["firm_last_year"] - df["year"] + 1

            df["event"]    = event_mask.astype(bool)
            df["duration"] = np.where(event_mask, dur_to_event, dur_to_censor).astype(float)
            df.loc[df["duration"] < 1, "duration"] = 1.0

            self.full_X['event'] = df['event'].values
            self.full_X['duration'] = df['duration'].values
        
        train_mask = (self.year >= self.train_start) & (self.year <= self.train_end)
        val_mask = (self.year >= self.val_start) & (self.year <= self.val_end)
        test_mask = (self.year >= self.test_start) & (self.year <= self.test_end)

        self.full_event = self.full_X['event'].astype(bool)
        self.full_duration = self.full_X['duration'].astype(float)

        self.event_train = self.full_event.loc[train_mask].to_numpy()
        self.event_val = self.full_event.loc[val_mask].to_numpy()
        self.event_test = self.full_event.loc[test_mask].to_numpy()

        self.duration_train = self.full_duration.loc[train_mask].to_numpy()
        self.duration_val = self.full_duration.loc[val_mask].to_numpy()
        self.duration_test = self.full_duration.loc[test_mask].to_numpy()

    def fit_xgboost(self, n_estimators_list: Optional[List[int]] = None, max_depth: int = 3, 
                learning_rate: float = 0.1) -> None:
    
        if n_estimators_list is None:
            n_estimators_list = [50, 100, 200, 300]
    
        best_n = None
        best_mis_rate = float('inf')
    
        for n_rounds in n_estimators_list:
            xgb = XGBClassifier(
                n_estimators=n_rounds,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric="logloss"
            )
            xgb.fit(self.X_train, self.y_train)
    
            y_prob = xgb.predict_proba(self.X_val)[:, 1]
            _, mis = self._best_cutoff_misrate(self.y_val.values, y_prob)
    
            if mis < best_mis_rate:
                best_mis_rate = mis
                best_n = n_rounds
    
        
        combined_mask = (self.year >= self.train_start) & (self.year <= self.val_end)
        feature_cols = self._get_feature_columns()
        X_combined = self.full_X.loc[combined_mask, feature_cols]
        y_combined = self.full_y.loc[combined_mask]
    
        best_xgb = XGBClassifier(
            n_estimators=best_n,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric="logloss"
        )
        best_xgb.fit(X_combined, y_combined)
    
        y_pred_proba = best_xgb.predict_proba(self.X_test)[:, 1]
    
        model_name = f'XGBoost (n={best_n})'
        self.models[model_name] = best_xgb
        self.predictions[model_name] = y_pred_proba
        self._summarize_model(model_name, self.y_test, y_pred_proba)

    def fit_lightgbm(self, max_depth_list: Optional[List[int]] = None, n_estimators: int = 300, 
                 learning_rate: float = 0.05) -> None:
        if max_depth_list is None:
            max_depth_list = [-1, 3, 5, 7, 9]
    
        best_md = None
        best_mis_rate = float('inf')
    
        for md in max_depth_list:
            lgbm = LGBMClassifier(
                objective="binary",
                max_depth=md,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=0.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            lgbm.fit(self.X_train, self.y_train)
    
            y_prob = lgbm.predict_proba(self.X_val)[:, 1] # type: ignore
            _, mis = self._best_cutoff_misrate(self.y_val.values, y_prob)
    
            if mis < best_mis_rate:
                best_mis_rate = mis
                best_md = md
        
        combined_mask = (self.year >= self.train_start) & (self.year <= self.val_end)
        feature_cols = self._get_feature_columns()
        X_combined = self.full_X.loc[combined_mask, feature_cols]
        y_combined = self.full_y.loc[combined_mask]
    
        best_lgbm = LGBMClassifier(
            objective="binary",
            max_depth=best_md, # type: ignore
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        best_lgbm.fit(X_combined, y_combined)
    
        y_pred_proba = best_lgbm.predict_proba(self.X_test)[:, 1] # type: ignore
    
        model_name = f'LightGBM (depth={best_md})'
        self.models[model_name] = best_lgbm
        self.predictions[model_name] = y_pred_proba
        self._summarize_model(model_name, self.y_test, y_pred_proba)

    def _ks_stat(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(np.max(np.abs(tpr - fpr)))

    def _best_cutoff_misrate(self, y_true: np.ndarray,y_score: np.ndarray) -> Tuple[float, float]:
        cutoffs = np.linspace(0, 1, 101)
        misrates = []
        for c in cutoffs:
            preds = (y_score >= c).astype(int)
            misrates.append((preds != y_true).mean())
        best_idx = int(np.argmin(misrates))
        return float(cutoffs[best_idx]), float(misrates[best_idx])

    def _best_cutoff_misrate_survival(self, y_true: np.ndarray, y_score: np.ndarray, 
                                      n_grid: int = 201) -> Tuple[float, float]:
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=float)

        cutoffs = np.unique(np.quantile(y_score, np.linspace(0, 1, n_grid)))
        mis = []
        for c in cutoffs:
            pred = (y_score >= c).astype(int)
            mis.append((pred != y_true).mean())
        i = int(np.argmin(mis))
        return float(cutoffs[i]), float(mis[i])

    def _summarize_model(self,name: str,y_true: np.ndarray,y_score: np.ndarray,
                         score_scale: str = 'prob') -> None:
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=float)

        auc = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else np.nan
        ks = self._ks_stat(y_true, y_score) if np.unique(y_true).size > 1 else np.nan

        if score_scale == 'prob':
            _, mis = self._best_cutoff_misrate(y_true, y_score)
        elif score_scale == 'score':
            _, mis = self._best_cutoff_misrate_survival(y_true, y_score)
        else:
            raise ValueError("score_scale must be 'prob' or 'score'")

        self.results.loc[len(self.results)] = {
            'name': str(name),
            'auc': auc,
            'ks': ks,
            'misclassification_rate': float(mis)
        }

    def plot_roc_curve(self, model_names: Optional[List[str]] = None) -> Dict[str, float]:
        
        if model_names is None:
            model_names = list(self.predictions.keys())

        plt.figure(figsize=(10, 8))
        aurocs = {}

        for name in model_names:
            if name not in self.predictions:
                print(f"Warning: {name} not found in predictions")
                continue

            y_score = self.predictions[name]

            if np.unique(self.y_test).size < 2:
                print(f"{name}: only one class present; ROC/AUC undefined.")
                continue

            fpr, tpr, _ = roc_curve(self.y_test, y_score)
            auc_val = roc_auc_score(self.y_test, y_score)
            aurocs[name] = auc_val

            plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'outputs/ROC-AUC Curve Plot')
        plt.close()

        return aurocs

    def get_results(self) -> pd.DataFrame:
        return self.results.copy()

    def fit_all_models(self) -> None:
        for mode in ['plain', 'rolling', 'fixed']:
            self.fit_logistic_regression(mode=mode)
            gc.collect()

        for mode in ['plain', 'rolling', 'fixed']:
            self.fit_lasso_logistic(mode=mode)
            gc.collect()

        for mode in ['plain', 'rolling', 'fixed']:
            self.fit_ridge_logistic(mode=mode)
            gc.collect()

        self.fit_knn()
        gc.collect()

        self.fit_random_forest()
        gc.collect()

        self.fit_survival_random_forest()
        gc.collect()

        self.fit_xgboost()
        gc.collect()

        self.fit_lightgbm()
        gc.collect()