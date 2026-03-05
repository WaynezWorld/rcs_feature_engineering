#!/usr/bin/env python3
"""
Standalone Forecasting Script

Loads a CSV of feature-engineered data, trains the time series model,
generates forecasts, and optionally runs daily disaggregation.

Usage
-----
# Basic forecast from a prepared CSV:
python run_forecasting.py --input results/processed_data_engineered.csv

# Override model type and forecast horizon:
python run_forecasting.py --input data/engineered.csv --model xgboost --periods 12

# With daily disaggregation:
python run_forecasting.py --input data/engineered.csv --daily

# Specify a segment group-by column:
python run_forecasting.py --input data/engineered.csv --group-by "Profit Center"

# Dry run (no model training):
python run_forecasting.py --input data/engineered.csv --dry-run
"""

import sys
import os
import argparse
import calendar
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
except ImportError:
    def load_dotenv(**kwargs) -> bool:  # type: ignore[misc]
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ConfigManager
# ─────────────────────────────────────────────────────────────────────────────

class ConfigManager:
    """Manages configuration loading and access"""

    def __init__(self, config_path: Optional[str] = None):
        load_dotenv()
        self.config_path = config_path or str(
            Path(__file__).resolve().parent / "config" / "default.yml"
        )
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        return config

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value: Any = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def reload(self):
        self.config = self._load_config()

    def update(self, updates: Dict[str, Any]):
        self._deep_update(self.config, updates)

    def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def convert_year_period_to_date(year_period: Union[str, int]) -> pd.Timestamp:
    """
    Convert Year Period format to date.
    Handles formats like "202101", "2021/01", or "2021-01" (YYYYPPP or YYYY/PPP or YYYY-PPP).
    Returns first day of the month.
    """
    year_period = str(year_period).strip()

    # If already in YYYYMMDD format, parse directly
    if len(year_period) == 8 and year_period.isdigit():
        try:
            return pd.to_datetime(year_period, format="%Y%m%d")
        except Exception:
            pass

    # Handle YYYY/PP or YYYY-PP or YYYY/PPP or YYYY-PPP formats
    if "/" in year_period or "-" in year_period:
        delimiter = "/" if "/" in year_period else "-"
        parts = year_period.split(delimiter)
        if len(parts) == 2:
            try:
                year = int(parts[0])
                period = int(parts[1])
                if 1 <= period <= 12:
                    return pd.Timestamp(year=year, month=period, day=1)
            except (ValueError, IndexError):
                pass

    # Parse YYYYPPP format (7 digits: YYYY + PPP, where PPP is 001-012)
    if len(year_period) == 7 and year_period.isdigit():
        try:
            year = int(year_period[:4])
            period = int(year_period[4:7])
            if 1 <= period <= 12:
                return pd.Timestamp(year=year, month=period, day=1)
        except (ValueError, IndexError):
            pass

    # Fallback: try to parse as string
    return pd.to_datetime(year_period)


def handle_outliers(
    series: pd.Series,
    method: str = "iqr",
    factor: float = 1.5,
) -> pd.Series:
    """
    Clip outliers in a pandas Series.

    method="iqr"  : winsorise using  Q1 - factor*IQR  /  Q3 + factor*IQR
    method="zscore": clip values beyond  factor  standard deviations from mean
    """
    s = pd.to_numeric(series, errors="coerce")
    if method == "iqr":
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return s.clip(lower=lower, upper=upper)
    elif method == "zscore":
        mean = s.mean()
        std = s.std()
        if std == 0:
            return s
        lower = mean - factor * std
        upper = mean + factor * std
        return s.clip(lower=lower, upper=upper)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Optional ML library imports
# ─────────────────────────────────────────────────────────────────────────────

try:
    from prophet import Prophet  # type: ignore[import-untyped]
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
    from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore[import-untyped]
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor  # type: ignore[import-untyped]
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor  # type: ignore[import-untyped]
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor  # type: ignore[import-untyped]
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _HWModel  # type: ignore[import-untyped]
    HOLTWINTERS_AVAILABLE = True
except ImportError:
    HOLTWINTERS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY: Dict[str, Any] = {}


def register_model(name: str, factory: Any, *, available: bool = True) -> None:
    _MODEL_REGISTRY[name] = {"factory": factory, "available": available}


def available_models() -> List[str]:
    return [k for k, v in _MODEL_REGISTRY.items() if v["available"]]


# ─────────────────────────────────────────────────────────────────────────────
# Forecast model classes
# ─────────────────────────────────────────────────────────────────────────────

class ForecastModel(ABC):
    """Abstract base class for forecast models"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        self.model = None
        self.is_fitted = False
        self.feature_columns: Optional[List[str]] = None

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_column: str,
            date_column: str, **kwargs) -> "ForecastModel":
        pass

    @abstractmethod
    def predict(self, df: Optional[pd.DataFrame] = None,
                periods: int = 12) -> pd.DataFrame:
        pass

    def save_model(self, path: str):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "model": self.model,
            "name": self.name,
            "params": self.params,
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, path: str):
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        self.model = model_data["model"]
        self.name = model_data["name"]
        self.params = model_data["params"]
        self.feature_columns = model_data["feature_columns"]
        self.is_fitted = model_data["is_fitted"]


class ProphetModel(ForecastModel):
    """Prophet-based forecasting model"""

    def __init__(self, name: str = "prophet", **kwargs):
        super().__init__(name, **kwargs)
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")

    def fit(self, df: pd.DataFrame, target_column: str,
            date_column: str, **kwargs) -> "ProphetModel":
        prophet_df = df[[date_column, target_column]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df = prophet_df.dropna()
        prophet_params = {
            "yearly_seasonality": self.params.get("yearly_seasonality", True),
            "weekly_seasonality": self.params.get("weekly_seasonality", False),
            "daily_seasonality": self.params.get("daily_seasonality", False),
            "growth": self.params.get("growth", "linear"),
        }
        self.model = Prophet(**prophet_params)
        additional_regressors = self.params.get("additional_regressors", [])
        for regressor in additional_regressors:
            if regressor in df.columns:
                self.model.add_regressor(regressor)
                prophet_df[regressor] = df[regressor]
        self.model.fit(prophet_df)
        self.is_fitted = True
        return self

    def predict(self, df: Optional[pd.DataFrame] = None,
                periods: int = 12) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if df is not None:
            future_df = df.copy()
            if "ds" not in future_df.columns:
                future_df.rename(columns={future_df.columns[0]: "ds"}, inplace=True)
        else:
            future_df = self.model.make_future_dataframe(periods=periods, freq="MS")
        forecast = self.model.predict(future_df)
        result_columns = ["ds", "yhat", "yhat_lower", "yhat_upper"]
        return forecast[result_columns].rename(columns={
            "ds": "date",
            "yhat": "prediction",
            "yhat_lower": "prediction_lower",
            "yhat_upper": "prediction_upper",
        })


class SklearnModel(ForecastModel):
    """Scikit-learn based forecasting model"""

    def __init__(self, name: str = "sklearn", model_type: str = "linear", **kwargs):
        super().__init__(name, **kwargs)
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is not available")
        self.model_type = model_type
        self.date_column: Optional[str] = None
        self._init_model()

    def _init_model(self):
        entry = _MODEL_REGISTRY.get(self.model_type)
        if entry is None:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                f"Available: {list(_MODEL_REGISTRY.keys())}"
            )
        if not entry["available"]:
            raise ImportError(f"Library for model '{self.model_type}' is not installed.")
        self.model = entry["factory"](**self.params)

    def _ensure_date_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df["_year"] = df[date_column].dt.year
        df["_month"] = df[date_column].dt.month
        df["_month_sin"] = np.sin(2 * np.pi * df["_month"] / 12.0)
        df["_month_cos"] = np.cos(2 * np.pi * df["_month"] / 12.0)
        df["_time_index"] = (df[date_column] - df[date_column].min()).dt.days
        return df

    def _prepare_features(self, df: pd.DataFrame, target_column: str,
                          date_column: str) -> Tuple[np.ndarray, np.ndarray]:
        self.date_column = date_column
        df = self._ensure_date_features(df, date_column)
        feature_columns = [
            col for col in df.columns
            if col not in [target_column, date_column]
            and df[col].dtype in ["int64", "float64"]
        ]
        self.feature_columns = feature_columns
        X = df[feature_columns].fillna(0).values
        y = np.asarray(df[target_column].fillna(0).values)
        return X, y

    def fit(self, df: pd.DataFrame, target_column: str,
            date_column: str, **kwargs) -> "SklearnModel":
        X, y = self._prepare_features(df, target_column, date_column)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, df: Optional[pd.DataFrame] = None,
                periods: int = 12) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if df is None:
            raise ValueError("DataFrame with features is required for sklearn prediction")
        if self.feature_columns is None:
            raise ValueError("Feature columns not available")
        if self.date_column:
            df = self._ensure_date_features(df, self.date_column)
        X = df[self.feature_columns].fillna(0).values
        predictions = self.model.predict(X)
        return pd.DataFrame({
            "date": df.iloc[:, 0],
            "prediction": predictions,
        })


class HoltWintersModel(ForecastModel):
    """Holt-Winters / Exponential Smoothing forecasting model."""

    def __init__(self, name: str = "holt_winters", **kwargs):
        super().__init__(name, **kwargs)
        if not HOLTWINTERS_AVAILABLE:
            raise ImportError(
                "statsmodels is not available. Install with: pip install statsmodels"
            )
        self.date_column: Optional[str] = None
        self._freq: str = "MS"

    def fit(self, df: pd.DataFrame, target_column: str,
            date_column: str, **kwargs) -> "HoltWintersModel":
        self.date_column = date_column
        ts = df.set_index(date_column)[target_column].sort_index().astype(float)
        if not isinstance(ts.index, pd.DatetimeIndex):
            ts.index = pd.DatetimeIndex(ts.index)
        inferred = pd.infer_freq(ts.index)
        if inferred is not None:
            ts.index = pd.DatetimeIndex(ts.index, freq=inferred)
        else:
            ts.index = pd.DatetimeIndex(ts.index, freq=self._freq)
        if len(ts) < 4:
            raise ValueError(f"HoltWinters requires at least 4 data points, got {len(ts)}")
        seasonal_periods = self.params.get("seasonal_periods", 12)
        trend = self.params.get("trend", "add")
        seasonal = self.params.get("seasonal", "add")
        damped_trend = self.params.get("damped_trend", False)
        if len(ts) < 2 * seasonal_periods:
            seasonal = None
        hw = _HWModel(
            ts,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
            damped_trend=damped_trend,
        )
        self.model = hw.fit(optimized=True)
        self.is_fitted = True
        return self

    def predict(self, df: Optional[pd.DataFrame] = None,
                periods: int = 12) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        forecast = self.model.forecast(periods)
        return pd.DataFrame({
            "date": forecast.index,
            "prediction": forecast.values,
        })


class EnsembleModel(ForecastModel):
    """Simple weighted-average ensemble of two sklearn-compatible models."""

    def __init__(self, name: str = "ensemble", **kwargs):
        super().__init__(name, **kwargs)
        self.weight_a: float = float(kwargs.get("weight_a", 0.5))
        self.model_a_obj: SklearnModel = SklearnModel(model_type=kwargs.get("model_a", "ridge"))
        self.model_b_obj: SklearnModel = SklearnModel(model_type=kwargs.get("model_b", "xgboost"))
        self.date_column: Optional[str] = None

    def fit(self, df: pd.DataFrame, target_column: str,
            date_column: str, **kwargs) -> "EnsembleModel":
        self.date_column = date_column
        self.model_a_obj.fit(df, target_column, date_column, **kwargs)
        self.model_b_obj.fit(df, target_column, date_column, **kwargs)
        self.feature_columns = self.model_a_obj.feature_columns
        self.is_fitted = True
        return self

    def predict(self, df: Optional[pd.DataFrame] = None,
                periods: int = 12) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        pred_a = self.model_a_obj.predict(df, periods)
        pred_b = self.model_b_obj.predict(df, periods)
        result = pred_a.copy()
        result["prediction"] = (
            self.weight_a * np.asarray(pred_a["prediction"].values)
            + (1 - self.weight_a) * np.asarray(pred_b["prediction"].values)
        )
        return result


# ─── Register built-in models ─────────────────────────────────────────────────
if SKLEARN_AVAILABLE:
    register_model("linear", LinearRegression, available=True)
    register_model(
        "ridge",
        lambda **kw: __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge(**kw),
        available=True,
    )
    register_model("random_forest", lambda **kw: RandomForestRegressor(**kw), available=True)
    register_model(
        "elastic_net",
        lambda **kw: __import__("sklearn.linear_model", fromlist=["ElasticNet"]).ElasticNet(**kw),
        available=True,
    )
    register_model(
        "lasso",
        lambda **kw: __import__("sklearn.linear_model", fromlist=["Lasso"]).Lasso(**kw),
        available=True,
    )
    register_model(
        "bayesian_ridge",
        lambda **kw: __import__("sklearn.linear_model", fromlist=["BayesianRidge"]).BayesianRidge(**kw),
        available=True,
    )
else:
    for _n in ("linear", "ridge", "random_forest", "elastic_net", "lasso", "bayesian_ridge"):
        register_model(_n, lambda **kw: None, available=False)

register_model("xgboost", lambda **kw: XGBRegressor(**kw), available=XGBOOST_AVAILABLE)
register_model("lightgbm", lambda **kw: LGBMRegressor(**kw), available=LIGHTGBM_AVAILABLE)
register_model(
    "catboost",
    lambda **kw: CatBoostRegressor(verbose=0, **kw),
    available=CATBOOST_AVAILABLE,
)


# ─────────────────────────────────────────────────────────────────────────────
# TimeSeriesForecaster
# ─────────────────────────────────────────────────────────────────────────────

class TimeSeriesForecaster:
    """Main forecasting orchestrator"""

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.model: Optional[ForecastModel] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.metrics: Dict[str, Any] = {}
        self.group_by: Optional[str] = None
        self.use_log_transform: bool = config_manager.get("forecasting.use_log_transform", False)
        self.target_transform_offset: float = config_manager.get(
            "forecasting.log_transform_offset", 0
        )
        self._agg_train_last: Optional[Dict[Any, Any]] = None
        self._agg_history: Optional[pd.DataFrame] = None
        self._max_target_log: Optional[float] = None
        self.split_meta: Optional[Dict[str, Any]] = None
        self._feature_engineer = None
        self._date_column: Optional[str] = None
        self._target_column: Optional[str] = None

    def _get_adaptive_xgb_params(self, n_train_rows: int) -> Dict[str, Any]:
        base_params = dict(self.config.get("forecasting.xgboost", {}) or {})
        tuning = self.config.get("forecasting.segment_tuning", {}) or {}
        if not tuning or not tuning.get("enabled", False):
            return base_params
        small_thresh = int(tuning.get("small_threshold", 30))
        medium_thresh = int(tuning.get("medium_threshold", 48))
        if n_train_rows < small_thresh:
            tier = "small"
        elif n_train_rows < medium_thresh:
            tier = "medium"
        else:
            tier = "large"
        overrides = tuning.get(tier, {}) or {}
        if overrides:
            return {**base_params, **overrides}
        return base_params

    def _populate_future_features(
        self, future_df: pd.DataFrame, target_column: str, date_column: str
    ) -> pd.DataFrame:
        if self._agg_history is None or self._agg_history.empty:
            return future_df
        if self.model is None:
            return future_df
        history = self._agg_history.sort_values(date_column).reset_index(drop=True)
        history_values = history[target_column].tolist()
        lag_pattern = re.compile(rf"^{re.escape(target_column)}_lag_(\d+)$")
        roll_pattern = re.compile(rf"^{re.escape(target_column)}_rolling_mean_(\d+)$")
        for feature in self.model.feature_columns or []:
            lag_match = lag_pattern.match(feature)
            roll_match = roll_pattern.match(feature)
            if lag_match:
                lag = int(lag_match.group(1))
                future_df[feature] = [
                    history_values[-lag] if len(history_values) >= lag else 0.0
                    for _ in range(len(future_df))
                ]
            elif roll_match:
                window = int(roll_match.group(1))
                if len(history_values) >= 1:
                    tail = history_values[-window:]
                    roll_value = float(np.mean(tail))
                else:
                    roll_value = 0.0
                future_df[feature] = [roll_value] * len(future_df)
            elif feature.startswith(f"{date_column}_month"):
                future_df[feature] = pd.to_datetime(future_df[date_column]).dt.month
                if feature.endswith("_sin"):
                    future_df[feature] = np.sin(2 * np.pi * future_df[feature] / 12)
                if feature.endswith("_cos"):
                    future_df[feature] = np.cos(2 * np.pi * future_df[feature] / 12)
            elif feature.startswith(f"{date_column}_quarter"):
                quarter = pd.to_datetime(future_df[date_column]).dt.quarter
                if feature.endswith("_sin"):
                    future_df[feature] = np.sin(2 * np.pi * quarter / 4)
                elif feature.endswith("_cos"):
                    future_df[feature] = np.cos(2 * np.pi * quarter / 4)
                else:
                    future_df[feature] = quarter
            elif feature == f"{date_column}_year":
                future_df[feature] = pd.to_datetime(future_df[date_column]).dt.year
        return future_df

    def _recursive_forecast(
        self,
        start_date: pd.Timestamp,
        periods: int,
        target_column: str,
        date_column: str,
    ) -> pd.DataFrame:
        if self._agg_history is None or self._agg_history.empty:
            raise ValueError("Aggregated history is required for recursive forecasting")
        assert self.model is not None
        history = self._agg_history.sort_values(date_column).reset_index(drop=True)
        predictions: List[Dict] = []
        date_range = pd.date_range(start=start_date, periods=periods, freq="MS")
        for forecast_date in date_range:
            row = {date_column: forecast_date}
            future_row = pd.DataFrame([row])
            if self._feature_engineer is not None and self._feature_engineer.fitted_transformers:
                future_features = self._feature_engineer.transform_future(
                    history_df=history,
                    future_df=future_row,
                    target_column=target_column,
                    date_column=date_column,
                )
                future_row = future_features
            else:
                future_row = self._populate_future_features(
                    future_row, target_column, date_column
                )
            for feature in self.model.feature_columns or []:
                if feature not in future_row.columns:
                    future_row[feature] = (
                        self._agg_train_last.get(feature, 0.0)
                        if self._agg_train_last
                        else 0.0
                    )
            pred_value = float(self.model.predict(future_row)["prediction"].iloc[0])
            predictions.append({"date": forecast_date, "prediction": pred_value})
            new_row = pd.DataFrame(
                {date_column: [forecast_date], target_column: [pred_value]}
            )
            history = pd.concat([history, new_row], ignore_index=True)
            self._agg_history = history
        return pd.DataFrame(predictions)

    def _prepare_eval_df(
        self, df: pd.DataFrame, target_column: str, date_column: str
    ) -> pd.DataFrame:
        eval_df = df
        if self.group_by:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            eval_df = df.groupby(date_column, as_index=False)[numeric_cols].sum()
        return eval_df

    def _split_data(
        self, df: pd.DataFrame, date_column: str, target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        backtest_months = int(self.config.get("forecasting.backtest_months", 0) or 0)
        validation_months = int(self.config.get("forecasting.validation_months", 0) or 0)

        if backtest_months > 0:
            cutoff_cfg = self.config.get("forecasting.backtest_cutoff_month", None)
            if cutoff_cfg:
                cutoff = pd.Timestamp(cutoff_cfg).to_period("M").to_timestamp()
            else:
                max_date = pd.Timestamp(df_sorted[date_column].max())
                cutoff = (max_date + pd.DateOffset(months=1)).to_period("M").to_timestamp()

            test_start = (
                cutoff - pd.DateOffset(months=backtest_months)
            ).to_period("M").to_timestamp()
            test_end = cutoff

            if validation_months > 0:
                validation_start = (
                    test_start - pd.DateOffset(months=validation_months)
                ).to_period("M").to_timestamp()
            else:
                validation_start = test_start

            train_data = df_sorted[df_sorted[date_column] < validation_start]
            validation_data = df_sorted[
                (df_sorted[date_column] >= validation_start)
                & (df_sorted[date_column] < test_start)
            ]
            test_data = df_sorted[
                (df_sorted[date_column] >= test_start)
                & (df_sorted[date_column] < test_end)
            ]

            if len(train_data) > 0 and len(test_data) > 0:
                self.split_meta = {
                    "cutoff": str(cutoff.date()),
                    "validation_start": str(validation_start.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                    "train_rows": len(train_data),
                    "validation_rows": len(validation_data),
                    "test_rows": len(test_data),
                }
                return train_data, validation_data, test_data

        test_size = self.config.get("forecasting.test_size", 0.2)
        validation_size = self.config.get("forecasting.validation_split", 0.1)
        n_total = len(df_sorted)
        n_test = int(n_total * test_size)
        n_validation = int(n_total * validation_size)
        n_train = n_total - n_test - n_validation
        train_data = df_sorted.iloc[:n_train]
        validation_data = df_sorted.iloc[n_train : n_train + n_validation]
        test_data = df_sorted.iloc[n_train + n_validation :]
        return train_data, validation_data, test_data

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, label: str = ""
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        y_true = pd.to_numeric(np.asarray(y_true), errors="coerce").astype(float)
        y_pred = pd.to_numeric(np.asarray(y_pred), errors="coerce").astype(float)
        mask = ~(
            np.isnan(y_true) | np.isnan(y_pred)
            | np.isinf(y_true) | np.isinf(y_pred)
        )
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        if len(y_true_clean) == 0:
            return metrics
        abs_err = np.abs(y_true_clean - y_pred_clean)
        if SKLEARN_AVAILABLE:
            metrics["mae"] = float(mean_absolute_error(y_true_clean, y_pred_clean))
            metrics["mse"] = float(mean_squared_error(y_true_clean, y_pred_clean))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            try:
                metrics["r2"] = float(r2_score(y_true_clean, y_pred_clean))
            except Exception:
                metrics["r2"] = float("nan")
        else:
            metrics["mae"] = float(np.mean(abs_err))
            metrics["mse"] = float(np.mean((y_true_clean - y_pred_clean) ** 2))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        mape_mask = y_true_clean != 0
        if np.any(mape_mask):
            metrics["mape"] = float(
                np.mean(
                    np.abs(
                        (y_true_clean[mape_mask] - y_pred_clean[mape_mask])
                        / y_true_clean[mape_mask]
                    )
                )
                * 100
            )
        sum_abs_actual = float(np.sum(np.abs(y_true_clean)))
        if sum_abs_actual > 0:
            metrics["wape"] = float(np.sum(abs_err) / sum_abs_actual * 100)
        try:
            seasonal_period = 12
            if len(y_true_clean) > seasonal_period:
                naive_errors = np.abs(
                    y_true_clean[seasonal_period:] - y_true_clean[:-seasonal_period]
                )
            else:
                naive_errors = np.abs(np.diff(y_true_clean))
            mean_naive = float(np.mean(naive_errors)) if len(naive_errors) > 0 else 0.0
            if mean_naive > 0:
                metrics["mase"] = float(np.mean(abs_err) / mean_naive)
        except Exception:
            pass
        signed_err = y_pred_clean - y_true_clean
        metrics["bias"] = float(np.mean(signed_err))
        if sum_abs_actual > 0:
            metrics["bias_pct"] = float(np.sum(signed_err) / sum_abs_actual * 100)
        mad = metrics.get("mae", 0.0)
        if mad > 0:
            metrics["tracking_signal"] = float(np.sum(signed_err) / mad)
        return metrics

    def _transform_target(
        self, y: Union[np.ndarray, pd.Series]
    ) -> Union[np.ndarray, pd.Series]:
        if not self.use_log_transform:
            return y
        return np.log1p(y + self.target_transform_offset)

    def _inverse_transform_target(
        self, y: Union[np.ndarray, pd.Series]
    ) -> Union[np.ndarray, pd.Series]:
        if not self.use_log_transform:
            return y
        if self._max_target_log is not None:
            y = np.clip(y, a_min=None, a_max=self._max_target_log + 2.0)
        return np.expm1(y) - self.target_transform_offset

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: str,
        group_by: Optional[str] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        date_format = self.config.get("data_source.date_format", "default")
        if date_format == "year_period":
            df[date_column] = df[date_column].apply(convert_year_period_to_date)
        else:
            df[date_column] = pd.to_datetime(df[date_column])
        if group_by:
            df = df.sort_values([group_by, date_column]).reset_index(drop=True)
        else:
            df = df.sort_values(date_column).reset_index(drop=True)
        df[target_column] = df[target_column].ffill().fillna(0)
        outlier_config = self.config.get("forecasting.outlier_handling", {}) or {}
        if outlier_config.get("enabled", False):
            method = outlier_config.get("method", "iqr")
            factor = float(outlier_config.get("factor", 1.5))
            df[target_column] = handle_outliers(
                df[target_column], method=method, factor=factor
            )
        if self.use_log_transform:
            min_value = df[target_column].min()
            required_offset = 0.0
            if pd.notna(min_value) and min_value <= -1:
                required_offset = float(-min_value + 1)
            if required_offset > self.target_transform_offset:
                self.target_transform_offset = required_offset
            max_value = df[target_column].max()
            if pd.notna(max_value):
                self._max_target_log = float(
                    np.log1p(max_value + self.target_transform_offset)
                )
            df[target_column] = self._transform_target(df[target_column])
        return df

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: str,
        group_by: Optional[str] = None,
        already_prepared: bool = False,
        feature_engineer=None,
    ) -> "TimeSeriesForecaster":
        if already_prepared:
            prepared_df = df
        else:
            prepared_df = self.prepare_data(df, target_column, date_column, group_by)
        self.group_by = group_by
        self._date_column = date_column
        self._target_column = target_column

        self.train_data, self.validation_data, self.test_data = self._split_data(
            prepared_df, date_column, target_column
        )
        assert self.train_data is not None
        assert self.validation_data is not None
        assert self.test_data is not None

        if feature_engineer is not None:
            self._feature_engineer = feature_engineer
            feature_engineer.fit(self.train_data, group_by=None)
            self.train_data = feature_engineer.transform(self.train_data)
            if len(self.validation_data) > 0:
                self.validation_data = feature_engineer.transform(self.validation_data)
            if len(self.test_data) > 0:
                self.test_data = feature_engineer.transform(self.test_data)
        else:
            self._feature_engineer = None
        assert self.train_data is not None
        assert self.validation_data is not None

        model_type = self.config.get("forecasting.model_type", "prophet")
        if model_type == "prophet":
            prophet_config = self.config.get("forecasting.prophet", {}) or {}
            self.model = ProphetModel(**prophet_config)
        elif model_type == "xgboost":
            xgb_params = self._get_adaptive_xgb_params(len(self.train_data))
            self.model = SklearnModel(model_type="xgboost", **xgb_params)
        elif model_type == "holt_winters":
            hw_params = self.config.get("forecasting.holt_winters", {}) or {}
            self.model = HoltWintersModel(**hw_params)
        elif model_type == "ensemble":
            ens_params = self.config.get("forecasting.ensemble", {}) or {}
            self.model = EnsembleModel(**ens_params)
        elif model_type in _MODEL_REGISTRY:
            model_params = self.config.get(f"forecasting.{model_type}", {}) or {}
            self.model = SklearnModel(model_type=model_type, **model_params)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Available: {['prophet', 'holt_winters', 'ensemble'] + available_models()}"
            )

        agg_data = None
        if group_by:
            numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
            agg_data = self.train_data.groupby(date_column, as_index=False)[numeric_cols].sum()
            agg_data = agg_data.sort_values(date_column).reset_index(drop=True)

        if agg_data is not None and len(agg_data) > 0:
            self._agg_train_last = agg_data.iloc[-1].to_dict()
            hist = agg_data[[date_column, target_column]].copy()
            hist[date_column] = pd.to_datetime(hist[date_column])
            self._agg_history = hist
        elif isinstance(self.model, (SklearnModel, EnsembleModel)) and len(self.train_data) > 0:
            self._agg_train_last = self.train_data.iloc[-1].to_dict()
            hist = self.train_data[[date_column, target_column]].copy()
            hist[date_column] = pd.to_datetime(hist[date_column])
            self._agg_history = hist

        if group_by and agg_data is not None and len(agg_data) > 0:
            self.model.fit(agg_data, target_column, date_column)
        else:
            self.model.fit(self.train_data, target_column, date_column)

        if len(self.validation_data) > 0:
            self._validate_model(target_column, date_column)

        return self

    def _validate_model(self, target_column: str, date_column: str):
        assert self.model is not None
        assert self.validation_data is not None
        validation_df = self._prepare_eval_df(
            self.validation_data, target_column, date_column
        )
        if isinstance(self.model, ProphetModel):
            val_predictions = self.model.predict(
                validation_df[[date_column]].rename(columns={date_column: "ds"})
            )
            y_pred = np.asarray(val_predictions["prediction"].values)
        elif isinstance(self.model, HoltWintersModel):
            n_periods = len(validation_df)
            val_predictions = self.model.predict(None, n_periods)
            y_pred = np.asarray(val_predictions["prediction"].values[:n_periods])
        else:
            val_predictions = self.model.predict(validation_df)
            y_pred = np.asarray(val_predictions["prediction"].values)
        y_true = np.asarray(validation_df[target_column].values)
        val_dates = np.asarray(validation_df[date_column].values)
        if self.use_log_transform:
            y_true = np.asarray(self._inverse_transform_target(y_true))
            y_pred = np.asarray(self._inverse_transform_target(y_pred))
        self.metrics["validation"] = self._calculate_metrics(y_true, y_pred, label="validation")
        self._log_horizon_errors(val_dates, y_true, y_pred, label="validation")

    def predict(
        self,
        periods: Optional[int] = None,
        future_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if not self.model or not self.model.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if periods is None:
            periods = int(self.config.get("forecasting.forecast_horizon", 12) or 12)
        periods = int(periods)

        if isinstance(self.model, HoltWintersModel):
            predictions = self.model.predict(None, periods)
            if self.use_log_transform and "prediction" in predictions.columns:
                predictions["prediction"] = self._inverse_transform_target(
                    np.asarray(predictions["prediction"].values)
                )
            return predictions

        _needs_features = isinstance(self.model, (SklearnModel, EnsembleModel))

        if future_df is None and _needs_features:
            forecast_start = self.config.get("forecasting.forecast_start")
            date_col = self._date_column or self.config.get("data_source.date_column", "date")
            target_col = self._target_column or self.config.get("data_source.amount_column", "Actual")
            if forecast_start == "current_month":
                start_date = pd.Timestamp.today().to_period("M").to_timestamp()
            else:
                last_date = self.train_data[date_col].max()  # type: ignore[index]
                start_date = (
                    pd.Timestamp(last_date) + pd.DateOffset(months=1)
                ).to_period("M").to_timestamp()

            if self._agg_history is not None and not self._agg_history.empty:
                predictions = self._recursive_forecast(
                    start_date, periods, target_col, date_col
                )
                if self.use_log_transform and "prediction" in predictions.columns:
                    predictions["prediction"] = self._inverse_transform_target(
                        np.asarray(predictions["prediction"].values)
                    )
                return predictions

            date_range = pd.date_range(start=start_date, periods=periods, freq="MS")
            future_df = pd.DataFrame({date_col: date_range})

        amount_col = self._target_column or self.config.get("data_source.amount_column", "Actual")
        date_col = self._date_column or self.config.get("data_source.date_column", "date")

        if future_df is not None and _needs_features and self.group_by:
            future_df = self._populate_future_features(future_df, amount_col, date_col)

        if future_df is not None and _needs_features and self._agg_train_last:
            feature_cols = getattr(self.model, "feature_columns", None) or []
            for feature in feature_cols:
                if feature not in future_df.columns:
                    future_df[feature] = self._agg_train_last.get(feature, 0.0)

        predictions = self.model.predict(future_df, periods)
        if self.use_log_transform and "prediction" in predictions.columns:
            predictions["prediction"] = self._inverse_transform_target(
                np.asarray(predictions["prediction"].values)
            )
        return predictions

    def evaluate(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: str,
    ) -> Dict[str, float]:
        if not self.model or not self.model.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        test_df = self._prepare_eval_df(df, target_column, date_column)
        if isinstance(self.model, ProphetModel):
            test_predictions = self.model.predict(
                test_df[[date_column]].rename(columns={date_column: "ds"})
            )
            y_pred = np.asarray(test_predictions["prediction"].values)
        elif isinstance(self.model, HoltWintersModel):
            n_periods = len(test_df)
            test_predictions = self.model.predict(None, n_periods)
            y_pred = np.asarray(test_predictions["prediction"].values[:n_periods])
        else:
            test_predictions = self.model.predict(test_df)
            y_pred = np.asarray(test_predictions["prediction"].values)
        y_true = np.asarray(test_df[target_column].values)
        test_dates = np.asarray(test_df[date_column].values)
        if self.use_log_transform:
            y_true = np.asarray(self._inverse_transform_target(y_true))
            y_pred = np.asarray(self._inverse_transform_target(y_pred))
        test_metrics = self._calculate_metrics(y_true, y_pred, label="test")
        self.metrics["test"] = test_metrics
        horizon_metrics = self._log_horizon_errors(test_dates, y_true, y_pred, label="test")
        self.metrics["test_horizon"] = horizon_metrics
        return test_metrics

    def _log_horizon_errors(
        self,
        dates: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label: str = "",
    ) -> List[Dict[str, Any]]:
        dates_ts = pd.to_datetime(dates)
        unique_months = sorted(dates_ts.unique())
        horizon_rows: List[Dict[str, Any]] = []
        for h, month in enumerate(unique_months, start=1):
            mask = dates_ts == month
            yt = y_true[mask]
            yp = y_pred[mask]
            if len(yt) == 0:
                continue
            mae_h = float(np.mean(np.abs(yt - yp)))
            actual_h = float(np.sum(np.abs(yt)))
            wape_h = (
                float(np.sum(np.abs(yt - yp))) / actual_h * 100
                if actual_h > 0
                else float("nan")
            )
            horizon_rows.append({
                "horizon": h,
                "month": str(month.date()) if hasattr(month, "date") else str(month),
                "actual_mean": float(np.mean(yt)),
                "pred_mean": float(np.mean(yp)),
                "mae": mae_h,
                "wape": wape_h,
            })
        return horizon_rows

    def predict_on_df(
        self, df: pd.DataFrame, target_column: str, date_column: str
    ) -> pd.DataFrame:
        if not self.model or not self.model.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        eval_df = self._prepare_eval_df(df, target_column, date_column)
        if isinstance(self.model, ProphetModel):
            predictions = self.model.predict(
                eval_df[[date_column]].rename(columns={date_column: "ds"})
            )
            pred_values = np.asarray(predictions["prediction"].values)
        elif isinstance(self.model, HoltWintersModel):
            n_periods = len(eval_df)
            predictions = self.model.predict(None, n_periods)
            pred_values = np.asarray(predictions["prediction"].values[:n_periods])
        else:
            predictions = self.model.predict(eval_df)
            pred_values = np.asarray(predictions["prediction"].values)
        if self.use_log_transform:
            pred_values = self._inverse_transform_target(pred_values)
        return pd.DataFrame({
            "date": np.asarray(eval_df[date_column].values),
            "prediction": pred_values,
        })

    def save_model(self, path: Optional[str] = None):
        if not self.model:
            raise ValueError("No model to save")
        if path is None:
            model_save_path = self.config.get("forecasting.model_save_path", "models/")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{model_save_path}forecast_model_{timestamp}.pkl"
        self.model.save_model(path)
        metadata_path = path.replace(".pkl", "_metadata.pkl")
        metadata = {
            "metrics": self.metrics,
            "config": self.config.config.get("forecasting", {}),
            "timestamp": datetime.now().isoformat(),
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

    def load_model(self, path: str):
        model_type = self.config.get("forecasting.model_type", "prophet")
        if model_type == "prophet":
            self.model = ProphetModel()
        else:
            self.model = SklearnModel(model_type=model_type)
        self.model.load_model(path)
        metadata_path = path.replace(".pkl", "_metadata.pkl")
        if Path(metadata_path).exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
                self.metrics = metadata.get("metrics", {})

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self.model or not self.model.is_fitted:
            return None
        if isinstance(self.model, SklearnModel) and hasattr(
            self.model.model, "feature_importances_"
        ):
            importances = self.model.model.feature_importances_
            feature_names = self.model.feature_columns or [
                f"feature_{i}" for i in range(len(importances))
            ]
            return dict(zip(feature_names, importances))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DailyAllocationConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DailyAllocationConfig:
    """Typed wrapper around the forecasting.daily_allocation YAML block."""

    method: str = "shape_topdown"
    history_years: int = 3
    min_history_months: int = 18
    weight_model: str = "dow_dom_moy"
    negative_handling_for_weights: str = "clip_to_zero"
    smoothing_epsilon: float = 1e-6
    fallback: List[str] = field(default_factory=lambda: ["dow", "uniform"])

    daily_enabled: bool = False
    as_of_date: str = "auto"
    posting_lag_days: int = 1

    current_month_strategy: str = "fixed"
    blend_alpha: float = 0.7
    min_expected_share: float = 0.05
    max_multiplier: float = 2.0
    allow_negative_totals: bool = True

    strategy_early_cutoff_days: int = 5
    strategy_min_share_for_blend: float = 0.15

    def resolve_strategy(self, as_of_day: int, expected_share: float) -> str:
        configured = self.current_month_strategy
        if configured == "fixed":
            return "fixed"
        if as_of_day <= self.strategy_early_cutoff_days:
            return "fixed"
        if expected_share < self.strategy_min_share_for_blend:
            return "fixed"
        return configured

    @classmethod
    def from_config(cls, config: ConfigManager) -> "DailyAllocationConfig":
        defaults = cls()
        alloc = config.get("forecasting.daily_allocation", {}) or {}
        cmt = config.get("forecasting.current_month_total", {}) or {}
        return cls(
            method=alloc.get("method", defaults.method),
            history_years=alloc.get("history_years", defaults.history_years),
            min_history_months=alloc.get("min_history_months", defaults.min_history_months),
            weight_model=alloc.get("weight_model", defaults.weight_model),
            negative_handling_for_weights=alloc.get(
                "negative_handling_for_weights", defaults.negative_handling_for_weights
            ),
            smoothing_epsilon=alloc.get("smoothing_epsilon", defaults.smoothing_epsilon),
            fallback=alloc.get("fallback", defaults.fallback) or ["dow", "uniform"],
            daily_enabled=config.get("forecasting.daily_enabled", False),
            as_of_date=config.get("forecasting.as_of_date", "auto"),
            posting_lag_days=config.get("forecasting.posting_lag_days", 1),
            current_month_strategy=cmt.get("strategy", defaults.current_month_strategy),
            blend_alpha=cmt.get("blend_alpha", defaults.blend_alpha),
            min_expected_share=cmt.get("min_expected_share", defaults.min_expected_share),
            max_multiplier=cmt.get("max_multiplier", defaults.max_multiplier),
            allow_negative_totals=cmt.get(
                "allow_negative_totals", defaults.allow_negative_totals
            ),
            strategy_early_cutoff_days=cmt.get(
                "strategy_early_cutoff_days", defaults.strategy_early_cutoff_days
            ),
            strategy_min_share_for_blend=cmt.get(
                "strategy_min_share_for_blend", defaults.strategy_min_share_for_blend
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Daily data-prep helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_totals(
    raw_df: pd.DataFrame,
    date_column: str = "TS",
    amount_column: str = "Actual",
    measure_column: Optional[str] = "Measure",
    measure_value: str = "MTD",
) -> pd.DataFrame:
    """Aggregate raw line-item data to daily totals by posting date."""
    df = raw_df.copy()
    if measure_column and measure_column in df.columns and measure_value:
        df = df[df[measure_column] == measure_value]
    if date_column not in df.columns:
        raise KeyError(
            f"Date column '{date_column}' not found.  Available: {list(df.columns)}"
        )
    df["_date"] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=["_date"])
    df["_date"] = df["_date"].dt.normalize()
    df[amount_column] = pd.to_numeric(df[amount_column], errors="coerce")
    daily = (
        df.groupby("_date", as_index=False)[amount_column]
        .sum()
        .rename(columns={"_date": "date", amount_column: "actual"})  # type: ignore[call-overload]
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


def daily_to_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Roll up a daily-totals frame to monthly totals."""
    df = daily_df.copy()
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("month_start", as_index=False)["actual"]
        .sum()
        .rename(columns={"actual": "actual_month_total"})  # type: ignore[call-overload]
        .sort_values("month_start")
        .reset_index(drop=True)
    )
    return monthly


def daily_to_weekly(
    daily_df: pd.DataFrame,
    date_column: str = "date",
    value_column: str = "y_pred",
    week_start: str = "Mon",
    segment_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Roll up a daily forecast frame to ISO-week weekly totals."""
    if daily_df.empty:
        cols = (segment_columns or []) + ["week_start", "week_end", value_column]
        return pd.DataFrame(columns=cols)

    df = daily_df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

    _start_to_end_anchor: Dict[str, str] = {
        "MON": "SUN",
        "TUE": "MON",
        "WED": "TUE",
        "THU": "WED",
        "FRI": "THU",
        "SAT": "FRI",
        "SUN": "SAT",
    }
    anchor = _start_to_end_anchor.get(week_start.upper(), "SUN")
    freq = f"W-{anchor}"

    grouper_keys: List = [pd.Grouper(key=date_column, freq=freq)]
    if segment_columns:
        grouper_keys = (segment_columns or []) + grouper_keys  # type: ignore[assignment]

    weekly = df.groupby(grouper_keys, as_index=False)[value_column].sum()
    weekly = weekly.rename(columns={date_column: "week_end"})  # type: ignore[call-overload]
    weekly["week_end"] = pd.to_datetime(weekly["week_end"])
    weekly["week_start"] = weekly["week_end"] - pd.Timedelta(days=6)

    out_cols = (segment_columns or []) + ["week_start", "week_end", value_column]
    weekly = (
        weekly[out_cols]
        .sort_values((segment_columns or []) + ["week_start"])
        .reset_index(drop=True)
    )
    return weekly


# ─────────────────────────────────────────────────────────────────────────────
# Shape-weight model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_shape_table(
    daily_history: pd.DataFrame,
    cfg: DailyAllocationConfig,
) -> pd.DataFrame:
    df = daily_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["moy"] = df["date"].dt.month
    df["dom"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()

    if cfg.negative_handling_for_weights == "clip_to_zero":
        df["actual_pos"] = df["actual"].clip(lower=0)
    else:
        df["actual_pos"] = df["actual"]

    month_totals = (
        df.groupby("month_start")["actual_pos"].sum().rename("month_pos_total")
    )
    df = df.merge(month_totals, on="month_start", how="left")
    df = df[df["month_pos_total"] > 0].copy()

    if df.empty:
        return pd.DataFrame(columns=["moy", "dom", "dow", "share"])

    df["share"] = df["actual_pos"] / df["month_pos_total"]
    shape = df.groupby(["moy", "dom", "dow"])["share"].mean().reset_index()
    return shape


def _lookup_weight(
    shape: pd.DataFrame,
    moy: int,
    dom: int,
    dow: int,
    cfg: DailyAllocationConfig,
) -> float:
    if shape.empty:
        return 1.0
    mask = (shape["moy"] == moy) & (shape["dom"] == dom) & (shape["dow"] == dow)
    vals = shape.loc[mask, "share"]
    if len(vals) > 0:
        return float(vals.mean())
    for fb in cfg.fallback:
        if fb == "dow":
            vals = shape.loc[shape["dow"] == dow, "share"]
            if len(vals) > 0:
                return float(vals.mean())
        elif fb == "dom":
            vals = shape.loc[shape["dom"] == dom, "share"]
            if len(vals) > 0:
                return float(vals.mean())
        elif fb == "uniform":
            return 1.0
    return 1.0


def _get_leakage_safe_history(
    daily_history: pd.DataFrame,
    target_month_start: pd.Timestamp,
    cfg: DailyAllocationConfig,
) -> pd.DataFrame:
    cutoff = target_month_start - pd.DateOffset(years=cfg.history_years)
    return daily_history[
        (daily_history["date"] >= cutoff)
        & (daily_history["date"] < target_month_start)
    ].copy()


def get_raw_scores_for_month(
    target_month_start: pd.Timestamp,
    daily_history: pd.DataFrame,
    cfg: DailyAllocationConfig,
) -> Dict[pd.Timestamp, float]:
    month_end = target_month_start + pd.offsets.MonthEnd(0)
    all_days = pd.date_range(target_month_start, month_end, freq="D")
    moy = target_month_start.month

    hist = _get_leakage_safe_history(daily_history, target_month_start, cfg)
    hist_months = hist["date"].dt.to_period("M").nunique() if len(hist) else 0
    shape = (
        _build_shape_table(hist, cfg)
        if hist_months >= cfg.min_history_months
        else pd.DataFrame()
    )

    scores: Dict[pd.Timestamp, float] = {}
    for d in all_days:
        d = pd.Timestamp(d)
        raw = _lookup_weight(shape, moy, d.day, d.dayofweek, cfg)
        raw = max(raw, 0.0) + cfg.smoothing_epsilon
        scores[d] = raw
    return scores


def normalize_scores(
    dates: Sequence[pd.Timestamp],
    raw_score_map: Dict[pd.Timestamp, float],
) -> np.ndarray:
    n = len(dates)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0])
    scores = np.array([raw_score_map.get(pd.Timestamp(d), 1.0) for d in dates])
    total = scores.sum()
    if total > 0:
        return scores / total
    return np.full(n, 1.0 / n)


def compute_daily_weights(
    target_month_start: pd.Timestamp,
    day_dates: Sequence[pd.Timestamp],
    daily_history: pd.DataFrame,
    cfg: DailyAllocationConfig,
) -> np.ndarray:
    if len(day_dates) == 0:
        return np.array([], dtype=float)
    raw_scores = get_raw_scores_for_month(target_month_start, daily_history, cfg)
    return normalize_scores(day_dates, raw_scores)


def compute_expected_mtd_share(
    raw_score_map: Dict[pd.Timestamp, float],
    observed_dates: Sequence[pd.Timestamp],
    min_expected_share: float = 0.05,
) -> float:
    all_total = sum(raw_score_map.values())
    if all_total <= 0:
        return min_expected_share
    obs_total = sum(raw_score_map.get(pd.Timestamp(d), 0.0) for d in observed_dates)
    share = obs_total / all_total
    return max(share, min_expected_share)


def compute_month_total_final(
    mtd_actual: float,
    month_total_model: float,
    expected_mtd_share: float,
    cfg: DailyAllocationConfig,
) -> float:
    strategy = cfg.current_month_strategy
    month_total_mtd = mtd_actual / expected_mtd_share
    if cfg.max_multiplier and month_total_model != 0:
        cap = abs(month_total_model) * cfg.max_multiplier
        month_total_mtd = max(min(month_total_mtd, cap), -cap)
    if strategy == "fixed":
        total = month_total_model
    elif strategy == "mtd_ratio":
        total = month_total_mtd
    elif strategy == "blend":
        alpha = cfg.blend_alpha
        total = alpha * month_total_model + (1 - alpha) * month_total_mtd
    else:
        total = month_total_model
    if not cfg.allow_negative_totals:
        total = max(total, 0.0)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# DailyAllocator
# ─────────────────────────────────────────────────────────────────────────────

class DailyAllocator:
    """Top-down daily forecast allocator with current-month nowcasting."""

    def __init__(self, config: ConfigManager):
        self.cfg = DailyAllocationConfig.from_config(config)
        self._daily_history: Optional[pd.DataFrame] = None
        self._monthly_forecasts: Optional[pd.DataFrame] = None
        self._month_total_final: Dict[pd.Timestamp, float] = {}

    def set_daily_history(self, daily_df: pd.DataFrame) -> None:
        required = {"date", "actual"}
        missing = required - set(daily_df.columns)
        if missing:
            raise ValueError(f"daily_df missing columns: {missing}")
        self._daily_history = daily_df.copy()
        self._daily_history["date"] = pd.to_datetime(self._daily_history["date"])

    def set_monthly_forecasts(self, monthly_forecast_df: pd.DataFrame) -> None:
        required = {"month_start", "forecast_total"}
        missing = required - set(monthly_forecast_df.columns)
        if missing:
            raise ValueError(f"monthly_forecast_df missing columns: {missing}")
        self._monthly_forecasts = monthly_forecast_df.copy()
        self._monthly_forecasts["month_start"] = pd.to_datetime(
            self._monthly_forecasts["month_start"]
        )

    def predict_daily(
        self,
        as_of_date: Optional[str] = None,
        horizon_months: int = 12,
    ) -> pd.DataFrame:
        if self._daily_history is None:
            raise RuntimeError("Call set_daily_history() before predict_daily()")
        if self._monthly_forecasts is None:
            raise RuntimeError("Call set_monthly_forecasts() before predict_daily()")

        if as_of_date is None or as_of_date == "auto":
            aod = pd.Timestamp.today().normalize()
        else:
            aod = pd.Timestamp(as_of_date)

        cutoff_date = aod - pd.Timedelta(days=self.cfg.posting_lag_days)
        current_month_start = aod.to_period("M").to_timestamp()
        target_months = pd.date_range(
            current_month_start, periods=1 + horizon_months, freq="MS"
        )

        self._month_total_final = {}
        rows: List[Dict] = []

        for month_start in target_months:
            month_end = month_start + pd.offsets.MonthEnd(0)
            is_current_month = month_start == current_month_start

            forecast_total = self._get_monthly_forecast(month_start)
            if forecast_total is None:
                continue

            all_days = pd.date_range(month_start, month_end, freq="D")

            if is_current_month:
                rows.extend(
                    self._allocate_current_month(
                        month_start, all_days, cutoff_date, forecast_total, aod
                    )
                )
            else:
                self._month_total_final[month_start] = forecast_total
                rows.extend(
                    self._allocate_future_month(month_start, all_days, forecast_total, aod)
                )

        result = pd.DataFrame(rows)
        if result.empty:
            return result

        result["horizon_day"] = (result["date"] - aod).dt.days
        result["horizon_month"] = (
            (result["month_start"].dt.to_period("M") - aod.to_period("M"))
            .apply(lambda x: x.n)
        )
        result["month_total_final"] = result["month_start"].map(self._month_total_final)
        self._check_coherence(result)
        return result

    def _get_monthly_forecast(self, month_start: pd.Timestamp) -> Optional[float]:
        mf = self._monthly_forecasts
        if mf is None:
            return None
        match = mf.loc[mf["month_start"] == month_start, "forecast_total"]
        if len(match) == 0:
            return None
        return float(match.iloc[0])

    def _get_daily_actual(self, d: pd.Timestamp) -> Optional[float]:
        hist = self._daily_history
        if hist is None:
            return None
        match = hist.loc[hist["date"] == d, "actual"]
        if len(match) == 0:
            return None
        return float(match.iloc[0])

    def _allocate_current_month(
        self,
        month_start: pd.Timestamp,
        all_days: pd.DatetimeIndex,
        cutoff_date: pd.Timestamp,
        forecast_total: float,
        aod: pd.Timestamp,
    ) -> List[Dict]:
        actual_days = all_days[all_days <= cutoff_date]
        remaining_days = all_days[all_days > cutoff_date]

        raw_score_map = get_raw_scores_for_month(month_start, self._daily_history, self.cfg)  # type: ignore[arg-type]
        full_weights = normalize_scores(list(all_days), raw_score_map)
        model_daily_alloc = forecast_total * full_weights
        _model_alloc_map = dict(zip(all_days, model_daily_alloc))

        mtd_actual = 0.0
        rows: List[Dict] = []
        for d in actual_days:
            val = self._get_daily_actual(d)
            actual_val = val if val is not None else 0.0
            mtd_actual += actual_val
            rows.append({
                "date": d,
                "month_start": month_start,
                "y_true": actual_val,
                "y_pred": actual_val,
                "forecast_amount": float(_model_alloc_map[d]),
                "is_actual": True,
                "as_of_date": aod,
                "month_total_model": forecast_total,
            })

        if len(remaining_days) == 0:
            self._month_total_final[month_start] = mtd_actual
            for r in rows:
                r["strategy"] = self.cfg.current_month_strategy
                r["effective_strategy"] = self.cfg.current_month_strategy
                r["expected_mtd_share"] = 1.0
                r["multiplier_applied"] = None
            return rows

        expected_share = compute_expected_mtd_share(
            raw_score_map, list(actual_days), self.cfg.min_expected_share
        )
        effective_strategy = self.cfg.resolve_strategy(aod.day, expected_share)

        multiplier_applied = None
        if effective_strategy in ("mtd_ratio", "blend") and expected_share > 0:
            mtd_ratio_est = mtd_actual / expected_share
            if self.cfg.max_multiplier and forecast_total != 0:
                cap = abs(forecast_total) * self.cfg.max_multiplier
                if abs(mtd_ratio_est) > cap:
                    multiplier_applied = self.cfg.max_multiplier

        orig_strategy = self.cfg.current_month_strategy
        self.cfg.current_month_strategy = effective_strategy
        month_total_final = compute_month_total_final(
            mtd_actual, forecast_total, expected_share, self.cfg
        )
        self.cfg.current_month_strategy = orig_strategy
        self._month_total_final[month_start] = month_total_final

        remaining_total = month_total_final - mtd_actual

        for r in rows:
            r["strategy"] = self.cfg.current_month_strategy
            r["effective_strategy"] = effective_strategy
            r["expected_mtd_share"] = expected_share
            r["multiplier_applied"] = multiplier_applied

        weights = normalize_scores(list(remaining_days), raw_score_map)
        daily_preds = remaining_total * weights
        for d, pred in zip(remaining_days, daily_preds):
            actual_val = self._get_daily_actual(d)
            rows.append({
                "date": d,
                "month_start": month_start,
                "y_true": actual_val,
                "y_pred": float(pred),
                "forecast_amount": float(_model_alloc_map[d]),
                "is_actual": False,
                "as_of_date": aod,
                "month_total_model": forecast_total,
                "strategy": self.cfg.current_month_strategy,
                "effective_strategy": effective_strategy,
                "expected_mtd_share": expected_share,
                "multiplier_applied": multiplier_applied,
            })
        return rows

    def _allocate_future_month(
        self,
        month_start: pd.Timestamp,
        all_days: pd.DatetimeIndex,
        forecast_total: float,
        aod: pd.Timestamp,
    ) -> List[Dict]:
        raw_scores = get_raw_scores_for_month(month_start, self._daily_history, self.cfg)  # type: ignore[arg-type]
        weights = normalize_scores(list(all_days), raw_scores)
        daily_preds = forecast_total * weights
        rows: List[Dict] = []
        for d, pred in zip(all_days, daily_preds):
            pred_val = float(pred)
            rows.append({
                "date": d,
                "month_start": month_start,
                "y_true": None,
                "y_pred": pred_val,
                "forecast_amount": pred_val,
                "is_actual": False,
                "as_of_date": aod,
                "month_total_model": forecast_total,
                "strategy": None,
                "effective_strategy": None,
                "expected_mtd_share": None,
                "multiplier_applied": None,
            })
        return rows

    def _check_coherence(self, result: pd.DataFrame, tol: float = 0.01) -> None:
        for month_start, grp in result.groupby("month_start"):
            ts_month = pd.Timestamp(month_start)  # type: ignore[arg-type]
            has_forecasted = (grp["is_actual"] == False).any()
            daily_sum = grp["y_pred"].sum()
            if has_forecasted:
                expected = self._month_total_final.get(ts_month)
                if expected is None:
                    expected = self._get_monthly_forecast(ts_month)
                if expected is None:
                    continue
                diff = abs(daily_sum - expected)
                if diff > tol:
                    print(
                        f"Warning: coherence violation {ts_month.strftime('%Y-%m')}: "
                        f"sum(y_pred)={daily_sum:.4f}  expected={expected:.4f}  diff={diff:.4f}"
                    )

    def backtest_intramonth_daily(
        self,
        as_of_day_offsets: Sequence[int] = (5, 10, 15, 20),
        strategies: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if self._daily_history is None:
            raise RuntimeError("Call set_daily_history() first")
        if self._monthly_forecasts is None:
            raise RuntimeError("Call set_monthly_forecasts() first")

        if strategies is None:
            strategies = [self.cfg.current_month_strategy]

        hist = self._daily_history
        monthly = daily_to_monthly(hist)
        records: List[Dict] = []

        for _, row in monthly.iterrows():
            ms = row["month_start"]
            month_actual_total = row["actual_month_total"]
            forecast_total = self._get_monthly_forecast(ms)
            if forecast_total is None:
                continue
            month_end = ms + pd.offsets.MonthEnd(0)
            all_days = pd.date_range(ms, month_end, freq="D")
            raw_score_map = get_raw_scores_for_month(ms, self._daily_history, self.cfg)

            for k in as_of_day_offsets:
                aod_nominal = ms + pd.Timedelta(days=k - 1)
                if aod_nominal > month_end:
                    continue
                cutoff = aod_nominal - pd.Timedelta(days=self.cfg.posting_lag_days)
                actual_days = all_days[all_days <= cutoff]
                remaining_days = all_days[all_days > cutoff]
                if len(remaining_days) == 0:
                    continue
                mtd_vals = hist[hist["date"].isin(actual_days)]
                mtd_actual = mtd_vals["actual"].sum() if len(mtd_vals) > 0 else 0.0
                rem_vals = hist[hist["date"].isin(remaining_days)]
                remaining_actual = rem_vals["actual"].sum() if len(rem_vals) > 0 else 0.0
                expected_share = compute_expected_mtd_share(
                    raw_score_map, actual_days, self.cfg.min_expected_share
                )

                for strat in strategies:
                    if strat == "scheduled":
                        effective_strat = self.cfg.resolve_strategy(k, expected_share)
                    else:
                        effective_strat = strat

                    orig_strat = self.cfg.current_month_strategy
                    self.cfg.current_month_strategy = effective_strat
                    month_total_final = compute_month_total_final(
                        mtd_actual, forecast_total, expected_share, self.cfg
                    )
                    remaining_total_fc = month_total_final - mtd_actual
                    self.cfg.current_month_strategy = orig_strat

                    weights = normalize_scores(list(remaining_days), raw_score_map)
                    daily_preds = remaining_total_fc * weights

                    rem_actual_series = []
                    for d in remaining_days:
                        val = self._get_daily_actual(d)
                        rem_actual_series.append(val if val is not None else 0.0)
                    rem_actual_arr = np.array(rem_actual_series)

                    abs_err = np.abs(daily_preds - rem_actual_arr)
                    remainder_mae = float(np.mean(abs_err))
                    sum_abs_actual = float(np.sum(np.abs(rem_actual_arr)))
                    remainder_wape = (
                        float(np.sum(abs_err) / sum_abs_actual * 100)
                        if sum_abs_actual > 0
                        else float(np.sum(abs_err))
                    )
                    full_month_error = month_total_final - month_actual_total
                    full_month_error_pct = (
                        (full_month_error / abs(month_actual_total) * 100)
                        if abs(month_actual_total) > 0
                        else 0.0
                    )
                    records.append({
                        "month_start": ms,
                        "as_of_day": k,
                        "strategy": strat,
                        "effective_strategy": effective_strat,
                        "posting_lag_days": self.cfg.posting_lag_days,
                        "mtd_actual": mtd_actual,
                        "remaining_actual": remaining_actual,
                        "month_actual_total": month_actual_total,
                        "month_total_model": forecast_total,
                        "month_total_final": month_total_final,
                        "remaining_forecast": remaining_total_fc,
                        "remainder_mae": remainder_mae,
                        "remainder_wape": remainder_wape,
                        "full_month_error": full_month_error,
                        "full_month_error_pct": full_month_error_pct,
                    })

        return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RCT forecasting pipeline on an engineered data CSV."
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yml",
        help="Path to YAML config file (default: config/default.yml)",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input CSV (feature-engineered data)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for forecast CSVs (default: results/)",
    )
    parser.add_argument(
        "--model", type=str,
        choices=["xgboost", "lightgbm", "random_forest", "linear", "prophet"],
        default=None,
        help="Override model type from config",
    )
    parser.add_argument(
        "--periods", type=int, default=None,
        help="Number of months to forecast (overrides config)",
    )
    parser.add_argument(
        "--group-by", type=str, default=None, dest="group_by",
        help="Segment column for per-segment forecasting (e.g. 'Profit Center')",
    )
    parser.add_argument(
        "--target-column", type=str, default=None, dest="target_column",
        help="Target revenue column name (default: from config)",
    )
    parser.add_argument(
        "--date-column", type=str, default=None, dest="date_column",
        help="Date column name (default: from config)",
    )
    parser.add_argument(
        "--daily", action="store_true",
        help="Run daily disaggregation after monthly forecast",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Load data and validate but skip model training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = ConfigManager(config_path=args.config)

    if args.model:
        config.update({"forecasting": {"model_type": args.model}})
    if args.dry_run:
        config.update({"forecasting": {"dry_run": True}})

    target_column = args.target_column or config.get("data_source.amount_column", "Actual")
    date_column = args.date_column or config.get("data_source.date_column", "Year Period")
    periods = args.periods or config.get("forecasting.forecast_horizon", 12)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns from {input_path}")

    group_by = args.group_by
    if group_by is None:
        key_cols = config.get("data_source.key_columns", [])
        segment_level = str(config.get("forecasting.segment_level", "company")).lower()
        if segment_level in {"rollup_shop", "pc", "profit_center"}:
            for candidate in ["Roll Up Shop", "roll_up_shop", "Profit Center", "profit_center"]:
                if candidate in df.columns:
                    group_by = candidate
                    break
        elif key_cols:
            group_by = key_cols[0] if key_cols[0] in df.columns else None

    if group_by:
        print(f"Segment column: {group_by}  ({df[group_by].nunique()} segments)")
    else:
        print("Segment column: None (single-series mode)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.dry_run:
        print("DRY RUN: skipping model training")
        print(f"  Columns available : {list(df.columns)}")
        print(f"  Target column     : {target_column}  (present={target_column in df.columns})")
        print(f"  Date column       : {date_column}  (present={date_column in df.columns})")
        return 0

    forecaster = TimeSeriesForecaster(config)
    forecaster.fit(
        df, target_column, date_column,
        group_by=group_by,
        already_prepared=True,
    )
    print(f"Model training complete  (type={config.get('forecasting.model_type')})")

    predictions = forecaster.predict(periods=periods)
    print(f"Generated {len(predictions)} prediction rows")

    if forecaster.test_data is not None and len(forecaster.test_data) > 0:
        metrics = forecaster.evaluate(
            forecaster.test_data, target_column, date_column
        )
        print(f"Test metrics: {metrics}")

    pred_file = output_dir / f"predictions_{ts}.csv"
    predictions.to_csv(pred_file, index=False)
    print(f"Predictions saved: {pred_file}")

    if args.daily:
        print("Running daily disaggregation")
        try:
            daily_cfg = DailyAllocationConfig.from_config(config)
            date_col_src = config.get("data_source.date_column", "Year Period")
            daily_df = build_daily_totals(
                df, date_column=date_col_src, amount_column=target_column
            )

            pred_date_col = next(
                (c for c in ["date", date_column, "ds"] if c in predictions.columns),
                None,
            )
            pred_val_col = next(
                (c for c in ["y_pred", "prediction", "yhat"] if c in predictions.columns),
                None,
            )
            if pred_date_col is None or pred_val_col is None:
                print(
                    f"Error: cannot find date/value columns in predictions: "
                    f"{list(predictions.columns)}",
                    file=sys.stderr,
                )
                return 1

            month_fc = predictions[[pred_date_col, pred_val_col]].copy()
            month_fc.columns = ["month_start", "forecast_total"]
            month_fc["month_start"] = pd.to_datetime(month_fc["month_start"])
            month_fc = month_fc.groupby("month_start")["forecast_total"].sum().reset_index()

            allocator = DailyAllocator(config)
            allocator.set_daily_history(daily_df)
            allocator.set_monthly_forecasts(month_fc)

            daily_preds = allocator.predict_daily(
                as_of_date=daily_cfg.as_of_date,
                horizon_months=periods,
            )

            daily_file = output_dir / f"daily_predictions_{ts}.csv"
            daily_preds.to_csv(daily_file, index=False)
            print(f"Daily predictions saved: {daily_file}  ({len(daily_preds)} rows)")

            weekly_preds = daily_to_weekly(
                daily_preds, date_column="date", value_column="y_pred"
            )
            weekly_file = output_dir / f"weekly_predictions_{ts}.csv"
            weekly_preds.to_csv(weekly_file, index=False)
            print(f"Weekly predictions saved: {weekly_file}  ({len(weekly_preds)} rows)")

        except Exception as exc:
            print(f"Error: daily disaggregation failed: {exc}", file=sys.stderr)
            return 1

    print("Forecasting complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
