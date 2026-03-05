"""
Time Series Forecasting Module

Extensible forecasting system supporting multiple models and approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import pickle
import joblib
from pathlib import Path
import re
from ..config.config_manager import ConfigManager
from ..utils.date_utils import convert_year_period_to_date
from ..utils.data_utils import handle_outliers

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Install with: pip install prophet")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model Registry — maps config string -> (constructor, availability_flag)
# New models: add one entry here + optional import above.  That's it.
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, Any] = {}  # populated after class defs, see below


def register_model(name: str, factory, *, available: bool = True) -> None:
    """Register a model constructor so it can be referenced by name in config.

    Args:
        name: Key used in ``forecasting.model_type`` YAML field.
        factory: Callable ``(**params) -> sklearn-compatible estimator``.
        available: Set False when the underlying library is not installed.
    """
    _MODEL_REGISTRY[name] = {"factory": factory, "available": available}


def available_models() -> List[str]:
    """Return the list of model names whose libraries are installed."""
    return [k for k, v in _MODEL_REGISTRY.items() if v["available"]]


class ForecastModel(ABC):
    """Abstract base class for forecast models"""
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize forecast model
        
        Args:
            name: Name of the forecast model
            **kwargs: Model-specific parameters
        """
        self.name = name
        self.params = kwargs
        self.model = None
        self.is_fitted = False
        self.feature_columns = None
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, target_column: str, 
            date_column: str, **kwargs) -> 'ForecastModel':
        """
        Fit the model to the data
        
        Args:
            df: Training DataFrame
            target_column: Name of target variable
            date_column: Name of date column
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, df: Optional[pd.DataFrame] = None, 
                periods: int = 12) -> pd.DataFrame:
        """
        Generate predictions
        
        Args:
            df: Optional DataFrame with future dates/features
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with predictions
        """
        pass
    
    def save_model(self, path: str):
        """Save the fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'name': self.name,
            'params': self.params,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str):
        """Load a fitted model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.name = model_data['name']
        self.params = model_data['params']
        self.feature_columns = model_data['feature_columns']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from: {path}")


class ProphetModel(ForecastModel):
    """Prophet-based forecasting model"""
    
    def __init__(self, name: str = "prophet", **kwargs):
        super().__init__(name, **kwargs)
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
    
    def fit(self, df: pd.DataFrame, target_column: str, 
            date_column: str, **kwargs) -> 'ProphetModel':
        """Fit Prophet model"""
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = df[[date_column, target_column]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Remove any rows with missing values
        prophet_df = prophet_df.dropna()
        
        # Initialize Prophet with parameters
        prophet_params = {
            'yearly_seasonality': self.params.get('yearly_seasonality', True),
            'weekly_seasonality': self.params.get('weekly_seasonality', False),
            'daily_seasonality': self.params.get('daily_seasonality', False),
            'growth': self.params.get('growth', 'linear')
        }
        
        self.model = Prophet(**prophet_params)
        
        # Add additional regressors if specified
        additional_regressors = self.params.get('additional_regressors', [])
        for regressor in additional_regressors:
            if regressor in df.columns:
                self.model.add_regressor(regressor)
                prophet_df[regressor] = df[regressor]
        
        # Fit the model
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        logger.info(f"Prophet model fitted with {len(prophet_df)} data points")
        return self
    
    def predict(self, df: Optional[pd.DataFrame] = None, 
                periods: int = 12) -> pd.DataFrame:
        """Generate Prophet predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if df is not None:
            # Use provided DataFrame
            future_df = df.copy()
            if 'ds' not in future_df.columns:
                # Assume first column is date
                future_df.rename(columns={future_df.columns[0]: 'ds'}, inplace=True)
        else:
            # Create future dataframe
            future_df = self.model.make_future_dataframe(periods=periods, freq='MS')
        
        # Generate forecast
        forecast = self.model.predict(future_df)
        
        # Return relevant columns
        result_columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
        return forecast[result_columns].rename(columns={
            'ds': 'date',
            'yhat': 'prediction',
            'yhat_lower': 'prediction_lower',
            'yhat_upper': 'prediction_upper'
        })


class SklearnModel(ForecastModel):
    """Scikit-learn based forecasting model"""
    
    def __init__(self, name: str = "sklearn", model_type: str = "linear", **kwargs):
        super().__init__(name, **kwargs)
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is not available")
        
        self.model_type = model_type
        self.date_column = None
        self._init_model()
    
    def _init_model(self):
        """Initialize the sklearn model via the model registry."""
        entry = _MODEL_REGISTRY.get(self.model_type)
        if entry is None:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. "
                f"Available: {list(_MODEL_REGISTRY.keys())}"
            )
        if not entry["available"]:
            raise ImportError(
                f"Library for model '{self.model_type}' is not installed."
            )
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
        """Prepare features for sklearn model"""

        self.date_column = date_column
        df = self._ensure_date_features(df, date_column)

        # Select numeric columns (excluding target and date)
        feature_columns = [col for col in df.columns 
                         if col not in [target_column, date_column] 
                         and df[col].dtype in ['int64', 'float64']]
        
        self.feature_columns = feature_columns
        
        X = df[feature_columns].fillna(0).values
        y = df[target_column].fillna(0).values
        
        return X, y
    
    def fit(self, df: pd.DataFrame, target_column: str, 
            date_column: str, **kwargs) -> 'SklearnModel':
        """Fit sklearn model"""
        
        X, y = self._prepare_features(df, target_column, date_column)
        
        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"Sklearn {self.model_type} model fitted with {len(X)} samples and {X.shape[1]} features")
        return self
    
    def predict(self, df: Optional[pd.DataFrame] = None, 
                periods: int = 12) -> pd.DataFrame:
        """Generate sklearn predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if df is None:
            raise ValueError("DataFrame with features is required for sklearn prediction")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not available")
        
        if self.date_column:
            df = self._ensure_date_features(df, self.date_column)

        # Prepare features
        X = df[self.feature_columns].fillna(0).values
        
        # Generate predictions
        predictions = self.model.predict(X)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'date': df.iloc[:, 0],  # Assume first column is date
            'prediction': predictions
        })
        
        return result_df


# ---------------------------------------------------------------------------
# Holt-Winters (Exponential Smoothing) — pure time-series model
# ---------------------------------------------------------------------------
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as _HWModel
    HOLTWINTERS_AVAILABLE = True
except ImportError:
    HOLTWINTERS_AVAILABLE = False


class HoltWintersModel(ForecastModel):
    """Holt-Winters / Exponential Smoothing forecasting model.

    Unlike sklearn models this operates on the target series directly
    (no feature columns). It handles trend + seasonality natively.
    """

    def __init__(self, name: str = "holt_winters", **kwargs):
        super().__init__(name, **kwargs)
        if not HOLTWINTERS_AVAILABLE:
            raise ImportError(
                "statsmodels is not available. Install with: pip install statsmodels"
            )
        self.date_column: Optional[str] = None
        self._freq: str = "MS"  # monthly start

    def fit(self, df: pd.DataFrame, target_column: str,
            date_column: str, **kwargs) -> "HoltWintersModel":
        self.date_column = date_column
        ts = df.set_index(date_column)[target_column].sort_index().astype(float)

        # Ensure a proper DatetimeIndex; infer freq if possible
        if not isinstance(ts.index, pd.DatetimeIndex):
            ts.index = pd.DatetimeIndex(ts.index)
        inferred = pd.infer_freq(ts.index)
        if inferred is not None:
            ts.index.freq = inferred
        else:
            ts.index = pd.DatetimeIndex(ts.index, freq=self._freq)

        # Need a minimum of 4 data points for any trend estimation
        if len(ts) < 4:
            raise ValueError(
                f"HoltWinters requires at least 4 data points, got {len(ts)}"
            )

        seasonal_periods = self.params.get("seasonal_periods", 12)
        trend = self.params.get("trend", "add")
        seasonal = self.params.get("seasonal", "add")
        damped_trend = self.params.get("damped_trend", False)

        # Need at least 2 full seasonal cycles for seasonal decomposition
        if len(ts) < 2 * seasonal_periods:
            seasonal = None
            logger.warning(
                "HoltWinters: not enough data for seasonal (%d < %d); "
                "falling back to trend-only",
                len(ts), 2 * seasonal_periods,
            )

        hw = _HWModel(
            ts,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
            damped_trend=damped_trend,
        )
        self.model = hw.fit(optimized=True)
        self.is_fitted = True
        logger.info(
            "HoltWinters model fitted with %d data points "
            "(trend=%s, seasonal=%s, damped=%s)",
            len(ts), trend, seasonal, damped_trend,
        )
        return self

    def predict(self, df: Optional[pd.DataFrame] = None,
                periods: int = 12) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        forecast = self.model.forecast(periods)
        result_df = pd.DataFrame({
            "date": forecast.index,
            "prediction": forecast.values,
        })
        return result_df


# ---------------------------------------------------------------------------
# Ensemble model — weighted average of two registry models
# ---------------------------------------------------------------------------
class EnsembleModel(ForecastModel):
    """Simple weighted-average ensemble of two sklearn-compatible models.

    Config example (in YAML)::

        forecasting:
          model_type: ensemble
          ensemble:
            model_a: ridge
            model_b: xgboost
            weight_a: 0.5       # weight_b = 1 - weight_a
    """

    def __init__(self, name: str = "ensemble", **kwargs):
        super().__init__(name, **kwargs)
        self.model_a_obj: Optional[SklearnModel] = None
        self.model_b_obj: Optional[SklearnModel] = None
        self.weight_a: float = kwargs.get("weight_a", 0.5)
        model_a_name = kwargs.get("model_a", "ridge")
        model_b_name = kwargs.get("model_b", "xgboost")
        self.model_a_obj = SklearnModel(model_type=model_a_name)
        self.model_b_obj = SklearnModel(model_type=model_b_name)
        self.date_column: Optional[str] = None

    def fit(self, df: pd.DataFrame, target_column: str,
            date_column: str, **kwargs) -> "EnsembleModel":
        self.date_column = date_column
        self.model_a_obj.fit(df, target_column, date_column, **kwargs)
        self.model_b_obj.fit(df, target_column, date_column, **kwargs)
        self.feature_columns = self.model_a_obj.feature_columns
        self.is_fitted = True
        logger.info(
            "Ensemble model fitted (model_a=%s w=%.2f, model_b=%s w=%.2f)",
            self.model_a_obj.model_type, self.weight_a,
            self.model_b_obj.model_type, 1 - self.weight_a,
        )
        return self

    def predict(self, df: Optional[pd.DataFrame] = None,
                periods: int = 12) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        pred_a = self.model_a_obj.predict(df, periods)
        pred_b = self.model_b_obj.predict(df, periods)
        result = pred_a.copy()
        result["prediction"] = (
            self.weight_a * pred_a["prediction"].values
            + (1 - self.weight_a) * pred_b["prediction"].values
        )
        return result


# ---------------------------------------------------------------------------
# Register built-in models.  To add a new model:
#   1. pip install <library>
#   2. Add an import + availability flag at the top of this file
#   3. Call register_model() here
#   4. Set  forecasting.model_type: <name>  in your YAML config
# ---------------------------------------------------------------------------
register_model("linear", LinearRegression, available=SKLEARN_AVAILABLE)
register_model("ridge", lambda **kw: __import__(
    'sklearn.linear_model', fromlist=['Ridge']).Ridge(**kw),
    available=SKLEARN_AVAILABLE)
register_model("random_forest", lambda **kw: RandomForestRegressor(**kw),
               available=SKLEARN_AVAILABLE)
register_model("xgboost", lambda **kw: XGBRegressor(**kw),
               available=XGBOOST_AVAILABLE)
register_model("lightgbm", lambda **kw: LGBMRegressor(**kw),
               available=LIGHTGBM_AVAILABLE)
register_model("elastic_net", lambda **kw: __import__(
    'sklearn.linear_model', fromlist=['ElasticNet']).ElasticNet(**kw),
    available=SKLEARN_AVAILABLE)
register_model("lasso", lambda **kw: __import__(
    'sklearn.linear_model', fromlist=['Lasso']).Lasso(**kw),
    available=SKLEARN_AVAILABLE)
register_model("bayesian_ridge", lambda **kw: __import__(
    'sklearn.linear_model', fromlist=['BayesianRidge']).BayesianRidge(**kw),
    available=SKLEARN_AVAILABLE)
register_model("catboost", lambda **kw: CatBoostRegressor(
    verbose=0, **kw), available=CATBOOST_AVAILABLE)


class TimeSeriesForecaster:
    """Main forecasting orchestrator"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize time series forecaster
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.model = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.metrics = {}
        self.group_by = None
        self.use_log_transform = config_manager.get('forecasting.use_log_transform', False)
        self.target_transform_offset = config_manager.get('forecasting.log_transform_offset', 0)
        self._agg_train_last = None
        self._agg_history = None
        self._max_target_log = None
        self.split_meta: Optional[Dict[str, Any]] = None  # set by _split_data
        self._feature_engineer = None  # set by fit() when caller passes feature_engineer

    def _get_adaptive_xgb_params(self, n_train_rows: int) -> Dict[str, Any]:
        """Return XGBoost hyper-parameters adapted to the training-set size.

        Small segments (few rows) are prone to overfitting, so we increase
        regularisation and reduce capacity.  The thresholds and per-tier
        overrides are configurable via ``forecasting.segment_tuning``.

        Tier lookup (first match wins):
          - ``small``  if n_train_rows < small_threshold  (default 30)
          - ``medium`` if n_train_rows < medium_threshold (default 48)
          - ``large``  otherwise

        Returns the base ``forecasting.xgboost`` dict merged with any
        tier-specific overrides.
        """
        base_params = dict(self.config.get('forecasting.xgboost', {}))
        tuning = self.config.get('forecasting.segment_tuning', {})
        if not tuning or not tuning.get('enabled', False):
            return base_params

        small_thresh = int(tuning.get('small_threshold', 30))
        medium_thresh = int(tuning.get('medium_threshold', 48))

        if n_train_rows < small_thresh:
            tier = 'small'
        elif n_train_rows < medium_thresh:
            tier = 'medium'
        else:
            tier = 'large'

        overrides = tuning.get(tier, {})
        if overrides:
            merged = {**base_params, **overrides}
            logger.debug(
                "Adaptive XGB params (tier=%s, n=%d): %s",
                tier, n_train_rows, overrides,
            )
            return merged
        return base_params

    def _populate_future_features(self, future_df: pd.DataFrame, target_column: str,
                                  date_column: str) -> pd.DataFrame:
        if self._agg_history is None or self._agg_history.empty:
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

    def _recursive_forecast(self, start_date: pd.Timestamp, periods: int,
                            target_column: str, date_column: str) -> pd.DataFrame:
        """Generate a multi-step forecast one period at a time.

        When a ``FeatureEngineer`` was supplied at ``fit`` time the method uses
        ``FeatureEngineer.transform_future`` to recompute lag / rolling / growth
        features from predicted values (Task D — recursive feature parity).
        Otherwise it falls back to the regex-based ``_populate_future_features``
        for backward compatibility.
        """
        if self._agg_history is None or self._agg_history.empty:
            raise ValueError("Aggregated history is required for recursive forecasting")

        history = self._agg_history.sort_values(date_column).reset_index(drop=True)
        predictions: List[Dict] = []
        date_range = pd.date_range(start=start_date, periods=periods, freq='MS')

        for forecast_date in date_range:
            row = {date_column: forecast_date}
            future_row = pd.DataFrame([row])

            if self._feature_engineer is not None and self._feature_engineer.fitted_transformers:
                # FE-based path: recompute all features from actual + predicted history
                future_features = self._feature_engineer.transform_future(
                    history_df=history,
                    future_df=future_row,
                    target_column=target_column,
                    date_column=date_column,
                )
                future_row = future_features
            else:
                # Legacy regex path
                future_row = self._populate_future_features(future_row, target_column, date_column)

            # Fill any remaining feature columns with last-known value or zero
            for feature in self.model.feature_columns or []:
                if feature not in future_row.columns:
                    future_row[feature] = (
                        self._agg_train_last.get(feature, 0.0) if self._agg_train_last else 0.0
                    )

            pred_value = float(self.model.predict(future_row)["prediction"].iloc[0])
            predictions.append({"date": forecast_date, "prediction": pred_value})

            # Append predicted value to working history for use in the next step
            new_row = pd.DataFrame(
                {date_column: [forecast_date], target_column: [pred_value]}
            )
            history = pd.concat([history, new_row], ignore_index=True)
            self._agg_history = history  # keep in sync

        return pd.DataFrame(predictions)

    def _prepare_eval_df(self, df: pd.DataFrame, target_column: str,
                         date_column: str) -> pd.DataFrame:
        eval_df = df
        if self.group_by:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            eval_df = df.groupby(date_column, as_index=False)[numeric_cols].sum()
        return eval_df
        
    def _split_data(self, df: pd.DataFrame, date_column: str,
                   target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.

        Backtest semantics — **Fixed-Origin Multi-Step** (implemented here)
        -------------------------------------------------------------------
        A **single** cutoff date is used.  The model is fit exactly once on all
        data before ``test_start`` and then evaluated on the full test window
        (``backtest_months`` consecutive months) in one shot.  No re-fitting
        occurs at each evaluation month.

        This is sometimes called *out-of-time evaluation with a fixed origin*.
        Its properties:

        * Origin is fixed at ``test_start``: forecast horizons h=1, 2, …, H
          all come from the **same** fitted model.
        * Captures how far ahead a model forecasts well from a single vantage
          point — appropriate for annual planning cycles.

        **Not** implemented here (but noted for reference):

        * *Rolling-origin single-step* (re-fit or pseudo-update each month).
          More data-hungry; useful for evaluating updating strategies.

        Configuration
        -------------
        ``forecasting.backtest_cutoff_month`` (YYYY-MM-01)
            Anchor date.  All months *before* ``test_start`` are train/val.
            Defaults to ``max(date) + 1 month`` when ``null``.
        ``forecasting.backtest_months``
            Length of the test window (H).  Default 12.
        ``forecasting.validation_months``
            Optional hold-out between train and test.  Default 0.

        Resulting split (``backtest_months=12, validation_months=3``)::

            [-------- train ---------][-- val --][------ test (h=1..12) -----]
                                      ^           ^
                                validation_start  test_start = cutoff - 12 months
        """

        # Sort by date
        df_sorted = df.sort_values(date_column).reset_index(drop=True)

        backtest_months = int(self.config.get('forecasting.backtest_months', 0) or 0)
        validation_months = int(self.config.get('forecasting.validation_months', 0) or 0)

        if backtest_months > 0:
            # --- Deterministic cutoff ---
            # Priority: explicit config > data-driven default (max date + 1 month)
            cutoff_cfg = self.config.get('forecasting.backtest_cutoff_month', None)
            if cutoff_cfg:
                cutoff = pd.Timestamp(cutoff_cfg).to_period("M").to_timestamp()
                logger.info("Using configured backtest_cutoff_month: %s", cutoff.date())
            else:
                max_date = pd.Timestamp(df_sorted[date_column].max())
                cutoff = (max_date + pd.DateOffset(months=1)).to_period("M").to_timestamp()
                logger.info(
                    "No backtest_cutoff_month configured; defaulting to max(date)+1month: %s",
                    cutoff.date(),
                )

            test_start = (cutoff - pd.DateOffset(months=backtest_months)).to_period("M").to_timestamp()
            test_end = cutoff

            if validation_months > 0:
                validation_start = (test_start - pd.DateOffset(months=validation_months)).to_period("M").to_timestamp()
            else:
                validation_start = test_start

            train_data = df_sorted[df_sorted[date_column] < validation_start]
            validation_data = df_sorted[(df_sorted[date_column] >= validation_start) & (df_sorted[date_column] < test_start)]
            test_data = df_sorted[(df_sorted[date_column] >= test_start) & (df_sorted[date_column] < test_end)]

            if len(train_data) == 0 or len(test_data) == 0:
                logger.warning("Backtest split resulted in empty train/test. Falling back to ratio split.")
            else:
                self.split_meta = {
                    'cutoff': str(cutoff.date()),
                    'validation_start': str(validation_start.date()),
                    'test_start': str(test_start.date()),
                    'test_end': str(test_end.date()),
                    'train_rows': len(train_data),
                    'validation_rows': len(validation_data),
                    'test_rows': len(test_data),
                }
                logger.info(
                    "Backtest split — cutoff=%s  validation_start=%s  test_start=%s  test_end=%s  |  Train=%d  Val=%d  Test=%d",
                    cutoff.date(), validation_start.date(), test_start.date(), test_end.date(),
                    len(train_data), len(validation_data), len(test_data),
                )
                return train_data, validation_data, test_data

        test_size = self.config.get('forecasting.test_size', 0.2)
        validation_size = self.config.get('forecasting.validation_split', 0.1)

        n_total = len(df_sorted)
        n_test = int(n_total * test_size)
        n_validation = int(n_total * validation_size)
        n_train = n_total - n_test - n_validation

        train_data = df_sorted.iloc[:n_train]
        validation_data = df_sorted.iloc[n_train:n_train + n_validation]
        test_data = df_sorted.iloc[n_train + n_validation:]

        logger.info(f"Data split - Train: {len(train_data)}, "
                   f"Validation: {len(validation_data)}, Test: {len(test_data)}")

        return train_data, validation_data, test_data
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           label: str = "") -> Dict[str, float]:
        """Calculate forecasting metrics with diagnostic logging"""
        
        metrics = {}

        # Coerce to numeric arrays (handles Decimal and object dtypes)
        y_true = pd.to_numeric(np.asarray(y_true), errors='coerce').astype(float)
        y_pred = pd.to_numeric(np.asarray(y_pred), errors='coerce').astype(float)
        
        # Remove any NaN / Inf values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No valid predictions for metric calculation")
            return metrics

        # --- Diagnostic logging (pass/fail scale check) ---
        abs_err = np.abs(y_true_clean - y_pred_clean)
        prefix = f"[{label}] " if label else ""
        logger.info(
            "%sEval arrays (n=%d): y_true  min=%.2f  mean=%.2f  max=%.2f",
            prefix, len(y_true_clean),
            float(np.min(y_true_clean)), float(np.mean(y_true_clean)), float(np.max(y_true_clean))
        )
        logger.info(
            "%sEval arrays (n=%d): y_pred  min=%.2f  mean=%.2f  max=%.2f",
            prefix, len(y_pred_clean),
            float(np.min(y_pred_clean)), float(np.mean(y_pred_clean)), float(np.max(y_pred_clean))
        )
        logger.info(
            "%sEval arrays (n=%d): abs_err min=%.2f  mean=%.2f  max=%.2f",
            prefix, len(abs_err),
            float(np.min(abs_err)), float(np.mean(abs_err)), float(np.max(abs_err))
        )

        # ---- Standard metrics ----
        metrics['mae'] = float(mean_absolute_error(y_true_clean, y_pred_clean))
        metrics['mse'] = float(mean_squared_error(y_true_clean, y_pred_clean))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        
        # MAPE (Mean Absolute Percentage Error) – mask zeros
        mape_mask = y_true_clean != 0
        if np.any(mape_mask):
            mape = float(np.mean(np.abs(
                (y_true_clean[mape_mask] - y_pred_clean[mape_mask]) /
                y_true_clean[mape_mask]
            )) * 100)
            metrics['mape'] = mape

        # WAPE / wMAPE  (volume-weighted, stable with near-zero months)
        sum_abs_actual = float(np.sum(np.abs(y_true_clean)))
        if sum_abs_actual > 0:
            metrics['wape'] = float(np.sum(abs_err) / sum_abs_actual * 100)

        # MASE  (Mean Absolute Scaled Error vs seasonal-naïve h=12, fallback h=1)
        try:
            seasonal_period = 12
            if len(y_true_clean) > seasonal_period:
                naive_errors = np.abs(y_true_clean[seasonal_period:] - y_true_clean[:-seasonal_period])
            else:
                naive_errors = np.abs(np.diff(y_true_clean))
            mean_naive = float(np.mean(naive_errors)) if len(naive_errors) > 0 else 0.0
            if mean_naive > 0:
                metrics['mase'] = float(np.mean(abs_err) / mean_naive)
        except Exception:
            pass

        # Bias  (mean signed error: positive = over-forecast)
        signed_err = y_pred_clean - y_true_clean
        metrics['bias'] = float(np.mean(signed_err))
        if sum_abs_actual > 0:
            metrics['bias_pct'] = float(np.sum(signed_err) / sum_abs_actual * 100)

        # Tracking signal  (cumulative forecast error / MAD)
        # A standard operational alert metric: |TS| > 4–6 suggests systematic bias
        cum_err = float(np.sum(signed_err))
        mad = metrics['mae']
        if mad > 0:
            metrics['tracking_signal'] = float(cum_err / mad)

        # R-squared
        try:
            metrics['r2'] = float(r2_score(y_true_clean, y_pred_clean))
        except Exception:
            metrics['r2'] = np.nan
        
        return metrics
    
    def _transform_target(self, y: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """
        Apply log1p transform to target variable if enabled
        
        Args:
            y: Target variable values
            
        Returns:
            Transformed target values
        """
        if not self.use_log_transform:
            return y
        
        # Apply log1p (handles zeros gracefully)
        if isinstance(y, pd.Series):
            return np.log1p(y + self.target_transform_offset)
        else:
            return np.log1p(y + self.target_transform_offset)
    
    def _inverse_transform_target(self, y: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
        """
        Apply expm1 to reverse log1p transform if enabled
        
        Args:
            y: Transformed target values
            
        Returns:
            Original scale target values
        """
        if not self.use_log_transform:
            return y
        
        # Apply expm1 (inverse of log1p)
        if self._max_target_log is not None:
            y = np.clip(y, a_min=None, a_max=self._max_target_log + 2.0)
        if isinstance(y, pd.Series):
            return np.expm1(y) - self.target_transform_offset
        else:
            return np.expm1(y) - self.target_transform_offset
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    date_column: str, group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare data for forecasting
        
        Args:
            df: Input DataFrame
            target_column: Name of target variable
            date_column: Name of date column
            group_by: Optional grouping column (e.g., customer_group)
            
        Returns:
            Prepared DataFrame
        """
        
        # Ensure date column is datetime
        df = df.copy()
        
        # Handle Year Period format if configured
        date_format = self.config.get('data_source.date_format', 'default')
        if date_format == 'year_period':
            df[date_column] = df[date_column].apply(convert_year_period_to_date)
        else:
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date and group if specified
        if group_by:
            df = df.sort_values([group_by, date_column]).reset_index(drop=True)
        else:
            df = df.sort_values(date_column).reset_index(drop=True)
        
        # Handle missing values in target
        df[target_column] = df[target_column].ffill().fillna(0)

        # Optional outlier handling before transforms
        outlier_config = self.config.get('forecasting.outlier_handling', {})
        if outlier_config.get('enabled', False):
            method = outlier_config.get('method', 'iqr')
            factor = float(outlier_config.get('factor', 1.5))
            df[target_column] = handle_outliers(df[target_column], method=method, factor=factor)

        # Optional log transform (auto-offset if needed)
        if self.use_log_transform:
            min_value = df[target_column].min()
            required_offset = 0.0
            if pd.notna(min_value) and min_value <= -1:
                required_offset = float(-min_value + 1)
            if required_offset > self.target_transform_offset:
                logger.warning(
                    "Adjusting log transform offset to %s to handle negative values",
                    required_offset
                )
                self.target_transform_offset = required_offset
            max_value = df[target_column].max()
            if pd.notna(max_value):
                self._max_target_log = float(np.log1p(max_value + self.target_transform_offset))
            df[target_column] = self._transform_target(df[target_column])
        
        logger.info(f"Data prepared: {len(df)} records from {df[date_column].min()} "
                   f"to {df[date_column].max()}")
        
        return df
    
    def fit(self, df: pd.DataFrame, target_column: str,
           date_column: str, group_by: Optional[str] = None,
           already_prepared: bool = False,
           feature_engineer=None) -> 'TimeSeriesForecaster':
        """Fit forecasting model.

        Parameters
        ----------
        df:
            Input DataFrame (raw or already prepared by the caller).
        target_column:
            Name of the target variable column.
        date_column:
            Name of the date column.
        group_by:
            Optional grouping column (e.g., segment / profit-centre key).
        already_prepared:
            If True, skip ``prepare_data`` (caller already handled date
            coercion, outlier handling, log transform).
        feature_engineer:
            Optional ``FeatureEngineer`` instance.  When supplied the method
            enforces the correct **split-first → fit-on-train-only** order::

                split(df) → FE.fit(train) → FE.transform(train/val/test)

            This prevents future-data leakage into engineered features.
            When ``None`` the method expects ``df`` to already contain all
            engineered feature columns (legacy behaviour, backward-compatible).
        """

        # 1. Prepare data (date coercion, outlier handling, optional log transform)
        if already_prepared:
            prepared_df = df
        else:
            prepared_df = self.prepare_data(df, target_column, date_column, group_by)
        self.group_by = group_by

        # 2. Split BEFORE feature engineering so FE cannot see future labels
        self.train_data, self.validation_data, self.test_data = self._split_data(
            prepared_df, date_column, target_column
        )

        # 3. Feature engineering — fit on train only, transform each split
        if feature_engineer is not None:
            self._feature_engineer = feature_engineer
            # For a single-segment series there is no meaningful group_by inside
            # the forecaster; group_by here refers to the caller-level segmentation
            # which is already resolved (one forecaster = one segment).
            fe_group_by = None
            logger.info(
                "FE fit-on-train-only: fitting on %d train rows, "
                "transforming train / val / test splits",
                len(self.train_data),
            )
            feature_engineer.fit(self.train_data, group_by=fe_group_by)
            self.train_data = feature_engineer.transform(self.train_data)
            if len(self.validation_data) > 0:
                self.validation_data = feature_engineer.transform(self.validation_data)
            if len(self.test_data) > 0:
                self.test_data = feature_engineer.transform(self.test_data)
        else:
            self._feature_engineer = None
        
        # Initialize model based on configuration (registry-aware)
        model_type = self.config.get('forecasting.model_type', 'prophet')

        if model_type == 'prophet':
            prophet_config = self.config.get('forecasting.prophet', {})
            self.model = ProphetModel(**prophet_config)
        elif model_type == 'xgboost':
            # XGBoost has special adaptive-param logic
            xgb_params = self._get_adaptive_xgb_params(len(self.train_data))
            self.model = SklearnModel(model_type='xgboost', **xgb_params)
        elif model_type == 'holt_winters':
            hw_params = self.config.get('forecasting.holt_winters', {})
            self.model = HoltWintersModel(**hw_params)
        elif model_type == 'ensemble':
            ens_params = self.config.get('forecasting.ensemble', {})
            self.model = EnsembleModel(**ens_params)
        elif model_type in _MODEL_REGISTRY:
            # All other registered models: pull params from config section
            model_params = self.config.get(f'forecasting.{model_type}', {})
            self.model = SklearnModel(model_type=model_type, **model_params)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Available: {['prophet', 'holt_winters', 'ensemble'] + available_models()}"
            )
        
        # Fit the model
        agg_data = None
        if group_by:
            # Fit on aggregated numeric features for the full time series
            numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
            agg_data = self.train_data.groupby(date_column, as_index=False)[numeric_cols].sum()
            agg_data = agg_data.sort_values(date_column).reset_index(drop=True)

        if agg_data is not None and len(agg_data) > 0:
            self._agg_train_last = agg_data.iloc[-1].to_dict()
            self._agg_history = agg_data[[date_column, target_column]].copy()
        elif isinstance(self.model, (SklearnModel, EnsembleModel)) and len(self.train_data) > 0:
            # Track history for recursive forecasting on a single aggregated series
            self._agg_train_last = self.train_data.iloc[-1].to_dict()
            self._agg_history = self.train_data[[date_column, target_column]].copy()

        if group_by and agg_data is not None and len(agg_data) > 0:
            self.model.fit(agg_data, target_column, date_column)
        else:
            self.model.fit(self.train_data, target_column, date_column)
        
        # Validate on validation set if available
        if len(self.validation_data) > 0:
            self._validate_model(target_column, date_column)
        
        logger.info(f"Forecasting model ({model_type}) fitted successfully")
        return self
    
    def _validate_model(self, target_column: str, date_column: str):
        """Validate model on validation set"""

        validation_df = self._prepare_eval_df(
            self.validation_data, target_column, date_column
        )
        
        # Generate predictions for validation set
        if isinstance(self.model, ProphetModel):
            # For Prophet, use validation dates
            val_predictions = self.model.predict(
                validation_df[[date_column]].rename(columns={date_column: 'ds'})
            )
            y_pred = val_predictions['prediction'].values
        elif isinstance(self.model, HoltWintersModel):
            # HoltWinters forecasts forward from training end; match periods
            n_periods = len(validation_df)
            val_predictions = self.model.predict(None, n_periods)
            y_pred = val_predictions['prediction'].values[:n_periods]
        else:
            # For sklearn / ensemble models, use validation features
            val_predictions = self.model.predict(validation_df)
            y_pred = val_predictions['prediction'].values
        
        y_true = validation_df[target_column].values
        val_dates = validation_df[date_column].values

        if self.use_log_transform:
            y_true = self._inverse_transform_target(y_true)
            y_pred = self._inverse_transform_target(y_pred)
        
        # Calculate metrics
        self.metrics['validation'] = self._calculate_metrics(y_true, y_pred, label="validation")
        
        logger.info(f"Validation metrics: {self.metrics['validation']}")

        # Horizon-wise errors (month-by-month within this split)
        self._log_horizon_errors(val_dates, y_true, y_pred, label="validation")
    
    def predict(self, periods: int = None, 
               future_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate forecasts
        
        Args:
            periods: Number of periods to forecast
            future_df: Optional DataFrame with future dates/features
            
        Returns:
            DataFrame with predictions
        """
        if not self.model or not self.model.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if periods is None:
            periods = self.config.get('forecasting.forecast_horizon', 12)
        
        # ── HoltWinters: purely time-series based, no feature DataFrame ──
        if isinstance(self.model, HoltWintersModel):
            predictions = self.model.predict(None, periods)
            if self.use_log_transform and 'prediction' in predictions.columns:
                predictions['prediction'] = self._inverse_transform_target(
                    predictions['prediction'].values
                )
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions

        # For EnsembleModel (wraps SklearnModels) treat like SklearnModel
        _needs_features = isinstance(self.model, (SklearnModel, EnsembleModel))

        if future_df is None and _needs_features:
            forecast_start = self.config.get('forecasting.forecast_start')
            if forecast_start == 'current_month':
                start_date = pd.Timestamp.today().to_period('M').to_timestamp()
            else:
                last_date = self.train_data[self.config.get('data_source.date_column')].max()
                start_date = (pd.Timestamp(last_date) + pd.DateOffset(months=1)).to_period('M').to_timestamp()

            if self._agg_history is not None and not self._agg_history.empty:
                predictions = self._recursive_forecast(
                    start_date, periods,
                    self.config.get('data_source.amount_column'),
                    self.config.get('data_source.date_column')
                )
                if self.use_log_transform and 'prediction' in predictions.columns:
                    predictions['prediction'] = self._inverse_transform_target(
                        predictions['prediction'].values
                    )
                logger.info(f"Generated {len(predictions)} predictions")
                return predictions

            date_range = pd.date_range(start=start_date, periods=periods, freq='MS')
            future_df = pd.DataFrame({
                self.config.get('data_source.date_column'): date_range
            })

        if future_df is not None and _needs_features and self.group_by:
            future_df = self._populate_future_features(
                future_df, self.config.get('data_source.amount_column'),
                self.config.get('data_source.date_column')
            )

        if future_df is not None and _needs_features and self._agg_train_last:
            feature_cols = getattr(self.model, 'feature_columns', None) or []
            for feature in feature_cols:
                if feature not in future_df.columns:
                    future_df[feature] = self._agg_train_last.get(feature, 0.0)

        # Generate predictions
        predictions = self.model.predict(future_df, periods)

        if self.use_log_transform and 'prediction' in predictions.columns:
            predictions['prediction'] = self._inverse_transform_target(
                predictions['prediction'].values
            )
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def evaluate(self, df: pd.DataFrame, target_column: str, 
                date_column: str) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            df: Test DataFrame
            target_column: Name of target variable
            date_column: Name of date column
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.model or not self.model.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        test_df = self._prepare_eval_df(df, target_column, date_column)

        # Generate predictions for test data
        if isinstance(self.model, ProphetModel):
            test_predictions = self.model.predict(
                test_df[[date_column]].rename(columns={date_column: 'ds'})
            )
            y_pred = test_predictions['prediction'].values
        elif isinstance(self.model, HoltWintersModel):
            n_periods = len(test_df)
            test_predictions = self.model.predict(None, n_periods)
            y_pred = test_predictions['prediction'].values[:n_periods]
        else:
            test_predictions = self.model.predict(test_df)
            y_pred = test_predictions['prediction'].values
        
        y_true = test_df[target_column].values
        test_dates = test_df[date_column].values

        if self.use_log_transform:
            y_true = self._inverse_transform_target(y_true)
            y_pred = self._inverse_transform_target(y_pred)
        
        # Calculate metrics
        test_metrics = self._calculate_metrics(y_true, y_pred, label="test")
        self.metrics['test'] = test_metrics
        
        logger.info(f"Test metrics: {test_metrics}")

        # Horizon-wise errors (h=1..N months)
        horizon_metrics = self._log_horizon_errors(test_dates, y_true, y_pred, label="test")
        self.metrics['test_horizon'] = horizon_metrics

        return test_metrics

    # ------------------------------------------------------------------ #
    #  Horizon-wise error reporting                                       #
    # ------------------------------------------------------------------ #
    def _log_horizon_errors(self, dates: np.ndarray, y_true: np.ndarray,
                            y_pred: np.ndarray, label: str = "") -> List[Dict[str, Any]]:
        """Log per-month (horizon) errors and return a list of dicts."""
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
            wape_h = (float(np.sum(np.abs(yt - yp))) / actual_h * 100) if actual_h > 0 else np.nan
            logger.info(
                "[%s h=%d] %s  actual=%.2f  pred=%.2f  MAE=%.2f  WAPE=%.2f%%",
                label, h, str(month.date()) if hasattr(month, 'date') else str(month),
                float(np.mean(yt)), float(np.mean(yp)), mae_h, wape_h
            )
            horizon_rows.append({
                'horizon': h,
                'month': str(month.date()) if hasattr(month, 'date') else str(month),
                'actual_mean': float(np.mean(yt)),
                'pred_mean': float(np.mean(yp)),
                'mae': mae_h,
                'wape': wape_h
            })
        return horizon_rows

    def predict_on_df(self, df: pd.DataFrame, target_column: str,
                      date_column: str) -> pd.DataFrame:
        """Generate predictions aligned to the provided dataframe."""
        if not self.model or not self.model.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        eval_df = self._prepare_eval_df(df, target_column, date_column)

        if isinstance(self.model, ProphetModel):
            predictions = self.model.predict(
                eval_df[[date_column]].rename(columns={date_column: 'ds'})
            )
            pred_values = predictions['prediction'].values
        elif isinstance(self.model, HoltWintersModel):
            # HoltWinters forecasts forward from training end; align to eval dates
            n_periods = len(eval_df)
            predictions = self.model.predict(None, n_periods)
            pred_values = predictions['prediction'].values[:n_periods]
        else:
            predictions = self.model.predict(eval_df)
            pred_values = predictions['prediction'].values

        if self.use_log_transform:
            pred_values = self._inverse_transform_target(pred_values)

        return pd.DataFrame({
            'date': eval_df[date_column].values,
            'prediction': pred_values
        })
    
    def save_model(self, path: Optional[str] = None):
        """Save the fitted model"""
        if not self.model:
            raise ValueError("No model to save")
        
        if path is None:
            model_save_path = self.config.get('forecasting.model_save_path', 'models/')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f"{model_save_path}forecast_model_{timestamp}.pkl"
        
        self.model.save_model(path)
        
        # Also save metrics and metadata
        metadata_path = path.replace('.pkl', '_metadata.pkl')
        metadata = {
            'metrics': self.metrics,
            'config': self.config.config.get('forecasting', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model and metadata saved to: {path}")
    
    def load_model(self, path: str):
        """Load a fitted model"""
        # Determine model type from config or path
        model_type = self.config.get('forecasting.model_type', 'prophet')
        
        if model_type == 'prophet':
            self.model = ProphetModel()
        else:
            self.model = SklearnModel(model_type=model_type)
        
        self.model.load_model(path)
        
        # Load metadata if available
        metadata_path = path.replace('.pkl', '_metadata.pkl')
        if Path(metadata_path).exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.metrics = metadata.get('metrics', {})
        
        logger.info(f"Model loaded from: {path}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if not self.model or not self.model.is_fitted:
            return None
        
        if isinstance(self.model, SklearnModel) and hasattr(self.model.model, 'feature_importances_'):
            importances = self.model.model.feature_importances_
            feature_names = self.model.feature_columns or [f"feature_{i}" for i in range(len(importances))]
            return dict(zip(feature_names, importances))
        
        return None