#!/usr/bin/env python3
"""
Standalone Feature Engineering Script

Loads a CSV of prepared data, applies the full feature engineering pipeline
defined in config, and writes the enriched dataset to a new CSV.

Usage
-----
# Use default config and a prepared-data CSV:
python run_feature_engineering.py --input data/prepared_data.csv

# Specify config and output path:
python run_feature_engineering.py \
    --config config/default.yml \
    --input  data/prepared_data.csv \
    --output results/engineered_data.csv \
    --group-by "Roll Up Shop"

# Skip validation after engineering:
python run_feature_engineering.py --input data/prepared.csv --skip-validation

# Just print new column names without saving:
python run_feature_engineering.py --input data/prepared.csv --show-features
"""

import sys
import os
import argparse
import calendar
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime, date as _date
from typing import Dict, List, Any, Optional, Union, Callable

import pandas as pd
import numpy as np
import yaml

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
except ImportError:
    def load_dotenv(**kwargs) -> bool:  # type: ignore[misc]
        return False

try:
    import holidays as _holidays_lib  # type: ignore[import-untyped]
    _US_HOLIDAYS_AVAILABLE = True
except ImportError:
    _holidays_lib = None  # type: ignore[assignment]
    _US_HOLIDAYS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# ConfigManager
# ─────────────────────────────────────────────────────────────────────────────

class ConfigManager:
    """Manages configuration loading and access"""

    def __init__(self, config_path: Optional[str] = None):
        load_dotenv()
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()

    def _get_default_config_path(self) -> str:
        return str(Path(__file__).resolve().parent / "config" / "default.yml")

    def _load_config(self) -> Dict[str, Any]:
        config = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        config = self._load_env_overrides(config)
        return config

    def _load_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        snowflake_config = config.setdefault('snowflake', {})
        snowflake_config['account'] = os.getenv('SNOWFLAKE_ACCOUNT', snowflake_config.get('account'))
        snowflake_config['user'] = os.getenv('SNOWFLAKE_USER', snowflake_config.get('user'))
        snowflake_config['password'] = os.getenv('SNOWFLAKE_PASSWORD', snowflake_config.get('password'))
        snowflake_config['warehouse'] = os.getenv('SNOWFLAKE_WAREHOUSE', snowflake_config.get('warehouse'))
        snowflake_config['database'] = os.getenv('SNOWFLAKE_DATABASE', snowflake_config.get('database'))
        snowflake_config['schema'] = os.getenv('SNOWFLAKE_SCHEMA', snowflake_config.get('schema'))
        snowflake_config['authenticator'] = os.getenv('SNOWFLAKE_AUTHENTICATOR', snowflake_config.get('authenticator', 'externalbrowser'))
        return config

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
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
# Feature Transformers
# ─────────────────────────────────────────────────────────────────────────────

class FeatureTransformer(ABC):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        self.is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'FeatureTransformer':
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class LagFeatureTransformer(FeatureTransformer):
    def fit(self, df: pd.DataFrame) -> 'LagFeatureTransformer':
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        df_result = df.copy()
        columns = self.params.get('columns', [])
        lags = self.params.get('lags', [1])
        group_by = self.params.get('group_by', None)
        for column in columns:
            if column not in df.columns:
                continue
            for lag in lags:
                lag_col_name = f"{column}_lag_{lag}"
                if group_by:
                    df_result[lag_col_name] = df_result.groupby(group_by)[column].shift(lag)
                else:
                    df_result[lag_col_name] = df_result[column].shift(lag)
        return df_result


class RollingFeatureTransformer(FeatureTransformer):
    def fit(self, df: pd.DataFrame) -> 'RollingFeatureTransformer':
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        df_result = df.copy()
        column = self.params.get('column')
        windows = self.params.get('windows', [3])
        group_by = self.params.get('group_by', None)
        aggregations = self.params.get('aggregations', ['mean'])
        if column not in df.columns:
            return df_result
        for window in windows:
            for agg in aggregations:
                feature_name = f"{column}_rolling_{agg}_{window}"
                if group_by:
                    df_result[feature_name] = (
                        df_result.groupby(group_by)[column]
                        .rolling(window=window, min_periods=1)
                        .agg(agg).reset_index(0, drop=True)
                    )
                else:
                    df_result[feature_name] = (
                        df_result[column].rolling(window=window, min_periods=1).agg(agg)
                    )
        return df_result


class SeasonalFeatureTransformer(FeatureTransformer):
    def fit(self, df: pd.DataFrame) -> 'SeasonalFeatureTransformer':
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        df_result = df.copy()
        date_column = self.params.get('date_column')
        features = self.params.get('features', ['month', 'quarter', 'year'])
        if date_column not in df.columns:
            return df_result
        date_series = pd.to_datetime(df_result[date_column])
        if 'month' in features:
            df_result[f'{date_column}_month'] = date_series.dt.month
            df_result[f'{date_column}_month_sin'] = np.sin(2 * np.pi * date_series.dt.month / 12)
            df_result[f'{date_column}_month_cos'] = np.cos(2 * np.pi * date_series.dt.month / 12)
        if 'quarter' in features:
            df_result[f'{date_column}_quarter'] = date_series.dt.quarter
            df_result[f'{date_column}_quarter_sin'] = np.sin(2 * np.pi * date_series.dt.quarter / 4)
            df_result[f'{date_column}_quarter_cos'] = np.cos(2 * np.pi * date_series.dt.quarter / 4)
        if 'year' in features:
            df_result[f'{date_column}_year'] = date_series.dt.year
        if 'day_of_week' in features:
            df_result[f'{date_column}_day_of_week'] = date_series.dt.dayofweek
            df_result[f'{date_column}_dow_sin'] = np.sin(2 * np.pi * date_series.dt.dayofweek / 7)
            df_result[f'{date_column}_dow_cos'] = np.cos(2 * np.pi * date_series.dt.dayofweek / 7)
        if 'day_of_month' in features:
            df_result[f'{date_column}_day_of_month'] = date_series.dt.day
        if 'week_of_year' in features:
            df_result[f'{date_column}_week_of_year'] = date_series.dt.isocalendar().week
        return df_result


class CalendarFeatureTransformer(FeatureTransformer):
    _FIXED_HOLIDAYS = [
        (1, 1),   # New Year's Day
        (6, 19),  # Juneteenth
        (7, 4),   # Independence Day
        (11, 11), # Veterans Day
        (12, 25), # Christmas Day
    ]
    _HIGH_SEASONALITY_MONTHS = {3, 9, 12}
    _SEASONAL_INTENSITY = {
        1: 0.92, 2: 0.95, 3: 1.12, 4: 0.98, 5: 0.96, 6: 1.05,
        7: 0.97, 8: 0.98, 9: 1.15, 10: 1.00, 11: 0.95, 12: 1.18,
    }

    @classmethod
    def _us_holidays_for_month(cls, year: int, month: int) -> int:
        if _US_HOLIDAYS_AVAILABLE and _holidays_lib is not None:
            us = _holidays_lib.US(years=year)
            return sum(1 for d in us.keys() if d.month == month)
        count = sum(1 for m, _ in cls._FIXED_HOLIDAYS if m == month)
        if month in {1, 2, 5, 9, 10, 11}:
            count += 1
        return count

    @staticmethod
    def _days_to_quarter_end(dt: datetime) -> int:
        month = dt.month
        quarter_end_month = ((month - 1) // 3 + 1) * 3
        last_day = calendar.monthrange(dt.year, quarter_end_month)[1]
        quarter_end = _date(dt.year, quarter_end_month, last_day)
        return (quarter_end - dt.date()).days

    @staticmethod
    def _days_to_year_end(dt: datetime) -> int:
        return (_date(dt.year, 12, 31) - dt.date()).days

    def fit(self, df: pd.DataFrame) -> 'CalendarFeatureTransformer':
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        df_result = df.copy()
        date_column = self.params.get('date_column')
        if date_column not in df.columns:
            return df_result
        date_series = pd.to_datetime(df_result[date_column])

        def _bdays(dt):
            y, m = dt.year, dt.month
            last_day = calendar.monthrange(y, m)[1]
            return int(np.busday_count(
                np.datetime64(f'{y:04d}-{m:02d}-01'),
                np.datetime64(f'{y:04d}-{m:02d}-{last_day:02d}') + np.timedelta64(1, 'D'),
            ))

        df_result[f'{date_column}_business_days'] = date_series.apply(_bdays)
        df_result[f'{date_column}_days_in_month'] = date_series.dt.days_in_month
        df_result[f'{date_column}_holidays'] = date_series.apply(
            lambda dt: self._us_holidays_for_month(dt.year, dt.month)
        )
        df_result[f'{date_column}_net_working_days'] = (
            df_result[f'{date_column}_business_days'] - df_result[f'{date_column}_holidays']
        ).clip(lower=1)
        df_result[f'{date_column}_month_in_quarter'] = ((date_series.dt.month - 1) % 3) + 1
        df_result[f'{date_column}_is_quarter_end'] = date_series.dt.is_quarter_end.astype(int)
        df_result[f'{date_column}_is_quarter_start'] = date_series.dt.is_quarter_start.astype(int)
        df_result[f'{date_column}_is_year_end'] = (date_series.dt.month == 12).astype(int)
        df_result[f'{date_column}_is_year_start'] = (date_series.dt.month == 1).astype(int)
        df_result[f'{date_column}_is_high_season'] = date_series.dt.month.isin(
            self._HIGH_SEASONALITY_MONTHS
        ).astype(int)
        df_result[f'{date_column}_seasonal_intensity'] = date_series.dt.month.map(
            self._SEASONAL_INTENSITY
        )
        df_result[f'{date_column}_is_q3_close'] = (date_series.dt.month == 9).astype(int)
        df_result[f'{date_column}_is_q1_close'] = (date_series.dt.month == 3).astype(int)
        df_result[f'{date_column}_days_to_quarter_end'] = date_series.apply(self._days_to_quarter_end)
        df_result[f'{date_column}_days_to_year_end'] = date_series.apply(self._days_to_year_end)
        days_to_qe = df_result[f'{date_column}_days_to_quarter_end']
        df_result[f'{date_column}_quarter_end_urgency'] = (1.0 / (days_to_qe + 1)).clip(upper=1.0)
        return df_result


class GrowthFeatureTransformer(FeatureTransformer):
    def fit(self, df: pd.DataFrame) -> 'GrowthFeatureTransformer':
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        df_result = df.copy()
        columns = self.params.get('columns', [])
        periods = self.params.get('periods', [1])
        group_by = self.params.get('group_by', None)
        for column in columns:
            if column not in df.columns:
                continue
            for period in periods:
                pct_change_name = f"{column}_pct_change_{period}"
                diff_name = f"{column}_diff_{period}"
                if group_by:
                    df_result[pct_change_name] = df_result.groupby(group_by)[column].pct_change(periods=period)
                    df_result[diff_name] = df_result.groupby(group_by)[column].diff(periods=period)
                else:
                    df_result[pct_change_name] = df_result[column].pct_change(periods=period)
                    df_result[diff_name] = df_result[column].diff(periods=period)
        return df_result


class StatisticalFeatureTransformer(FeatureTransformer):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.stats = {}

    def fit(self, df: pd.DataFrame) -> 'StatisticalFeatureTransformer':
        columns = self.params.get('columns', [])
        group_by = self.params.get('group_by', None)
        for column in columns:
            if column not in df.columns:
                continue
            if group_by:
                self.stats[column] = df.groupby(group_by)[column].agg(['mean', 'std', 'median'])
            else:
                self.stats[column] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'median': df[column].median(),
                }
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        df_result = df.copy()
        columns = self.params.get('columns', [])
        group_by = self.params.get('group_by', None)
        for column in columns:
            if column not in self.stats:
                continue
            if group_by:
                group_stats = self.stats[column].reset_index()
                for stat in ['mean', 'std', 'median']:
                    df_result = df_result.merge(
                        group_stats[[group_by, stat]].rename(columns={stat: f"{column}_group_{stat}"}),
                        on=group_by, how='left'
                    )
                df_result[f"{column}_zscore"] = (
                    (df_result[column] - df_result[f"{column}_group_mean"]) /
                    df_result[f"{column}_group_std"]
                )
            else:
                for stat_name, stat_value in self.stats[column].items():
                    df_result[f"{column}_global_{stat_name}"] = stat_value
                df_result[f"{column}_zscore"] = (
                    (df_result[column] - self.stats[column]['mean']) /
                    self.stats[column]['std']
                )
        return df_result


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.transformers: List[FeatureTransformer] = []
        self.fitted_transformers: List[FeatureTransformer] = []
        self._load_transformers_from_config()

    def _load_transformers_from_config(self):
        if not self.config.get('feature_engineering.enabled', True):
            return
        for feature_config in self.config.get('feature_engineering.features', []):
            feature_type = feature_config.get('type')
            feature_name = feature_config.get('name')
            if feature_type == 'lag':
                transformer = LagFeatureTransformer(feature_name,
                    columns=feature_config.get('columns', []),
                    lags=feature_config.get('lags', [1]))
            elif feature_type == 'rolling_mean':
                transformer = RollingFeatureTransformer(feature_name,
                    column=feature_config.get('column'),
                    windows=feature_config.get('windows', [3]),
                    aggregations=['mean'])
            elif feature_type == 'rolling':
                transformer = RollingFeatureTransformer(feature_name,
                    column=feature_config.get('column'),
                    windows=feature_config.get('windows', [3]),
                    aggregations=feature_config.get('aggregations', ['mean']))
            elif feature_type == 'seasonal':
                transformer = SeasonalFeatureTransformer(feature_name,
                    date_column=feature_config.get('date_column'),
                    features=feature_config.get('features', ['month', 'quarter']))
            elif feature_type == 'calendar':
                transformer = CalendarFeatureTransformer(feature_name,
                    date_column=feature_config.get('date_column'))
            elif feature_type == 'growth':
                transformer = GrowthFeatureTransformer(feature_name,
                    columns=feature_config.get('columns', []),
                    periods=feature_config.get('periods', [1]))
            elif feature_type == 'statistical':
                transformer = StatisticalFeatureTransformer(feature_name,
                    columns=feature_config.get('columns', []))
            else:
                continue
            self.transformers.append(transformer)

    def add_transformer(self, transformer: FeatureTransformer):
        self.transformers.append(transformer)

    def fit(self, df: pd.DataFrame,
            group_by: Optional[Union[str, List[str]]] = None) -> 'FeatureEngineer':
        if not self.config.get('feature_engineering.enabled', True):
            return self
        if group_by:
            for transformer in self.transformers:
                if 'group_by' not in transformer.params:
                    transformer.params['group_by'] = group_by
        self.fitted_transformers = []
        current_df = df.copy()
        for transformer in self.transformers:
            fitted_transformer = transformer.fit(current_df)
            self.fitted_transformers.append(fitted_transformer)
            current_df = fitted_transformer.transform(current_df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('feature_engineering.enabled', True):
            return df
        if not self.fitted_transformers:
            raise ValueError("Feature engineer must be fitted before transform")
        current_df = df.copy()
        for transformer in self.fitted_transformers:
            current_df = transformer.transform(current_df)
        return current_df

    def fit_transform(self, df: pd.DataFrame,
                      group_by: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        return self.fit(df, group_by).transform(df)

    def get_feature_names(self, original_features: List[str]) -> List[str]:
        if not self.fitted_transformers:
            return original_features
        dummy_df = pd.DataFrame(columns=original_features)
        for transformer in self.fitted_transformers:
            dummy_df = transformer.transform(dummy_df)
        return list(dummy_df.columns)

    def get_feature_importance_info(self) -> Dict[str, Any]:
        return {
            'total_transformers': len(self.fitted_transformers),
            'transformer_names': [t.name for t in self.fitted_transformers],
            'feature_engineering_enabled': self.config.get('feature_engineering.enabled', True),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

class ValidationRule(ABC):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        pass


class NullCheckRule(ValidationRule):
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        columns = self.params.get('columns', [])
        results = {}
        for column in columns:
            if column in df.columns:
                null_count = df[column].isnull().sum()
                results[f"{column}_null_count"] = null_count
                results[f"{column}_null_percentage"] = (null_count / len(df)) * 100
                results[f"{column}_has_nulls"] = null_count > 0
            else:
                results[f"{column}_missing_column"] = True
        return {
            'rule_name': self.name,
            'rule_type': 'null_check',
            'passed': not any(results.get(f"{col}_has_nulls", False) for col in columns),
            'details': results,
        }


class DateRangeRule(ValidationRule):
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        column = self.params.get('column')
        min_date = self.params.get('min_date')
        max_date = self.params.get('max_date')
        if column not in df.columns:
            return {'rule_name': self.name, 'rule_type': 'date_range', 'passed': False,
                    'details': {'error': f'Column {column} not found'}}
        date_series = pd.to_datetime(df[column])
        results = {}
        if min_date:
            below_min = (date_series < pd.to_datetime(min_date)).sum()
            results['below_min_date'] = below_min
            results['min_date_violations'] = below_min > 0
        if max_date:
            above_max = (date_series > pd.to_datetime(max_date)).sum()
            results['above_max_date'] = above_max
            results['max_date_violations'] = above_max > 0
        results['data_min_date'] = date_series.min().strftime('%Y-%m-%d')
        results['data_max_date'] = date_series.max().strftime('%Y-%m-%d')
        passed = not (results.get('min_date_violations', False) or
                      results.get('max_date_violations', False))
        return {'rule_name': self.name, 'rule_type': 'date_range', 'passed': passed, 'details': results}


class PositiveValueRule(ValidationRule):
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        column = self.params.get('column')
        allow_negative = self.params.get('allow_negative', False)
        allow_zero = self.params.get('allow_zero', True)
        if column not in df.columns:
            return {'rule_name': self.name, 'rule_type': 'positive_value', 'passed': False,
                    'details': {'error': f'Column {column} not found'}}
        numeric_series = pd.to_numeric(df[column], errors='coerce')
        negative_count = (numeric_series < 0).sum()
        zero_count = (numeric_series == 0).sum()
        return {
            'rule_name': self.name,
            'rule_type': 'positive_value',
            'passed': bool((allow_negative or negative_count == 0) and (allow_zero or zero_count == 0)),
            'details': {
                'negative_values': negative_count, 'zero_values': zero_count,
                'min_value': numeric_series.min(), 'max_value': numeric_series.max(),
            },
        }


class DuplicateCheckRule(ValidationRule):
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        columns = self.params.get('columns', [])
        if not columns:
            duplicate_count = df.duplicated().sum()
        else:
            missing_cols = [c for c in columns if c not in df.columns]
            if missing_cols:
                return {'rule_name': self.name, 'rule_type': 'duplicate_check', 'passed': False,
                        'details': {'error': f'Columns not found: {missing_cols}'}}
            duplicate_count = df.duplicated(subset=columns).sum()
        return {
            'rule_name': self.name,
            'rule_type': 'duplicate_check',
            'passed': bool(duplicate_count == 0),
            'details': {'duplicate_count': duplicate_count, 'total_records': len(df)},
        }


class CustomValidationRule(ValidationRule):
    def __init__(self, name: str, validation_func: Callable, **kwargs):
        super().__init__(name, **kwargs)
        self.validation_func = validation_func

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            result = self.validation_func(df, **self.params)
            if isinstance(result, dict):
                return {'rule_name': self.name, 'rule_type': 'custom',
                        'passed': result.get('passed', False), 'details': result.get('details', {})}
            return {'rule_name': self.name, 'rule_type': 'custom',
                    'passed': bool(result), 'details': {'result': result}}
        except Exception as e:
            return {'rule_name': self.name, 'rule_type': 'custom',
                    'passed': False, 'details': {'error': str(e)}}


class ValidationManager:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.rules: List[ValidationRule] = []
        self._load_rules_from_config()

    def _load_rules_from_config(self):
        if not self.config.get('validation.enabled', True):
            return
        for rule_config in self.config.get('validation.rules', []):
            rule_type = rule_config.get('type')
            rule_name = rule_config.get('name')
            if rule_type == 'null':
                rule = NullCheckRule(rule_name, columns=rule_config.get('columns', []))
            elif rule_type == 'date_range':
                rule = DateRangeRule(rule_name, column=rule_config.get('column'),
                                     min_date=rule_config.get('min_date'),
                                     max_date=rule_config.get('max_date'))
            elif rule_type == 'positive':
                rule = PositiveValueRule(rule_name, column=rule_config.get('column'))
            elif rule_type == 'duplicate':
                rule = DuplicateCheckRule(rule_name, columns=rule_config.get('columns', []))
            else:
                continue
            self.rules.append(rule)

    def add_rule(self, rule: ValidationRule):
        self.rules.append(rule)

    def add_custom_rule(self, name: str, validation_func: Callable, **kwargs):
        self.add_rule(CustomValidationRule(name, validation_func, **kwargs))

    def validate(self, df: pd.DataFrame, stage: Optional[str] = None) -> Dict[str, Any]:
        if not self.config.get('validation.enabled', True):
            return {'validation_enabled': False, 'passed': True, 'stage': stage,
                    'timestamp': datetime.now().isoformat(), 'results': []}
        results = []
        all_passed = True
        for rule in self.rules:
            try:
                result = rule.validate(df)
                results.append(result)
                if not result['passed']:
                    all_passed = False
            except Exception as e:
                all_passed = False
                results.append({'rule_name': rule.name, 'rule_type': 'error',
                                 'passed': False, 'details': {'error': str(e)}})
        return {
            'validation_enabled': True,
            'passed': all_passed,
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'total_rules': len(self.rules),
            'passed_rules': sum(1 for r in results if r['passed']),
            'failed_rules': sum(1 for r in results if not r['passed']),
            'data_shape': df.shape,
            'results': results,
        }

    def get_failed_validations(self, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [r for r in validation_results.get('results', []) if not r['passed']]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RCT feature engineering pipeline on a prepared CSV."
    )
    parser.add_argument("--config", type=str, default="config/default.yml",
                        help="Path to YAML config file (default: config/default.yml)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV (output of data preparation)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for the engineered output CSV. "
                             "Defaults to <input_stem>_engineered.csv in the same directory.")
    parser.add_argument("--group-by", type=str, default=None, dest="group_by",
                        help="Column name to use as the segment group-by key "
                             "(e.g. 'Roll Up Shop' or 'Profit Center'). "
                             "When set, lag/rolling features are computed per segment.")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip post-engineering validation step")
    parser.add_argument("--show-features", action="store_true",
                        help="Print the list of engineered feature columns added and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── Configuration ────────────────────────────────────────────────────
    config = ConfigManager(config_path=args.config)

    # ── Load input data ──────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        return 1
    df = pd.read_csv(input_path)

    # ── Resolve group-by column ───────────────────────────────────────────
    group_by = args.group_by
    if group_by is None:
        key_columns = config.get("data_source.key_columns", [])
        segment_level = str(config.get("forecasting.segment_level", "company")).lower()
        if segment_level in {"rollup_shop", "pc", "profit_center"}:
            for candidate in ["Roll Up Shop", "roll_up_shop", "Profit Center", "profit_center"]:
                if candidate in df.columns:
                    group_by = candidate
                    break
        elif key_columns:
            group_by = key_columns[0] if key_columns[0] in df.columns else None

    # ── Feature Engineering ───────────────────────────────────────────────
    feature_engineer = FeatureEngineer(config)
    engineered_df = feature_engineer.fit_transform(df, group_by=group_by)

    original_cols = set(df.columns)
    new_cols = [c for c in engineered_df.columns if c not in original_cols]

    if args.show_features:
        print("\nNew feature columns added:")
        for col in new_cols:
            print(f"  {col}")
        print()

    # ── Post-engineering validation ───────────────────────────────────────
    if not args.skip_validation:
        validator = ValidationManager(config)
        validator.validate(engineered_df, stage="feature_engineering")

    # ── Save output ───────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_engineered.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_csv(output_path, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
