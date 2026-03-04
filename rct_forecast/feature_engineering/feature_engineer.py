"""
Feature Engineering Module

Extensible feature engineering system for time series forecasting.
"""

import calendar
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date as _date
import logging
from abc import ABC, abstractmethod
from ..config.config_manager import ConfigManager

try:
    import holidays as _holidays_lib
    _US_HOLIDAYS_AVAILABLE = True
except ImportError:
    _holidays_lib = None
    _US_HOLIDAYS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureTransformer(ABC):
    """Abstract base class for feature transformers"""
    
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
    """Create lag features for time series data"""
    
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
                logger.warning(f"Column {column} not found for lag features")
                continue
            
            for lag in lags:
                lag_col_name = f"{column}_lag_{lag}"
                
                if group_by:
                    df_result[lag_col_name] = df_result.groupby(group_by)[column].shift(lag)
                else:
                    df_result[lag_col_name] = df_result[column].shift(lag)
                
                logger.debug(f"Created lag feature: {lag_col_name}")
        
        return df_result


class RollingFeatureTransformer(FeatureTransformer):
    """Create rolling window features"""
    
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
            logger.warning(f"Column {column} not found for rolling features")
            return df_result
        
        for window in windows:
            for agg in aggregations:
                feature_name = f"{column}_rolling_{agg}_{window}"
                
                if group_by:
                    df_result[feature_name] = (df_result.groupby(group_by)[column]
                                             .rolling(window=window, min_periods=1)
                                             .agg(agg).reset_index(0, drop=True))
                else:
                    df_result[feature_name] = (df_result[column]
                                             .rolling(window=window, min_periods=1)
                                             .agg(agg))
                
                logger.debug(f"Created rolling feature: {feature_name}")
        
        return df_result


class SeasonalFeatureTransformer(FeatureTransformer):
    """Create seasonal/cyclical features from date columns"""
    
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
            logger.warning(f"Date column {date_column} not found for seasonal features")
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
        
        logger.debug(f"Created seasonal features for {date_column}")
        return df_result


class CalendarFeatureTransformer(FeatureTransformer):
    """Create calendar-based features: business days, holidays, quarter position, seasonality."""

    _FIXED_HOLIDAYS = [
        (1, 1),   # New Year's Day
        (6, 19),  # Juneteenth
        (7, 4),   # Independence Day
        (11, 11), # Veterans Day
        (12, 25), # Christmas Day
    ]

    _HIGH_SEASONALITY_MONTHS = {3, 9, 12}
    
    _SEASONAL_INTENSITY = {
        1: 0.92,
        2: 0.95,
        3: 1.12,
        4: 0.98,
        5: 0.96,
        6: 1.05,
        7: 0.97,
        8: 0.98,
        9: 1.15,
        10: 1.00,
        11: 0.95,
        12: 1.18,
    }

    @classmethod
    def _us_holidays_for_month(cls, year: int, month: int) -> int:
        if _US_HOLIDAYS_AVAILABLE:
            us = _holidays_lib.US(years=year)  # type: ignore[union-attr]
            return sum(1 for d in us.keys() if d.month == month)

        count = 0
        for m, d in cls._FIXED_HOLIDAYS:
            if m == month:
                count += 1

        if month == 1:   count += 1
        if month == 2:   count += 1
        if month == 5:   count += 1
        if month == 9:   count += 1
        if month == 10:  count += 1
        if month == 11:  count += 1
        return count

    @staticmethod
    def _days_to_quarter_end(dt: datetime) -> int:
        month = dt.month
        quarter_end_month = ((month - 1) // 3 + 1) * 3
        year = dt.year
        last_day = calendar.monthrange(year, quarter_end_month)[1]
        quarter_end = _date(year, quarter_end_month, last_day)
        return (quarter_end - dt.date()).days

    @staticmethod
    def _days_to_year_end(dt: datetime) -> int:
        year_end = _date(dt.year, 12, 31)
        return (year_end - dt.date()).days

    def fit(self, df: pd.DataFrame) -> 'CalendarFeatureTransformer':
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        df_result = df.copy()
        date_column = self.params.get('date_column')

        if date_column not in df.columns:
            logger.warning(f"Date column {date_column} not found for calendar features")
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
        
        df_result[f'{date_column}_days_to_quarter_end'] = date_series.apply(
            self._days_to_quarter_end
        )
        df_result[f'{date_column}_days_to_year_end'] = date_series.apply(
            self._days_to_year_end
        )
        
        days_to_qe = df_result[f'{date_column}_days_to_quarter_end']
        df_result[f'{date_column}_quarter_end_urgency'] = (
            1.0 / (days_to_qe + 1)
        ).clip(upper=1.0)

        logger.debug(f"Created calendar features for {date_column}")
        return df_result


class GrowthFeatureTransformer(FeatureTransformer):
    """Create growth/change features"""
    
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
                logger.warning(f"Column {column} not found for growth features")
                continue
            
            for period in periods:
                pct_change_name = f"{column}_pct_change_{period}"
                if group_by:
                    df_result[pct_change_name] = (df_result.groupby(group_by)[column]
                                                .pct_change(periods=period))
                else:
                    df_result[pct_change_name] = df_result[column].pct_change(periods=period)
                
                diff_name = f"{column}_diff_{period}"
                if group_by:
                    df_result[diff_name] = (df_result.groupby(group_by)[column]
                                          .diff(periods=period))
                else:
                    df_result[diff_name] = df_result[column].diff(periods=period)
                
                logger.debug(f"Created growth features: {pct_change_name}, {diff_name}")
        
        return df_result


class StatisticalFeatureTransformer(FeatureTransformer):
    """Create statistical features"""
    
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
                group_stats = df.groupby(group_by)[column].agg(['mean', 'std', 'median'])
                self.stats[column] = group_stats
            else:
                self.stats[column] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'median': df[column].median()
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
                        group_stats[[group_by, stat]].rename(
                            columns={stat: f"{column}_group_{stat}"}
                        ),
                        on=group_by,
                        how='left'
                    )
            else:
                for stat_name, stat_value in self.stats[column].items():
                    df_result[f"{column}_global_{stat_name}"] = stat_value
            
            if group_by:
                df_result[f"{column}_zscore"] = (
                    (df_result[column] - df_result[f"{column}_group_mean"]) / 
                    df_result[f"{column}_group_std"]
                )
            else:
                df_result[f"{column}_zscore"] = (
                    (df_result[column] - self.stats[column]['mean']) / 
                    self.stats[column]['std']
                )
        
        return df_result


class FeatureEngineer:
    """Main feature engineering orchestrator"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.transformers = []
        self.fitted_transformers = []
        self._load_transformers_from_config()
    
    def _load_transformers_from_config(self):
        if not self.config.get('feature_engineering.enabled', True):
            logger.info("Feature engineering is disabled in configuration")
            return
        
        features_config = self.config.get('feature_engineering.features', [])
        
        for feature_config in features_config:
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
                logger.warning(f"Unknown feature transformer type: {feature_type}")
                continue
            
            self.transformers.append(transformer)
            logger.info(f"Loaded feature transformer: {feature_name}")
    
    def add_transformer(self, transformer: FeatureTransformer):
        self.transformers.append(transformer)
        logger.info(f"Added feature transformer: {transformer.name}")
    
    def fit(self, df: pd.DataFrame, 
            group_by: Optional[Union[str, List[str]]] = None) -> 'FeatureEngineer':
        if not self.config.get('feature_engineering.enabled', True):
            logger.info("Feature engineering is disabled")
            return self
        
        logger.info("Fitting feature transformers")
        
        if group_by:
            for transformer in self.transformers:
                if 'group_by' not in transformer.params:
                    transformer.params['group_by'] = group_by
        
        self.fitted_transformers = []
        current_df = df.copy()
        
        for transformer in self.transformers:
            try:
                fitted_transformer = transformer.fit(current_df)
                self.fitted_transformers.append(fitted_transformer)
                current_df = fitted_transformer.transform(current_df)
                logger.info(f"Fitted transformer: {transformer.name}")
            except Exception as e:
                logger.error(f"Failed to fit transformer {transformer.name}: {e}")
                raise
        
        logger.info("All feature transformers fitted successfully")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('feature_engineering.enabled', True):
            logger.info("Feature engineering is disabled")
            return df
        
        if not self.fitted_transformers:
            raise ValueError("Feature engineer must be fitted before transform")
        
        logger.info("Transforming data with feature engineering")
        
        current_df = df.copy()
        
        for transformer in self.fitted_transformers:
            try:
                current_df = transformer.transform(current_df)
                logger.debug(f"Applied transformer: {transformer.name}")
            except Exception as e:
                logger.error(f"Failed to apply transformer {transformer.name}: {e}")
                raise
        
        original_features = len(df.columns)
        new_features = len(current_df.columns)
        logger.info(f"Feature engineering complete: {original_features} -> {new_features} features")
        
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
            'feature_engineering_enabled': self.config.get('feature_engineering.enabled', True)
        }
