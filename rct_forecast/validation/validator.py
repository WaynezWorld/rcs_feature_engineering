"""
Validation Framework

Extensible validation system for data quality and process integrity checks.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
from abc import ABC, abstractmethod
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class ValidationRule(ABC):
    """Abstract base class for validation rules"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        pass


class NullCheckRule(ValidationRule):
    """Check for null values in specified columns"""
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        columns = self.params.get('columns', [])
        results = {}
        
        for column in columns:
            if column in df.columns:
                null_count = df[column].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                results[f"{column}_null_count"] = null_count
                results[f"{column}_null_percentage"] = null_percentage
                results[f"{column}_has_nulls"] = null_count > 0
            else:
                logger.warning(f"Column {column} not found in DataFrame")
                results[f"{column}_missing_column"] = True
        
        return {
            'rule_name': self.name,
            'rule_type': 'null_check',
            'passed': not any(results.get(f"{col}_has_nulls", False) for col in columns),
            'details': results
        }


class DateRangeRule(ValidationRule):
    """Check if dates are within specified range"""
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        column = self.params.get('column')
        min_date = self.params.get('min_date')
        max_date = self.params.get('max_date')
        
        if column not in df.columns:
            return {
                'rule_name': self.name,
                'rule_type': 'date_range',
                'passed': False,
                'details': {'error': f'Column {column} not found'}
            }
        
        date_series = pd.to_datetime(df[column])
        results = {}
        
        if min_date:
            min_date_dt = pd.to_datetime(min_date)
            below_min = (date_series < min_date_dt).sum()
            results['below_min_date'] = below_min
            results['min_date_violations'] = below_min > 0
        
        if max_date:
            max_date_dt = pd.to_datetime(max_date)
            above_max = (date_series > max_date_dt).sum()
            results['above_max_date'] = above_max
            results['max_date_violations'] = above_max > 0
        
        results['data_min_date'] = date_series.min().strftime('%Y-%m-%d')
        results['data_max_date'] = date_series.max().strftime('%Y-%m-%d')
        
        passed = not (results.get('min_date_violations', False) or
                      results.get('max_date_violations', False))
        
        return {
            'rule_name': self.name,
            'rule_type': 'date_range',
            'passed': passed,
            'details': results
        }


class PositiveValueRule(ValidationRule):
    """Check if numeric values are positive"""
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        column = self.params.get('column')
        allow_negative = self.params.get('allow_negative', False)
        allow_zero = self.params.get('allow_zero', True)
        
        if column not in df.columns:
            return {
                'rule_name': self.name,
                'rule_type': 'positive_value',
                'passed': False,
                'details': {'error': f'Column {column} not found'}
            }
        
        numeric_series = pd.to_numeric(df[column], errors='coerce')
        negative_count = (numeric_series < 0).sum()
        zero_count = (numeric_series == 0).sum()
        
        results = {
            'negative_values': negative_count,
            'zero_values': zero_count,
            'has_negative_values': negative_count > 0,
            'has_zero_values': zero_count > 0,
            'min_value': numeric_series.min(),
            'max_value': numeric_series.max(),
            'allow_negative': allow_negative,
            'allow_zero': allow_zero
        }
        
        negative_ok = allow_negative or negative_count == 0
        zero_ok = allow_zero or zero_count == 0
        
        return {
            'rule_name': self.name,
            'rule_type': 'positive_value',
            'passed': bool(negative_ok and zero_ok),
            'details': results
        }


class DuplicateCheckRule(ValidationRule):
    """Check for duplicate records based on specified columns"""
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        columns = self.params.get('columns', [])
        
        if not columns:
            duplicate_count = df.duplicated().sum()
        else:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {
                    'rule_name': self.name,
                    'rule_type': 'duplicate_check',
                    'passed': False,
                    'details': {'error': f'Columns not found: {missing_cols}'}
                }
            duplicate_count = df.duplicated(subset=columns).sum()
        
        results = {
            'duplicate_count': duplicate_count,
            'has_duplicates': duplicate_count > 0,
            'total_records': len(df),
            'unique_records': len(df) - duplicate_count
        }
        
        return {
            'rule_name': self.name,
            'rule_type': 'duplicate_check',
            'passed': bool(duplicate_count == 0),
            'details': results
        }


class CustomValidationRule(ValidationRule):
    """Custom validation rule using user-defined function"""
    
    def __init__(self, name: str, validation_func: Callable, **kwargs):
        super().__init__(name, **kwargs)
        self.validation_func = validation_func
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            result = self.validation_func(df, **self.params)
            if isinstance(result, dict):
                return {
                    'rule_name': self.name,
                    'rule_type': 'custom',
                    'passed': result.get('passed', False),
                    'details': result.get('details', {})
                }
            else:
                return {
                    'rule_name': self.name,
                    'rule_type': 'custom',
                    'passed': bool(result),
                    'details': {'result': result}
                }
        except Exception as e:
            logger.error(f"Custom validation rule {self.name} failed: {e}")
            return {
                'rule_name': self.name,
                'rule_type': 'custom',
                'passed': False,
                'details': {'error': str(e)}
            }


class ValidationManager:
    """Manages validation rules and execution"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.rules = []
        self._load_rules_from_config()
    
    def _load_rules_from_config(self):
        if not self.config.get('validation.enabled', True):
            logger.info("Validation is disabled in configuration")
            return
        
        rules_config = self.config.get('validation.rules', [])
        
        for rule_config in rules_config:
            rule_type = rule_config.get('type')
            rule_name = rule_config.get('name')
            
            if rule_type == 'null':
                rule = NullCheckRule(rule_name, columns=rule_config.get('columns', []))
            elif rule_type == 'date_range':
                rule = DateRangeRule(rule_name,
                                     column=rule_config.get('column'),
                                     min_date=rule_config.get('min_date'),
                                     max_date=rule_config.get('max_date'))
            elif rule_type == 'positive':
                rule = PositiveValueRule(rule_name, column=rule_config.get('column'))
            elif rule_type == 'duplicate':
                rule = DuplicateCheckRule(rule_name, columns=rule_config.get('columns', []))
            else:
                logger.warning(f"Unknown validation rule type: {rule_type}")
                continue
            
            self.rules.append(rule)
            logger.info(f"Loaded validation rule: {rule_name}")
    
    def add_rule(self, rule: ValidationRule):
        self.rules.append(rule)
        logger.info(f"Added validation rule: {rule.name}")
    
    def add_custom_rule(self, name: str, validation_func: Callable, **kwargs):
        rule = CustomValidationRule(name, validation_func, **kwargs)
        self.add_rule(rule)
    
    def validate(self, df: pd.DataFrame,
                 stage: Optional[str] = None) -> Dict[str, Any]:
        if not self.config.get('validation.enabled', True):
            logger.info("Validation is disabled")
            return {
                'validation_enabled': False,
                'passed': True,
                'stage': stage,
                'timestamp': datetime.now().isoformat(),
                'results': []
            }
        
        logger.info(f"Running validation{f' for stage: {stage}' if stage else ''}")
        
        results = []
        all_passed = True
        
        for rule in self.rules:
            try:
                result = rule.validate(df)
                results.append(result)
                if not result['passed']:
                    all_passed = False
                    logger.warning(f"Validation rule {rule.name} failed: {result['details']}")
                else:
                    logger.info(f"Validation rule {rule.name} passed")
            except Exception as e:
                all_passed = False
                results.append({
                    'rule_name': rule.name,
                    'rule_type': 'error',
                    'passed': False,
                    'details': {'error': str(e)}
                })
                logger.error(f"Validation rule {rule.name} error: {e}")
        
        validation_summary = {
            'validation_enabled': True,
            'passed': all_passed,
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'total_rules': len(self.rules),
            'passed_rules': sum(1 for r in results if r['passed']),
            'failed_rules': sum(1 for r in results if not r['passed']),
            'data_shape': df.shape,
            'results': results
        }
        
        if all_passed:
            logger.info(f"All validation rules passed{f' for stage: {stage}' if stage else ''}")
        else:
            logger.error(f"Validation failed{f' for stage: {stage}' if stage else ''}")
        
        return validation_summary
    
    def get_failed_validations(self, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [r for r in validation_results.get('results', []) if not r['passed']]
