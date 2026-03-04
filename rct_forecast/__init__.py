"""
RCT Revenue Forecasting Package — Feature Engineering subset
"""
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

__version__ = "0.1.0"

from .config.config_manager import ConfigManager

try:
    from .validation.validator import ValidationManager
    VALIDATION_AVAILABLE = True
except ImportError:
    ValidationManager = None
    VALIDATION_AVAILABLE = False

try:
    from .feature_engineering.feature_engineer import FeatureEngineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FeatureEngineer = None
    FEATURE_ENGINEERING_AVAILABLE = False

__all__ = ["ConfigManager"]
if VALIDATION_AVAILABLE:
    __all__.append("ValidationManager")
if FEATURE_ENGINEERING_AVAILABLE:
    __all__.append("FeatureEngineer")
