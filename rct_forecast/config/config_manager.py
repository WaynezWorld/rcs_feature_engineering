"""
Configuration Manager

Handles loading and managing configuration from YAML files and environment variables.
"""

import os
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(**kwargs):  # type: ignore[misc]
        """python-dotenv not installed — .env loading skipped silently."""
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and access"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load .env file if present
        load_dotenv()
        """
        Initialize ConfigManager
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        current_dir = Path(__file__).parent.parent.parent
        return str(current_dir / "config" / "default.yml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file and environment variables"""
        config = {}
        
        # Load from YAML file
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
        else:
            logger.warning(f"Configuration file not found: {self.config_path}")
        
        # Override with environment variables
        config = self._load_env_overrides(config)
        
        return config
    
    def _load_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load environment variable overrides, supporting external browser connection."""
        # Snowflake credentials
        snowflake_config = config.setdefault('snowflake', {})
        snowflake_config['account'] = os.getenv('SNOWFLAKE_ACCOUNT', snowflake_config.get('account'))
        snowflake_config['user'] = os.getenv('SNOWFLAKE_USER', snowflake_config.get('user'))
        snowflake_config['password'] = os.getenv('SNOWFLAKE_PASSWORD', snowflake_config.get('password'))
        snowflake_config['warehouse'] = os.getenv('SNOWFLAKE_WAREHOUSE', snowflake_config.get('warehouse'))
        snowflake_config['database'] = os.getenv('SNOWFLAKE_DATABASE', snowflake_config.get('database'))
        snowflake_config['schema'] = os.getenv('SNOWFLAKE_SCHEMA', snowflake_config.get('schema'))
        # Optional: support external browser authentication
        snowflake_config['authenticator'] = os.getenv('SNOWFLAKE_AUTHENTICATOR', snowflake_config.get('authenticator', 'externalbrowser'))
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation)
        
        Args:
            key: Configuration key (e.g., 'snowflake.account')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        logger.info("Configuration reloaded")
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
        """
        self._deep_update(self.config, updates)
        logger.info("Configuration updated")

    def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]):
        """Recursively merge updates into the base dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
