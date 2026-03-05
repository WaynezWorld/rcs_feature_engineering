#!/usr/bin/env python3
"""
Standalone Data Loading Script

Connects to Snowflake, loads revenue data (full or delta), and saves to CSV.

Usage
-----
# Full load — pull all available data:
python run_data_loading.py --mode full

# Delta load — only pull new records since last checkpoint:
python run_data_loading.py --mode delta

# Use a custom config file:
python run_data_loading.py --config config/local.yml --mode delta

# Override output path:
python run_data_loading.py --mode full --output results/my_raw_data.csv

# Test connection only (no data extraction):
python run_data_loading.py --test-connection
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import yaml

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
except ImportError:
    def load_dotenv(**kwargs) -> bool:  # type: ignore[misc]
        return False

try:
    import snowflake.connector  # type: ignore[import-untyped]
except ImportError as _sf_err:
    raise ImportError(
        "snowflake-connector-python is required: pip install snowflake-connector-python"
    ) from _sf_err


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
        return self._load_env_overrides(config)

    def _load_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        sf = config.setdefault("snowflake", {})
        sf["account"] = os.getenv("SNOWFLAKE_ACCOUNT", sf.get("account"))
        sf["user"] = os.getenv("SNOWFLAKE_USER", sf.get("user"))
        sf["password"] = os.getenv("SNOWFLAKE_PASSWORD", sf.get("password"))
        sf["warehouse"] = os.getenv("SNOWFLAKE_WAREHOUSE", sf.get("warehouse"))
        sf["database"] = os.getenv("SNOWFLAKE_DATABASE", sf.get("database"))
        sf["schema"] = os.getenv("SNOWFLAKE_SCHEMA", sf.get("schema"))
        sf["role"] = os.getenv("SNOWFLAKE_ROLE", sf.get("role"))
        sf["authenticator"] = os.getenv(
            "SNOWFLAKE_AUTHENTICATOR", sf.get("authenticator", "externalbrowser")
        )
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
# SnowflakeLoader
# ─────────────────────────────────────────────────────────────────────────────

class SnowflakeLoader:
    """Loads data from Snowflake into a pandas DataFrame."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self._conn = None

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self):
        sf = self.config.get("snowflake", {})
        connect_kwargs: Dict[str, Any] = {
            "account": sf.get("account", ""),
            "user": sf.get("user", ""),
            "warehouse": sf.get("warehouse") or None,
            "database": sf.get("database") or None,
            "schema": sf.get("schema") or None,
        }
        # Role is optional
        role = sf.get("role")
        if role:
            connect_kwargs["role"] = role

        authenticator = sf.get("authenticator", "")
        if authenticator and authenticator.lower() == "externalbrowser":
            connect_kwargs["authenticator"] = "externalbrowser"
        else:
            connect_kwargs["password"] = sf.get("password", "")
            if authenticator:
                connect_kwargs["authenticator"] = authenticator

        self._conn = snowflake.connector.connect(**connect_kwargs)

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    # ── Query helpers ──────────────────────────────────────────────────────

    def _table_ref(self) -> str:
        db = self.config.get("snowflake.database", "")
        schema = self.config.get("snowflake.schema", "")
        table = self.config.get("data_source.table", "")
        if db and schema:
            return f'"{db}"."{schema}"."{table}"'
        if schema:
            return f'"{schema}"."{table}"'
        return f'"{table}"'

    def _execute(self, sql: str) -> pd.DataFrame:
        if self._conn is None:
            raise RuntimeError("Not connected — call connect() first")
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            col_names = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=col_names)
        finally:
            cur.close()

    # ── Full load ──────────────────────────────────────────────────────────

    def load_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        sql = f"SELECT * FROM {self._table_ref()}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self._execute(sql)

    # ── Delta load ─────────────────────────────────────────────────────────

    def load_delta(self, limit: Optional[int] = None) -> pd.DataFrame:
        checkpoint_column = self.config.get("delta_loading.checkpoint_column", "")
        checkpoint_file = Path(
            self.config.get("delta_loading.checkpoint_file", "data/delta_checkpoint.json")
        )

        last_value = self._read_checkpoint(checkpoint_file)

        sql = f'SELECT * FROM {self._table_ref()}'
        if last_value is not None:
            # Use parameterised quoting — values are column values, not SQL
            safe_value = str(last_value).replace("'", "''")
            sql += f""" WHERE "{checkpoint_column}" > '{safe_value}'"""
        if limit:
            sql += f" LIMIT {int(limit)}"

        df = self._execute(sql)

        if not df.empty and checkpoint_column in df.columns:
            self._write_checkpoint(checkpoint_file, df[checkpoint_column].max())

        return df

    # ── Checkpoint helpers ─────────────────────────────────────────────────

    @staticmethod
    def _read_checkpoint(path: Path) -> Optional[Any]:
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("last_value")
        return None

    @staticmethod
    def _write_checkpoint(path: Path, value: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"last_value": value, "updated_at": datetime.now().isoformat()}, f)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load revenue data from Snowflake and save to CSV."
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yml",
        help="Path to YAML config file (default: config/default.yml)",
    )
    parser.add_argument(
        "--mode", type=str, choices=["full", "delta"], default="delta",
        help="Load mode: 'full' or 'delta' (default: delta)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (overrides config output.raw_data_file)",
    )
    parser.add_argument(
        "--test-connection", action="store_true", dest="test_connection",
        help="Test Snowflake connection and exit without loading data",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit rows fetched (useful for testing)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── Configuration ─────────────────────────────────────────────────────
    config = ConfigManager(config_path=args.config)
    loader = SnowflakeLoader(config)

    # ── Test connection only ───────────────────────────────────────────────
    if args.test_connection:
        try:
            loader.connect()
            loader.close()
            print("Connection successful.")
        except Exception as exc:
            print(f"Connection failed: {exc}", file=sys.stderr)
            return 1
        return 0

    # ── Connect ────────────────────────────────────────────────────────────
    try:
        loader.connect()
    except Exception as exc:
        print(f"Failed to connect to Snowflake: {exc}", file=sys.stderr)
        return 1

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        if args.mode == "full":
            df = loader.load_data(limit=args.limit)
        else:
            df = loader.load_delta(limit=args.limit)
    except Exception as exc:
        print(f"Data load failed: {exc}", file=sys.stderr)
        loader.close()
        return 1
    finally:
        loader.close()

    if df.empty:
        print("No data returned — nothing to save.")
        return 0

    # ── Save output ────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = config.get("output.raw_data_file", f"results/raw_data_{ts}.csv")
    output_path = Path(args.output or default_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} rows x {len(df.columns)} columns → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
