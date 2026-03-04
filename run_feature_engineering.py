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
import argparse
import logging
from pathlib import Path

# Resolve package root (same directory as this script)
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import pandas as pd

from rct_forecast.config.config_manager import ConfigManager
from rct_forecast.feature_engineering.feature_engineer import FeatureEngineer
from rct_forecast.validation.validator import ValidationManager
from rct_forecast.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RCT feature engineering pipeline on a prepared CSV."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yml",
        help="Path to YAML config file (default: config/default.yml)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV (output of data preparation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path for the engineered output CSV. "
            "Defaults to <input_stem>_engineered.csv in the same directory."
        ),
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default=None,
        dest="group_by",
        help=(
            "Column name to use as the segment group-by key "
            "(e.g. 'Roll Up Shop' or 'Profit Center'). "
            "When set, lag/rolling features are computed per segment."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-engineering validation step",
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Print the list of engineered feature columns added and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── Configuration ────────────────────────────────────────────────────
    config = ConfigManager(config_path=args.config)
    logger = setup_logging(config)
    logger.info("Feature Engineering Script")
    logger.info("  Config  : %s", args.config)
    logger.info("  Input   : %s", args.input)

    # ── Load input data ──────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    logger.info("Loading input data from: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows x %d columns", *df.shape)

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

    if group_by:
        logger.info("Group-by column: %s  (%d unique values)", group_by, df[group_by].nunique())
    else:
        logger.info("Group-by column: None (global fit)")

    # ── Feature Engineering ───────────────────────────────────────────────
    logger.info("Initialising FeatureEngineer from config")
    feature_engineer = FeatureEngineer(config)

    logger.info("Fitting and transforming data")
    engineered_df = feature_engineer.fit_transform(df, group_by=group_by)

    original_cols = set(df.columns)
    new_cols = [c for c in engineered_df.columns if c not in original_cols]
    logger.info(
        "Feature engineering complete: %d -> %d columns  (%d new features)",
        len(original_cols),
        len(engineered_df.columns),
        len(new_cols),
    )

    if args.show_features:
        print("\nNew feature columns added:")
        for col in new_cols:
            print(f"  {col}")
        print()

    # ── Post-engineering validation ───────────────────────────────────────
    if not args.skip_validation:
        logger.info("Running post-engineering validation")
        validator = ValidationManager(config)
        fe_validation = validator.validate(engineered_df, stage="feature_engineering")
        if fe_validation["passed"]:
            logger.info("Validation passed")
        else:
            logger.warning("Validation failed: %s", fe_validation)

    # ── Save output ───────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_engineered.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_csv(output_path, index=False)
    logger.info(
        "Engineered data saved to: %s  (%d rows x %d columns)",
        output_path,
        *engineered_df.shape,
    )

    # ── Transformer summary ───────────────────────────────────────────────
    fi_info = feature_engineer.get_feature_importance_info()
    logger.info(
        "Transformers applied: %d  |  %s",
        fi_info["total_transformers"],
        ", ".join(fi_info["transformer_names"]),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
