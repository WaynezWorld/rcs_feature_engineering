"""
Daily top-down temporal disaggregation for revenue forecasting.

Phase 1+2 implementation:
    monthly_total_forecast -> daily shape weights -> coherent daily forecasts

Phase 2 adds **current-month nowcasting**: the system uses MTD actuals
and the shape model's expected-share to estimate a month-end total before
allocating the remaining days.

The module is intentionally simple and deterministic:
  - No heavy ML model for daily series.
  - Shape weights are learned from historical within-month proportions
    using a (month-of-year, day-of-month, day-of-week) lookup with
    ordered fallbacks.
  - Coherence invariant: sum(daily_pred) == month_total_final  (to float tol).
"""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DailyAllocationConfig:
    """Typed wrapper around the ``forecasting.daily_allocation`` YAML block."""

    method: str = "shape_topdown"
    history_years: int = 3
    min_history_months: int = 18
    weight_model: str = "dow_dom_moy"
    negative_handling_for_weights: str = "clip_to_zero"
    smoothing_epsilon: float = 1e-6
    fallback: List[str] = field(default_factory=lambda: ["dow", "uniform"])

    # Top-level daily settings (not nested)
    daily_enabled: bool = False
    as_of_date: str = "auto"
    posting_lag_days: int = 1

    # Current-month nowcast settings
    current_month_strategy: str = "fixed"          # fixed | mtd_ratio | blend
    blend_alpha: float = 0.7                       # 1.0 = all model, 0.0 = all mtd
    min_expected_share: float = 0.05               # floor to avoid blow-ups
    max_multiplier: float = 2.0                    # cap mtd_ratio total
    allow_negative_totals: bool = True

    # Strategy-scheduling guardrails
    # -----------------------------------------------------------------------
    # If as_of_day <= strategy_early_cutoff_days OR expected_share is below
    # strategy_min_share_for_blend, the configured strategy is overridden to
    # "fixed" because there is too little MTD signal for blend/mtd_ratio.
    strategy_early_cutoff_days: int = 5       # day-of-month threshold
    strategy_min_share_for_blend: float = 0.15  # expected-share threshold

    def resolve_strategy(self, as_of_day: int, expected_share: float) -> str:
        """
        Apply guardrails and return the *effective* strategy to use.

        Rules (evaluated in order):
          1. Configured strategy is already ``"fixed"`` -> return ``"fixed"``.
          2. ``as_of_day <= strategy_early_cutoff_days`` -> force ``"fixed"``.
          3. ``expected_share < strategy_min_share_for_blend`` -> force ``"fixed"``.
          4. Otherwise return the configured strategy unchanged.
        """
        configured = self.current_month_strategy
        if configured == "fixed":
            return "fixed"
        if as_of_day <= self.strategy_early_cutoff_days:
            return "fixed"
        if expected_share < self.strategy_min_share_for_blend:
            return "fixed"
        return configured

    @classmethod
    def from_config(cls, config) -> "DailyAllocationConfig":
        """Build from a :class:`ConfigManager` instance."""
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
            allow_negative_totals=cmt.get("allow_negative_totals", defaults.allow_negative_totals),
            strategy_early_cutoff_days=cmt.get(
                "strategy_early_cutoff_days", defaults.strategy_early_cutoff_days
            ),
            strategy_min_share_for_blend=cmt.get(
                "strategy_min_share_for_blend", defaults.strategy_min_share_for_blend
            ),
        )


# ---------------------------------------------------------------------------
# Data-prep helpers
# ---------------------------------------------------------------------------

def build_daily_totals(
    raw_df: pd.DataFrame,
    date_column: str = "TS",
    amount_column: str = "Actual",
    measure_column: Optional[str] = "Measure",
    measure_value: str = "MTD",
) -> pd.DataFrame:
    """
    Aggregate the raw line-item data to **daily** totals by posting date.

    Parameters
    ----------
    raw_df : DataFrame
        Raw data as loaded from Snowflake (line-item level).
    date_column : str
        Column that contains the posting date (day-level).
        In the sandbox table this is ``TS`` (end-of-period timestamp).
    amount_column : str
        Revenue column (e.g. ``"Actual"``).
    measure_column, measure_value : str
        If the source has a Measure dimension, keep only ``measure_value``
        (typically ``"MTD"``).

    Returns
    -------
    DataFrame with columns ``[date, actual]``, one row per calendar day,
    sorted by date.  Values are dollar-scale (no log transform).
    """
    df = raw_df.copy()

    # Optional Measure filter
    if measure_column and measure_column in df.columns and measure_value:
        df = df[df[measure_column] == measure_value]

    # Parse date
    if date_column not in df.columns:
        raise KeyError(f"Date column '{date_column}' not found.  Available: {list(df.columns)}")

    df["_date"] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=["_date"])
    df["_date"] = df["_date"].dt.normalize()  # drop HH:MM:SS

    # Coerce amount
    df[amount_column] = pd.to_numeric(df[amount_column], errors="coerce")

    daily = (
        df.groupby("_date", as_index=False)[amount_column]
        .sum()
        .rename(columns={"_date": "date", amount_column: "actual"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    logger.info(
        "build_daily_totals: %d raw rows -> %d daily totals (%s - %s)",
        len(raw_df), len(daily),
        daily["date"].min().strftime("%Y-%m-%d") if len(daily) else "N/A",
        daily["date"].max().strftime("%Y-%m-%d") if len(daily) else "N/A",
    )
    return daily


def daily_to_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up a daily-totals frame to monthly totals.

    Parameters
    ----------
    daily_df : DataFrame  with columns ``[date, actual]``

    Returns
    -------
    DataFrame ``[month_start, actual_month_total]`` sorted by month_start.
    """
    df = daily_df.copy()
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("month_start", as_index=False)["actual"]
        .sum()
        .rename(columns={"actual": "actual_month_total"})
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
    """
    Roll up a daily forecast frame to **ISO-week** (or configurable) weekly totals.

    Weekly sums are always coherent with the input: the total across all weeks
    equals the total across all days for any given group.

    Parameters
    ----------
    daily_df : DataFrame
        Daily forecast output (e.g. from ``DailyAllocator.predict_daily()``).
        Must contain ``date_column`` and ``value_column``.
    date_column : str
        Name of the date column (default ``"date"``).
    value_column : str
        Numeric column to aggregate (default ``"y_pred"``).
    week_start : str
        Day abbreviation that begins each week.  Standard pandas ``freq``
        alias anchor: ``"Mon"`` (ISO default), ``"Sun"``, ``"Sat"`` etc.
        Passed directly to ``pd.Grouper(freq="W-<week_start>")``.
    segment_columns : list of str, optional
        Additional columns to group by (e.g. ``["roll_up_shop"]``).
        If *None*, the output is a single aggregate series.

    Returns
    -------
    DataFrame with columns:
        ``week_start, week_end, <value_column>`` (+ any segment columns).
        ``week_start`` is the Monday (or configured anchor) of each ISO week.
        ``week_end``   is the Sunday (or 6 days after week_start).
        Values are summed within each (week, segment) group.

    Notes
    -----
    Partial-week rows at the head/tail of the series are included.
    Coherence invariant: ``weekly[value_column].sum() == daily[value_column].sum()``
    (within float tolerance) for any segment.
    """
    if daily_df.empty:
        cols = (segment_columns or []) + ["week_start", "week_end", value_column]
        return pd.DataFrame(columns=cols)

    df = daily_df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

    # pandas W-{anchor} groups each week so that {anchor} is the LAST day.
    # Map the user's desired week-start day to the correct anchor (= start - 1 day).
    _start_to_end_anchor: Dict[str, str] = {
        "MON": "SUN",  # ISO Mon–Sun  → label on Sunday
        "TUE": "MON",
        "WED": "TUE",
        "THU": "WED",
        "FRI": "THU",
        "SAT": "FRI",
        "SUN": "SAT",  # Sun–Sat      → label on Saturday
    }
    anchor = _start_to_end_anchor.get(week_start.upper(), "SUN")
    freq = f"W-{anchor}"

    grouper_keys: List = [pd.Grouper(key=date_column, freq=freq)]
    if segment_columns:
        grouper_keys = (segment_columns or []) + grouper_keys  # type: ignore[assignment]

    weekly = (
        df.groupby(grouper_keys, as_index=False)[value_column]
        .sum()
    )

    # The freq grouper labels the week by its *last* day (the anchor day).
    # Rename that to week_end and derive week_start.
    weekly = weekly.rename(columns={date_column: "week_end"})
    weekly["week_end"] = pd.to_datetime(weekly["week_end"])
    weekly["week_start"] = weekly["week_end"] - pd.Timedelta(days=6)

    # Re-order columns: segment cols | week_start | week_end | value
    out_cols = (segment_columns or []) + ["week_start", "week_end", value_column]
    weekly = weekly[out_cols].sort_values(
        (segment_columns or []) + ["week_start"]
    ).reset_index(drop=True)

    logger.debug(
        "daily_to_weekly: %d daily rows -> %d weekly rows (%s - %s)",
        len(daily_df), len(weekly),
        weekly["week_start"].min().strftime("%Y-%m-%d") if len(weekly) else "N/A",
        weekly["week_end"].max().strftime("%Y-%m-%d") if len(weekly) else "N/A",
    )
    return weekly


# ---------------------------------------------------------------------------
# Shape-weight model
# ---------------------------------------------------------------------------

def _build_shape_table(
    daily_history: pd.DataFrame,
    cfg: DailyAllocationConfig,
) -> pd.DataFrame:
    """
    Build a lookup table of average daily share keyed by
    ``(month_of_year, day_of_month, day_of_week)``.

    Each historical month contributes proportional shares where
    the denominator is ``sum(clipped_daily_actual)`` for that month.
    """
    df = daily_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["moy"] = df["date"].dt.month
    df["dom"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek  # Mon=0 .. Sun=6
    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # Optional clip negatives when computing proportions
    if cfg.negative_handling_for_weights == "clip_to_zero":
        df["actual_pos"] = df["actual"].clip(lower=0)
    else:
        df["actual_pos"] = df["actual"]

    # Month totals (positive part)
    month_totals = df.groupby("month_start")["actual_pos"].sum().rename("month_pos_total")
    df = df.merge(month_totals, on="month_start", how="left")

    # Drop months where month_pos_total is <= 0
    df = df[df["month_pos_total"] > 0].copy()

    if df.empty:
        return pd.DataFrame(columns=["moy", "dom", "dow", "share"])

    df["share"] = df["actual_pos"] / df["month_pos_total"]

    # Group-level mean share
    shape = (
        df.groupby(["moy", "dom", "dow"], as_index=False)["share"]
        .mean()
    )
    return shape


def _lookup_weight(
    shape: pd.DataFrame,
    moy: int,
    dom: int,
    dow: int,
    cfg: DailyAllocationConfig,
) -> float:
    """Hierarchical lookup with ordered fallback."""
    if shape.empty:
        return 1.0  # will be normalised later

    # Level 1: (moy, dom, dow)
    mask = (shape["moy"] == moy) & (shape["dom"] == dom) & (shape["dow"] == dow)
    vals = shape.loc[mask, "share"]
    if len(vals) > 0:
        return float(vals.mean())

    # Iterate through fallbacks
    for fb in cfg.fallback:
        if fb == "dow":
            mask_dow = shape["dow"] == dow
            vals = shape.loc[mask_dow, "share"]
            if len(vals) > 0:
                return float(vals.mean())
        elif fb == "dom":
            mask_dom = shape["dom"] == dom
            vals = shape.loc[mask_dom, "share"]
            if len(vals) > 0:
                return float(vals.mean())
        elif fb == "uniform":
            return 1.0  # normalised later

    # Ultimate fallback
    return 1.0


def _get_leakage_safe_history(
    daily_history: pd.DataFrame,
    target_month_start: pd.Timestamp,
    cfg: DailyAllocationConfig,
) -> pd.DataFrame:
    """Return history strictly before ``target_month_start``, bounded by
    ``cfg.history_years`` lookback."""
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
    """
    Compute a raw (pre-normalisation) score for **every** day in the target
    month.  Scores are lookup-value + epsilon and come from a single shape
    table so that within-month shares are comparable.

    Parameters
    ----------
    target_month_start : Timestamp
        First day of the month.
    daily_history : DataFrame ``[date, actual]``
        Full daily history (will be filtered internally for leakage safety).
    cfg : DailyAllocationConfig

    Returns
    -------
    dict  ``{Timestamp -> float}`` for every calendar day in the month.
    """
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
    """
    Normalise raw scores for a subset of dates so they sum to 1.

    Parameters
    ----------
    dates : sequence of Timestamps
        The subset of days to normalise over.
    raw_score_map : dict
        ``{date -> raw_score}`` produced by :func:`get_raw_scores_for_month`.

    Returns
    -------
    1-D numpy array of length ``len(dates)``, summing to 1.
    """
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
    """
    Compute normalised daily allocation weights for ``day_dates`` within
    a single month.

    This is the **public convenience wrapper** that builds raw scores for
    the full month and normalises over the requested subset.

    Parameters
    ----------
    target_month_start : Timestamp
        First day of the target month.
    day_dates : sequence of Timestamps
        The specific calendar days for which we need weights.
    daily_history : DataFrame ``[date, actual]``
        Historical daily totals used for shape estimation.
    cfg : DailyAllocationConfig

    Returns
    -------
    1-D numpy array of length ``len(day_dates)`` that sums to 1.
    """
    if len(day_dates) == 0:
        return np.array([], dtype=float)
    raw_scores = get_raw_scores_for_month(target_month_start, daily_history, cfg)
    return normalize_scores(day_dates, raw_scores)


# ---------------------------------------------------------------------------
# Current-month nowcast helper
# ---------------------------------------------------------------------------

def compute_expected_mtd_share(
    raw_score_map: Dict[pd.Timestamp, float],
    observed_dates: Sequence[pd.Timestamp],
    min_expected_share: float = 0.05,
) -> float:
    """
    From a single set of raw scores for the full month, compute the
    fraction of the month's "expected shape" that has been observed.

    ``expected_share = sum(scores for observed_dates) / sum(all scores)``

    Floored at ``min_expected_share`` to prevent blow-ups when very few
    days have been observed.
    """
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
    """
    Blend the model's monthly forecast with an MTD-ratio estimate to
    produce ``month_total_final``.

    Strategies:
      - ``"fixed"``:     use the model total unmodified.
      - ``"mtd_ratio"``: ``MTD_actual / expected_share``,
                         optionally capped by ``max_multiplier * model``.
      - ``"blend"``:     ``alpha * model + (1 - alpha) * mtd_ratio``.
    """
    strategy = cfg.current_month_strategy

    # MTD-ratio estimate
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
        logger.warning("Unknown current_month_total strategy '%s'; using fixed.", strategy)
        total = month_total_model

    if not cfg.allow_negative_totals:
        total = max(total, 0.0)

    return total


# ---------------------------------------------------------------------------
# Daily forecast generation
# ---------------------------------------------------------------------------

class DailyAllocator:
    """
    Top-down daily forecast allocator with current-month nowcasting.

    Given monthly forecast totals and (optionally) MTD actuals for the
    current month, produce coherent daily forecasts.

    Phase 2 adds MTD-aware month-end estimation for the current month via
    ``current_month_total.strategy`` (fixed / mtd_ratio / blend).

    Usage::

        allocator = DailyAllocator(config)
        allocator.set_daily_history(daily_df)
        allocator.set_monthly_forecasts(monthly_forecast_df)
        daily_forecast = allocator.predict_daily(as_of_date, horizon_months=12)
    """

    def __init__(self, config):
        self.cfg = DailyAllocationConfig.from_config(config)
        self._daily_history: Optional[pd.DataFrame] = None
        self._monthly_forecasts: Optional[pd.DataFrame] = None
        # Populated after predict_daily(); maps month_start -> month_total_final
        self._month_total_final: Dict[pd.Timestamp, float] = {}

    # -- setters ----------------------------------------------------------

    def set_daily_history(self, daily_df: pd.DataFrame) -> None:
        """
        Provide the historical daily totals ``[date, actual]``.
        """
        required = {"date", "actual"}
        missing = required - set(daily_df.columns)
        if missing:
            raise ValueError(f"daily_df missing columns: {missing}")
        self._daily_history = daily_df.copy()
        self._daily_history["date"] = pd.to_datetime(self._daily_history["date"])
        logger.info(
            "DailyAllocator: loaded %d daily history rows (%s - %s)",
            len(self._daily_history),
            self._daily_history["date"].min().strftime("%Y-%m-%d"),
            self._daily_history["date"].max().strftime("%Y-%m-%d"),
        )

    def set_monthly_forecasts(self, monthly_forecast_df: pd.DataFrame) -> None:
        """
        Provide monthly forecast totals ``[month_start, forecast_total]``.
        """
        required = {"month_start", "forecast_total"}
        missing = required - set(monthly_forecast_df.columns)
        if missing:
            raise ValueError(f"monthly_forecast_df missing columns: {missing}")
        self._monthly_forecasts = monthly_forecast_df.copy()
        self._monthly_forecasts["month_start"] = pd.to_datetime(
            self._monthly_forecasts["month_start"]
        )
        logger.info(
            "DailyAllocator: loaded %d monthly forecasts (%s - %s)",
            len(self._monthly_forecasts),
            self._monthly_forecasts["month_start"].min().strftime("%Y-%m"),
            self._monthly_forecasts["month_start"].max().strftime("%Y-%m"),
        )

    # -- core prediction ---------------------------------------------------

    def predict_daily(
        self,
        as_of_date: Optional[str] = None,
        horizon_months: int = 12,
    ) -> pd.DataFrame:
        """
        Produce daily forecasts coherent with monthly totals.

        Current month uses nowcast logic (strategy-dependent month_total_final).
        Future months use the raw monthly model forecast.

        Returns
        -------
        DataFrame with columns:
            date, month_start, y_true, y_pred, forecast_amount, is_actual,
            as_of_date, horizon_day, horizon_month, month_total_model,
            month_total_final, strategy, expected_mtd_share,
            multiplier_applied
        """
        if self._daily_history is None:
            raise RuntimeError("Call set_daily_history() before predict_daily()")
        if self._monthly_forecasts is None:
            raise RuntimeError("Call set_monthly_forecasts() before predict_daily()")

        # Resolve as-of date
        if as_of_date is None or as_of_date == "auto":
            aod = pd.Timestamp.today().normalize()
        else:
            aod = pd.Timestamp(as_of_date)

        cutoff_date = aod - pd.Timedelta(days=self.cfg.posting_lag_days)
        logger.info(
            "predict_daily: as_of_date=%s  cutoff_date=%s  posting_lag=%d",
            aod.strftime("%Y-%m-%d"), cutoff_date.strftime("%Y-%m-%d"),
            self.cfg.posting_lag_days,
        )

        # Build the list of target months
        current_month_start = aod.to_period("M").to_timestamp()
        target_months = pd.date_range(
            current_month_start,
            periods=1 + horizon_months,
            freq="MS",
        )

        self._month_total_final = {}
        rows: List[Dict] = []

        for month_start in target_months:
            month_end = month_start + pd.offsets.MonthEnd(0)
            is_current_month = (month_start == current_month_start)

            # Monthly model forecast
            forecast_total = self._get_monthly_forecast(month_start)
            if forecast_total is None:
                logger.warning(
                    "No monthly forecast for %s - skipping.",
                    month_start.strftime("%Y-%m"),
                )
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
                    self._allocate_future_month(
                        month_start, all_days, forecast_total, aod
                    )
                )

        result = pd.DataFrame(rows)
        if result.empty:
            logger.warning("predict_daily produced no rows")
            return result

        # Assign horizon columns
        result["horizon_day"] = (result["date"] - aod).dt.days
        result["horizon_month"] = (
            (result["month_start"].dt.to_period("M") - aod.to_period("M"))
            .apply(lambda x: x.n)
        )

        # Stamp month_total_final for downstream consumers
        result["month_total_final"] = result["month_start"].map(self._month_total_final)

        # Verify coherence
        self._check_coherence(result)
        return result

    # -- private helpers ---------------------------------------------------

    def _get_monthly_forecast(self, month_start: pd.Timestamp) -> Optional[float]:
        """Look up the monthly forecast total for a given month."""
        mf = self._monthly_forecasts
        match = mf.loc[mf["month_start"] == month_start, "forecast_total"]
        if len(match) == 0:
            return None
        return float(match.iloc[0])

    def _get_daily_actual(self, d: pd.Timestamp) -> Optional[float]:
        """Retrieve the actual daily value if it exists in history."""
        hist = self._daily_history
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
        """
        For the current (partial) month, apply the nowcast strategy:
          1. Gather MTD actuals (days <= cutoff_date).
          2. Compute raw_score_map for the full month (leakage-safe).
          3. Derive expected_mtd_share from the same raw scores.
          4. Compute month_total_final via the configured strategy.
          5. remaining_total = month_total_final - mtd_actual.
          6. Allocate remaining_total across remaining days using
             normalised scores from the same raw_score_map.

        Each row carries both ``y_pred`` (best available: actual for past days,
        shape-allocated forecast for future) and ``forecast_amount`` (what the
        model *would have* allocated to that day regardless of actuals).
        """
        actual_days = all_days[all_days <= cutoff_date]
        remaining_days = all_days[all_days > cutoff_date]

        # Build raw scores for the full month from a single shape table
        raw_score_map = get_raw_scores_for_month(
            month_start, self._daily_history, self.cfg
        )

        # Pre-compute shape-allocated model forecast for ALL days
        # This gives the "what model would have said" amount per day.
        full_weights = normalize_scores(all_days, raw_score_map)
        model_daily_alloc = forecast_total * full_weights
        _model_alloc_map = dict(zip(all_days, model_daily_alloc))

        # Collect MTD actuals
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
            # Fully observed: month_total_final = actual total
            self._month_total_final[month_start] = mtd_actual
            # Back-fill strategy metadata on already-built rows
            for r in rows:
                r["strategy"] = self.cfg.current_month_strategy
                r["effective_strategy"] = self.cfg.current_month_strategy
                r["expected_mtd_share"] = 1.0
                r["multiplier_applied"] = None
            logger.info(
                "Current month %s: fully observed, MTD_actual=%.2f",
                month_start.strftime("%Y-%m"), mtd_actual,
            )
            return rows

        # Compute expected MTD share
        expected_share = compute_expected_mtd_share(
            raw_score_map, actual_days, self.cfg.min_expected_share
        )

        # Apply strategy guardrails: too early in month or MTD share too low
        # means there is insufficient signal for blend/mtd_ratio.
        effective_strategy = self.cfg.resolve_strategy(aod.day, expected_share)
        if effective_strategy != self.cfg.current_month_strategy:
            logger.info(
                "Current month %s: strategy guardrail applied — "
                "configured=%s -> effective=%s  "
                "(as_of_day=%d, expected_share=%.4f)",
                month_start.strftime("%Y-%m"),
                self.cfg.current_month_strategy, effective_strategy,
                aod.day, expected_share,
            )

        # Compute multiplier_applied (for audit trail)
        multiplier_applied = None
        if effective_strategy in ("mtd_ratio", "blend") and expected_share > 0:
            mtd_ratio_est = mtd_actual / expected_share
            if self.cfg.max_multiplier and forecast_total != 0:
                cap = abs(forecast_total) * self.cfg.max_multiplier
                if abs(mtd_ratio_est) > cap:
                    multiplier_applied = self.cfg.max_multiplier

        # Nowcast: compute month_total_final using effective strategy
        orig_strategy = self.cfg.current_month_strategy
        self.cfg.current_month_strategy = effective_strategy
        month_total_final = compute_month_total_final(
            mtd_actual, forecast_total, expected_share, self.cfg
        )
        self.cfg.current_month_strategy = orig_strategy
        self._month_total_final[month_start] = month_total_final

        remaining_total = month_total_final - mtd_actual

        logger.info(
            "Current month %s: strategy=%s (effective=%s)  MTD_actual=%.2f  "
            "expected_share=%.4f  model_total=%.2f  month_total_final=%.2f  "
            "remaining=%.2f  actual_days=%d  remaining_days=%d",
            month_start.strftime("%Y-%m"), self.cfg.current_month_strategy,
            effective_strategy, mtd_actual, expected_share, forecast_total,
            month_total_final, remaining_total, len(actual_days), len(remaining_days),
        )

        # Back-fill strategy metadata on actual-day rows
        for r in rows:
            r["strategy"] = self.cfg.current_month_strategy
            r["effective_strategy"] = effective_strategy
            r["expected_mtd_share"] = expected_share
            r["multiplier_applied"] = multiplier_applied

        # Allocate remaining days using normalised scores from the same map
        weights = normalize_scores(remaining_days, raw_score_map)
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
        """For a fully future month: distribute the total across all days."""
        raw_scores = get_raw_scores_for_month(
            month_start, self._daily_history, self.cfg
        )
        weights = normalize_scores(all_days, raw_scores)
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
        """
        Verify daily-to-monthly coherence, gated by month type.

        Three month types and their invariants:

        1. **Fully future** (all days forecasted, zero actuals):
           ``sum(y_pred) == monthly_forecast_total``
        2. **Partially observed** (some actuals, some forecasted):
           ``sum(y_pred) == month_total_final`` (the nowcasted total)
        3. **Fully observed** (all days actual, zero forecasted):
           ``sum(y_pred) == sum(y_true) == actual_month_total``
           No requirement that it equals the monthly forecast total;
           the gap is the forecast error.
        """
        for month_start, grp in result.groupby("month_start"):
            has_forecasted = (grp["is_actual"] == False).any()
            daily_sum = grp["y_pred"].sum()

            if has_forecasted:
                # Types 1 & 2: coherence to month_total_final
                expected = self._month_total_final.get(month_start)
                if expected is None:
                    expected = self._get_monthly_forecast(month_start)
                if expected is None:
                    continue
                diff = abs(daily_sum - expected)
                if diff > tol:
                    logger.warning(
                        "COHERENCE VIOLATION %s: sum(y_pred)=%.4f  "
                        "month_total_final=%.4f  diff=%.4f",
                        month_start.strftime("%Y-%m"), daily_sum, expected, diff,
                    )
                else:
                    logger.debug(
                        "Coherence OK %s (forecast): diff=%.6f",
                        month_start.strftime("%Y-%m"), diff,
                    )
            else:
                # Type 3: fully observed - sum should equal actuals
                actual_sum = grp["y_true"].sum()
                diff = abs(daily_sum - actual_sum)
                if diff > tol:
                    logger.warning(
                        "ACTUAL-SUM MISMATCH %s: sum(y_pred)=%.4f  "
                        "sum(y_true)=%.4f  diff=%.4f",
                        month_start.strftime("%Y-%m"), daily_sum, actual_sum, diff,
                    )
                else:
                    logger.debug(
                        "Coherence OK %s (fully observed): diff=%.6f",
                        month_start.strftime("%Y-%m"), diff,
                    )

    # -- backtest ----------------------------------------------------------

    def backtest_intramonth_daily(
        self,
        as_of_day_offsets: Sequence[int] = (5, 10, 15, 20),
        strategies: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        For each completed historical month, simulate partial-month
        forecasts at several ``as_of_day`` offsets and compute remainder
        WAPE, MAE, and full-month total error.

        Optionally evaluates multiple ``current_month_total.strategy``
        variants side-by-side.

        Parameters
        ----------
        as_of_day_offsets : sequence of int
            Day-of-month values to simulate (e.g. [5, 10, 15, 20]).
        strategies : sequence of str or None
            If provided, evaluates each strategy (e.g.
            ``["fixed", "mtd_ratio", "blend"]``).
            Defaults to the single configured strategy.

        Returns
        -------
        DataFrame with columns:
            month_start, as_of_day, strategy, mtd_actual, remaining_actual,
            month_actual_total, month_total_model, month_total_final,
            remaining_forecast, remainder_mae, remainder_wape,
            full_month_error, full_month_error_pct
        """
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

            # Build raw scores once per month (leakage-safe)
            raw_score_map = get_raw_scores_for_month(
                ms, self._daily_history, self.cfg
            )

            for k in as_of_day_offsets:
                # Nominal as-of calendar day (1-indexed)
                aod_nominal = ms + pd.Timedelta(days=k - 1)
                if aod_nominal > month_end:
                    continue  # as_of_day lies beyond end of this month

                # Simulate posting lag: only data through (aod - lag) is available.
                # This matches the behaviour of predict_daily() so the backtest
                # is production-faithful.
                cutoff = aod_nominal - pd.Timedelta(days=self.cfg.posting_lag_days)

                actual_days = all_days[all_days <= cutoff]
                remaining_days = all_days[all_days > cutoff]
                if len(remaining_days) == 0:
                    continue

                # MTD from history (respects lag-adjusted cutoff)
                mtd_vals = hist[hist["date"].isin(actual_days)]
                mtd_actual = mtd_vals["actual"].sum() if len(mtd_vals) > 0 else 0.0

                # Actual remaining
                rem_vals = hist[hist["date"].isin(remaining_days)]
                remaining_actual = rem_vals["actual"].sum() if len(rem_vals) > 0 else 0.0

                # Expected MTD share (from same raw scores)
                expected_share = compute_expected_mtd_share(
                    raw_score_map, actual_days, self.cfg.min_expected_share
                )

                for strat in strategies:
                    # "scheduled" resolves to the guardrail-chosen strategy for
                    # this specific (as_of_day, expected_share) combination.
                    if strat == "scheduled":
                        effective_strat = self.cfg.resolve_strategy(k, expected_share)
                    else:
                        effective_strat = strat

                    # Override strategy temporarily
                    orig_strat = self.cfg.current_month_strategy
                    self.cfg.current_month_strategy = effective_strat

                    month_total_final = compute_month_total_final(
                        mtd_actual, forecast_total, expected_share, self.cfg
                    )
                    remaining_total_fc = month_total_final - mtd_actual

                    self.cfg.current_month_strategy = orig_strat

                    # Weights from the same raw_score_map
                    weights = normalize_scores(remaining_days, raw_score_map)
                    daily_preds = remaining_total_fc * weights

                    # Per-day actuals for remaining period
                    rem_actual_series = []
                    for d in remaining_days:
                        val = self._get_daily_actual(d)
                        rem_actual_series.append(val if val is not None else 0.0)
                    rem_actual_arr = np.array(rem_actual_series)

                    # Remainder-level metrics
                    abs_err = np.abs(daily_preds - rem_actual_arr)
                    remainder_mae = float(np.mean(abs_err))
                    sum_abs_actual = float(np.sum(np.abs(rem_actual_arr)))
                    remainder_wape = (
                        float(np.sum(abs_err) / sum_abs_actual * 100)
                        if sum_abs_actual > 0
                        else float(np.sum(abs_err))
                    )

                    # Full-month total error
                    full_month_error = month_total_final - month_actual_total
                    full_month_error_pct = (
                        (full_month_error / abs(month_actual_total) * 100)
                        if abs(month_actual_total) > 0
                        else 0.0
                    )

                    records.append({
                        "month_start": ms,
                        "as_of_day": k,
                        "strategy": strat,            # requested strategy name
                        "effective_strategy": effective_strat,  # after guardrails
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
