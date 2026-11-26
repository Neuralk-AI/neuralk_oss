# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0
#
# Ported from tabpfn-time-series (https://github.com/PriorLabs/tabpfn-time-series,
# Apache-2.0), tabpfn_time_series/features/ at commit 258a074c4ebe74636641c61db86df99e964cc477.

"""The time features of TabPFN-TS: a series as a table, one row per timestamp, each row's features known before the forecast.

Every generator takes and returns a frame indexed by ``(item_id, timestamp)``
holding every series, the context rows then the horizon rows (whose
``target`` is NaN), and groups by ``item_id`` for any per-series computation.
``featurize`` runs a list of them over a context and a horizon, as
TabPFN-TS's ``FeatureTransformer``; ``default_features`` is TabPFN-TS's
default list:

- ``KnownCovariateLagFeature``: lags {1, season} and a trailing mean over
  ``season`` steps of each known covariate (none without covariates);
- ``RunningIndexFeature``: the row's position in its series;
- ``CalendarFeature``: the year, and sin and cos of 8 calendar fields;
- ``AutoSeasonalFeature``: sin and cos of the running index over the 12
  strongest periods of the series' spectrum, zeros for fewer.

The formulas are TabPFN-TS's, quirks included (a calendar field's index is
divided by its period minus one; ``PeriodicSinCosineFeature`` counts rows
over the whole frame, not per series); they are what its forecasts were
evaluated with. gluonts' ``*_index`` time features are inlined as
``_CALENDAR_INDEX``, with the same values.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import fft
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

ITEM_ID = "item_id"
TIMESTAMP = "timestamp"
TARGET = "target"
# Marks the context rows of the frame the generators see.
IS_TRAIN = "_is_train"

DetrendType = Literal["first_diff", "loess", "linear", "constant"]


class FeatureGenerator(ABC):
    """A step adding columns to the ``(item_id, timestamp)``-indexed frame of every series."""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """``df`` with this generator's columns added."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate(df)

    def __repr__(self) -> str:
        return f"{type(self).__name__}_{self.__dict__}"


class RunningIndexFeature(FeatureGenerator):
    """``running_index``: the row's position in its series, from 0 at its first kept context row, continued over the horizon."""

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["running_index"] = df.groupby(level=ITEM_ID, sort=False).cumcount()
        return df


# gluonts.time_feature's zero-based ``<name>_index`` features.
_CALENDAR_INDEX: dict[str, Callable[[pd.DatetimeIndex], np.ndarray]] = {
    "second_of_minute": lambda t: np.asarray(t.second),
    "minute_of_hour": lambda t: np.asarray(t.minute),
    "hour_of_day": lambda t: np.asarray(t.hour),
    "day_of_week": lambda t: np.asarray(t.dayofweek),
    "day_of_month": lambda t: np.asarray(t.day) - 1,
    "day_of_year": lambda t: np.asarray(t.dayofyear) - 1,
    "week_of_year": lambda t: np.asarray(t.isocalendar().week, dtype=np.int64) - 1,
    "month_of_year": lambda t: np.asarray(t.month) - 1,
}


class CalendarFeature(FeatureGenerator):
    """The timestamp's ``components`` (the year by default) as they are, and sin and cos of each calendar field over its period.

    A field's zero-based index is divided by its period minus one (TabPFN-TS's
    convention: day of week 6 is angle 2 pi). A field mapped to None is added
    as its raw index.
    """

    def __init__(
        self,
        components: Sequence[str] | None = None,
        seasonal_features: dict[str, list[float] | None] | None = None,
    ) -> None:
        self.components = list(components or ["year"])
        self.seasonal_features = seasonal_features or {
            "second_of_minute": [60],
            "minute_of_hour": [60],
            "hour_of_day": [24],
            "day_of_week": [7],
            "day_of_month": [30.5],
            "day_of_year": [365],
            "week_of_year": [52],
            "month_of_year": [12],
        }

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        timestamps = pd.DatetimeIndex(df.index.get_level_values(TIMESTAMP))
        for component in self.components:
            df[component] = getattr(timestamps, component)
        for name, periods in self.seasonal_features.items():
            feature = _CALENDAR_INDEX[name](timestamps).astype(np.int32)
            if periods is None:
                df[name] = feature
                continue
            # Several periods write the same two columns, the last one kept (as TabPFN-TS).
            for period in periods:
                period = period - 1
                df[f"{name}_sin"] = np.sin(2 * np.pi * feature / period)
                df[f"{name}_cos"] = np.cos(2 * np.pi * feature / period)
        return df


class PeriodicSinCosineFeature(FeatureGenerator):
    """``sin_<suffix>``, ``cos_<suffix>``: sin and cos over each of ``periods`` of the row's position in the whole frame (not per series, as TabPFN-TS)."""

    def __init__(self, periods: Sequence[float], name_suffix: str | None = None) -> None:
        self.periods = list(periods)
        self.name_suffix = name_suffix

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = np.arange(len(df))
        columns = {}
        for i, period in enumerate(self.periods):
            suffix = f"{self.name_suffix}_{i}" if self.name_suffix else f"{period}"
            angle = 2 * np.pi * idx / period
            columns[f"sin_{suffix}"] = np.sin(angle)
            columns[f"cos_{suffix}"] = np.cos(angle)
        if not columns:
            return df.copy()
        return pd.concat([df, pd.DataFrame(columns, index=df.index)], axis=1)


class AutoSeasonalFeature(FeatureGenerator):
    """``sin_#i``, ``cos_#i`` for i < ``max_top_k``: sin and cos of the running index over each series' i-th strongest period, 0 where it has fewer.

    The periods come from the context's targets (``find_seasonal_periods``);
    ``config`` overrides the keys of ``DEFAULTS``.
    """

    DEFAULTS: dict[str, Any] = {
        "max_top_k": 12,
        "do_detrend": True,
        "detrend_type": "linear",
        "use_peaks_only": True,
        "apply_hann_window": True,
        "zero_padding_factor": 2,
        "round_to_closest_integer": True,
        "validate_with_acf": False,
        "sampling_interval": 1.0,
        "magnitude_threshold": 0.05,
        "relative_threshold": True,
        "exclude_zero": True,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if unknown := sorted(set(config or {}) - set(self.DEFAULTS)):
            raise ValueError(f"AutoSeasonalFeature has no option {unknown}")
        self.config = {**self.DEFAULTS, **(config or {})}
        if self.config["max_top_k"] < 1:
            logger.warning("max_top_k must be at least 1, setting to 1")
            self.config["max_top_k"] = 1
        if self.config["zero_padding_factor"] < 1:
            logger.warning("zero_padding_factor must be at least 1, setting to 1")
            self.config["zero_padding_factor"] = 1
        if self.config["detrend_type"] not in ("first_diff", "loess", "linear", "constant"):
            logger.warning(f"Invalid detrend_type: {self.config['detrend_type']}, using 'linear'")
            self.config["detrend_type"] = "linear"

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        max_top_k = self.config["max_top_k"]
        pos = df.groupby(level=ITEM_ID, sort=False).cumcount().to_numpy()
        # The codes follow the sort=False group order of the loop below.
        codes, _ = pd.factorize(df.index.get_level_values(ITEM_ID), sort=False)
        n_series = int(codes.max()) + 1 if n else 0
        periods_by_series = np.full((n_series, max_top_k), np.nan)
        for code, (_, target) in enumerate(df[TARGET].groupby(level=ITEM_ID, sort=False)):
            periods = [p for p, _ in find_seasonal_periods(target, **self.config)]
            for i, period in enumerate(periods[:max_top_k]):
                periods_by_series[code, i] = period
        period_per_row = periods_by_series[codes]

        with np.errstate(divide="ignore", invalid="ignore"):
            angle = 2 * np.pi * pos[:, None] / period_per_row
        valid = ~np.isnan(period_per_row)
        sin = np.where(valid, np.sin(angle), 0.0)
        cos = np.where(valid, np.cos(angle), 0.0)
        columns = {}
        for i in range(max_top_k):
            columns[f"sin_#{i}"] = sin[:, i]
            columns[f"cos_#{i}"] = cos[:, i]
        return pd.concat([df, pd.DataFrame(columns, index=df.index)], axis=1)


def find_seasonal_periods(
    target_values: pd.Series | np.ndarray,
    max_top_k: int = 10,
    do_detrend: bool = True,
    detrend_type: DetrendType = "first_diff",
    use_peaks_only: bool = True,
    apply_hann_window: bool = True,
    zero_padding_factor: int = 2,
    round_to_closest_integer: bool = True,
    validate_with_acf: bool = False,
    sampling_interval: float = 1.0,
    magnitude_threshold: float | None = 0.05,
    relative_threshold: bool = True,
    exclude_zero: bool = False,
) -> list[tuple[float, float]]:
    """The dominant periods of a series and their spectrum magnitudes, ``[(period, magnitude), ...]``, strongest first.

    TabPFN-TS's FFT detection: NaNs dropped (the horizon's), the signal
    detrended, Hann-windowed and zero-padded by ``zero_padding_factor``; the
    peaks of its magnitude spectrum (DC excluded) of at least
    ``magnitude_threshold`` (a fraction of the highest with
    ``relative_threshold``), all bins when there is none; their periods
    rounded and kept once each. The keyword defaults are TabPFN-TS's
    function's; ``AutoSeasonalFeature`` passes its own (``detrend_type``
    ``"linear"``, ``max_top_k`` 12, ``exclude_zero``). ``"loess"`` detrending
    and ``validate_with_acf`` need statsmodels.
    """
    values = np.array(target_values, dtype=float)
    values = values[~np.isnan(values)]
    n_original = len(values)

    if do_detrend:
        values = detrend(values, detrend_type)
    if apply_hann_window:
        values = values * np.hanning(n_original)
    if zero_padding_factor > 1:
        n = int(n_original * zero_padding_factor)
        padded = np.zeros(n)
        padded[:n_original] = values
        values = padded
    else:
        n = n_original

    magnitudes = np.abs(fft.rfft(values))
    freqs = np.fft.rfftfreq(n, d=sampling_interval)
    magnitudes[0] = 0.0

    if magnitude_threshold is not None and relative_threshold:
        threshold = magnitude_threshold * np.max(magnitudes)
    else:
        threshold = magnitude_threshold

    if use_peaks_only:
        if threshold is not None:
            peaks, _ = find_peaks(magnitudes, height=threshold)
        else:
            peaks, _ = find_peaks(magnitudes)
        if len(peaks) == 0:
            peaks = np.arange(len(magnitudes))
        top_indices = peaks[np.argsort(magnitudes[peaks])[::-1]][:max_top_k]
    else:
        order = np.argsort(magnitudes)[::-1]
        if threshold is not None:
            order = np.array([i for i in order if magnitudes[i] >= threshold], dtype=np.int64)
        top_indices = order[:max_top_k]

    periods = np.zeros_like(freqs)
    non_zero = freqs > 0
    periods[non_zero] = 1.0 / freqs[non_zero]
    top_periods = periods[top_indices]
    if round_to_closest_integer:
        top_periods = np.round(top_periods)
    if exclude_zero:
        keep = top_periods != 0
        top_periods, top_indices = top_periods[keep], top_indices[keep]
    if len(top_periods) > 0:
        first = np.unique(top_periods, return_index=True)[1]
        top_periods, top_indices = top_periods[first], top_indices[first]

    results = [(top_periods[i], magnitudes[top_indices[i]]) for i in range(len(top_indices))]

    if validate_with_acf:
        from statsmodels.tsa.stattools import acf

        acf_values = acf(
            np.array(target_values, dtype=float)[:n_original], nlags=n_original, fft=True
        )
        acf_peaks, _ = find_peaks(acf_values, height=1.96 / np.sqrt(n_original))
        validated = [
            (period, mag)
            for period, mag in results
            if int(round(period)) < len(acf_values)
            and any(abs(int(round(period)) - peak) <= 1 for peak in acf_peaks)
        ]
        if validated:
            results = validated

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def detrend(x: np.ndarray, detrend_type: DetrendType) -> np.ndarray:
    """``x`` without its trend: first differences, a LOESS fit (statsmodels), a line, or its mean."""
    if detrend_type == "first_diff":
        return np.diff(x, prepend=x[0])
    if detrend_type == "loess":
        from statsmodels.api import nonparametric

        indices = np.arange(len(x))
        return x - nonparametric.lowess(x, indices, frac=0.1)[:, 1]
    if detrend_type == "linear":
        indices = np.arange(len(x))
        return x - np.polyval(np.polyfit(indices, x, 1, rcond=None), indices)
    if detrend_type == "constant":
        return x - np.mean(x)
    raise ValueError(f"Invalid detrend method: {detrend_type}")


def season_from_freq(freq: Any) -> int:
    """Steps of a covariate's lag and trailing mean at a sampling frequency, at least 4.

    A day for sub-daily data, a week for daily, a year for monthly and
    quarterly; 4 for weekly data and unknown frequencies.
    """
    name, n = _offset_name(freq)
    if name in {"T", "MIN"}:
        season = 1440 / n
    elif name == "H":
        season = 24 / n
    elif name in {"D", "B"}:
        season = 7 / n
    elif name in {"M", "MS", "ME"}:
        season = 12 / n
    elif name in {"Q", "QS", "QE"}:
        season = 4 / n
    else:
        season = 4
    return max(4, int(round(season)))


def _offset_name(freq: Any) -> tuple[str, int]:
    offset = pd.tseries.frequencies.to_offset(freq)
    return offset.name.split("-")[0].upper(), max(1, getattr(offset, "n", 1))


def _freq_from_step(step: pd.Timedelta) -> Any:
    """The frequency of a typical step; calendar months and quarters snap to MS and QS."""
    # step / 1 day, which pandas computes through a deprecated numpy path.
    days = pd.Timedelta(step).total_seconds() / 86_400
    if 28 <= days <= 31:
        return "MS"
    if 89 <= days <= 92:
        return "QS"
    return pd.tseries.frequencies.to_offset(step)


def _infer_freq_for_lags(timestamps: pd.DatetimeIndex, item_ids: pd.Index) -> Any:
    """The sampling frequency of the first series pandas infers one for; else the most common step over all series (context rows with a missing target are dropped, so no series may be gap-free)."""
    by_item = pd.Series(timestamps, index=item_ids).groupby(level=0, sort=False)
    steps = []
    for _, ts in by_item:
        ts = pd.DatetimeIndex(ts.unique()).sort_values()
        if len(ts) >= 3:
            freq = pd.infer_freq(ts)
            if freq is not None:
                return freq
        if len(ts) >= 2:
            steps.append(ts.to_series().diff().dropna())
    if not steps:
        return None
    step = pd.concat(steps).mode().iloc[0]
    return _freq_from_step(step) if step > pd.Timedelta(0) else None


class KnownCovariateLagFeature(FeatureGenerator):
    """``<c>_lag_<k>`` and ``<c>_mean_<k>``: lags {1, season} and a trailing mean over ``season`` steps (at least one value) of each known covariate, per series.

    Known covariates have values over the context and the horizon, so a
    horizon row's lag reads the covariate, never the target. ``columns=None``
    takes the numeric and boolean columns, the target aside, with a value on
    some horizon row (a past-only covariate has none); it must then run
    before the other generators, whose columns it would take too.
    ``season=None`` derives it from the sampling frequency
    (``season_from_freq``), 4 when none is found.
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        season: int | None = None,
        lags: Sequence[int] | None = None,
        means: Sequence[int] | None = None,
        target_column: str = TARGET,
    ) -> None:
        self.columns = list(columns) if columns is not None else None
        self.season = season
        self.lags = list(lags) if lags is not None else None
        self.means = list(means) if means is not None else None
        self.target_column = target_column

    def _known_columns(self, df: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        horizon = ~df[IS_TRAIN] if IS_TRAIN in df.columns else None
        columns = []
        for c in df.columns:
            if c in (self.target_column, IS_TRAIN):
                continue
            if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])):
                continue
            if horizon is not None and not df.loc[horizon, c].notna().any():
                continue
            columns.append(c)
        return columns

    def _season(self, df: pd.DataFrame) -> int:
        if self.season is not None:
            return max(1, int(self.season))
        freq = _infer_freq_for_lags(
            pd.DatetimeIndex(df.index.get_level_values(TIMESTAMP)),
            df.index.get_level_values(ITEM_ID),
        )
        return season_from_freq(freq) if freq is not None else 4

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = self._known_columns(df)
        if not columns:
            return df.copy()
        season = self._season(df)
        lags = self.lags if self.lags is not None else sorted({1, season})
        means = self.means if self.means is not None else [season]

        x = df[columns].astype(float)
        grouped = x.groupby(level=ITEM_ID, sort=False)
        out = {}
        for k in lags:
            shifted = grouped.shift(k)
            for c in columns:
                out[f"{c}_lag_{k}"] = shifted[c].to_numpy()
        for k in means:
            rolled = grouped.rolling(k, min_periods=1).mean().droplevel(0).reindex(df.index)
            for c in columns:
                out[f"{c}_mean_{k}"] = rolled[c].to_numpy()
        return pd.concat([df, pd.DataFrame(out, index=df.index, dtype=np.float64)], axis=1)


def default_features() -> list[FeatureGenerator]:
    """TabPFN-TS's default generators, ``TABPFN_TS_DEFAULT_FEATURES``: covariate lags, running index, calendar, seasonal periods."""
    return [
        KnownCovariateLagFeature(),
        RunningIndexFeature(),
        CalendarFeature(),
        AutoSeasonalFeature(),
    ]


def featurize(
    context: pd.DataFrame,
    future: pd.DataFrame,
    generators: Sequence[FeatureGenerator],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The context's and the horizon's rows with the generators' columns, and the names of those columns, the generators' only (TabPFN-TS's ``FeatureTransformer.transform``).

    Both frames are indexed by ``(item_id, timestamp)``; the context's
    ``target`` has no NaN, the horizon's is all NaN. The generators see one
    frame, every context row then every horizon row. Generated float64
    columns are cast to float32, the input columns keep their dtypes.
    """
    if TARGET not in context.columns:
        raise ValueError(f"Target column '{TARGET}' not found in the context")
    if not future[TARGET].isna().all():
        raise ValueError("The horizon rows must not hold target values")

    input_columns = set(context.columns)
    train = context.assign(**{IS_TRAIN: True})
    test = future.assign(**{IS_TRAIN: False})
    target_dtype = train[TARGET].dtype
    if pd.api.types.is_integer_dtype(target_dtype):
        target_dtype = "float64"
    test[TARGET] = test[TARGET].astype(target_dtype)
    frame = pd.concat([train, test])
    del train, test

    for generator in generators:
        frame = generator(frame)

    # As TabPFN-TS, a horizon-only input column counts as generated for the downcast.
    new_columns = [c for c in frame.columns if c not in input_columns and c != IS_TRAIN]
    downcast = {c: np.float32 for c in new_columns if frame[c].dtype == np.float64}
    if downcast:
        frame = frame.astype(downcast)

    is_train = frame[IS_TRAIN].to_numpy(dtype=bool)
    frame = frame.drop(columns=[IS_TRAIN])
    train, test = frame[is_train], frame[~is_train]
    assert not train[TARGET].isna().any(), "every context target must be present"
    assert test[TARGET].isna().all()
    generated = [c for c in new_columns if c not in future.columns]
    return train, test, generated
