# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0
#
# The preprocessing follows tabpfn-time-series (https://github.com/PriorLabs/tabpfn-time-series,
# Apache-2.0), tabpfn_time_series/pipeline.py at commit 258a074c4ebe74636641c61db86df99e964cc477.

"""Seldon as a forecaster: TabPFN-TS's pipeline, a series made a table of time features, with Seldon's regressor fit on each series' history."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from neuralk_oss.forecasting.features import (
    ITEM_ID,
    TARGET,
    TIMESTAMP,
    FeatureGenerator,
    default_features,
    featurize,
)
from neuralk_oss.inference._inference import _CONFIGS, _read_config
from neuralk_oss.inference.recipe import TooManyFeatures
from neuralk_oss.inference.regressor import SeldonRegressor
from neuralk_oss.model import Seldon
from neuralk_oss.model.seldon import HUB_CHECKPOINTS

# TabPFN-TS's DEFAULT_QUANTILE_CONFIG.
DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# TabPFN-TS's default max_context_length (its TabPFN-TS-3 configuration).
DEFAULT_MAX_CONTEXT_LENGTH = 32_768
_FORECASTER_KEYS = ("max_context_length", "temporal_features", "point_forecast")


class SeldonForecaster:
    """A Seldon regressor forecasting time series the TabPFN-TS way: each series' history is a table, one row per timestamp and its value the label, and the regressor fit on it predicts the rows of the horizon.

    For each series, as TabPFN-TS's ``TabPFNTSPipeline.predict``:

    1. the context rows whose target is missing are dropped, unless the
       series has at most one value, whose missing ones are then 0;
    2. the series keeps its last ``max_context_length`` rows;
    3. ``temporal_features`` (TabPFN-TS's defaults, see
       ``neuralk_oss.forecasting.features``) add each row's time features,
       computed over every series at once;
    4. the regressor is fit on the context rows, every column but the
       target its features (the covariates included), and predicts the
       quantiles of the horizon rows.

    The target's past values are never features: the forecast is a
    regression on time and covariates, not an autoregression. A covariate
    with values on the horizon is a known one (the lag features read it); a
    context column without is past-only, kept as TabPFN-TS keeps it, NaN on
    the horizon rows.

    Nothing is subsampled: every kept context row is in every member's
    context, and every feature column is read. A series whose context
    passes the regressor's ``max_context_size`` (where it would give each
    member a random subset) raises instead; so does a table past the
    recipe's ``max_features``, which must be set not to cut
    (``SeldonRecipe(cut_features=False)``, as the packaged configs do).

    Parameters
    ----------
    regressor : SeldonRegressor
        The regressor fit on each series; its recipe must have
        ``cut_features=False``. Its ``clip_margin`` should be None, or a trend
        could not leave the context's range; one set warns.
    max_context_length : int or None, default=32768
        Context rows each series keeps, its latest; None keeps them all.
    temporal_features : sequence of FeatureGenerator, optional
        The feature generators, TabPFN-TS's defaults
        (``features.default_features()``) by default.
    point_forecast : {"median", "mean"}, default="median"
        The point forecast: the regressor's median (TabPFN-TS's default
        output selection) or the mean of its quantiles.
    """

    def __init__(
        self,
        regressor: SeldonRegressor,
        *,
        max_context_length: int | None = DEFAULT_MAX_CONTEXT_LENGTH,
        temporal_features: Sequence[FeatureGenerator] | None = None,
        point_forecast: Literal["median", "mean"] = "median",
    ) -> None:
        if not isinstance(regressor, SeldonRegressor):
            raise TypeError(f"regressor must be a SeldonRegressor, got {type(regressor).__name__}")
        if max_context_length is not None and max_context_length < 1:
            raise ValueError(
                f"max_context_length must be None or at least 1, got {max_context_length}"
            )
        if max_context_length is not None and max_context_length > regressor.max_context_size:
            raise ValueError(
                f"max_context_length ({max_context_length}) passes the regressor's "
                f"max_context_size ({regressor.max_context_size}), past which each member "
                "would read a random subset of the context: raise max_context_size"
            )
        if regressor.estimator.recipe.cut_features:
            raise ValueError(
                "the regressor's recipe cuts the columns past max_features: give it "
                "SeldonRecipe(cut_features=False, ...), as the packaged forecaster configs do"
            )
        if point_forecast not in ("median", "mean"):
            raise ValueError(f"point_forecast must be 'median' or 'mean', got {point_forecast!r}")
        if regressor.clip_margin is not None:
            # A warning, not an error: a bounded target may want the clamp.
            warnings.warn(
                f"the regressor's clip_margin ({regressor.clip_margin}) clamps the forecasts "
                "to the context's range widened by that margin, so a trend past it is capped: "
                "set clip_margin=None, as the packaged forecaster configs do",
                UserWarning,
                stacklevel=2,
            )
        self.regressor = regressor
        self.max_context_length = max_context_length
        self.temporal_features = (
            default_features() if temporal_features is None else list(temporal_features)
        )
        self.point_forecast = point_forecast

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | os.PathLike,
        *,
        subfolder: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        **kwargs: Any,
    ) -> SeldonForecaster:
        """The forecaster of a saved regressor: a released model's name, ``"seldon"`` or ``"seldon-mini"``, or what ``Seldon.from_pretrained`` takes (a local directory, a checkpoint name such as ``"seldon-mini-regressor"``, a Hub repository); ``kwargs`` are the constructor's and ``SeldonRegressor``'s.

        A released model's name, or its regression checkpoint's, takes the
        settings of its packaged forecaster config
        (``configs/<model>-forecaster.yaml``), which ``kwargs`` override. That
        config is the only one read: the regressor's packaged config
        (``<model>-regressor.yaml``) does not apply, so a key the forecaster
        config leaves out takes the constructor's default. Anything else takes
        the constructors' defaults.
        """
        name = str(name_or_path)
        model = name.removesuffix("-regressor")
        if (
            subfolder is None
            and f"{model}-regressor" in HUB_CHECKPOINTS
            and not Path(name_or_path).is_dir()
        ):
            settings = _read_config(_CONFIGS / f"{model}-forecaster.yaml")
            settings.pop("checkpoint", None)
            kwargs = {**settings, **kwargs}
            name_or_path = f"{model}-regressor"
        own = {key: kwargs.pop(key) for key in _FORECASTER_KEYS if key in kwargs}
        # Not SeldonRegressor.from_pretrained, which would merge the regressor's packaged config under kwargs.
        weights = Seldon.from_pretrained(
            name_or_path, task="regression", subfolder=subfolder, revision=revision, token=token
        )
        regressor = SeldonRegressor(weights, **kwargs)
        return cls(regressor, **own)

    @classmethod
    def from_config(cls, config: str | os.PathLike, **kwargs: Any) -> SeldonForecaster:
        """The forecaster of a YAML config: a packaged one by name, such as ``"seldon-mini-forecaster"``, or a file.

        The config maps ``checkpoint`` to ``from_pretrained``'s first
        argument, ``recipe`` to ``SeldonRecipe``'s keyword arguments, and any
        other key to a keyword argument of ``from_pretrained``; ``kwargs``
        override its keys.
        """
        path = Path(config)
        if path.suffix not in (".yaml", ".yml") and path.parent == Path():
            names = sorted(
                name
                for p in _CONFIGS.iterdir()
                if (name := p.name.removesuffix(".yaml")).endswith("-forecaster")
            )
            if str(config) not in names:
                raise ValueError(
                    f"unknown forecaster config {config!r}; packaged: {names}, or a .yaml path"
                )
            path = _CONFIGS / f"{config}.yaml"
        params = {**_read_config(path), **kwargs}
        if "checkpoint" not in params:
            raise ValueError(f"config {config!r} names no checkpoint")
        return cls.from_pretrained(params.pop("checkpoint"), **params)

    def to(self, device: str | torch.device) -> SeldonForecaster:
        """Move the regressor to ``device``."""
        self.regressor.to(device)
        return self

    def predict(
        self,
        context: pd.DataFrame,
        future: pd.DataFrame,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """The forecasts of the series of ``future``, each fit on its rows of ``context`` (TabPFN-TS's ``TabPFNTSPipeline.predict``).

        Parameters
        ----------
        context : pd.DataFrame
            The history, indexed by ``(item_id, timestamp)``: a ``target``
            column, and covariates.
        future : pd.DataFrame
            The horizon's rows, indexed by ``(item_id, timestamp)``, of series
            of ``context``: the known covariates' values; a ``target`` column,
            if any, all NaN.
        quantiles : sequence of float
            Levels in [0, 1].

        Returns
        -------
        pd.DataFrame
            Indexed as ``future``: ``target``, the point forecast, and one
            column per level (named by the float), in the series order of
            ``future``.
        """
        levels = _check_quantiles(quantiles)
        context = _check_frame(context, "context")
        future = _check_frame(future, "future")
        if TARGET not in context.columns:
            raise ValueError(f"context has no '{TARGET}' column")
        items = future.index.unique(ITEM_ID)
        if missing := sorted(map(str, set(items) - set(context.index.unique(ITEM_ID)))):
            raise ValueError(f"future has series without context: {missing}")
        context = context[context.index.get_level_values(ITEM_ID).isin(items)]
        context = _handle_missing_values(context)
        if self.max_context_length is not None:
            context = context.groupby(level=ITEM_ID, sort=False).tail(self.max_context_length)
        future = _preprocess_future(future)
        _warn_missing_future_covariates(context, future)

        train, test, generated = featurize(context, future, self.temporal_features)
        train_by_item = dict(iter(train.groupby(level=ITEM_ID, sort=False)))
        test_by_item = dict(iter(test.groupby(level=ITEM_ID, sort=False)))
        outputs = []
        for item in items:
            outputs.append(
                self._forecast_one(train_by_item[item], test_by_item[item], generated, levels)
            )
        return pd.concat(outputs)

    def predict_df(
        self,
        context_df: pd.DataFrame,
        future_df: pd.DataFrame | None = None,
        prediction_length: int | None = None,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """The forecasts of long-format frames (TabPFN-TS's ``TabPFNTSPipeline.predict_df``).

        ``context_df`` has columns ``timestamp``, ``target``, ``item_id``
        (optional, for one series) and covariates. The horizon is either
        ``future_df`` (``timestamp``, ``item_id``, the known covariates' future
        values) or ``prediction_length`` steps after each series' last
        timestamp, at the frequency of the context's series (all regular and
        equal, else this raises). Returns, indexed by ``(item_id, timestamp)``,
        ``target`` (the point forecast) and one column per level.
        """
        if (future_df is None) == (prediction_length is None):
            raise ValueError("Provide exactly one of future_df or prediction_length")
        context = _indexed(context_df)
        if prediction_length is not None:
            future = _future_rows(context, prediction_length)
        else:
            future = _indexed(future_df)
        return self.predict(context, future, quantiles)

    def forecast(
        self,
        x_context: pd.DataFrame | None,
        y_context: pd.DataFrame | pd.Series,
        x_query: pd.DataFrame | pd.DatetimeIndex,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        """sdm's forecasting protocol (as ``KumoForecast``): the rows are time steps, ``y_context`` the targets, ``x_query``'s rows the horizon.

        Parameters
        ----------
        x_context : pd.DataFrame or None
            The covariates over the history, indexed as ``y_context``. Its
            columns ``x_query`` also has are known covariates, the others
            past-only.
        y_context : pd.DataFrame or pd.Series
            The history of one or more targets, on a DatetimeIndex. Each
            target is forecast on its own, from its own history and the
            covariates (the other targets are not read).
        x_query : pd.DataFrame or pd.DatetimeIndex
            The horizon's timestamps (its index), and the known covariates'
            values over it.
        quantiles : sequence of float
            Levels in [0, 1].

        Returns
        -------
        pd.DataFrame
            Indexed as ``x_query``: per target ``t``, the point forecast ``t``
            and sdm's quantile columns ``t__q<round(100 * level)>``.
        """
        levels = _check_quantiles(quantiles)
        names = [f"q{round(100 * level)}" for level in levels]
        if len(set(names)) != len(names):
            raise ValueError(f"quantile levels {levels} share a column name {names}")
        if isinstance(y_context, pd.Series):
            y_context = y_context.to_frame(TARGET if y_context.name is None else y_context.name)
        if isinstance(x_query, pd.DatetimeIndex):
            x_query = pd.DataFrame(index=x_query)
        if not isinstance(y_context.index, pd.DatetimeIndex):
            raise ValueError("y_context must be indexed by timestamps (a DatetimeIndex)")
        if not isinstance(x_query.index, pd.DatetimeIndex):
            raise ValueError("x_query must be indexed by timestamps (a DatetimeIndex)")
        if x_context is None:
            x_context = pd.DataFrame(index=y_context.index)
        if not x_context.index.equals(y_context.index):
            raise ValueError("x_context and y_context must share their index")
        if extra := sorted(map(str, set(x_query.columns) - set(x_context.columns))):
            raise ValueError(f"x_query has columns x_context does not: {extra}")
        if clash := sorted(map(str, set(x_context.columns) & {TARGET, *y_context.columns})):
            raise ValueError(f"x_context columns named like a target: {clash}")

        targets = list(y_context.columns)
        contexts, futures = [], []
        for i, target in enumerate(targets):
            frame = x_context.assign(**{TARGET: y_context[target].to_numpy()})
            contexts.append(_long(frame, i))
            futures.append(_long(x_query, i))
        out = self.predict(pd.concat(contexts), pd.concat(futures), levels)

        result = {}
        for i, target in enumerate(targets):
            rows = out.xs(i, level=ITEM_ID)
            result[target] = rows[TARGET].to_numpy()
            for level, name in zip(levels, names, strict=True):
                result[f"{target}__{name}"] = rows[level].to_numpy()
        return pd.DataFrame(result, index=x_query.index)

    def _forecast_one(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        generated: list[str],
        levels: list[float],
    ) -> pd.DataFrame:
        """One series' forecast: the regressor fit on its context rows, its quantiles at ``levels`` on the horizon rows."""
        X, y = train.drop(columns=[TARGET]), train[TARGET].to_numpy(dtype=np.float64)
        X_test = test.drop(columns=[TARGET])
        item = train.index.get_level_values(ITEM_ID)[0]
        reg = self.regressor
        if len(X) > reg.max_context_size:
            raise ValueError(
                f"series {item} keeps {len(X)} context rows, past the regressor's "
                f"max_context_size ({reg.max_context_size}), past which each member would read "
                "a random subset: lower max_context_length or raise max_context_size"
            )
        stypes = {name: "numerical" for name in generated}
        try:
            reg.fit(X, y, stypes=stypes)
        except TooManyFeatures as error:
            raise TooManyFeatures(
                f"series {item}: {error}; raise max_features or pass fewer covariates"
            ) from error
        assert not reg._subsampled, "the regressor subsampled the context"
        q = reg.predict_quantiles(X_test).astype(np.float64)
        point = q.mean(axis=1) if self.point_forecast == "mean" else _at_levels(q, [0.5])[:, 0]
        out = pd.DataFrame({TARGET: point}, index=test.index)
        values = _at_levels(q, levels)
        for i, level in enumerate(levels):
            out[level] = values[:, i]
        return out


def _check_quantiles(quantiles: Sequence[float]) -> list[float]:
    levels = [float(q) for q in quantiles]
    if not levels or not all(0 <= q <= 1 for q in levels):
        raise ValueError(f"quantiles must be a non-empty list of levels in [0, 1], got {quantiles}")
    return levels


def _at_levels(q: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """The quantiles at ``levels``, interpolated linearly between the model's, at levels k / (Q + 1), k = 1..Q (clamped to the outermost).

    For the released heads (999 quantiles), a level that is a multiple of
    0.001 is one of the model's own.
    """
    n = q.shape[1]
    pos = np.round(np.asarray(levels, dtype=np.float64) * (n + 1) - 1, 9).clip(0, n - 1)
    low = np.floor(pos).astype(np.int64)
    high = np.minimum(low + 1, n - 1)
    weight = pos - low
    return q[:, low] * (1 - weight) + q[:, high] * weight


def _check_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """``frame`` indexed by ``(item_id, timestamp)``, each series in time order (sorted if it is not); duplicated timestamps raise."""
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != [
        ITEM_ID,
        TIMESTAMP,
    ]:
        raise ValueError(f"{name} must be indexed by ({ITEM_ID!r}, {TIMESTAMP!r})")
    stamps = frame.index.get_level_values(TIMESTAMP)
    if not isinstance(stamps, pd.DatetimeIndex):
        frame = frame.set_axis(
            pd.MultiIndex.from_arrays(
                [frame.index.get_level_values(ITEM_ID), pd.to_datetime(stamps)],
                names=[ITEM_ID, TIMESTAMP],
            )
        )
    if frame.index.duplicated().any():
        raise ValueError(f"{name} has a series with a timestamp twice")
    codes, _ = pd.factorize(frame.index.get_level_values(ITEM_ID), sort=False)
    order = np.lexsort((frame.index.get_level_values(TIMESTAMP).asi8, codes))
    contiguous = (np.diff(codes[order]) >= 0).all()
    if not (contiguous and (order == np.arange(len(order))).all()):
        # Series in their first appearance's order, each in time order.
        frame = frame.iloc[order]
    return frame


def _handle_missing_values(context: pd.DataFrame) -> pd.DataFrame:
    """TabPFN-TS's ``_handle_missing_values``: a series with at most one target value has its missing ones set to 0; other series drop their rows without one."""
    valid = context[TARGET].notna().groupby(level=ITEM_ID, sort=False).sum()
    sparse = valid[valid <= 1].index
    context = context.copy()
    if len(sparse) > 0:
        fill = context.index.get_level_values(ITEM_ID).isin(sparse) & context[TARGET].isna()
        context.loc[fill, TARGET] = 0
    return context[context[TARGET].notna()]


def _preprocess_future(future: pd.DataFrame) -> pd.DataFrame:
    if TARGET in future.columns and future[TARGET].notna().any():
        raise ValueError(f"future: every '{TARGET}' must be NaN over the horizon")
    return future.assign(**{TARGET: np.nan})


def _warn_missing_future_covariates(context: pd.DataFrame, future: pd.DataFrame) -> None:
    known = context.columns.intersection(future.columns).drop(TARGET).tolist()
    if future[known].isna().any().any():
        warnings.warn(
            "Some covariate values in future are missing (NaN). This may affect prediction quality.",
            UserWarning,
            stacklevel=3,
        )


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    """A long-format frame indexed by ``(item_id, timestamp)``; without an ``item_id`` column, one series, ``0``."""
    if TIMESTAMP not in frame.columns:
        raise ValueError(f"the frame has no '{TIMESTAMP}' column")
    frame = frame.copy()
    if ITEM_ID not in frame.columns:
        frame[ITEM_ID] = 0
    frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP])
    return frame.set_index([ITEM_ID, TIMESTAMP])


def _long(frame: pd.DataFrame, item: Any) -> pd.DataFrame:
    """A one-series frame on a DatetimeIndex, indexed by ``(item, timestamp)``."""
    index = pd.MultiIndex.from_arrays(
        [np.full(len(frame), item), frame.index], names=[ITEM_ID, TIMESTAMP]
    )
    return frame.set_axis(index)


def _infer_frequency(context: pd.DataFrame) -> str:
    """The frequency every series of ``context`` follows; raises if one is irregular or they differ.

    TabPFN-TS reads it on a random sample of 50 series (autogluon's
    ``TimeSeriesDataFrame.freq``); every series is read here.
    """
    frequencies, irregular = set(), []
    for item, rows in context.groupby(level=ITEM_ID, sort=False):
        freq = pd.DatetimeIndex(rows.index.get_level_values(TIMESTAMP)).inferred_freq
        if freq is None:
            irregular.append(item)
        else:
            frequencies.add(freq)
    if irregular or len(frequencies) != 1:
        detail = (
            f"irregular series {irregular[:5]}"
            if irregular
            else f"frequencies {sorted(frequencies)}"
        )
        raise ValueError(
            f"cannot infer one frequency for prediction_length ({detail}): pass future_df"
        )
    return pd.tseries.frequencies.to_offset(frequencies.pop()).freqstr


def _future_rows(context: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    """``prediction_length`` timestamps after each series' last (TabPFN-TS's ``generate_test_X``)."""
    if prediction_length < 1:
        raise ValueError(f"prediction_length must be at least 1, got {prediction_length}")
    context = _check_frame(context, "context_df")
    offset = pd.tseries.frequencies.to_offset(_infer_frequency(context))
    last = context.index.to_frame(index=False).groupby(ITEM_ID, sort=False)[TIMESTAMP].max()
    items, stamps = [], []
    for item, end in last.items():
        items.append(np.full(prediction_length, item, dtype=object))
        stamps.append(pd.date_range(start=end + offset, periods=prediction_length, freq=offset))
    index = pd.MultiIndex.from_arrays(
        [np.concatenate(items), pd.DatetimeIndex(np.concatenate(stamps))],
        names=[ITEM_ID, TIMESTAMP],
    )
    return pd.DataFrame({TARGET: np.full(len(index), np.nan)}, index=index)
