# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

"""SeldonForecaster and TabPFN-TS's time features.

The feature tests pin each formula of tabpfn-time-series (commit 258a074) on
values worked out by hand.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from neuralk_oss import (
    Seldon,
    SeldonClassifier,
    SeldonForecaster,
    SeldonRecipe,
    SeldonRegressor,
)
from neuralk_oss.forecasting import features as F
from neuralk_oss.forecasting.forecaster import DEFAULT_QUANTILES, _at_levels
from neuralk_oss.inference import TooManyFeatures

CONFIGS = Path(__file__).parents[1] / "src" / "neuralk_oss" / "configs"
TINY = Path(__file__).parent / "golden" / "seldon-tiny-regressor"
N, HORIZON = 120, 6


def _context(n=N, items=("a", "b"), freq="h", covariates=False):
    """Long-format series: a daily cycle, one level per series; with covariates, a known ``promo`` and a past-only ``temp``."""
    frames = []
    for k, item in enumerate(items):
        steps = np.arange(n)
        frame = pd.DataFrame(
            {
                "item_id": item,
                "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq),
                "target": 10.0 + 3.0 * k + np.sin(2 * np.pi * steps / 24),
            }
        )
        if covariates:
            frame["promo"] = (steps % 5 == 0).astype(float)
            frame["temp"] = np.cos(steps / 3)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _future(context, horizon=HORIZON, freq="h", promo=True):
    frames = []
    for item, rows in context.groupby("item_id", sort=False):
        stamps = pd.date_range(rows["timestamp"].max(), periods=horizon + 1, freq=freq)[1:]
        frame = pd.DataFrame({"item_id": item, "timestamp": stamps})
        if promo:
            frame["promo"] = np.arange(horizon) % 2.0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _indexed(frame):
    return frame.set_index(["item_id", "timestamp"])


def _forecaster(model, **kwargs):
    regressor_kwargs = {
        k: kwargs.pop(k) for k in ("max_context_size", "recipe", "clip_margin") if k in kwargs
    }
    regressor_kwargs.setdefault("recipe", SeldonRecipe(cut_features=False))
    regressor = SeldonRegressor(model, n_estimators=2, device="cpu", **regressor_kwargs)
    return SeldonForecaster(regressor, **kwargs)


# --- Features -----------------------------------------------------------------


def test_calendar_feature_values():
    # A Tuesday, day 65 of a leap year, ISO week 10.
    stamp = pd.Timestamp("2024-03-05 13:45:30")
    frame = pd.DataFrame(
        {"target": [1.0]},
        index=pd.MultiIndex.from_tuples([("a", stamp)], names=["item_id", "timestamp"]),
    )
    out = F.CalendarFeature().generate(frame).iloc[0]
    # TabPFN-TS divides gluonts' zero-based index by the period minus one.
    expected = {
        "second_of_minute": 30 / 59,
        "minute_of_hour": 45 / 59,
        "hour_of_day": 13 / 23,
        "day_of_week": 1 / 6,
        "day_of_month": 4 / 29.5,
        "day_of_year": 64 / 364,
        "week_of_year": 9 / 51,
        "month_of_year": 2 / 11,
    }
    assert out["year"] == 2024
    for name, fraction in expected.items():
        assert out[f"{name}_sin"] == pytest.approx(np.sin(2 * np.pi * fraction))
        assert out[f"{name}_cos"] == pytest.approx(np.cos(2 * np.pi * fraction))


def test_running_index_starts_per_series_and_continues_over_the_horizon():
    context = _indexed(_context(n=10))
    future = _indexed(_future(_context(n=10), promo=False)).assign(target=np.nan)
    train, test, _ = F.featurize(context, future, [F.RunningIndexFeature()])
    for item in ("a", "b"):
        assert train.loc[item, "running_index"].tolist() == list(range(10))
        assert test.loc[item, "running_index"].tolist() == list(range(10, 10 + HORIZON))


def test_find_seasonal_periods_strongest_first():
    steps = np.arange(500)
    y = 3 * np.sin(2 * np.pi * steps / 24) + np.sin(2 * np.pi * steps / 7) + 0.01 * steps
    periods = [p for p, _ in F.find_seasonal_periods(y, **{**F.AutoSeasonalFeature.DEFAULTS})]
    assert periods[:2] == [24.0, 7.0]
    assert len(periods) <= 12 and 0.0 not in periods


def test_auto_seasonal_feature_has_twelve_slots_zero_filled():
    # 20 cycles: on fewer, the linear detrend's residue adds the window's length as a period.
    context = _indexed(_context(n=480, items=("a",)))
    future = _indexed(_future(_context(n=480, items=("a",)), promo=False)).assign(target=np.nan)
    train, test, _ = F.featurize(context, future, [F.AutoSeasonalFeature()])
    columns = [f"{t}_#{i}" for i in range(12) for t in ("sin", "cos")]
    assert set(columns) <= set(train.columns)
    # The target's only period is 24: slot 0 holds it over the running index, the rest are 0.
    pos = np.arange(480 + HORIZON)
    both = pd.concat([train, test])
    np.testing.assert_allclose(both["sin_#0"], np.sin(2 * np.pi * pos / 24), atol=1e-6)
    assert (both[[c for c in columns if not c.endswith("#0")]] == 0).all().all()


def test_known_covariate_lags_read_the_covariate_never_the_target():
    context = _indexed(_context(n=48, items=("a",), covariates=True))
    future = _indexed(_future(_context(n=48, items=("a",)))).assign(target=np.nan)
    train, test, generated = F.featurize(context, future, F.default_features())
    # Hourly: season 24; temp has no horizon value, it is past-only and not lagged.
    assert {"promo_lag_1", "promo_lag_24", "promo_mean_24"} <= set(generated)
    assert not any(c.startswith("temp_") for c in generated)
    promo = np.r_[context["promo"].to_numpy(), future["promo"].to_numpy()]
    both = pd.concat([train, test])
    np.testing.assert_allclose(both["promo_lag_1"].to_numpy()[1:], promo[:-1])
    assert np.isnan(both["promo_lag_24"].to_numpy()[:24]).all()
    np.testing.assert_allclose(both["promo_lag_24"].to_numpy()[24:], promo[:-24])
    trailing = pd.Series(promo).rolling(24, min_periods=1).mean().to_numpy()
    np.testing.assert_allclose(both["promo_mean_24"], trailing, rtol=1e-6)
    # The past-only covariate is kept, NaN on the horizon (as TabPFN-TS).
    assert test["temp"].isna().all() and train["temp"].notna().all()


@pytest.mark.parametrize(
    "freq, season",
    [("h", 24), ("15min", 96), ("D", 7), ("B", 7), ("W", 4), ("MS", 12), ("QS", 4), ("YS", 4)],
)
def test_season_from_freq(freq, season):
    assert F.season_from_freq(freq) == season


def test_featurize_downcasts_generated_floats_only():
    context = _indexed(_context(n=48, items=("a",), covariates=True))
    future = _indexed(_future(_context(n=48, items=("a",)))).assign(target=np.nan)
    train, _, generated = F.featurize(context, future, F.default_features())
    assert train["temp"].dtype == np.float64 and train["target"].dtype == np.float64
    assert train["promo_lag_1"].dtype == np.float32 and train["sin_#0"].dtype == np.float32
    # Columns in TabPFN-TS's order: inputs, covariate lags, index, year, calendar, seasons.
    assert generated[:4] == ["promo_lag_1", "promo_lag_24", "promo_mean_24", "running_index"]
    assert generated[4:6] == ["year", "second_of_minute_sin"] and generated[-1] == "cos_#11"


def test_featurize_refuses_horizon_targets():
    context = _indexed(_context(n=20, items=("a",)))
    future = _indexed(_future(_context(n=20, items=("a",)), promo=False)).assign(target=1.0)
    with pytest.raises(ValueError, match="horizon"):
        F.featurize(context, future, F.default_features())
    with pytest.raises(ValueError, match="Target column 'target' not found"):
        F.featurize(context.drop(columns="target"), future, F.default_features())


def test_featurize_integer_target():
    context = _indexed(_context(n=10, items=("a",))).assign(target=np.arange(10))
    future = _indexed(_future(_context(n=10, items=("a",)), promo=False)).assign(target=np.nan)
    train, test, _ = F.featurize(context, future, [F.RunningIndexFeature()])
    # The horizon's NaN targets need a float column: the context's integers are read as floats.
    assert train["target"].tolist() == list(range(10)) and test["target"].isna().all()


def _frame(n=4, items=("a",)):
    stamps = pd.date_range("2024-01-01", periods=n, freq="D")
    index = pd.MultiIndex.from_product([items, stamps], names=["item_id", "timestamp"])
    return pd.DataFrame({"target": np.arange(len(index), dtype=float)}, index=index)


def test_calendar_feature_raw_index():
    frame = _frame()  # 2024-01-01 is a Monday
    out = F.CalendarFeature(components=["month"], seasonal_features={"day_of_week": None})(frame)
    # A field without periods is its raw zero-based index, no sin or cos.
    assert out["day_of_week"].tolist() == [0, 1, 2, 3] and out["month"].tolist() == [1] * 4
    assert "day_of_week_sin" not in out and "year" not in out


def test_periodic_sin_cosine_counts_rows_over_the_whole_frame():
    frame = _frame(n=3, items=("a", "b"))
    out = F.PeriodicSinCosineFeature([4, 6])(frame)
    # Series b starts at row 3, not 0 (TabPFN-TS's quirk).
    np.testing.assert_allclose(out["sin_4"], np.sin(2 * np.pi * np.arange(6) / 4))
    np.testing.assert_allclose(out["cos_6"], np.cos(2 * np.pi * np.arange(6) / 6))
    named = F.PeriodicSinCosineFeature([4, 6], name_suffix="p")(frame)
    assert [c for c in named.columns if c != "target"] == [
        "sin_p_0",
        "cos_p_0",
        "sin_p_1",
        "cos_p_1",
    ]
    empty = F.PeriodicSinCosineFeature([])(frame)
    pd.testing.assert_frame_equal(empty, frame)
    assert empty is not frame


def test_auto_seasonal_feature_config(caplog):
    with pytest.raises(ValueError, match=r"no option \['max_top'\]"):
        F.AutoSeasonalFeature({"max_top": 3})
    # Out-of-range values are logged and replaced, as TabPFN-TS.
    with caplog.at_level("WARNING", logger=F.__name__):
        feature = F.AutoSeasonalFeature(
            {"max_top_k": 0, "zero_padding_factor": 0, "detrend_type": "cubic"}
        )
    assert feature.config["max_top_k"] == 1 and feature.config["zero_padding_factor"] == 1
    assert feature.config["detrend_type"] == "linear"
    assert len(caplog.records) == 3 and "cubic" in caplog.text
    assert repr(feature).startswith("AutoSeasonalFeature_{'config': {'max_top_k': 1")
    out = feature(_indexed(_context(n=48, items=("a",))))
    assert {"sin_#0", "cos_#0"} <= set(out.columns) and "sin_#1" not in out


def _two_periods(n=480):
    steps = np.arange(n)
    return np.sin(2 * np.pi * steps / 24) + 0.5 * np.sin(2 * np.pi * steps / 8)


@pytest.mark.parametrize(
    "options, periods",
    [
        ({}, [24, 8]),
        # Without padding the bins are coarser; the periods fall on bins either way.
        ({"zero_padding_factor": 1}, [24, 8]),
        ({"detrend_type": "constant"}, [24, 8]),
        # First differences weigh a frequency by itself: the shorter period comes first.
        ({"detrend_type": "first_diff"}, [8, 24]),
        # An absolute threshold, in units of the spectrum: only the strongest peak passes 70.
        ({"magnitude_threshold": 70.0, "relative_threshold": False}, [24]),
        # Every bin over the threshold, not only peaks: a peak's neighbours leak in.
        ({"use_peaks_only": False}, [24, 23, 25, 8, 26, 22]),
    ],
)
def test_find_seasonal_periods_options(options, periods):
    config = {**F.AutoSeasonalFeature.DEFAULTS, **options}
    assert [p for p, _ in F.find_seasonal_periods(_two_periods(), **config)] == periods


def test_find_seasonal_periods_without_threshold():
    config = {**F.AutoSeasonalFeature.DEFAULTS, "magnitude_threshold": None}
    found = [p for p, _ in F.find_seasonal_periods(_two_periods(), **config)]
    # Every peak, the window's side lobes and its length among them, the strongest first.
    assert found[:3] == [24, 8, 480] and len(found) > 3


def test_detrend():
    x = np.array([1.0, 3.0, 6.0])
    np.testing.assert_allclose(F.detrend(x, "first_diff"), [0.0, 2.0, 3.0])
    np.testing.assert_allclose(F.detrend(x, "constant"), x - x.mean())
    np.testing.assert_allclose(F.detrend(2.0 * np.arange(5) + 1.0, "linear"), 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="Invalid detrend method: cubic"):
        F.detrend(x, "cubic")


@pytest.mark.parametrize(
    "freq, dropped, season",
    [
        # Gaps leave pandas no frequency: the most common step, snapped to calendar months and quarters.
        ("MS", [5, 11], 12),
        ("QS", [5], 4),
        ("D", [3], 7),
    ],
)
def test_covariate_season_from_the_most_common_step(freq, dropped, season):
    stamps = pd.date_range("2020-01-01", periods=24, freq=freq).delete(dropped)
    index = pd.MultiIndex.from_arrays([["a"] * len(stamps), stamps], names=["item_id", "timestamp"])
    frame = pd.DataFrame({"target": 1.0, "promo": np.arange(len(stamps), dtype=float)}, index)
    out = F.KnownCovariateLagFeature()(frame)
    assert {f"promo_lag_{season}", f"promo_mean_{season}"} <= set(out.columns)


def test_covariate_lag_columns_and_season():
    # One timestamp per series: no step to read, season 4.
    frame = _frame(n=1, items=("a", "b")).assign(promo=1.0, temp=2.0)
    assert "promo_lag_4" in F.KnownCovariateLagFeature()(frame)
    frame = _frame(n=6).assign(promo=np.arange(6.0), temp=2.0)
    # Named columns (an absent one skipped) and season.
    out = F.KnownCovariateLagFeature(columns=["promo", "absent"], season=3)(frame)
    assert [c for c in out.columns if c not in frame] == [
        "promo_lag_1",
        "promo_lag_3",
        "promo_mean_3",
    ]
    np.testing.assert_allclose(out["promo_lag_3"], [np.nan, np.nan, np.nan, 0, 1, 2])
    np.testing.assert_allclose(out["promo_mean_3"], [0, 0.5, 1, 2, 3, 4])
    # No covariate: the frame as it is.
    pd.testing.assert_frame_equal(F.KnownCovariateLagFeature()(_frame()), _frame())


# --- Forecaster ------------------------------------------------------------------


def test_predict_df_shapes_and_order(tiny_regressor):
    out = _forecaster(tiny_regressor).predict_df(_context(), prediction_length=HORIZON)
    # Indexed by (item_id, timestamp), as TabPFN-TS's predict_df.
    assert list(out.index.names) == ["item_id", "timestamp"]
    out = out.reset_index()
    assert list(out.columns) == ["item_id", "timestamp", "target", *DEFAULT_QUANTILES]
    assert out["item_id"].tolist() == ["a"] * HORIZON + ["b"] * HORIZON
    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-01-06")  # N hours after the start
    q = out[list(DEFAULT_QUANTILES)].to_numpy()
    assert np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all()
    np.testing.assert_allclose(out["target"], out[0.5])  # the median


def test_point_forecast_mean(tiny_regressor):
    out = _forecaster(tiny_regressor, point_forecast="mean").predict_df(
        _context(items=("a",)), prediction_length=HORIZON, quantiles=[0.5]
    )
    assert not np.allclose(out["target"], out[0.5])


def test_each_series_reads_only_its_own_history(tiny_regressor):
    forecaster = _forecaster(tiny_regressor)
    context = _context()
    out = forecaster.predict_df(context, prediction_length=HORIZON).reset_index()
    changed = context.copy()
    changed.loc[changed["item_id"] == "b", "target"] *= -3.0
    out2 = forecaster.predict_df(changed, prediction_length=HORIZON).reset_index()
    a = out["item_id"] == "a"
    np.testing.assert_array_equal(out.loc[a, "target"], out2.loc[a, "target"])
    assert not np.allclose(out.loc[~a, "target"], out2.loc[~a, "target"])


def test_missing_targets_and_context_length(tiny_regressor, monkeypatch):
    seen = []
    original = SeldonRegressor.fit

    def spy(self, X, y, **kwargs):
        seen.append((X.copy(), np.asarray(y).copy(), kwargs))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(SeldonRegressor, "fit", spy)
    context = _context()
    context.loc[[3, 7], "target"] = np.nan  # series a: two rows dropped
    b = context["item_id"] == "b"
    context.loc[b, "target"] = np.nan
    context.loc[context.index[b][5], "target"] = 4.0  # series b: one value, the others 0
    _forecaster(tiny_regressor, max_context_length=100).predict_df(
        context, prediction_length=HORIZON
    )

    (Xa, ya, kwargs_a), (Xb, yb, _) = seen
    assert len(Xa) == 100 and np.isfinite(ya).all()  # the latest 100 of 118 kept rows
    assert Xa["running_index"].tolist() == list(range(100))
    assert len(Xb) == 100 and yb.sum() == 0.0  # b's value is row 5, before its latest 100
    # The generated features are read as numbers, not categories (year has one value).
    assert kwargs_a["stypes"]["year"] == "numerical"


def test_no_subsampling_of_rows(tiny_regressor):
    with pytest.raises(ValueError, match="max_context_size"):
        _forecaster(tiny_regressor, max_context_size=50, max_context_length=100)
    forecaster = _forecaster(tiny_regressor, max_context_size=50, max_context_length=None)
    with pytest.raises(ValueError, match="random subset"):
        forecaster.predict_df(_context(items=("a",)), prediction_length=HORIZON)
    # At the bound, every row is every member's context.
    forecaster = _forecaster(tiny_regressor, max_context_size=N, max_context_length=None)
    forecaster.predict_df(_context(items=("a",)), prediction_length=HORIZON)
    assert not forecaster.regressor._subsampled


def test_no_subsampling_of_columns(tiny_regressor):
    recipe = SeldonRecipe(max_features=10, cut_features=False)
    forecaster = _forecaster(tiny_regressor, recipe=recipe)
    with pytest.raises(TooManyFeatures, match="series 0: .* past max_features"):
        forecaster.predict_df(
            _context(items=("a",)).drop(columns="item_id"), prediction_length=HORIZON
        )
    # A recipe that cuts is refused up front.
    with pytest.raises(ValueError, match="cut_features=False"):
        _forecaster(tiny_regressor, recipe=SeldonRecipe())


def test_clip_margin_warns(tiny_regressor):
    # The clamp would cap a trend at the context's range: allowed, but warned.
    with pytest.warns(UserWarning, match="clip_margin"):
        _forecaster(tiny_regressor, clip_margin=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _forecaster(tiny_regressor)


def test_recipe_raises_on_the_columns_the_model_reads(tiny_regressor):
    # 12 columns, 4 of them constant: the recipe drops those before counting.
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 8)), columns=[f"x{i}" for i in range(8)])
    X[[f"c{i}" for i in range(4)]] = 1.0
    y = rng.normal(size=60)
    for max_features, raises in ((8, False), (7, True)):
        recipe = SeldonRecipe(max_features=max_features, cut_features=False)
        reg = SeldonRegressor(tiny_regressor, n_estimators=2, device="cpu", recipe=recipe)
        if raises:
            with pytest.raises(TooManyFeatures, match="8 feature columns"):
                reg.fit(X, y)
        else:
            reg.fit(X, y)
    # The default cuts, as before.
    reg = SeldonRegressor(
        tiny_regressor, n_estimators=2, device="cpu", recipe=SeldonRecipe(max_features=7)
    )
    assert reg.fit(X, y).predict(X[:3]).shape == (3,)


def test_known_and_past_only_covariates(tiny_regressor):
    forecaster = _forecaster(tiny_regressor)
    context = _context(covariates=True)
    out = forecaster.predict_df(context, future_df=_future(context))
    assert len(out) == 2 * HORIZON and np.isfinite(out["target"]).all()
    with pytest.warns(UserWarning, match="missing"):
        future = _future(context)
        future.loc[0, "promo"] = np.nan
        forecaster.predict_df(context, future_df=future)


def test_horizon_only_column_keeps_its_type(tiny_regressor, monkeypatch):
    seen = []
    original = SeldonRegressor.fit

    def spy(self, X, y, **kwargs):
        seen.append(kwargs["stypes"])
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(SeldonRegressor, "fit", spy)
    context = _context(items=("a",))
    future = _future(context, promo=False).assign(event=pd.Categorical(["x", "y"] * (HORIZON // 2)))
    _forecaster(tiny_regressor).predict_df(context, future_df=future)
    # Not a generated feature: sdm infers its type, it is not forced numerical.
    assert "event" not in seen[0] and seen[0]["running_index"] == "numerical"


def test_predict_df_input_checks(tiny_regressor):
    forecaster = _forecaster(tiny_regressor)
    context = _context(items=("a",))
    with pytest.raises(ValueError, match="exactly one"):
        forecaster.predict_df(context)
    irregular = context.drop(index=[5, 9])
    with pytest.raises(ValueError, match="future_df"):
        forecaster.predict_df(irregular, prediction_length=3)
    with pytest.raises(ValueError, match="timestamp twice"):
        forecaster.predict_df(pd.concat([context, context.iloc[:1]]), prediction_length=3)
    with pytest.raises(ValueError, match="without context"):
        future = _future(context).assign(item_id="z")
        forecaster.predict_df(context, future_df=future)
    with pytest.raises(ValueError, match="quantiles"):
        forecaster.predict_df(context, prediction_length=3, quantiles=[1.5])
    with pytest.raises(ValueError, match="prediction_length must be at least 1, got 0"):
        forecaster.predict_df(context, prediction_length=0)
    with pytest.raises(ValueError, match="no 'timestamp' column"):
        forecaster.predict_df(context.drop(columns="timestamp"), prediction_length=3)
    with pytest.raises(ValueError, match="every 'target' must be NaN"):
        forecaster.predict_df(context, future_df=_future(context).assign(target=1.0))
    indexed, future = _indexed(context), _indexed(_future(context))
    with pytest.raises(ValueError, match=r"context must be indexed by \('item_id', 'timestamp'\)"):
        forecaster.predict(indexed.reset_index(level="item_id"), future)
    with pytest.raises(ValueError, match="context has no 'target' column"):
        forecaster.predict(indexed.drop(columns="target"), future)


def test_predict_reads_string_timestamps(tiny_regressor):
    forecaster = _forecaster(tiny_regressor)
    context = _indexed(_context(n=48, items=("a",)))
    future = _indexed(_future(_context(n=48, items=("a",)), promo=False))
    as_text = context.set_axis(
        pd.MultiIndex.from_arrays(
            [context.index.get_level_values(0), context.index.get_level_values(1).astype(str)],
            names=context.index.names,
        )
    )
    pd.testing.assert_frame_equal(
        forecaster.predict(as_text, future), forecaster.predict(context, future)
    )


def test_constructor_checks(tiny_regressor):
    regressor = _forecaster(tiny_regressor).regressor
    with pytest.raises(TypeError, match="must be a SeldonRegressor, got Seldon"):
        SeldonForecaster(tiny_regressor)
    with pytest.raises(ValueError, match="max_context_length must be None or at least 1, got 0"):
        SeldonForecaster(regressor, max_context_length=0)
    with pytest.raises(ValueError, match="point_forecast must be 'median' or 'mean', got 'mode'"):
        SeldonForecaster(regressor, point_forecast="mode")
    forecaster = SeldonForecaster(regressor)
    assert forecaster.to("cpu") is forecaster and forecaster.regressor.device.type == "cpu"


def test_unsorted_context_is_sorted_per_series(tiny_regressor):
    forecaster = _forecaster(tiny_regressor)
    context = _context()
    shuffled = context.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(
        forecaster.predict_df(context, prediction_length=HORIZON),
        forecaster.predict_df(shuffled, prediction_length=HORIZON),
    )


def test_single_series_without_item_id(tiny_regressor):
    out = (
        _forecaster(tiny_regressor)
        .predict_df(_context(items=("a",)).drop(columns="item_id"), prediction_length=HORIZON)
        .reset_index()
    )
    assert (out["item_id"] == 0).all() and len(out) == HORIZON


def test_forecast_protocol(tiny_regressor):
    forecaster = _forecaster(tiny_regressor)
    context = _context(items=("a",), covariates=True).set_index("timestamp")
    y = pd.DataFrame({"sales": context["target"], "returns": -context["target"]})
    x = context[["promo", "temp"]]  # temp is past-only
    query = _indexed(_future(_context(items=("a",)))).loc["a"]
    out = forecaster.forecast(x, y, query, quantiles=[0.1, 0.5, 0.9])
    assert list(out.columns) == [
        "sales", "sales__q10", "sales__q50", "sales__q90",
        "returns", "returns__q10", "returns__q50", "returns__q90",
    ]  # fmt: skip
    assert out.index.equals(query.index)
    # Each target is forecast on its own: as alone.
    alone = forecaster.forecast(x, y[["sales"]], query, quantiles=[0.1, 0.5, 0.9])
    pd.testing.assert_frame_equal(out[alone.columns], alone)
    with pytest.raises(ValueError, match="share a column name"):
        forecaster.forecast(x, y, query, quantiles=[0.101, 0.102])


def test_forecast_protocol_inputs(tiny_regressor):
    forecaster = _forecaster(tiny_regressor)
    context = _context(n=48, items=("a",), covariates=True).set_index("timestamp")
    stamps = _indexed(_future(_context(n=48, items=("a",)))).loc["a"].index
    # A Series is one target, named after it (``target`` unnamed); a DatetimeIndex is the horizon.
    out = forecaster.forecast(None, context["target"].rename(None), stamps, quantiles=[0.5])
    assert list(out.columns) == ["target", "target__q50"] and out.index.equals(stamps)
    y = context[["target"]]
    x = context[["promo"]]
    query = pd.DataFrame({"promo": 1.0}, index=stamps)
    checks = [
        ((x, y.reset_index(drop=True), query), "y_context must be indexed by timestamps"),
        ((x, y, query.reset_index(drop=True)), "x_query must be indexed by timestamps"),
        ((x[1:], y, query), "x_context and y_context must share their index"),
        ((x, y, query.assign(temp=0.0)), r"x_query has columns x_context does not: \['temp'\]"),
        ((x.assign(target=0.0), y, query), r"x_context columns named like a target: \['target'\]"),
    ]
    for args, message in checks:
        with pytest.raises(ValueError, match=message):
            forecaster.forecast(*args)


def test_at_levels():
    q = np.arange(9, dtype=np.float64)[None]  # levels 0.1, ..., 0.9
    np.testing.assert_allclose(
        _at_levels(q, [0.1, 0.5, 0.9, 0.15, 0.0, 1.0]), [[0, 4, 8, 0.5, 0, 8]]
    )
    q = np.arange(999, dtype=np.float64)[None]  # the released heads: levels 0.001, ..., 0.999
    np.testing.assert_array_equal(
        _at_levels(q, [0.1, 0.3, 0.5, 0.7, 0.9, 0.025]), [[99, 299, 499, 699, 899, 24]]
    )


# --- Configs and the regressor's stypes ------------------------------------------------


@pytest.mark.parametrize("name", ["seldon-forecaster", "seldon-mini-forecaster"])
def test_packaged_forecaster_configs(name):
    config = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    regressor = yaml.safe_load(
        (CONFIGS / f"{name.replace('forecaster', 'regressor')}.yaml").read_text()
    )
    assert config["clip_margin"] is None and config["max_context_length"] == 32768
    assert config["recipe"].pop("cut_features") is False
    assert config["max_context_length"] <= config["max_context_size"]
    # The regressor's settings otherwise.
    for key in ("checkpoint", "n_estimators", "recipe", "amp", "random_state"):
        assert config[key] == regressor[key]
    # Not offered to the classifier or the regressor.
    with pytest.raises(ValueError, match="unknown"):
        SeldonClassifier.from_config(name)
    with pytest.raises(ValueError, match="unknown"):
        SeldonRegressor.from_config(name)


@pytest.mark.parametrize("name", ["seldon-mini", "seldon-mini-regressor"])
def test_from_pretrained_reads_only_the_forecaster_config(name, tiny_regressor, monkeypatch):
    loaded = []

    def fake_load(name_or_path, **kwargs):
        loaded.append((str(name_or_path), kwargs["task"]))
        return tiny_regressor

    def refuse(*args, **kwargs):
        raise AssertionError("the regressor's packaged config was read")

    monkeypatch.setattr(Seldon, "from_pretrained", fake_load)
    monkeypatch.setattr(SeldonRegressor, "from_pretrained", refuse)
    forecaster = SeldonForecaster.from_pretrained(name, device="cpu")
    config = yaml.safe_load((CONFIGS / "seldon-mini-forecaster.yaml").read_text())
    reg = forecaster.regressor
    assert loaded == [("seldon-mini-regressor", "regression")]
    assert reg.clip_margin is None and reg.n_estimators == config["n_estimators"]
    assert reg.max_context_size == config["max_context_size"]
    assert reg.estimator.recipe.half_shuffle and not reg.estimator.recipe.cut_features
    assert forecaster.max_context_length == config["max_context_length"]
    # Keyword arguments override the config.
    forecaster = SeldonForecaster.from_pretrained(name, device="cpu", n_estimators=2)
    assert forecaster.regressor.n_estimators == 2


def test_from_config_file(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        f"checkpoint: {TINY}\nn_estimators: 2\ndevice: cpu\nclip_margin: null\n"
        "max_context_length: 64\npoint_forecast: mean\n"
        "recipe:\n  half_shuffle: true\n  cut_features: false\n"
    )
    forecaster = SeldonForecaster.from_config(path)
    assert forecaster.max_context_length == 64 and forecaster.point_forecast == "mean"
    assert forecaster.regressor.n_estimators == 2 and forecaster.regressor.clip_margin is None
    assert forecaster.regressor.estimator.recipe.half_shuffle
    out = forecaster.predict_df(_context(items=("a",)), prediction_length=3)
    assert len(out) == 3
    with pytest.raises(ValueError, match="unknown forecaster config"):
        SeldonForecaster.from_config("seldon")
    path.write_text("n_estimators: 2\n")
    with pytest.raises(ValueError, match="names no checkpoint"):
        SeldonForecaster.from_config(path)


def test_from_config_packaged_name(tiny_regressor, monkeypatch):
    loaded = []
    monkeypatch.setattr(
        Seldon, "from_pretrained", lambda name, **kwargs: loaded.append(name) or tiny_regressor
    )
    forecaster = SeldonForecaster.from_config("seldon-forecaster", device="cpu", n_estimators=2)
    config = yaml.safe_load((CONFIGS / "seldon-forecaster.yaml").read_text())
    # The config's "seldon" names the regression checkpoint, as from_pretrained's names do.
    assert config["checkpoint"] == "seldon" and loaded == ["seldon-regressor"]
    assert forecaster.max_context_length == config["max_context_length"]
    assert forecaster.regressor.n_estimators == 2  # a keyword argument over the config


def test_regressor_stypes_override(tiny_regressor):
    # Two values over 200 rows: sdm reads it as categorical.
    X = pd.DataFrame({"year": [2024.0] * 100 + [2025.0] * 100, "x": np.linspace(0, 1, 200)})
    y = np.linspace(0, 1, 200)
    reg = SeldonRegressor(tiny_regressor, n_estimators=2, device="cpu")
    assert reg.fit(X, y)._stypes["year"] == "categorical"  # sdm's inference
    assert reg.fit(X, y, stypes={"year": "numerical"})._stypes["year"] == "numerical"
    with pytest.raises(ValueError, match="does not have"):
        reg.fit(X, y, stypes={"nope": "numerical"})
