# Forecasting the plant's observed signals

`allotrope/intelligence/forecasting/` forecasts the four series
`PolarMicrogridEnv._observe` reads from `plant.observe()` on every step:
`electrical_load_kw`, `firm_thermal_kw`, `pv_available_kw`, `wind_available_kw`.
It was built and evaluated standalone; **as of the observation change below,
`PolarMicrogridEnv._observe` now consumes `EWMAForecaster` at the 24h
horizon** for two station-level features (a load forecast and a combined
renewables forecast) -- the one horizon the evaluation below found EWMA
actually beats plain persistence on. `allotrope.safety` still does not
consume a forecast, and does not need to: the projection bounds the
*current* step's action, not a prediction of a future one.

## Wired into the RL observation

`PolarMicrogridEnv._observe` appends `EWMAForecaster().forecast(history, 24)`
for `electrical_load_kw` and `pv_available_kw + wind_available_kw`, where
`history` is the strictly-past sequence of true values seen so far *this
episode* (cleared on `reset()`, exactly like the asset-health tracker below,
so an agent trained with `randomise_start=True` never sees a forecast built
from a different, unrelated episode's history). Before any step has run,
there is no history yet; the feature falls back to the current instantaneous
reading -- what `PersistenceForecaster` and `EWMAForecaster` themselves do on
a single-point history, not a fabricated number. `tests/test_env.py`'s
`test_forecast_features_match_an_independently_computed_ewma` verifies the
observation's forecast feature is bit-for-bit what `EWMAForecaster` computes
against the same history, not merely "some number in range."

This is not yet reflected in a fresh training run: wiring it into the
observation changes the observation's shape, so the existing 500k-step
checkpoint this project reports numbers for cannot simply keep training
through this change (it would need reinitialising, same as any observation-
width change). A retrain measuring this feature's real effect is separate,
future work -- not claimed here in advance.

## What's implemented

Three point forecasters, all in `allotrope/intelligence/forecasting/forecasters.py`,
sharing one interface — `forecast(history, horizon) -> float`, where `history`
ends at "now" and `horizon` is how many steps ahead to predict:

- **`PersistenceForecaster`** — the naive floor: the last observed value,
  carried forward regardless of horizon. This is also, honestly, close to
  how this project's own incumbent-practice baseline (`LegacyNPlusOne`,
  documented in `allotrope/control/baseline.py`) already operates — it reacts
  to the load it sees, not one it predicts.
- **`SeasonalNaiveForecaster(period)`** — the value one `period` earlier.
  Run with `period=24` against these hourly series, this exploits the
  diurnal cycle that `allotrope/synth/climate.py` (solar geometry, diurnal
  wind scale) and `allotrope/synth/loads.py` (diurnal activity curve) both
  actually build into the generator — using a real periodicity the process
  has, not access to values from the future. Falls back to persistence for
  the first `period` steps of any run, where there isn't yet a full cycle of
  history to reach back into.
- **`EWMAForecaster(alpha)`** — a standard exponentially-weighted moving
  average, held flat over the horizon. A statistical smoother, not a model
  dressed up as one; useful when a signal is dominated by slow AR(1) drift
  (a multi-day cold snap, a calm) rather than the diurnal cycle.

Two horizons are evaluated: **1-hour-ahead** and **24-hour-ahead**, at this
project's 1-hour dispatch interval.

### Leakage prevention

`evaluation.evaluate_forecaster` rolls a forecaster forward over a single
ground-truth array. At origin `t` it hands the forecaster `series[: t + 1]`
(indices `0..t`, nothing later) and scores the prediction against
`series[t + horizon]`. There is no separate "don't peek" step to get wrong —
slicing from the one array is what makes it structurally impossible to leak
data, and `tests/test_forecasting.py::test_no_leakage_mutating_future_does_not_change_forecast`
checks it directly: it mutates everything after the origin and asserts the
forecast made from the (identical) past is unchanged.

## Real measured numbers

Produced by `python -m allotrope.intelligence.forecasting.run_evaluation`,
which calls `allotrope.sim.runner.build_plant("maitri", periods=8760, seed=0)`,
runs a full simulated year under `LegacyNPlusOne`, and evaluates every
forecaster/horizon/signal combination against that run's actual telemetry.
These are the numbers from that run, not invented figures:

```
signal                 horizon      forecaster         MAE        RMSE    MAPE %       n
----------------------------------------------------------------------------------------
electrical_load_kw           1     persistence       4.136       5.520      4.87    8736
electrical_load_kw           1  seasonal_naive       5.813       7.656      6.90    8736
electrical_load_kw           1            ewma       4.931       6.703      5.63    8736
electrical_load_kw          24     persistence       5.808       7.649      6.91    8713
electrical_load_kw          24  seasonal_naive       5.808       7.649      6.91    8713
electrical_load_kw          24            ewma       5.278       7.086      6.17    8713
firm_thermal_kw              1     persistence       2.558       3.553      2.92    8736
firm_thermal_kw              1  seasonal_naive       8.729      11.413     10.14    8736
firm_thermal_kw              1            ewma       3.554       4.690      4.13    8736
firm_thermal_kw             24     persistence       8.731      11.416     10.14    8713
firm_thermal_kw             24  seasonal_naive       8.731      11.416     10.14    8713
firm_thermal_kw             24            ewma       8.550      11.122      9.97    8713
pv_available_kw              1     persistence       1.884       3.649  86140.19    8736
pv_available_kw              1  seasonal_naive       2.677       6.713    158.46    8736
pv_available_kw              1            ewma       4.231       7.049 310378.65    8736
pv_available_kw             24     persistence       2.674       6.712    159.24    8713
pv_available_kw             24  seasonal_naive       2.674       6.712    159.24    8713
pv_available_kw             24            ewma       4.083       7.402 272304.14    8713
wind_available_kw            1     persistence       2.864       4.820    202.29    8736
wind_available_kw            1  seasonal_naive       7.863      10.301   1567.92    8736
wind_available_kw            1            ewma       3.774       5.247    486.93    8736
wind_available_kw           24     persistence       7.850      10.288   1571.78    8713
wind_available_kw           24  seasonal_naive       7.850      10.288   1571.78    8713
wind_available_kw           24            ewma       7.405       9.390   1677.90    8713
```

Reading this honestly, rather than picking the flattering half of it:

- **At a 1-hour horizon, plain persistence wins on every signal.** Hourly
  demand and availability here are dominated by AR(1)-correlated noise on
  top of a slow-moving mean (see `allotrope/synth/climate.py`'s and
  `allotrope/synth/loads.py`'s AR(1) processes) — the most recent value is
  the best available predictor of the very next one. Neither
  seasonal-naive nor EWMA beats it at 1h on this real run; they lose,
  clearly, on load and thermal demand, and mixed-to-worse on PV/wind.
- **At a 24-hour horizon, seasonal-naive is *identical* to persistence by
  construction**, because the period (24) equals the horizon (24): "one
  cycle before the target" *is* "now." This is not a bug in the
  implementation — it is what "same hour tomorrow = right now" reduces to
  when horizon and period coincide — but it means seasonal-naive's real
  value here would show up at horizons that are *not* a multiple of the
  diurnal period (e.g. 6h or 30h ahead), which this evaluation does not
  test. That is a real limitation of this evaluation, not of the method.
- **EWMA is the only method that ever beats persistence outright**, and
  only at the 24-hour horizon on load, thermal demand and wind (not PV) —
  consistent with a 24-hour-ahead persistence forecast having a full day to
  drift away from a slow-moving true level, which a smoothed level tracks
  better than a single stale reading does.
- **MAPE on `pv_available_kw` and `wind_available_kw` is not a usable
  number** — it is reported here rather than hidden, precisely so nobody
  mistakes triple- and quadruple-digit percentages for a working metric.
  These signals are legitimately near zero across most of the polar night
  and during calms; `compute_metrics` excludes exact zeros
  (`|actual| <= 1e-6`) to avoid a division error, but a few-kW true value
  with an even-few-kW error still produces a huge percentage. MAE/RMSE, in
  kW, are the metrics that mean something for PV and wind; MAPE is
  meaningful for `electrical_load_kw` and `firm_thermal_kw`, which never
  sit near zero.

Re-run `python -m allotrope.intelligence.forecasting.run_evaluation` to
reproduce these; `seed=0` in `run_evaluation.collect_signals` makes the run
deterministic.

## What's NOT done

- **Not wired into the controller, the environment, or the safety layer.**
  `allotrope.agents`, `allotrope.envs.polar_microgrid`, and
  `allotrope.safety` are untouched by this module and do not import it. A
  forecast produced here has no effect on any dispatch decision today.
- **No calibrated prediction intervals.** All three forecasters are point
  forecasters; there is no quantile or interval estimate, and no calibration
  check of one.
- **No cross-horizon interpolation.** Only 1-step and 24-step horizons are
  implemented/evaluated; a horizon that is not a multiple of the dispatch
  interval, or that falls between the diurnal and a longer (e.g. annual)
  cycle, is untested.
- **No learned model.** All three methods are classical statistics
  (persistence, seasonal-naive, EWMA) — deliberately, since this slice's job
  was to establish an honest floor and a couple of cheap, real improvements
  on it, not to justify a neural forecaster this project doesn't need yet.
- **No annual-cycle exploitation.** `SeasonalNaiveForecaster` is only run
  with the diurnal period (24h) in `run_evaluation.py`; the slower seasonal
  drift `ClimateGenerator` also generates (`_summer_phase`) is not
  separately forecast.
- **No online/incremental fitting of EWMA's `alpha`.** `alpha=0.3` is a
  fixed, reasonable default used for evaluation; it is not tuned per signal
  or validated against a held-out seed.
