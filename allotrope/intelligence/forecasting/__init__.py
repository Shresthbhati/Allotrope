"""Forecasting of the plant's observed demand/availability signals.

Forecasts `electrical_load_kw`, `firm_thermal_kw`, `pv_available_kw` and
`wind_available_kw` -- the four series `PolarMicrogridEnv._observe` reads
from `plant.observe()` -- at 1-step and 24-step (1h/24h, at this project's
1h dispatch interval) horizons, using only strictly-past data.

`EWMAForecaster` at the 24h horizon is now consumed directly by
`allotrope.envs.polar_microgrid.PolarMicrogridEnv._observe` (two station-
level observation features: a load forecast and a combined renewables
forecast) -- the one horizon this module's own evaluation
(`docs/forecasting.md`) found EWMA actually beats plain persistence on, so
it is the one place forecasting has been shown to add information the
current instantaneous reading doesn't already carry. `allotrope.safety`
still does not consume it, and does not need to: the projection layer
bounds the *current* step's action, not a forecast of a future one.
"""

from allotrope.intelligence.forecasting.evaluation import (
    ForecastMetrics,
    compute_metrics,
    evaluate_forecaster,
)
from allotrope.intelligence.forecasting.forecasters import (
    EWMAForecaster,
    Forecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)

__all__ = [
    "Forecaster",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "EWMAForecaster",
    "ForecastMetrics",
    "compute_metrics",
    "evaluate_forecaster",
]
