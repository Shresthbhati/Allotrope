"""Short-horizon forecasters for the four signals the controller cares about.

`PolarMicrogridEnv._observe` reads `electrical_load_kw`, `firm_thermal_kw`,
`pv_available_kw` and `wind_available_kw` from `plant.observe()` on every
step. A controller that could see a few hours or a day of where those are
headed -- rather than only their instantaneous value -- could pre-position
storage and commit gensets ahead of a load ramp instead of reacting to it.
This module is that forecasting capability, built and evaluated standalone.
`EWMAForecaster` at the 24h horizon is now wired into
`allotrope.envs.polar_microgrid.PolarMicrogridEnv._observe`; see
`docs/forecasting.md` for exactly what is and is not done.

Two honest, non-learned methods are implemented, both cheap enough to run on
station hardware without touching the satellite link:

  * `PersistenceForecaster` -- tomorrow looks like right now. The floor any
    other method must beat, not a strawman: `allotrope/control/baseline.py`
    already documents that this project's actual incumbent practice runs on
    exactly this kind of no-forecast reasoning.
  * `SeasonalNaiveForecaster` -- tomorrow looks like the corresponding hour
    one cycle ago. `allotrope/synth/climate.py` builds solar and wind from a
    diurnal cycle plus slower AR(1)/seasonal drift, and `allotrope/synth/
    loads.py` builds electrical and thermal demand from a diurnal activity
    curve plus crew occupancy; exploiting a period the generator itself uses
    is honest forecasting, not access to future values -- the forecaster is
    still only ever given data strictly before the time it predicts.
  * `EWMAForecaster` -- an exponentially-weighted moving average, a
    complementary statistical baseline for signals (or horizons) where the
    diurnal period is a poor guide, e.g. a signal in the middle of a
    multi-day cold snap or blizzard that seasonal-naive would ignore.

All three share one interface: `forecast(history, horizon) -> float`, where
`history` is a 1-D array of chronologically ordered past observations ending
at "now" (`history[-1]`), and `horizon` is how many steps past "now" to
predict. No forecaster ever receives an array that extends past "now" --
enforcing that is the caller's job (see `evaluation.evaluate_forecaster`,
which slices `history` from a single ground-truth series so it is
structurally impossible to leak).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    """A point forecaster: strictly-past history in, one future value out."""

    name: str

    def forecast(self, history: np.ndarray, horizon: int) -> float:
        ...


@dataclass
class PersistenceForecaster:
    """Naive baseline: the value `horizon` steps ahead equals the last observed value.

    This is the honest floor every other method here is measured against. If
    a fancier forecaster cannot beat this on a real signal, it is not adding
    information.
    """

    name: str = "persistence"

    def forecast(self, history: np.ndarray, horizon: int) -> float:
        if len(history) == 0:
            raise ValueError("persistence forecaster needs at least one observation")
        return float(history[-1])


@dataclass
class SeasonalNaiveForecaster:
    """Same-phase-last-cycle: the value `period` steps before the target.

    `period` should match a periodicity the underlying process actually has
    (24 for an hourly series' diurnal cycle, for instance). When there is not
    yet enough history to reach back a full period from the target time, this
    falls back to persistence rather than fabricating a number from data that
    does not exist -- that fallback is exercised for the first `period`
    steps of any run and is exactly as honest as `PersistenceForecaster`
    itself for those steps.
    """

    period: int
    name: str = "seasonal_naive"

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")

    def forecast(self, history: np.ndarray, horizon: int) -> float:
        if len(history) == 0:
            raise ValueError("seasonal-naive forecaster needs at least one observation")
        n = len(history)
        # Index, within `history`, of the observation one period before the
        # target time t+horizon. `history[-1]` is time t, so the target sits
        # at conceptual index n-1+horizon, and one period earlier is
        # n-1+horizon-period.
        idx = n - 1 + horizon - self.period
        if idx < 0:
            return float(history[-1])
        return float(history[idx])


@dataclass
class EWMAForecaster:
    """Exponentially-weighted moving average, held flat over the horizon.

    `alpha` in (0, 1] is the weight on the most recent observation; smaller
    values average over a longer effective window. This is a standard,
    non-learned statistical smoother -- not a model dressed up as one -- and
    it is the honest choice when a signal is dominated by slow drift (e.g. an
    AR(1) cold snap) rather than by the diurnal cycle `SeasonalNaiveForecaster`
    exploits.
    """

    alpha: float = 0.3
    name: str = "ewma"

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

    def forecast(self, history: np.ndarray, horizon: int) -> float:
        if len(history) == 0:
            raise ValueError("EWMA forecaster needs at least one observation")
        # `adjust=False` is the standard recursive form
        # level[t] = alpha*x[t] + (1-alpha)*level[t-1]; pandas evaluates the
        # whole recursion in C, which matters here since the evaluator calls
        # this once per timestep over a growing history.
        level = pd.Series(history).ewm(alpha=self.alpha, adjust=False).mean().iloc[-1]
        # Held flat: an EWMA has no trend term, so its best estimate of any
        # future step is simply its current smoothed level.
        return float(level)


__all__ = [
    "Forecaster",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "EWMAForecaster",
]
