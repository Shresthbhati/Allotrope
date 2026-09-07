"""The Gymnasium environment: the plant, as a reinforcement-learning problem.

The action space is a Dict rather than a flattened Box, because the control
problem genuinely is hybrid and flattening it would hide that. Which sets are
committed is a discrete decision with minimum up and down times attached; how
hard they are worked, and what storage does, is continuous. Those are different
problems and the project solves them with different algorithms -- DQN for the
switching, SDDPG for the dispatch -- so the environment presents them as
different spaces.

The safety projection is applied inside `step`, not left to the agent. This is
deliberate and has a consequence worth stating plainly: the agent trains against
a plant it *cannot* damage, so its exploration is safe from the first random
action, and the policy that results has never needed to learn the constraints
that were enforced for it. What it learns is how to be efficient inside them.

Observations are normalised into roughly unit range, and the normalisation
constants are all drawn from the station configuration, so an agent trained at
Maitri sees the same numerical scale at Bharati despite the plants differing in
size by a factor of two.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from allotrope.config import StationConfig, load_station
from allotrope.envs.reward import RewardFunction, RewardWeights
from allotrope.intelligence.asset_health import AssetHealthTracker
from allotrope.intelligence.forecasting import EWMAForecaster
from allotrope.safety.projection import SafetyProjection
from allotrope.sim.plant import DispatchCommand, PolarMicrogrid
from allotrope.sim.runner import build_plant

SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.25

# How far ahead the observation's forecast features look. 24 steps (24h at
# this project's 1h dispatch interval) is not an arbitrary choice: 1-step
# forecasts add nothing an agent doesn't already see in the current
# observation (docs/forecasting.md measures plain persistence winning at 1h
# on every signal, since the forecast and "now" are then the same number),
# while 24h is exactly the horizon that same evaluation found EWMA actually
# beats persistence on -- the one place forecasting has been shown to add
# real information the instantaneous reading doesn't already carry.
FORECAST_HORIZON_STEPS = 24


class PolarMicrogridEnv(gym.Env):
    """A polar station microgrid as a Gymnasium environment.

    Action
        `genset_on`  MultiBinary(n_gensets)   -- commitment, the discrete layer
        `dispatch`   Box(-1, 1, n_gensets + n_storage + 1) -- the continuous layer:
                     one loading fraction per set, one power fraction per pack,
                     and one melting rate.

    Observation
        A flat Box. Demand, renewables, weather, storage state, machine state and
        cyclical time features, each scaled by a constant from the station config.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        station: str | StationConfig = "maitri",
        start: str = "2026-01-01",
        periods: int = 8760,
        freq: str = "1h",
        seed: int | None = 0,
        weights: RewardWeights | None = None,
        apply_safety: bool = True,
        episode_steps: int | None = None,
        randomise_start: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = station if isinstance(station, StationConfig) else load_station(station)
        self._build_kwargs = dict(start=start, periods=periods, freq=freq)
        self._seed = seed
        self.apply_safety = apply_safety
        self.episode_steps = episode_steps
        self.randomise_start = randomise_start

        self.plant: PolarMicrogrid = build_plant(self.cfg, seed=seed, **self._build_kwargs)
        self.projection = SafetyProjection(self.cfg)
        self.reward_fn = RewardFunction(weights)

        n_gensets = len(self.cfg.gensets)
        n_storage = len(self.cfg.storage)
        self.action_space = spaces.Dict(
            {
                "genset_on": spaces.MultiBinary(n_gensets),
                "dispatch": spaces.Box(
                    low=-1.0, high=1.0, shape=(n_gensets + n_storage + 1,), dtype=np.float32
                ),
            }
        )
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self._observation_width(),), dtype=np.float32
        )

        self._episode_start = 0
        self._prev_deposit = 0.0
        self._prev_unmet_water = 0.0
        self.last_breakdown = None
        self.last_safety_report = None

        # Read-only, per-episode: accumulates exactly like
        # allotrope.intelligence.asset_health's own tracker (same class,
        # same formulas -- no second bookkeeping system), reset fresh each
        # episode so the wear/FEC features an agent sees reflect only what
        # happened since the current episode started, matching how an
        # agent trained on randomised episode starts (`randomise_start`)
        # cannot see stress accumulated in a different, unrelated window.
        self._asset_tracker = AssetHealthTracker(self.cfg, dt_h=self.plant.dt_h)
        self._forecaster = EWMAForecaster()
        self._load_history: list[float] = []
        self._renewable_history: list[float] = []

    # -- scaling ----------------------------------------------------------

    @property
    def _power_scale_kw(self) -> float:
        """One number to scale every power quantity, so stations stay comparable."""
        return max(self.cfg.total_genset_kw, 1.0)

    def _observation_width(self) -> int:
        # +2 station-level forecast features (load, renewables); +1 per
        # genset (wear score) and +1 per battery (full-equivalent-cycles) --
        # both already computed by allotrope.intelligence.asset_health,
        # wired in here rather than duplicated.
        return 14 + 6 * len(self.cfg.gensets) + 3 * len(self.cfg.storage)

    # -- gym api ----------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.plant = build_plant(self.cfg, seed=seed, **self._build_kwargs)

        start_index = 0
        if self.randomise_start and self.episode_steps:
            latest = max(self.plant.n_steps - self.episode_steps - 1, 0)
            start_index = int(self.np_random.integers(0, latest + 1)) if latest else 0

        initial_soc = None
        if options and "initial_soc" in options:
            initial_soc = float(options["initial_soc"])
        self.plant.reset(start_index=start_index, initial_soc=initial_soc)

        self._episode_start = start_index
        self._prev_deposit = self._mean_deposit()
        self._prev_unmet_water = self.plant.state.unmet_water_kwh
        self.last_breakdown = None
        self.last_safety_report = None
        self._asset_tracker = AssetHealthTracker(self.cfg, dt_h=self.plant.dt_h)
        self._load_history = []
        self._renewable_history = []
        return self._observe(), {"start_index": start_index}

    def step(
        self, action: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        command = self.decode_action(action)

        info: dict[str, Any] = {}
        if self.apply_safety:
            command, report = self.projection.project(command, self.plant.observe(), self.plant)
            self.last_safety_report = report
            info["safety"] = report.as_dict()

        telemetry = self.plant.step(command)
        telemetry["min_indoor_temp_c"] = self.cfg.criticality.min_indoor_temp_c
        self._asset_tracker.update(telemetry)
        self._load_history.append(telemetry["electrical_load_kw"])
        self._renewable_history.append(telemetry["pv_available_kw"] + telemetry["wind_available_kw"])

        deposit = self._mean_deposit()
        unmet_water = self.plant.state.unmet_water_kwh
        day_rolled_over = unmet_water > self._prev_unmet_water

        reward, breakdown = self.reward_fn(
            telemetry,
            dt_h=self.plant.dt_h,
            deposit_delta=deposit - self._prev_deposit,
            day_rolled_over=day_rolled_over,
            unmet_water_kwh=unmet_water - self._prev_unmet_water,
        )
        self._prev_deposit = deposit
        self._prev_unmet_water = unmet_water
        self.last_breakdown = breakdown

        steps_taken = self.plant.state.step_index - self._episode_start
        truncated = bool(self.episode_steps and steps_taken >= self.episode_steps)
        terminated = self.plant.done

        info.update({"telemetry": telemetry, "reward_breakdown": breakdown.as_dict()})
        observation = (
            self._observe() if not terminated else np.zeros(self._observation_width(), np.float32)
        )
        return observation, float(reward), terminated, truncated, info

    # -- action decoding --------------------------------------------------

    def decode_action(
        self, action: dict[str, Any], obs: dict[str, Any] | None = None
    ) -> DispatchCommand:
        """Map a normalised agent action onto a physical dispatch command.

        Continuous values arrive in [-1, 1]. Loading fractions are mapped onto
        each set's own stable band rather than onto [0, rating], so that the
        agent cannot express an unreachable setpoint and does not have to learn
        where the minimum stable load is -- it is a property of the machine, and
        the environment already knows it.

        Accepts an already-fetched `plant.observe()` dict for the same reason
        `_observe` does: a caller that already has one (`HybridAgent`) should
        not have to trigger a second call to get the battery envelope this
        needs.
        """
        genset_on = np.asarray(action["genset_on"]).astype(bool).ravel()
        dispatch = np.asarray(action["dispatch"], dtype=float).ravel()

        n_g = len(self.cfg.gensets)
        n_s = len(self.cfg.storage)
        loading = np.clip(dispatch[:n_g], -1.0, 1.0)
        storage = np.clip(dispatch[n_g : n_g + n_s], -1.0, 1.0)
        melt = float(np.clip(dispatch[n_g + n_s], -1.0, 1.0))

        setpoints = []
        for k, g in enumerate(self.cfg.gensets):
            span = g.rated_kw - g.min_stable_kw
            setpoints.append(g.min_stable_kw + span * (loading[k] + 1.0) / 2.0)

        observation = self.plant.observe() if obs is None else obs
        battery = []
        for k in range(n_s):
            limit = (
                observation["battery_max_discharge_kw"][k]
                if storage[k] >= 0
                else observation["battery_max_charge_kw"][k]
            )
            battery.append(float(storage[k] * limit))

        melt_kw = self.projection.melt_ceiling_kw() * (melt + 1.0) / 2.0

        return DispatchCommand(
            genset_on=tuple(bool(v) for v in genset_on),
            genset_setpoint_kw=tuple(float(v) for v in setpoints),
            battery_kw=tuple(battery),
            snow_melt_kw=float(melt_kw),
        )

    # -- observation ------------------------------------------------------

    def _observe(self, obs: dict[str, Any] | None = None) -> np.ndarray:
        """Encode the plant's raw observation into the flat feature vector.

        Accepts an already-fetched `plant.observe()` dict so a caller that has
        one in hand (`HybridAgent`, notably) is not forced to trigger a second,
        redundant call just to get it encoded.
        """
        obs = self.plant.observe() if obs is None else obs
        cfg = self.cfg
        scale = self._power_scale_kw
        timestamp = obs["timestamp"]

        seconds = timestamp.hour * 3600 + timestamp.minute * 60
        day_angle = 2.0 * np.pi * seconds / SECONDS_PER_DAY
        year_angle = 2.0 * np.pi * timestamp.dayofyear / DAYS_PER_YEAR

        # Forecast features: EWMA at FORECAST_HORIZON_STEPS ahead, the one
        # horizon docs/forecasting.md's own evaluation found actually beats
        # plain persistence (see that module's real measured numbers) --
        # not the instantaneous current value already carried above, which
        # is what a 1-step "forecast" would just duplicate. Before any step
        # has run this episode (`reset`'s initial observation), there is no
        # history yet; fall back to the current reading itself, exactly
        # what PersistenceForecaster would do with a single-point history
        # and what SeasonalNaiveForecaster/EWMAForecaster already do for
        # their own first steps -- an honest "no information yet" default,
        # not a fabricated forecast.
        load_forecast = (
            self._forecaster.forecast(np.asarray(self._load_history), FORECAST_HORIZON_STEPS)
            if self._load_history
            else obs["electrical_load_kw"]
        )
        renewable_forecast = (
            self._forecaster.forecast(np.asarray(self._renewable_history), FORECAST_HORIZON_STEPS)
            if self._renewable_history
            else obs["pv_available_kw"] + obs["wind_available_kw"]
        )

        features = [
            obs["electrical_load_kw"] / scale,
            obs["critical_load_kw"] / scale,
            obs["firm_thermal_kw"] / scale,
            obs["pv_available_kw"] / scale,
            obs["wind_available_kw"] / scale,
            obs["air_temp_c"] / 40.0,
            obs["wind_speed_ms"] / 25.0,
            (obs["indoor_temp_c"] - cfg.thermal.indoor_setpoint_c) / 10.0,
            obs["snow_melt_remaining_kwh"] / max(self.projection.melt_ceiling_kw() * 24.0, 1.0),
            np.sin(day_angle),
            np.cos(day_angle),
            np.sin(year_angle),
            load_forecast / scale,
            renewable_forecast / scale,
        ]
        features += [float(v) for v in obs["genset_online"]]
        features += [p / g.rated_kw for p, g in zip(obs["genset_power_kw"], cfg.gensets)]
        features += [float(d) for d in obs["genset_deposit"]]
        # Whether a commit/decommit request is even feasible right now
        # (minimum up/down time). Diagnosed as a real gap during this
        # project's own RL performance audit: the safety layer already
        # reads these two fields to decide whether a stop/start would
        # actually take effect (allotrope.safety.projection's
        # _effective_online), but the agent never saw them -- it had to
        # infer switching feasibility blind, purely from reward, which is
        # consistent with the observed failure mode (6,599 stop requests
        # blocked for breaching reserve across one evaluation year, versus
        # only 495 actual starts -- a policy repeatedly asking for a stop
        # it has no way to know is currently locked out).
        features += [float(v) for v in obs["genset_can_start"]]
        features += [float(v) for v in obs["genset_can_stop"]]
        features += [float(s) for s in obs["battery_soc"]]
        features += [
            obs["battery_max_discharge_kw"][k] / max(s.max_discharge_kw, 1.0)
            for k, s in enumerate(cfg.storage)
        ]
        # Accumulated stress, from allotrope.intelligence.asset_health --
        # the same wear-score/full-equivalent-cycles formulas that module
        # already reports read-only, now visible to the agent so it can see
        # stress accumulated *this episode* rather than only the current
        # step's deposit. Divided down from the wear score's native rupee
        # scale (thousands to tens of thousands over an episode) into the
        # observation's usual range; both still clip at the same [-5, 5]
        # every other feature does for an unusually stressed episode,
        # rather than getting a bespoke unclipped scale.
        features += [
            self._asset_tracker.gensets[g.id].wear_score().value / 5000.0 for g in cfg.gensets
        ]
        features += [
            self._asset_tracker.batteries[s.id].full_equivalent_cycles().value for s in cfg.storage
        ]
        return np.clip(np.asarray(features, dtype=np.float32), -5.0, 5.0)

    def _mean_deposit(self) -> float:
        return float(np.mean([g.state.deposit for g in self.plant.gensets]))

    # -- convenience ------------------------------------------------------

    def encode_command(self, command: DispatchCommand) -> dict[str, Any]:
        """Invert `decode_action`, so a rule-based controller can be scored here.

        This is what lets the rule-based baselines and a learned policy be
        compared inside the same environment, under the same reward, rather than
        through two separate code paths that might not agree.
        """
        n_g = len(self.cfg.gensets)
        n_s = len(self.cfg.storage)
        dispatch = np.zeros(n_g + n_s + 1, dtype=np.float32)

        for k, g in enumerate(self.cfg.gensets):
            span = max(g.rated_kw - g.min_stable_kw, 1e-9)
            frac = (command.genset_setpoint_kw[k] - g.min_stable_kw) / span
            dispatch[k] = np.clip(frac * 2.0 - 1.0, -1.0, 1.0)

        observation = self.plant.observe()
        for k in range(n_s):
            power = command.battery_kw[k]
            limit = (
                observation["battery_max_discharge_kw"][k]
                if power >= 0
                else observation["battery_max_charge_kw"][k]
            )
            dispatch[n_g + k] = np.clip(power / limit, -1.0, 1.0) if limit > 1e-9 else 0.0

        ceiling = max(self.projection.melt_ceiling_kw(), 1e-9)
        dispatch[n_g + n_s] = np.clip(command.snow_melt_kw / ceiling * 2.0 - 1.0, -1.0, 1.0)

        return {
            "genset_on": np.asarray(command.genset_on, dtype=np.int8),
            "dispatch": dispatch,
        }

    def summary(self) -> dict[str, float]:
        return self.plant.summary()


__all__ = ["PolarMicrogridEnv"]
