# Current-state audit — 2026-09-07

This audit reflects the actual code on `main` as of this commit, traced by
reading execution paths rather than trusting filenames or docstrings. Every
claim below was checked by reading the relevant call sites; nothing here is
inferred from a module existing.

Legend: **IMPLEMENTED+TESTED** / **IMPLEMENTED, UNTESTED** / **PARTIAL** /
**PRESENT BUT UNUSED** / **MISSING**.

## Core simulation and control

| Capability | Status | Notes |
|---|---|---|
| Physical plant simulator (`allotrope.sim.plant`) | IMPLEMENTED+TESTED | Willans-line fuel, genset fouling/deposit, battery thermal derating, CHP heat recovery, boiler fallback. Energy-balance and renewable-accounting invariants tested in `tests/test_plant.py`. |
| Rule-based baselines (`LegacyNPlusOne`, `EfficientRuleBased`) | IMPLEMENTED+TESTED | Both implement the shared `.act(observation, plant) -> DispatchCommand` interface `allotrope.sim.runner.run_episode` drives. |
| Hybrid RL agent (`BranchingDQN` + `SDDPG` via `HybridAgent`) | IMPLEMENTED+TESTED | Same controller interface as the baselines. Trained checkpoints exist under `runs/`; a fresh 500k-step training run at Maitri completed this session on the current (post-forecast/asset-health) observation space, but has not yet been evaluated against the baselines or the resilience benchmark. |
| Safety projection (`allotrope.safety.projection.SafetyProjection`) | IMPLEMENTED+TESTED | Analytic, closed-form, no solver. Sanitises both malformed commands and corrupted/non-finite observations before any bound reads them. Adversarially audited (`scripts/run_safety_audit.py`): 0 kWh critical-load loss across 5 attack policies, guarded vs. unguarded. |
| Deterministic fallback (`allotrope.safety.fallback.GuardedController`) | IMPLEMENTED+TESTED | Wraps any agent; this is what's actually deployed/evaluated, not a bare `HybridAgent`. |
| Gymnasium environment (`PolarMicrogridEnv`) | IMPLEMENTED+TESTED | Observation width `14 + 6*n_gensets + 3*n_storage`; includes forecast (EWMA, 24h horizon) and asset-health (wear score, FEC) features as of the most recent observation-shape change. |

## Forecasting

| Capability | Status | Notes |
|---|---|---|
| Forecasters (`PersistenceForecaster`, `SeasonalNaiveForecaster`, `EWMAForecaster`) | IMPLEMENTED+TESTED | `docs/forecasting.md` documents an honest evaluation: EWMA only beats persistence at the 24h horizon; 1-step forecasts add nothing over the current reading. |
| Forecast wired into the actual controller path | IMPLEMENTED+TESTED | `PolarMicrogridEnv._observe` computes `EWMAForecaster.forecast(history, 24)` for load and renewables and appends both as observation features, using only history available at decision time (no future-state leakage — verified by `tests/test_env.py`'s forecast tests). This reaches every controller trained or evaluated through the env, including `HybridAgent`. |
| MAE/RMSE/bias reporting | IMPLEMENTED, UNTESTED-AS-A-REGRESSION | `allotrope.intelligence.forecasting.evaluation`/`run_evaluation.py` compute these; not wired into CI as a regression gate. |

## Asset health

| Capability | Status | Notes |
|---|---|---|
| `AssetHealthTracker` (genset wear score, battery FEC) | IMPLEMENTED+TESTED | Formulas: `wear_score = 1500*starts + 8000*deposit`, `full_equivalent_cycles()` from throughput. |
| Wired into the controller's actual observation | IMPLEMENTED+TESTED | `PolarMicrogridEnv._observe` appends wear-score and FEC features per unit; reused (not duplicated) by the training path and by `allotrope.resilience`'s `run_episode`-based path indirectly via the tracker's presence in the env. A fresh RL retrain to measure the *effect* of this wiring on learned behaviour has completed but not yet been analysed. |
| Wired into the reward/objective | PARTIAL | The reward function's prices (`allotrope.envs.reward.RewardWeights`) already include per-start and per-deposit-growth terms independent of this tracker; the tracker's wear score is an observation feature, not yet a second reward term layered on top. No evidence this was ever meant to double-count the same physical quantity twice. |

## Optimization (MILP)

| Capability | Status | Notes |
|---|---|---|
| MILP formulation (`allotrope.optimization`) | IMPLEMENTED+TESTED | Genset commitment (binary on/start/stop, min-up/down time via windowed-sum constraints), genset output bounds, battery SOC dynamics with one-way efficiency and charge/discharge exclusivity, reserve margin, hard critical-load balance (no unserved-load variable — the balance constraint *is* the guarantee), hard firm-thermal service, boiler heat, renewable availability/curtailment. Objective reuses `RewardWeights`' exact prices. |
| Documented physical simplifications | IMPLEMENTED | Three explicit, honest simplifications versus the real simulator: no fouling penalty on fuel (clean Willans line only), clean-engine black-carbon factor only, no deposit/wet-stacking accounting at all (`wet_stacking_fraction` absent from output rather than faked as zero). Both nonlinearities excluded for the same reason: they are path-dependent, which a linear program cannot represent. |
| Water/deferrable-demand service in the MILP | MISSING | Not modelled; the real simulator's snow-melt/water deferral logic has no MILP equivalent yet. |
| Integrated into the resilience/scientific benchmark as a comparison point | MISSING | `allotrope.resilience.benchmark` currently drives only `run_episode`-compatible step controllers (rule-based today, RL equally supported). The MILP oracle solves a whole horizon in one call and has a different interface (`solve(instance) -> OptimizationResult`, not `.act(observation, plant)`); nothing yet adapts an `OptimizationResult`'s schedule into a replayable per-step controller for a head-to-head comparison table. This is real, scoped follow-up work, not a rounding error. |

## Resilience / fault injection

| Capability | Status | Notes |
|---|---|---|
| Fault injection (`allotrope.faults`) | IMPLEMENTED+TESTED | Three fault kinds enforced at the physical layer: `genset_trip` (forces `GensetState.online=False`, bypassing anti-cycling since a trip is a hardware failure), `battery_lockout` (binds `max_charge_kw`/`max_discharge_kw` to zero on the instance), `renewable_derate` (scales the plant's own PV/wind arrays). One shared implementation (`allotrope/faults/core.py`) used identically by the Gymnasium wrapper (`FaultInjector`) and by `allotrope.resilience.benchmark`'s direct `run_episode` path. Deterministic under a seed (confirmed by test). |
| Fault types beyond the three above (multi-generator failure, PV/wind collapse as distinct from derate, forecast error injection, load spike, extreme thermal demand, sensor corruption/staleness *as a fault*, comms loss, controller crash, actuator failure, cascading failure) | MISSING | Not implemented. `single_genset_trip`/`compound_failure` scenarios exercise one genset at a time; nothing currently fails two gensets simultaneously as a named scenario (though the mechanism supports it — a schedule is just a list of `FaultEvent`s, so a two-genset scenario is a config, not new code). Sensor corruption exists as a *classification* concept (`allotrope.safety.sensor_health`) but is not wired in as an injectable fault a benchmark can trigger. |
| Resilience benchmark (`allotrope.resilience`) | IMPLEMENTED+TESTED | Runs any `run_episode`-compatible controller through named fault schedules, reporting real summary metrics (fuel, critical-unserved kWh, genset starts, etc.) per (controller, scenario) pair, including an unfaulted baseline row for comparison. Currently only exercises the two rule-based baselines in this session's own testing; has not yet been run against the trained `HybridAgent`/`GuardedController`. |
| Recovery-time, safety-intervention-count, controller-switch-count, latency metrics in the benchmark | MISSING | `run_benchmark`'s rows currently carry whatever `PolarMicrogrid.summary()` reports (fuel, black carbon, genset starts/hours, renewable/curtailed kWh, critical/total unserved kWh). Recovery time, safety intervention counts, and controller-switch counts are not yet computed or reported — the summary dict has no such keys, and nothing in the benchmark path computes them separately. |

## Sensor health / observation validation

| Capability | Status | Notes |
|---|---|---|
| NaN/Inf sanitisation on both commands and observations | IMPLEMENTED+TESTED | `SafetyProjection._sanitise_observation`/`_sanitise_floats`/`_sanitise_bools`, fixed earlier this session after the project's own adversarial audit found a real gap (a NaN sensor reading reaching an unguarded `np.clip` and corrupting battery SOC with zero recorded intervention). Always fails toward the conservative direction. |
| Named severity classification (VALID/DEGRADED/STALE/CORRUPTED/UNSAFE) | PARTIAL | `allotrope.safety.sensor_health` implements this fully and correctly for range/NaN/Inf checking (`CORRUPTED`), live by default. Staleness detection (`STALE`) is implemented and tested but shipped **disabled by default** (`max_stale_steps=0` for every field in `specs_for_station`) after a real finding: a naive repeat-count threshold false-triggers on this simulator's legitimate flat stretches (polar-night zero PV, a nameplate-pinned battery envelope, a flat baseload) because the simulator has no sensor noise to calibrate a threshold against. `DEGRADED` and `UNSAFE` are defined in the enum but never emitted by any code path yet — reserved, not implemented. |
| Wired into the actual safety/control path | PARTIAL | `sensor_health` classifies; it does not alter what reaches the plant (`SafetyProjection` alone has that authority, unchanged). No controller currently *consults* a `SensorStatus` to change its own behaviour — that consumption is exactly what `allotrope.intelligence.confidence.ControllerConfidence` (added this session) does, but `ControllerConfidence` itself is not yet wired into any controller's decision loop. |

## Confidence / OOD detection

| Capability | Status | Notes |
|---|---|---|
| Confidence/OOD assessment (`allotrope.intelligence.confidence`) | IMPLEMENTED+TESTED | Combines three real, independently-checkable signals: observation validity (via `sensor_health`), envelope distance (mean-squared z-score of the encoded observation against a caller-supplied reference set — explicitly *not* a claim about an RL agent's true training distribution, since this module has no access to a training run's replay buffer), and safety-intervention rate over a rolling window (a real, already-recorded `SafetyReport` signal). |
| Ensemble disagreement | MISSING, HONESTLY REPORTED AS SUCH | Always `None`. Neither `BranchingDQN` nor `SDDPG` expose an ensemble or multiple value-estimate heads through `.act()` (SDDPG's twin critics exist internally for TD3-style training stability, not as a surfaced uncertainty signal). Reported as absent rather than fabricated. |
| Wired into an actual controller decision (switching, fallback trigger) | MISSING | `ControllerConfidence.assess()` is a standalone, tested function; nothing currently calls it from inside a controller's action loop or uses its output to switch controllers. This is the natural next step (adaptive controller switching), not yet built. |

## Controller switching / adaptive hierarchy

| Capability | Status | Notes |
|---|---|---|
| RL → rule-based → deterministic fallback → safety projection hierarchy | PARTIAL | The bottom two layers exist and are used (`GuardedController` wraps an agent with the deterministic fallback and safety projection). The top switch — RL handing off to `EfficientRuleBased` specifically on low confidence/OOD, with hysteresis and logged switch events — does not exist. `GuardedController`'s existing fallback logic switches on a different, narrower trigger set (see `allotrope/safety/fallback.py`) than "confidence-aware switching between a learned policy and a rule-based one." |
| Switch history/logging (timestamp, previous/new controller, trigger, confidence) | MISSING | No such record type exists yet. |

## Long-horizon reserve management

| Capability | Status | Notes |
|---|---|---|
| Deterministic multi-day/multi-week fuel-reserve planning | MISSING | The safety projection enforces an instantaneous reserve margin (committed capacity vs. current load + configured margin) every step; nothing projects fuel consumption forward against a renewable forecast over a multi-day-to-multi-week polar-night horizon, or preemptively tightens discretionary demand ahead of a projected shortfall. |

## Edge/model integrity

| Capability | Status | Notes |
|---|---|---|
| Model checksum/version/rollback/watchdog/inference-timeout | MISSING | No such module exists in `allotrope/`. |

## Federated learning

| Capability | Status | Notes |
|---|---|---|
| `allotrope.federated` | IMPLEMENTED, PARTIALLY TESTED | `allotrope/federated/round.py`, `tests/test_federated.py` exist from before this session; not re-audited in this pass beyond confirming the module and its test file are present and the test suite (which includes it) passes. A "matched-budget fair evaluation" claim was flagged earlier in this project's own history as not yet built; not re-verified here since no work touched this area this session. |

## Frontend

| Capability | Status | Notes |
|---|---|---|
| `frontend/` (React/TypeScript Command Center) | IMPLEMENTED, PARTIALLY TESTED | Pre-existing; not touched this session. Not re-audited line-by-line in this pass. |
| `webapp/frontend/` | PRESENT BUT UNUSED | Landed via an external merge earlier in this project's history (from a parallel `Dang-ctrl/Allotrope` branch). Not referenced by `run.bat` or the README. Architecturally orphaned — flagged, not yet triaged (consolidate, remove, or wire in is still an open decision). |
| "Break Allotrope" live fault-injection UI | MISSING | No frontend control currently triggers `allotrope.faults.FaultInjector` or the resilience benchmark against a live backend session. |
| Dashboard fault controls / event timeline / benchmark comparison view | MISSING | Not built. |

## What changed this session (for context on this audit's freshness)

Merged to `main`, in order: forecasting+asset-health wiring into the RL
observation (PR #14), a `--checkpoint-every` training-resilience fix (PR
#13), the offline MILP oracle (PR #15), the fault-injection framework (PR
#16), the resilience benchmark (PR #17), and per-sensor
staleness/plausibility classification (PR #18). The confidence/OOD module
described above is implemented and tested on a branch not yet merged as of
this audit.

## Honest summary of the highest-value gaps

In rough priority order, based on what actually blocks the target
architecture in the mission brief:

1. **MILP is not in the comparison benchmark yet.** It solves a different
   problem shape (whole-horizon, perfect foresight) than the step-by-step
   controllers `run_episode` drives; adapting its schedule into a
   replayable controller (or building a parallel comparison path) is
   necessary before any "how close to optimal" claim can be made.
2. **Confidence/OOD exists but isn't consumed.** No controller currently
   changes its behaviour based on `ControllerConfidence.assess()`'s output
   — the adaptive-switching hierarchy in the mission brief has no switch
   yet, only the signal that would drive one.
3. **The resilience benchmark doesn't yet measure recovery time, safety
   interventions, or controller switches** — only the physical/economic
   summary metrics `PolarMicrogrid.summary()` already reported for any
   run. Those three are exactly what a fault-focused benchmark most needs
   to add over an ordinary evaluation run.
4. **Long-horizon reserve management is entirely missing.** The safety
   projection's reserve margin is instantaneous, not a multi-day
   fuel-depletion projection against a polar-night forecast.
5. **The frontend has no live connection to any of this session's new
   backend capability** (faults, resilience benchmark, confidence/OOD,
   MILP). It remains whatever state it was in before this session's
   backend work began.

Nothing above is a claim that these are impossible or even especially
hard — several are natural, scoped next PRs. They are listed because the
mission brief's target architecture depends on all of them existing and
being connected, and right now several exist without being connected.
