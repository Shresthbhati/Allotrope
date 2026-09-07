# Asset health tracking

This document describes `allotrope/intelligence/asset_health/`, what it
measures versus estimates versus stands in for, and — as with every other
claim in this repository — what it does not yet do.

## What this is

A read-only observer of a running `PolarMicrogrid`. `AssetHealthTracker` is
built against a `StationConfig` (for nameplate ratings and envelopes) and fed
the plant's own per-step telemetry dict (exactly what `PolarMicrogrid.step()`
returns) one step at a time via `.update(telemetry)`. It accumulates that
telemetry into a per-genset and per-battery health record and returns it,
fully labeled, from `.report()`.

It adds no parallel physics. Every accumulated number is either read straight
off telemetry the plant already produces, or derived from it with a named,
documented formula. It does not touch dispatch, the safety layer, or any
agent — see "What is not done" below.

## Why these fields, not others

The plant (`allotrope/sim/assets.py`, `allotrope/sim/plant.py`) already tracks
the state that actually matters for genset and battery wear: cumulative
starts and run hours live on `GensetState`; exhaust fouling ("wet stacking")
lives in `GensetState.deposit`, accumulated by `Genset._update_deposit`
whenever a set runs below `GensetSpec.wet_stack_threshold_frac`; battery state
of charge and cumulative energy throughput live on `BatteryState`. This
tracker reuses all of it rather than inventing a second bookkeeping system:

- **Low-load hours** are defined against `wet_stack_threshold_frac`, the
  project's own existing physical definition of harmful part-load running
  (it is precisely the boundary `_update_deposit` uses to decide whether
  deposit is accruing). Using any other cutoff would create a second,
  disagreeing notion of "harmful" alongside the deposit model this same
  tracker reports.
- **Full-equivalent-cycles** (battery) is the standard industry proxy —
  cumulative absolute throughput divided by twice nameplate capacity — applied
  to the exact `abs(power_kw) * dt_h` quantity `Battery.step` already
  accumulates into `BatteryState.throughput_kwh`.
- **SOC-extreme hours** are time spent within 5 percentage points of either
  bound of the battery's own configured `soc_min`/`soc_max` envelope — the
  same envelope `Battery.max_charge_kw` / `max_discharge_kw` already enforce
  as the chemistry's hard limits, not an arbitrary percentage of full range.
- **Wear score** (genset) reuses two of `allotrope.envs.reward.RewardWeights`'
  own prices — `genset_start_per_event` (1500) and `deposit_growth_per_unit`
  (8000) — as the combination weights: `wear = 1500 * starts + 8000 * deposit`.
  Those two numbers are already this project's stated valuation of "maintenance
  life consumed by a cold start" and "fouling... as deferred maintenance."
  Reusing them keeps the wear score consistent with the economic weighting the
  reward function already trusts, instead of inventing a fresh, ungrounded set
  of weights. Low-load hours are reported as their own field rather than
  folded into the score a second time, because they are already the mechanism
  by which deposit accumulates — adding them again would double-count the same
  physical effect.

## Metric labels

Every metric `AssetHealthTracker.report()` returns carries one of four labels,
both as a `MetricLabel` on the `Metric` dataclass in code and as a `label`
string in the returned dict:

| Label | Meaning | Examples here |
|---|---|---|
| `measured` | Read directly off simulator state | genset `run_hours`, `starts`; battery `soc`, `throughput_kwh`, SOC-extreme hours, `cold_charge_blocks` |
| `modeled` | Computed by a physical/engineering model already in this codebase | genset `deposit` (from `Genset._update_deposit`) |
| `proxy` | A stand-in for something the simulator does not model directly | genset `wear_score` |
| `estimated` | A derived statistic from a named, standard formula | battery `full_equivalent_cycles` |

No metric is ever presented without its label attached in code (`Metric.label`,
`Metric.as_dict()`), and none should ever be read as more certain than its
label says.

## What the wear score is not

`wear_score` is higher for a genset that has started more often and fouled
more — that is the entire claim. It is not, and must never be read as:

- a probability of failure,
- a remaining-useful-life estimate, in hours, days, or any other unit,
- a calibrated measurement of physical wear (bearing wear, ring wear, etc.) —
  nothing in this simulator models those, so nothing here can estimate them.

This project's own house rule against fabricated results forbids inventing
numbers like "73% chance of failure in 30 days," and no evidence exists in
this codebase to support one. The wear score is a proxy, and is labeled as
one everywhere it appears.

## Real numbers from a real run

Ran `EfficientRuleBased` against Maitri for 30 days (720 hourly steps, seed 7),
feeding every step's telemetry through the tracker (`tracker.update(telemetry)`
called once per `plant.step()`), then reconciling battery cold-charge-block
counts against the live plant. Full output reproduced from an actual run —
nothing here is hand-typed or invented.

Plant's own `summary()` for the same run, for cross-check:

```
fuel_kl: 23.59
black_carbon_g: 979.5
genset_run_hours: 1161.0
genset_starts: 83
wet_stacking_fraction: 0.0
mean_deposit: 0.0
renewable_fraction: 0.141
unmet_water_kwh: 141.1
critical_unserved_kwh: 0.0
freeze_violation_steps: 0.0
cold_charge_blocks: 0.0
```

Tracker's per-genset report (`AssetHealthTracker.report()["gensets"]`):

| Genset | run_hours (measured) | starts (measured) | low_load_hours (measured) | deposit (modeled) | wear_score (proxy) |
|---|---|---|---|---|---|
| G1 | 720.0 | 1 | 0.0 | 0.0 | 1,500 |
| G2 | 441.0 | 82 | 0.0 | 0.0 | 123,000 |
| G3 | 0.0 | 0 | 0.0 | 0.0 | 0 |

Summing the tracker's per-genset `starts` gives 1 + 82 + 0 = 83, matching
`plant.summary()["genset_starts"]` exactly (this equality is also asserted in
`tests/test_asset_health.py::test_genset_starts_match_plant_summary`). G2's 82
starts over 30 days is a real artifact of this rule-based controller
cycling a second set on and off against the base-load set (G1, which never
stopped) rather than holding it committed — a real number this tracker
surfaced, not a flattering one, and not smoothed over.

Deposit stayed at 0.0 for all three sets in this run because none of them was
ever loaded below its 30% wet-stacking threshold (`wet_stacking_fraction: 0.0`
in the plant's own summary agrees) — `EfficientRuleBased` is specifically
designed to avoid exactly that operating band.

Tracker's per-battery report (`AssetHealthTracker.report()["batteries"]`):

| Battery | soc (measured) | throughput_kwh (measured) | full_equivalent_cycles (estimated) | low_soc_hours (measured) | high_soc_hours (measured) | cold_charge_blocks (measured) |
|---|---|---|---|---|---|---|
| BESS_LFP | 0.50 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| BESS_LTO | 0.50 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

Both batteries saw zero throughput in this particular run: `EfficientRuleBased`
only draws its `battery_kw` reserve down when committed genset capacity plus
renewables plus that reserve is needed to avoid a shortfall, and it never
was needed against this 30-day, seed-7 realisation. `plant.summary()`'s
`cold_charge_blocks: 0.0` agrees with the tracker's reconciled count. This is
reported honestly rather than replaced with a "more interesting" run — a
tracker that only reports flattering numbers when they happen to occur is not
one that can be trusted when they don't.

## Now wired into the RL observation

`PolarMicrogridEnv` runs its own `AssetHealthTracker` instance (reset fresh
each episode, exactly like this module's own usage pattern above) and
appends each genset's `wear_score()` (scaled down from its native rupee
units) and each battery's `full_equivalent_cycles()` to the observation
vector, so a learned policy can see *accumulated* stress rather than only
the current step's deposit or SOC. This is still read-only: the tracker
observes telemetry the plant already produced; it does not feed back into
dispatch, the reward, or the safety layer. `tests/test_env.py`'s
`test_wear_score_and_fec_features_track_the_asset_health_tracker` verifies
the observation's features are bit-for-bit what this tracker computes.

## What is not done in this pass

- **Not wired into the reward or dispatch decisions.** The agent can now
  *see* wear/FEC, but nothing yet prices battery cycling in the reward the
  way genset starts and deposit already are (`allotrope.envs.reward`), and
  no controller changes its behaviour based on a high wear score --
  predictive-maintenance-aware dispatch is still future work.
- **No failure probabilities or remaining-useful-life estimates**, for the
  reasons given above — there is no evidence in this codebase to support
  either, and inventing one would violate this project's own rules.
- **No cross-episode persistence.** A tracker instance accumulates over
  exactly the steps it is fed; nothing here saves health state to disk or
  carries it between separate `run_episode` calls. A caller who wants
  multi-episode wear tracking must feed the tracker every episode's telemetry
  in sequence, or maintain their own accumulation across trackers.
- **No calibration against real hardware.** Every label above describes
  provenance *within this simulator*, not fidelity to a real genset or
  battery pack; that calibration question is out of scope for this module.

## Tests

`tests/test_asset_health.py` runs a real 240-step episode
(`allotrope.sim.runner.build_plant` + `EfficientRuleBased`, the same real
controller used elsewhere in this project) through the tracker and checks,
among other things, that:

- summed per-genset tracked starts equal `plant.summary()["genset_starts"]`
  exactly, and match each genset's own `GensetState.total_starts`;
- tracked run hours and deposit match each genset's live state exactly;
- the battery full-equivalent-cycle estimate matches an independent
  hand-computation (`throughput_kwh / (2 * capacity_kwh)`) against the
  plant's own `BatteryState.throughput_kwh`;
- every metric in `.report()` carries one of the four valid labels.

Run with:

```
pytest tests/test_asset_health.py -v
```
