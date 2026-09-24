"""Offline PID tuning tools.

Grid search, random search, and multi-objective evaluation.
All tuning is OFFLINE — never modifies production config at runtime.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field


@dataclass
class TuningObjective:
    """Weights for multi-objective controller score."""

    w_rmse: float = 0.30
    w_settling_time: float = 0.20
    w_overshoot: float = 0.20
    w_control_effort: float = 0.15
    w_loss_penalty: float = 0.15

    max_rmse_px: float = 50.0
    max_settling_time_s: float = 5.0
    max_overshoot_px: float = 30.0
    max_control_effort: float = 100.0


@dataclass
class TuningCandidate:
    """A candidate parameter set with its evaluation results."""

    pan_kp: float = 0.5
    pan_ki: float = 0.01
    pan_kd: float = 0.1
    tilt_kp: float = 0.5
    tilt_ki: float = 0.01
    tilt_kd: float = 0.1

    rmse_px: float = 0.0
    p95_error_px: float = 0.0
    max_error_px: float = 0.0
    settling_time_s: float = 0.0
    overshoot_px: float = 0.0
    control_effort: float = 0.0
    loss_count: int = 0
    lock_retention_pct: float = 0.0
    acquisition_time_s: float = 0.0

    objective_score: float = 0.0
    scenario_results: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pan_kp": self.pan_kp, "pan_ki": self.pan_ki, "pan_kd": self.pan_kd,
            "tilt_kp": self.tilt_kp, "tilt_ki": self.tilt_ki, "tilt_kd": self.tilt_kd,
            "rmse_px": round(self.rmse_px, 3),
            "p95_error_px": round(self.p95_error_px, 3),
            "max_error_px": round(self.max_error_px, 3),
            "settling_time_s": round(self.settling_time_s, 3),
            "overshoot_px": round(self.overshoot_px, 3),
            "loss_count": self.loss_count,
            "lock_retention_pct": round(self.lock_retention_pct, 1),
            "objective_score": round(self.objective_score, 4),
        }


class PIDTuningEvaluator:
    """Evaluates a PID configuration on a set of scenarios.

    Offline only — does not affect production configuration.
    """

    def __init__(self, objective: TuningObjective | None = None) -> None:
        self._objective = objective or TuningObjective()

    def evaluate_candidate(
        self,
        errors: list[float],
        control_commands: list[float],
        timestamps: list[float],
        loss_events: int = 0,
        total_frames: int = 1,
        acquisition_time_s: float = 0.0,
    ) -> TuningCandidate:
        """Evaluate a candidate from collected frame data.

        Args:
            errors: Per-frame tracking errors (px).
            control_commands: Per-frame control command magnitudes.
            timestamps: Per-frame timestamps (s).
            loss_events: Number of target loss events.
            total_frames: Total frames processed.
            acquisition_time_s: Time to first lock (s).

        Returns:
            TuningCandidate with computed metrics.
        """
        import numpy as np

        candidate = TuningCandidate()

        if not errors:
            return candidate

        err_arr = np.array(errors)
        candidate.rmse_px = float(np.sqrt(np.mean(err_arr ** 2)))
        candidate.p95_error_px = float(np.percentile(err_arr, 95))
        candidate.max_error_px = float(np.max(err_arr))

        candidate.settling_time_s = self._compute_settling_time(errors, timestamps)
        candidate.overshoot_px = self._compute_overshoot(errors)
        candidate.control_effort = float(np.sum(np.abs(np.array(control_commands)))) if control_commands else 0.0
        candidate.loss_count = loss_events
        candidate.acquisition_time_s = acquisition_time_s

        locked_frames = sum(1 for e in errors if e < 20.0)
        candidate.lock_retention_pct = (locked_frames / total_frames * 100.0) if total_frames > 0 else 0.0

        candidate.objective_score = self._compute_objective(candidate)

        return candidate

    def _compute_settling_time(self, errors: list[float], timestamps: list[float]) -> float:
        threshold = 10.0
        for i, (err, ts) in enumerate(zip(errors, timestamps, strict=True)):
            if err < threshold:
                all_settled = all(e < threshold for e in errors[i:i + 10])
                if all_settled:
                    return ts - timestamps[0] if timestamps else 0.0
        return timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    def _compute_overshoot(self, errors: list[float]) -> float:
        if len(errors) < 3:
            return 0.0
        peak = max(errors)
        converged = errors[-1] if errors else 0.0
        if converged < peak * 0.5:
            return peak
        return 0.0

    def _compute_objective(self, c: TuningCandidate) -> float:
        obj = self._objective
        rmse_score = max(0.0, 1.0 - c.rmse_px / obj.max_rmse_px)
        settling_score = max(0.0, 1.0 - c.settling_time_s / obj.max_settling_time_s)
        overshoot_score = max(0.0, 1.0 - c.overshoot_px / obj.max_overshoot_px)
        effort_score = max(0.0, 1.0 - c.control_effort / obj.max_control_effort)
        loss_score = max(0.0, 1.0 - c.loss_count * 0.2)

        return (
            obj.w_rmse * rmse_score
            + obj.w_settling_time * settling_score
            + obj.w_overshoot * overshoot_score
            + obj.w_control_effort * effort_score
            + obj.w_loss_penalty * loss_score
        )


class PIDSweepRunner:
    """Runs grid/random parameter sweep over scenarios.

    Usage:
        runner = PIDSweepRunner()
        results = runner.grid_search(param_grid, eval_fn)
        best = runner.select_best(results)
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def grid_search(
        self,
        param_grid: dict[str, list[float]],
        eval_fn: callable,
    ) -> list[TuningCandidate]:
        """Run grid search over parameter combinations.

        Args:
            param_grid: Dict of param_name -> list of values.
            eval_fn: callable(**params) -> TuningCandidate.

        Returns:
            List of evaluated candidates, sorted by score descending.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        candidates = []

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo, strict=True))
            try:
                candidate = eval_fn(**params)
                candidates.append(candidate)
            except Exception:
                continue

        candidates.sort(key=lambda c: c.objective_score, reverse=True)
        return candidates

    def random_search(
        self,
        param_ranges: dict[str, tuple[float, float]],
        eval_fn: callable,
        n_trials: int = 50,
    ) -> list[TuningCandidate]:
        """Run random search over parameter ranges.

        Args:
            param_ranges: Dict of param_name -> (min, max).
            eval_fn: callable(**params) -> TuningCandidate.
            n_trials: Number of random trials.

        Returns:
            List of evaluated candidates, sorted by score descending.
        """
        rng = random.Random(self._seed)
        candidates = []

        for _ in range(n_trials):
            params = {}
            for name, (lo, hi) in param_ranges.items():
                params[name] = rng.uniform(lo, hi)
            try:
                candidate = eval_fn(**params)
                candidates.append(candidate)
            except Exception:
                continue

        candidates.sort(key=lambda c: c.objective_score, reverse=True)
        return candidates

    def select_best(
        self,
        candidates: list[TuningCandidate],
        min_lock_retention: float = 80.0,
    ) -> TuningCandidate | None:
        """Select best candidate with minimum lock retention filter.

        Prefers candidates that maintain lock retention above threshold.
        """
        valid = [c for c in candidates if c.lock_retention_pct >= min_lock_retention]
        if not valid:
            valid = candidates
        if not valid:
            return None
        return valid[0]

    def cross_validate(
        self,
        candidates: list[TuningCandidate],
        scenario_results: dict[str, dict[str, float]],
    ) -> list[TuningCandidate]:
        """Rank candidates by cross-scenario consistency.

        Penalizes candidates with high variance across scenarios.
        """
        import numpy as np

        for c in candidates:
            scores = list(c.scenario_results.values())
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                c.objective_score = mean_score - 0.1 * std_score

        candidates.sort(key=lambda c: c.objective_score, reverse=True)
        return candidates
