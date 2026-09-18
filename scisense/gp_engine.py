"""Gaussian Process engine and uncertainty gate.

For each feature -> target pair SciSense fits a one-dimensional Gaussian
Process, reports held-out accuracy and interval calibration, and converts
the result into a confidence score. The gate compares that score with a
threshold; only pairs that pass are handed to the LLM layer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Matern,
    WhiteKernel,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

Z_95 = 1.959964  # two-sided 95% normal quantile


def build_kernels() -> dict:
    """Candidate kernels. Inputs are standardised before fitting, so the
    length-scale bounds are in units of standard deviations of x."""

    def with_noise(k):
        return ConstantKernel(1.0, (1e-3, 1e3)) * k + WhiteKernel(
            noise_level=1e-2, noise_level_bounds=(1e-6, 10.0)
        )

    return {
        "RBF": with_noise(RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))),
        "Matern-1.5": with_noise(
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
        ),
        "Matern-2.5": with_noise(
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        ),
    }


@dataclass
class GPResult:
    """Everything SciSense knows about one predictor -> target relationship."""

    predictor: str
    target: str
    kernel_name: str
    log_marginal_likelihood: float
    x_train: np.ndarray
    y_train: np.ndarray
    x_grid: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    r2_train: float
    r2_test: float
    rmse_test: float
    coverage_95_test: float  # share of held-out points inside the 95% interval
    noise_level: float
    length_scale: float  # in standardised x units
    mean_uncertainty: float  # mean predictive std over the plotting grid
    confidence: float
    n_train: int
    n_test: int
    fit_warnings: list = field(default_factory=list)

    @property
    def y_lower(self) -> np.ndarray:
        return self.y_mean - Z_95 * self.y_std

    @property
    def y_upper(self) -> np.ndarray:
        return self.y_mean + Z_95 * self.y_std


class GPEngine:
    """Fits one GP per feature, scores it, and applies the uncertainty gate.

    Kernel choice uses the log marginal likelihood on the training split only,
    so the test split is touched once, for reporting. (The original notebook
    picked the kernel by test R^2 and then reported that same R^2, which is
    optimistic.)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        test_size: float = 0.2,
        n_grid_points: int = 200,
        n_restarts_optimizer: int = 3,
        random_state: int = 42,
        r2_weight: float = 0.65,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 <= r2_weight <= 1.0:
            raise ValueError("r2_weight must be in [0, 1]")
        self.confidence_threshold = confidence_threshold
        self.test_size = test_size
        self.n_grid_points = n_grid_points
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.r2_weight = r2_weight
        self.results: dict[str, GPResult] = {}

    # ------------------------------------------------------------------ fit
    def fit_and_analyse(
        self,
        x: np.ndarray,
        y: np.ndarray,
        predictor_name: str,
        target_name: str,
    ) -> GPResult:
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 20:
            raise ValueError(
                f"{predictor_name}: need at least 20 complete rows, got {len(x)}"
            )
        if np.ptp(x) == 0:
            raise ValueError(f"{predictor_name}: feature is constant")

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

        # Standardise x with training statistics only.
        mu, sd = x_train.mean(), x_train.std()
        sd = sd if sd > 0 else 1.0

        def z(v):
            return ((v - mu) / sd).reshape(-1, 1)

        best_gp, best_lml, best_name, best_warnings = None, -np.inf, None, []
        for name, kernel in build_kernels().items():
            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                try:
                    gp.fit(z(x_train), y_train)
                except (np.linalg.LinAlgError, ValueError):
                    continue
            lml = float(gp.log_marginal_likelihood_value_)
            if lml > best_lml:
                best_gp, best_lml, best_name = gp, lml, name
                best_warnings = sorted(
                    {
                        str(w.message).split(". ")[0]
                        for w in caught
                        if issubclass(w.category, ConvergenceWarning)
                    }
                )

        if best_gp is None:
            raise RuntimeError(f"All kernels failed for {predictor_name}")

        # Held-out evaluation (test split used here only).
        y_pred_test, y_std_test = best_gp.predict(z(x_test), return_std=True)
        r2_test = float(r2_score(y_test, y_pred_test))
        rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        inside = np.abs(y_test - y_pred_test) <= Z_95 * y_std_test
        coverage = float(inside.mean())
        r2_train = float(r2_score(y_train, best_gp.predict(z(x_train))))

        # Smooth curve for plotting and for the uncertainty summary.
        x_grid = np.linspace(x.min(), x.max(), self.n_grid_points)
        y_mean, y_std = best_gp.predict(z(x_grid), return_std=True)

        noise, length = self._kernel_params(best_gp)
        confidence = self.compute_confidence(y_std, y_mean, r2_test)

        result = GPResult(
            predictor=predictor_name,
            target=target_name,
            kernel_name=best_name,
            log_marginal_likelihood=best_lml,
            x_train=x_train,
            y_train=y_train,
            x_grid=x_grid,
            y_mean=y_mean,
            y_std=y_std,
            r2_train=r2_train,
            r2_test=r2_test,
            rmse_test=rmse_test,
            coverage_95_test=coverage,
            noise_level=noise,
            length_scale=length,
            mean_uncertainty=float(np.mean(y_std)),
            confidence=confidence,
            n_train=len(x_train),
            n_test=len(x_test),
            fit_warnings=best_warnings,
        )
        self.results[predictor_name] = result
        return result

    # ------------------------------------------------------------ scoring
    def compute_confidence(
        self, y_std: np.ndarray, y_mean: np.ndarray, r2_test: float
    ) -> float:
        """Heuristic confidence score in [0, 1].

        confidence = w * max(0, R^2_test) + (1 - w) / (1 + relative_uncertainty)
        relative_uncertainty = mean(predictive std) / mean(|predicted mean|)

        This is a hand-designed score, not a calibrated probability. The
        weight w (default 0.65) and the gate threshold are design choices.
        """
        r2_part = max(0.0, float(r2_test))
        signal = float(np.abs(y_mean).mean()) + 1e-8
        rel_unc = float(np.mean(y_std)) / signal
        unc_part = 1.0 / (1.0 + rel_unc)
        score = self.r2_weight * r2_part + (1.0 - self.r2_weight) * unc_part
        return float(np.clip(score, 0.0, 1.0))

    def gate_passes(self, result: GPResult) -> bool:
        """The uncertainty gate: True means the LLM may be called."""
        return result.confidence >= self.confidence_threshold

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _kernel_params(gp: GaussianProcessRegressor) -> tuple[float, float]:
        """Read learned noise level and length scale from the fitted kernel."""
        noise, length = float("nan"), float("nan")
        for name, value in gp.kernel_.get_params().items():
            if name.endswith("noise_level") and not name.endswith("_bounds"):
                noise = float(value)
            elif name.endswith("length_scale") and not name.endswith("_bounds"):
                length = float(np.atleast_1d(value)[0])
        return noise, length

    def summary(self, result: GPResult) -> str:
        gate = "PASS (LLM may generate)" if self.gate_passes(result) else "BLOCKED (LLM not called)"
        lines = [
            f"{result.predictor} -> {result.target}",
            f"  Kernel (by log marginal likelihood): {result.kernel_name}",
            f"  R2 train / test:        {result.r2_train:.4f} / {result.r2_test:.4f}",
            f"  RMSE test:              {result.rmse_test:.6f}",
            f"  95% interval coverage:  {result.coverage_95_test:.3f} (ideal ~0.95)",
            f"  Noise level:            {result.noise_level:.3g}",
            f"  Length scale (std x):   {result.length_scale:.3g}",
            f"  Mean predictive std:    {result.mean_uncertainty:.6f}",
            f"  Confidence:             {result.confidence:.4f} (threshold {self.confidence_threshold})",
            f"  Gate:                   {gate}",
        ]
        if result.fit_warnings:
            lines.append("  Optimiser warnings:     " + "; ".join(result.fit_warnings))
        return "\n".join(lines)
