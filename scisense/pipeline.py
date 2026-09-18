"""End-to-end pipeline: CSV -> GP per feature -> gate -> gated LLM summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gp_engine import GPEngine, GPResult
from .llm_layer import LLMReasoningLayer, SciInsight

NOISE_CONTROL = "random_noise_control"
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]


@dataclass
class PipelineOutput:
    dataset_name: str
    n_rows_used: int
    n_rows_total: int
    target_col: str
    feature_cols: list
    gp_results: dict
    insights: dict
    engine: GPEngine


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def run_scisense(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[list] = None,
    dataset_name: str = "Dataset",
    confidence_threshold: float = 0.75,
    max_rows: int = 1500,
    add_noise_control: bool = False,
    variable_notes: Optional[dict] = None,
    complete_fn: Optional[Callable[[str, str], str]] = None,
    random_state: int = 42,
) -> PipelineOutput:
    """Run SciSense on a DataFrame.

    GP fitting scales as O(n^3), so datasets larger than max_rows are
    randomly subsampled (with a fixed seed) before fitting.
    """
    nums = numeric_columns(df)
    if target_col not in nums:
        raise ValueError(f"Target '{target_col}' is not a numeric column. Numeric columns: {nums}")
    if feature_cols is None:
        feature_cols = [c for c in nums if c != target_col]
    missing = [c for c in feature_cols if c not in nums]
    if missing:
        raise ValueError(f"Non-numeric or missing feature columns: {missing}")
    if not feature_cols:
        raise ValueError("Need at least one numeric feature besides the target.")

    data = df[feature_cols + [target_col]].copy()
    n_total = len(data)
    if n_total > max_rows:
        data = data.sample(n=max_rows, random_state=random_state)

    features = list(feature_cols)
    if add_noise_control:
        rng = np.random.default_rng(random_state)
        data[NOISE_CONTROL] = rng.normal(size=len(data))
        features.append(NOISE_CONTROL)

    engine = GPEngine(confidence_threshold=confidence_threshold, random_state=random_state)
    llm = LLMReasoningLayer(engine, complete_fn=complete_fn)

    results: dict[str, GPResult] = {}
    insights: dict[str, SciInsight] = {}
    for col in features:
        res = engine.fit_and_analyse(data[col].values, data[target_col].values, col, target_col)
        results[col] = res
        insights[col] = llm.generate_insight(
            res, dataset_name=dataset_name, n_rows=len(data), variable_notes=variable_notes
        )

    return PipelineOutput(
        dataset_name=dataset_name,
        n_rows_used=len(data),
        n_rows_total=n_total,
        target_col=target_col,
        feature_cols=features,
        gp_results=results,
        insights=insights,
        engine=engine,
    )


def plot_results(output: PipelineOutput):
    """One panel per feature: data, GP mean, 68% and 95% predictive bands."""
    n = len(output.feature_cols)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.6), squeeze=False)
    for i, col in enumerate(output.feature_cols):
        ax, r = axes[0][i], output.gp_results[col]
        c = PALETTE[i % len(PALETTE)]
        ax.scatter(r.x_train, r.y_train, s=6, alpha=0.25, color=c, label="Training data")
        ax.plot(r.x_grid, r.y_mean, color=c, lw=2, label="GP mean")
        ax.fill_between(r.x_grid, r.y_lower, r.y_upper, color=c, alpha=0.18, label="95% band")
        ax.fill_between(r.x_grid, r.y_mean - r.y_std, r.y_mean + r.y_std, color=c, alpha=0.28, label="68% band")
        passed = output.engine.gate_passes(r)
        colour = "#2E7D32" if passed else "#C62828"
        ax.text(
            0.03, 0.97,
            f"R\u00b2 test = {r.r2_test:.3f}\nCoverage95 = {r.coverage_95_test:.2f}\n"
            f"Conf = {r.confidence:.3f} {'PASS' if passed else 'BLOCKED'}",
            transform=ax.transAxes, va="top", fontsize=9, color=colour, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=colour, alpha=0.9),
        )
        ax.set_title(f"{col} \u2192 {output.target_col}", fontweight="bold", fontsize=10)
        ax.set_xlabel(col)
        if i == 0:
            ax.set_ylabel(output.target_col)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"SciSense: {output.dataset_name}", fontweight="bold")
    fig.tight_layout()
    return fig
