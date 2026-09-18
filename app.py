"""SciSense Gradio app (Hugging Face Spaces entry point).

Set GROQ_API_KEY as a Space secret to enable the LLM layer. Without it the
app still runs the GP analysis and the gate, and says the LLM is disabled.
"""

from __future__ import annotations

import os

import gradio as gr
import pandas as pd

from scisense import run_scisense
from scisense.llm_layer import DEFAULT_MODEL, groq_completion
from scisense.pipeline import NOISE_CONTROL, numeric_columns, plot_results

EXAMPLE_CSV = os.path.join(os.path.dirname(__file__), "data", "tc_dataset_1200_updated.csv")
EXAMPLE_TARGET = "direct_t_c"
EXAMPLE_NOTES = {
    "lambda_0": "bare electron-phonon coupling constant before disorder is applied",
    "mean_lambda": "coupling constant averaged over disorder realisations",
    "mean_t_c": "critical temperature averaged over disorder realisations (normalised)",
    "direct_t_c": "critical temperature computed directly (normalised)",
}
COMPLETE_FN = groq_completion(DEFAULT_MODEL)


def load_df(file_path):
    if file_path:
        return pd.read_csv(file_path), os.path.basename(file_path)
    if os.path.exists(EXAMPLE_CSV):
        return pd.read_csv(EXAMPLE_CSV), "Superconductivity and Disorder (example)"
    raise gr.Error("Upload a CSV file.")


def on_upload(file_path):
    try:
        df, _ = load_df(file_path)
    except gr.Error:
        return gr.update(choices=[], value=None)
    cols = numeric_columns(df)
    default = EXAMPLE_TARGET if EXAMPLE_TARGET in cols else (cols[-1] if cols else None)
    return gr.update(choices=cols, value=default)


def analyse(file_path, target, threshold, noise_control):
    df, name = load_df(file_path)
    if not target:
        raise gr.Error("Choose a target column.")
    notes = EXAMPLE_NOTES if file_path is None else None
    try:
        out = run_scisense(
            df,
            target_col=target,
            dataset_name=name,
            confidence_threshold=float(threshold),
            add_noise_control=noise_control,
            variable_notes=notes,
            complete_fn=COMPLETE_FN,
        )
    except ValueError as exc:
        raise gr.Error(str(exc))

    rows = [
        "| Feature | Kernel | R² test | 95% coverage | Confidence | Gate |",
        "|---|---|---|---|---|---|",
    ]
    for col, r in out.gp_results.items():
        gate = "PASS" if out.engine.gate_passes(r) else "BLOCKED"
        label = f"{col} (control)" if col == NOISE_CONTROL else col
        rows.append(
            f"| {label} | {r.kernel_name} | {r.r2_test:.3f} | "
            f"{r.coverage_95_test:.2f} | {r.confidence:.3f} | {gate} |"
        )
    sub = (
        f" (random sample of {out.n_rows_used:,} from {out.n_rows_total:,})"
        if out.n_rows_used < out.n_rows_total
        else ""
    )
    summary = (
        f"**{name}**: {out.n_rows_used:,} rows used{sub}, target `{target}`, "
        f"gate threshold {threshold:.2f}\n\n" + "\n".join(rows)
    )

    parts = []
    for col, ins in out.insights.items():
        status = "Gate passed" if ins.gate_passed else "Gate blocked"
        block = f"#### {col}\n*{status} · tier: {ins.certainty_level} · confidence {ins.confidence:.3f}*\n\n{ins.insight}"
        if ins.unsupported_numbers:
            block += (
                "\n\n> Check: the reply contains numbers not present in the GP output: "
                + ", ".join(ins.unsupported_numbers)
            )
        parts.append(block)
    llm_note = (
        f"LLM: {DEFAULT_MODEL} via Groq, temperature 0."
        if COMPLETE_FN
        else "LLM disabled: no GROQ_API_KEY set. GP analysis and gate still run."
    )
    return summary, plot_results(out), "\n\n---\n\n".join(parts) + f"\n\n*{llm_note}*"


with gr.Blocks(title="SciSense") as demo:
    gr.Markdown(
        "# SciSense\n"
        "Gaussian Process uncertainty gating for LLM-generated insights. "
        "A GP is fitted to each feature → target pair. The LLM is only called "
        "for pairs whose confidence score passes the gate, and its wording is "
        "tied to that score.\n\n"
        "Leave the upload empty to run the bundled superconductivity example."
    )
    with gr.Row():
        with gr.Column():
            file_in = gr.File(label="CSV file (optional)", file_types=[".csv"], type="filepath")
            target_in = gr.Dropdown(label="Target column", choices=[], allow_custom_value=False)
            threshold_in = gr.Slider(0.5, 0.95, value=0.75, step=0.01, label="Gate threshold")
            noise_in = gr.Checkbox(
                value=True, label="Add a random-noise control feature (should be blocked)"
            )
            run_btn = gr.Button("Run SciSense", variant="primary")
        with gr.Column():
            summary_out = gr.Markdown()
    plot_out = gr.Plot(label="GP mean with 68% and 95% predictive bands")
    insights_out = gr.Markdown()

    file_in.change(on_upload, inputs=file_in, outputs=target_in)
    demo.load(on_upload, inputs=file_in, outputs=target_in)
    run_btn.click(
        analyse,
        inputs=[file_in, target_in, threshold_in, noise_in],
        outputs=[summary_out, plot_out, insights_out],
    )

if __name__ == "__main__":
    demo.launch()
