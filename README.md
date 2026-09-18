# SciSense

**Gaussian Process uncertainty gating for LLM-generated scientific insights.**

<!-- After deploying, add: **Live demo:** https://huggingface.co/spaces/<your-username>/scisense -->

SciSense fits a Gaussian Process (GP) to each feature → target relationship in a numeric dataset, turns the fit into a confidence score, and uses that score as a gate in front of a large language model:

* **Below the threshold** the LLM is never called. The relationship is reported as blocked.
* **Above the threshold** the LLM writes a short summary, and the strength of its wording is tied to the confidence tier (indicative, strong, definitive).
* **After generation** every number in the LLM reply is checked against the numbers the GP engine produced, and any unsupported number is flagged.

The idea: a language model should only make a claim about data when a probabilistic model of that data is confident enough to back it, and should not sound more certain than the evidence.

This is a self-directed portfolio project. It builds on my MSc dissertation (Aston University), which applied GP regression and other methods to a synthetic superconductivity dataset. SciSense reuses that dataset and adds the gating and LLM layers.

## How it works

```
CSV ──► per-feature GP (sklearn) ──► held-out R², RMSE, 95% interval coverage
                                         │
                                         ▼
                                confidence score ──► gate (default 0.75)
                                                        │
                                   blocked ◄────────────┴────────────► passed
                                (no LLM call)                             │
                                                                          ▼
                                             prompt built only from GP numbers
                                             + wording rule for the tier
                                                                          │
                                                                          ▼
                                             LLM (Llama 3.3 70B via Groq, temperature 0)
                                                                          │
                                                                          ▼
                                             unsupported-number check on the reply
```

**GP engine** (`scisense/gp_engine.py`)
* One GP per feature, inputs standardised on the training split, `normalize_y=True`.
* Three candidate kernels (RBF, Matérn 1.5, Matérn 2.5), each with a constant scale and a white-noise term.
* The kernel is chosen by **log marginal likelihood on the training split**. The 20% test split is used once, for reporting.
* Reported per feature: train and test R², test RMSE, **empirical coverage of the 95% predictive interval on test points**, learned noise level and length scale, and any optimiser convergence warnings.

**Confidence score and gate**

```
confidence = 0.65 · max(0, R²_test) + 0.35 · 1 / (1 + relative_uncertainty)
relative_uncertainty = mean(predictive std) / mean(|predicted mean|)
```

The gate passes when `confidence ≥ threshold` (default 0.75). Tiers: ≥ 0.95 definitive, ≥ 0.85 strong, ≥ threshold indicative.

**Negative control.** The pipeline can add a column of pure random noise as an extra feature. Its GP fit has R² ≈ 0, so the gate should block it. This is how the gate is shown to work rather than assumed to work.

## Results on the Superconductivity and Disorder dataset

Dataset: 1,200 synthetic samples generated from stochastic BCS equations ([Kaggle](https://www.kaggle.com/datasets/karanmin/superconductivity-and-disorder)). Target: `direct_t_c`.

| Feature | Kernel | R² test | 95% coverage | Confidence | Gate |
|---|---|---|---|---|---|
| mean_lambda | RBF | 0.917 | 0.93 | 0.873 | PASS |
| mean_t_c | RBF | 0.914 | 0.92 | 0.875 | PASS |
| random_noise_control | Matérn 1.5 | -0.010 | 0.95 | 0.177 | BLOCKED |

Held-out test split: 240 of 1,200 rows. The random-noise control is blocked (confidence 0.177 against a 0.75 threshold), which shows the gate rejects an uninformative feature. With 240 test points, coverage estimates carry roughly ±0.014 sampling error: mean_lambda (0.93) is consistent with nominal 95% coverage, mean_t_c (0.92) is slightly overconfident, and lambda_0 (1.00) is conservative because the relationship is almost noise-free.


<!-- Fill this table from `python -m scisense.report data/tc_dataset_1200_updated.csv direct_t_c` -->

How to read this: `direct_t_c` is computed deterministically from `lambda_0`, so a near-perfect fit there is a sanity check, not a discovery. The informative comparison is between `lambda_0` and the disorder-averaged `mean_lambda`: the drop in R² and the much wider predictive band quantify how much information about Tc is lost once disorder is averaged over.

## Limitations

* The confidence score is a hand-designed heuristic, not a calibrated probability. The 0.65 / 0.35 weights and the 0.75 threshold are design choices and have not been tuned against labelled outcomes.
* The relative-uncertainty term depends on the target's offset (it divides by the mean predicted value), so it is not scale-free.
* Each GP is one-dimensional. Interactions between features are not modelled.
* GP fitting is O(n³); the app subsamples datasets above 1,500 rows with a fixed seed.
* The unsupported-number check catches invented numbers, not invented claims. Whether the LLM's wording actually matches the tier has not been evaluated systematically.
* Results so far are on one synthetic dataset.

## Run it

```bash
git clone https://github.com/Karanm5/Scisense.git
cd Scisense
pip install -r requirements.txt
export GROQ_API_KEY=...        # optional; without it the GP analysis and gate still run
python app.py                  # Gradio app on http://127.0.0.1:7860
```

Put the dataset at `data/tc_dataset_1200_updated.csv` to use it as the built-in example.

From Python:

```python
import pandas as pd
from scisense import run_scisense

df = pd.read_csv("data/tc_dataset_1200_updated.csv")
out = run_scisense(df, target_col="direct_t_c", add_noise_control=True)
for r in out.gp_results.values():
    print(out.engine.summary(r))
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests use synthetic data and a stub LLM, so they need no API key. They check that a deterministic relationship passes the gate, an unrelated feature is blocked, a blocked feature never triggers an LLM call, interval coverage is sensible, and the unsupported-number check flags invented values.

## Repository layout

```
scisense/        GP engine, gate, LLM layer, pipeline and plotting
app.py           Gradio app (Hugging Face Spaces entry point)
tests/           pytest suite
notebooks/       original Colab development notebook (history)
figures/         figures from the development notebook
```

## Stack

Python, scikit-learn (GaussianProcessRegressor), NumPy, pandas, Matplotlib, Gradio, Groq API (Llama 3.3 70B), pytest.

## Licence

MIT
