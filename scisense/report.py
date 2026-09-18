"""Print the README results table for a CSV.

Usage: python -m scisense.report data/tc_dataset_1200_updated.csv direct_t_c
"""

import sys

import pandas as pd

from .pipeline import NOISE_CONTROL, run_scisense


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 1
    path, target = argv
    out = run_scisense(pd.read_csv(path), target_col=target, add_noise_control=True)
    print("| Feature | Kernel | R² test | 95% coverage | Confidence | Gate |")
    print("|---|---|---|---|---|---|")
    for col, r in out.gp_results.items():
        gate = "PASS" if out.engine.gate_passes(r) else "BLOCKED"
        print(f"| {col} | {r.kernel_name} | {r.r2_test:.3f} | {r.coverage_95_test:.2f} | {r.confidence:.3f} | {gate} |")
    print()
    for r in out.gp_results.values():
        print(out.engine.summary(r), end="\n\n")
    if NOISE_CONTROL in out.gp_results and out.engine.gate_passes(out.gp_results[NOISE_CONTROL]):
        print("WARNING: the noise control passed the gate. Investigate before publishing results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
