"""
AMDEC — Multi-Trial Statistical Analysis & LaTeX Exporter.

Reads raw trial CSV outputs from results/raw/, calculates multi-trial statistical metrics
(Mean, Sample Std, 95% Student's t Confidence Intervals, Median, Min, Max), exports
structured processed CSV summaries to results/processed/, and generates publication-grade LaTeX tables.
"""

import csv
import os
import sys
from typing import Any, Dict, List
import numpy as np
import scipy.stats as stats

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_column_stats(values: List[float], confidence: float = 0.95) -> Dict[str, Any]:
    """Calculates comprehensive statistical aggregations for a continuous sample."""
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = float(stats.sem(arr)) if n > 1 else 0.0
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, n - 1)) if n > 1 else 0.0
    ci_hw = float(t_crit * se) if n > 1 else 0.0

    return {
        "n": n,
        "mean": m,
        "std": s,
        "sem": se,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci_lower": m - ci_hw,
        "ci_upper": m + ci_hw,
        "ci_half_width": ci_hw,
    }


def process_primary_trials():
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "raw"))
    proc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "processed"))
    os.makedirs(proc_dir, exist_ok=True)

    trials_file = os.path.join(raw_dir, "primary_trials.csv")
    if not os.path.exists(trials_file):
        print(f"Warning: {trials_file} does not exist. Please run 'python experiments/run_full_eval.py' first.")
        return

    with open(trials_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    n_trials = len(rows)
    metrics_to_agg = [
        ("memory_savings_percent", "Logical Memory Savings (%)"),
        ("compression_ratio", "Effective Compression Ratio"),
        ("amdec_memory_mb", "AMDEC Compressed Footprint (MB)"),
        ("num_clusters", "Similarity Clusters Formed"),
        ("mean_delta_bytes", "Mean Delta Size (w/ metadata, B)"),
        ("m1_reduction_pct", "M1 Volatile Scan Reduction (%)"),
        ("pair_reduction_percent", "M2 LSH Candidate Reduction (%)"),
        ("mean_reconstruction_us", "Mean Reconstruction Latency (µs)"),
        ("p95_reconstruction_us", "P95 Reconstruction Latency (µs)"),
        ("p99_reconstruction_us", "P99 Reconstruction Latency (µs)"),
        ("simd_speedup", "M4 SIMD Differencing Speedup (x)"),
        ("pipeline_time_ms", "Total Pipeline Runtime (ms)"),
        ("pipeline_throughput_pages_s", "Pipeline Throughput (pages/s)"),
    ]

    summary_rows = []
    print("=" * 95)
    print(f"  AMDEC STATISTICAL SUMMARY ({n_trials} Independent Trials)")
    print("=" * 95)
    print(f"  {'Metric':<36} | {'Mean ± Std':<18} | {'95% Confidence Interval':<24} | {'Median [Min, Max]'}")
    print("-" * 95)

    for col_key, label in metrics_to_agg:
        vals = [float(r[col_key]) for r in rows if col_key in r and r[col_key]]
        if vals:
            st = compute_column_stats(vals)
            summary_rows.append({
                "metric_key": col_key,
                "metric_label": label,
                "n_trials": st["n"],
                "mean": st["mean"],
                "std": st["std"],
                "ci_lower": st["ci_lower"],
                "ci_upper": st["ci_upper"],
                "ci_half_width": st["ci_half_width"],
                "median": st["median"],
                "min": st["min"],
                "max": st["max"],
            })
            print(f"  {label:<36} | {st['mean']:6.2f} ± {st['std']:5.2f}    | [{st['ci_lower']:6.2f}, {st['ci_upper']:6.2f}]           | {st['median']:6.2f} [{st['min']:6.2f}, {st['max']:6.2f}]")

    # Export Processed Summary CSV
    summary_csv = os.path.join(proc_dir, "primary_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Export LaTeX Table Snippet
    latex_path = os.path.join(proc_dir, "primary_statistical_table.tex")
    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{AMDEC Multi-Trial System Evaluation ($N=10,000$ pages, 10 independent trials, 95\% CI)}",
        r"\label{tab:amdec_evaluation}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Mean $\pm$ Std} & \textbf{95\% CI} & \textbf{Median} & \textbf{[Min, Max]} \\",
        r"\hline",
    ]

    for sr in summary_rows:
        latex_lines.append(
            f"{sr['metric_label']} & {sr['mean']:.2f} $\\pm$ {sr['std']:.2f} & [{sr['ci_lower']:.2f}, {sr['ci_upper']:.2f}] & {sr['median']:.2f} & [{sr['min']:.2f}, {sr['max']:.2f}] \\\\"
        )

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))

    print("-" * 95)
    print(f"  [Summary Exported] : {summary_csv}")
    print(f"  [LaTeX Exported]   : {latex_path}")
    print("=" * 95)


if __name__ == "__main__":
    process_primary_trials()
