"""
AMDEC — Literature Baseline Comparison Generator.

Generates:
1. Terminal-formatted comparative analysis table.
2. LaTeX table snippet formatted for IEEE / ACM conference submissions.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_baseline_comparison():
    print("=" * 95)
    print("  AMDEC: CONCEPTUAL COMPARISON AGAINST LITERATURE BASELINES")
    print("=" * 95)

    headers = [
        "System", "Target Domain", "Candidate Selection", "Matching Technique",
        "Reported Savings", "Reported CPU Overhead", "Primary Limitation"
    ]

    rows = [
        [
            "VMware CBPS (2002)",
            "Hypervisor VMs",
            "Periodic full scan",
            "Exact SHA-1 Hash",
            "~20-60% (homogeneous)",
            "Periodic CPU spikes",
            "Exact-only; static binaries"
        ],
        [
            "Linux KSM (2009)",
            "Linux Kernel",
            "Uniform red-black scan",
            "Exact memcmp",
            "~20-30% (workloads)",
            "10-25% single core",
            "Single byte diff blocks merge"
        ],
        [
            "UKSM (2018)",
            "Linux Kernel",
            "Hierarchical region scan",
            "Adaptive partial hash",
            "~25-35% (dynamic)",
            "5-10% CPU load",
            "Region-aware, but exact-match only"
        ],
        [
            "Difference Engine (2008)",
            "Xen Hypervisor",
            "Unguided search",
            "Pairwise delta (bsdiff)",
            "~40-60% footprint",
            "High pairwise cost",
            "O(N^2) unguided delta search"
        ],
        [
            "SiLo (2011)",
            "Backup Storage",
            "Locality-aware LSH",
            "Chunk MinHash",
            "~90% disk reduction",
            "Offline batch",
            "Secondary storage, not live RAM"
        ],
        [
            "AMDEC (This Work)",
            "Anonymous Heaps",
            "Region-aware tiering (M1)",
            "MinHash + LSH (M2) + Base-Delta (M3)",
            "70-75% logical savings",
            "<5% scan overhead (sim)",
            "Userspace simulation artifact"
        ],
    ]

    # Format Terminal Table
    col_widths = [24, 18, 25, 34, 22, 22, 32]
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)

    print(header_line)
    print(sep_line)
    for r in rows:
        print(" | ".join(f"{c:<{w}}" for c, w in zip(r, col_widths)))

    print("=" * 95)
    print("\n* Note: Baselines cited from peer-reviewed literature (not same-machine benchmarks).")

    # Generate LaTeX Table Code
    latex_code = r"""
\begin{table*}[t]
\centering
\caption{Architectural and Quantitative Comparison with Published Literature Baselines}
\label{tab:baseline_comparison}
\small
\begin{tabular}{p{2.6cm}p{2.2cm}p{3.2cm}p{3.2cm}p{2.5cm}p{2.5cm}}
\hline
\textbf{System} & \textbf{Environment} & \textbf{Candidate Selection} & \textbf{Matching Strategy} & \textbf{Memory Reduction} & \textbf{Primary Limitation} \\
\hline
VMware CBPS~\cite{waldspurger2002} & Hypervisor & Periodic full scan & Exact SHA-1 hash & 20--60\% & Exact-match only \\
Linux KSM~\cite{arcangeli2009} & Kernel & Uniform red-black scan & Exact \texttt{memcmp} & 20--30\% & Misses near-duplicates \\
UKSM~\cite{xia2018} & Kernel & Hierarchical distilling & Adaptive partial hash & 25--35\% & Exact-match only \\
Difference Engine~\cite{gupta2008} & Xen Hypervisor & Unguided search & Pairwise delta (\texttt{bsdiff}) & 40--60\% & $O(N^2)$ search overhead \\
SiLo~\cite{xia2011} & Storage & Locality clustering & MinHash chunking & $\sim$90\% (disk) & Backup storage only \\
\textbf{AMDEC (Ours)} & Userspace Sim & \textbf{Region stability tiering} & \textbf{MinHash-LSH + Base-Delta} & \textbf{70--75\% (logical)} & Simulation artifact \\
\hline
\end{tabular}
\end{table*}
"""
    latex_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "baseline_comparison_table.tex"))
    with open(latex_path, "w") as f:
        f.write(latex_code.strip())

    print(f"\n[LaTeX Table Exported]: {latex_path}")


if __name__ == "__main__":
    generate_baseline_comparison()
