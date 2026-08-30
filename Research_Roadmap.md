# AMDEC — End-to-End Research & Patent Project Roadmap
### Adaptive Memory Deduplication via Locality-Sensitive Similarity Clustering and Base-Delta Encoding

> **Platform:** macOS / Linux, Python 3.x userspace simulation (NumPy, datasketch, scipy, pandas, matplotlib, pytest)  
> **Methodology:** Design Science Research (DSR)  
> **Scope:** Algorithmic simulation & evaluation artifact — pure userspace, no kernel module, no hypervisor/VM  
> **Target Output:** Peer-reviewed research paper (Scopus/UGC-CARE indexed) + optional patent disclosure  
> **Timeline:** ~10 weeks

---

## 📋 Quick Overview

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7
 Setup      Workload     Module 2    Module 3    Module 5    M4 & Full   Analysis    Paper &
 & Scope    & M1         (LSH)       (Delta)     (COW)       Evaluation  & Plots     Publish
 (Wk 0)     (Wk 1-2)     (Wk 3-4)    (Wk 5)      (Wk 6)      (Wk 7)      (Wk 8)      (Wk 9-10)
                            ▲                                               ▲
                            │                                               │
                        REVIEW 2                                        REVIEW 3
                       CHECKPOINT                                      CHECKPOINT
```

---

## 1. Scope & Execution Clarification

1. **Simulated Logical Memory Savings:** Algorithmic footprint of raw base pages, unmerged pages, delta byte payloads, and metadata overhead.
2. **Python Process Memory (RSS):** Memory occupied by the Python runtime interpreter — *not to be confused with algorithmic savings*.
3. **Execution Runtime:** Measured wall-clock time of the Python userspace simulation on the testbed machine.
4. **Kernel / Baseline Comparison:** Published literature figures from KSM/UKSM cited strictly as conceptual benchmarks, not same-machine runs.

---

## 2. Canonical Repository Structure

```
AMDEC/
├── src/
│   ├── __init__.py
│   ├── workload/
│   │   ├── __init__.py
│   │   └── page_generator.py         # Synthetic anonymous heap workload & Zipf write traffic
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── m1_sampling.py            # Region-aware sampling & feedback engine
│   │   ├── m2_lsh_clustering.py      # MinHash + LSH feature extraction & clustering
│   │   ├── m3_delta_storage.py       # Base selection & metadata-aware delta storage
│   │   ├── m4_acceleration.py        # Vectorized vs scalar diff benchmark
│   │   └── m5_cow.py                 # Simulated COW fault handler & page table
│   └── common/
│       ├── __init__.py
│       ├── types.py                  # Page dataclasses, enums, record structures
│       └── metrics.py                # Metric collectors and export utilities
│
├── experiments/
│   ├── demo_week2.py                 # Week 2 verification demo
│   ├── demo_review2.py               # Review 2 live demo script (M1 + M2)
│   ├── run_full_eval.py              # Single-command end-to-end evaluation
│   ├── parameter_sweep.py            # Multi-parameter workload sweep
│   ├── ablation_m1.py                # M1 enabled vs disabled experiment
│   ├── ablation_delta.py             # Adaptive vs fixed delta thresholding
│   └── ablation_lsh.py               # LSH vs brute-force candidate matching
│
├── tests/
│   ├── __init__.py
│   ├── test_page_generator.py        # Workload distribution & write tests
│   ├── test_m1_sampling.py           # Stability tiering & feedback tests
│   ├── test_m2_lsh.py                # MinHash, LSH indexing & clustering tests
│   ├── test_m3_delta.py              # Base selection, delta RLE & fidelity tests
│   └── test_m5_cow.py                # COW write handling & page reconstruction tests
│
├── analysis/
│   ├── plot_results.py               # Publication-grade chart generation
│   ├── compare_baselines.py          # Literature comparison table generator
│   └── statistical_analysis.py       # Mean, standard error, confidence intervals
│
├── results/
│   ├── raw/                          # Timestamped CSV logs
│   └── figures/                      # High-resolution PNG/PDF charts
│
├── docs/
│   ├── architecture.md               # Detailed architectural writeup
│   └── INVENTION_DISCLOSURE.md       # Dated invention disclosure record
│
├── paper/                            # LaTeX manuscript source (IEEE format)
├── references/                       # BibTeX bibliography & reference PDFs
├── requirements.txt                  # Python dependencies
├── .gitignore
├── README.md
└── Research_Roadmap.md
```

---

## 3. Formal Metrics Table

| Metric Category | Metric Name | Formal Definition | Unit | Target / Objective |
|---|---|---|---|---|
| **Storage** | Logical Memory Savings | $(\text{Baseline} - \text{AMDEC}) / \text{Baseline} \times 100$ | $\%$ | Quantify reduction |
| **Storage** | Compression Ratio | $\text{Baseline} / \text{AMDEC}$ | ratio | Higher is better |
| **Storage** | Average Delta Size | Mean byte footprint of delta records (incl. metadata) | Bytes | Compact representation |
| **Sampling (M1)** | Candidate Reduction | $1 - (\text{Candidates to M2} / \text{Total Pages})$ | $\%$ | $>50\%$ volatile pages skipped |
| **Sampling (M1)** | Tier Distribution | $\% \text{ HOT}, \text{WARM}, \text{COLD}$ | $\%$ | Skewed stability |
| **Clustering (M2)** | Cluster Precision | True Near-Duplicates / Total Clustered Pairs | ratio | $\ge 0.90$ |
| **Clustering (M2)** | Cluster Recall | Identified Similar / Ground-Truth Similar | ratio | High coverage |
| **Latency (M5)** | Reconstruction Latency | Wall-clock time to reconstruct $P_{\text{base}} \oplus \text{Delta}$ | $\mu\text{s}$ | Low retrieval delay |
| **Latency (M5)** | Tail Latency | 95th (P95) and 99th (P99) percentile reconstruction | $\mu\text{s}$ | Bounded tail |
| **Runtime** | Component Execution Time | Wall-clock time for M1, M2, M3, and M5 | $\text{ms}$ | Measured simulation timing |

---

## 4. Required Ablation Experiments

1. **Module 1 Ablation (Sampling Efficacy):**
   * AMDEC with M1 vs. AMDEC with M1 disabled (uniform full scan).
   * Measure: Candidate count entering M2, M2 clustering execution time, and false COW breaks.
2. **Module 3 Ablation (Adaptive Thresholding):**
   * AMDEC with adaptive fallback vs. Fixed threshold / no fallback.
   * Measure: Logical memory savings, tail delta sizes, and reconstruction latency.
3. **Module 2 Ablation (LSH vs. Brute Force):**
   * MinHash + LSH clustering vs. Brute-force pairwise similarity search ($N = 1,000–5,000$).
   * Measure: Total pair comparisons, clustering wall-clock runtime, and cluster recall.

---

## 5. Review Checkpoints

### ⚑ Review 2 Checkpoint (End of Week 4)
* **Deliverables:**
  * M1 & M2 fully operational and integrated in Python.
  * Synthetic workload generating ground-truth near-duplicate clusters.
  * M1 stability tiering filtering out hot pages.
  * M2 producing validated similarity clusters.
  * Unit tests passing for Workload, M1, and M2.
  * Live demo script (`experiments/demo_review2.py`).

### ⚑ Review 3 Checkpoint (End of Week 10 — Final)
* **Deliverables:**
  * All 5 modules (M1–M5) integrated into single-command evaluation (`experiments/run_full_eval.py`).
  * 100% round-trip reconstruction correctness verified.
  * Full parameter sweep results (mutation rate, similarity mix, scalability to 100K pages).
  * 3 ablation studies completed and charted.
  * Academic paper draft in IEEE two-column format.
  * Live demo (3 scenarios) + backup recorded video.

---

*Last Updated: August 2026 | AMDEC Research Roadmap | Pure Python Userspace Simulation*
