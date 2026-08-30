# Adaptive Memory Deduplication using Similarity Clustering (AMDEC)
## Technical Architecture, Research Methodology, and Implementation Roadmap

**Platform:** macOS / Linux, Python 3.x userspace simulation (NumPy, datasketch, scipy, pandas, matplotlib, pytest).  
**Scope:** Algorithmic simulation & evaluation artifact — pure userspace, no kernel module, no hypervisor/VM, no root privileges.

---

## 1. Project Specification & Scope

### 1.1 Official Project Scope
**AMDEC (Adaptive Memory Deduplication via Locality-Sensitive Similarity Clustering and Base-Delta Encoding)** is a Python-based userspace simulation of an adaptive memory deduplication system. It identifies near-duplicate anonymous memory pages using locality-sensitive similarity clustering and stores them using compact base-delta representations.

The project is an **algorithmic simulation and evaluation artifact**, not a production operating system memory-management implementation.

### 1.2 Clear Memory & Execution Distinctions
To maintain technical defense and scientific integrity, this project strictly distinguishes between four independent concepts:

1. **Simulated Logical Memory Consumption:** The algorithmic memory footprint computed by the model (accounting for raw base pages, unmerged pages, delta byte payloads, and metadata overhead).
2. **Simulator Python Process Footprint (RSS):** The physical memory occupied by the Python runtime interpreter, object wrappers, and data structures while executing the simulation. *Python RSS does not represent the memory savings of the deduplication algorithm.*
3. **Algorithm Execution Time:** The measured wall-clock and CPU processing time of the Python/NumPy simulation pipeline running on the experimental testbed.
4. **Production OS / Kernel Memory Behavior:** Theoretical physical memory subsystem behavior, which is modeled via discrete-event simulation rather than live kernel-space interception.

### 1.3 Role of Python in Systems Research
The transition to a Python-first simulation affects execution speed, but **not theoretical storage reduction or clustering correctness**:

| Category | Characteristic | Impact of Python Implementation |
|---|---|---|
| **Language-Independent Core Metrics** | Similarity detection, cluster membership, delta size, compression ratio, logical memory savings, algorithmic precision/recall, byte-for-byte reconstruction fidelity | **Zero impact.** Algorithmic math, hash distributions, and compression ratios are mathematically identical regardless of language. |
| **Language-Dependent Metrics** | Wall-clock execution time, simulation throughput (MB/s), reconstruction latency, absolute CPU utilization | **Dependent.** Measured numbers represent the performance of the Python userspace simulator on the testbed and are reported as such (not claimed as kernel-level C/assembly timings). |

---

## 2. Research Methodology

AMDEC follows the **Design Science Research (DSR)** framework standard for systems and applied computer science research:

| DSR Phase | AMDEC Implementation | Deliverable |
|---|---|---|
| **1. Problem Identification** | Literature analysis showing exact-match deduplicators miss up to 86% of near-duplicate anonymous memory pages | Literature matrix, Gap formalization (Gaps 1–4) |
| **2. Design** | 5-module adaptive architecture (M1–M5) integrating sampling, LSH clustering, base-delta storage, and COW reconstruction | Architecture specification & data flow design |
| **3. Development** | Userspace simulation engine in Python (NumPy, datasketch) | Modular codebase with automated test suite |
| **4. Demonstration** | Execution against controlled synthetic memory workloads with varied similarity, mutation, and skew parameters | Experiment orchestration harnesses |
| **5. Evaluation** | Measurement of logical memory reduction, candidate reduction, cluster precision/recall, reconstruction latency, tail latencies, and ablation studies | CSV dataset logs, publication figures, baseline comparison |
| **6. Communication** | Academic paper, reproducible repository, and live interactive demo | Research paper draft, presentation slides |

> [!IMPORTANT]
> **Methodological Positioning:** The research contribution is evaluated at the algorithmic and simulation level. The absence of kernel integration limits claims about production operating-system throughput, but does not invalidate the scientific evaluation of the proposed representations, filtering heuristics, and clustering strategies.

---

## 3. System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │   Synthetic Anonymous Memory Workload   │
                    │   Generator (Python / NumPy)            │
                    │   • Controllable near/exact/unique mix  │
                    │   • Zipf-distributed write traffic      │
                    └────────────────────┬────────────────────┘
                                         │ List[Page]
                    ┌────────────────────▼────────────────────┐
                    │ MODULE 1: Region-Aware Sampling &       │
                    │ Adaptive Feedback Engine                │
                    │ • Age, write-rate, idle-time scoring    │
                    │ • HOT / WARM / COLD classification      │
                    │ • COW break dynamic quarantine          │
                    └────────────────────┬────────────────────┘
                                         │ Candidate Pages (COLD / WARM)
                    ┌────────────────────▼────────────────────┐
                    │ MODULE 2: Feature Extraction &          │
                    │ MinHash + LSH Clustering Engine         │
                    │ • 128-byte sub-block chunking           │
                    │ • MinHash signature generation (K=32)   │
                    │ • LSH banding (b=8) + Union-Find        │
                    └────────────────────┬────────────────────┘
                                         │ Similarity Clusters
                    ┌────────────────────▼────────────────────┐
                    │ MODULE 3: Base-Delta Storage &          │
                    │ Adaptive Patching Manager               │
                    │ • Representative base selection         │
                    │ • XOR diff + run-length encoding        │
                    │ • Metadata-aware delta sizing           │
                    │ • Adaptive fallback thresholding        │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │ MODULE 4: Computational Acceleration    │
                    │ Benchmark                               │
                    │ • Python loop vs. NumPy vectorized diff │
                    │ • Optional: native C extension          │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │ MODULE 5: Simulated COW Fault Handler & │
                    │ Reconstruction Engine                   │
                    │ • Simulated Page Table (Base/Delta/Full)│
                    │ • Byte-for-byte exact reconstruction    │
                    │ • Latency & tail percentile profiling   │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
                             Raw CSV Experiment Data
                                         │
                                         ▼
                       Python Analysis & Visualizations
```

---

## 4. Module-Level Design

### Module 1: Region-Aware Sampling & Adaptive Feedback Engine
* **Purpose:** Address Gap 2 (CPU Overhead) and Gap 3 (Semantic Ignorance) by filtering out volatile pages before expensive similarity analysis.
* **Page Lifecycle Metrics:**
  $$\text{age} = \max(1.0, t_{\text{current}} - t_{\text{alloc}})$$
  $$\text{idle\_time} = \max(0.0, t_{\text{current}} - t_{\text{last\_write}})$$
  $$\text{effective\_writes} = \text{write\_count} + (\text{cow\_faults} \times w_{\text{cow\_penalty}})$$
  $$\text{write\_rate} = \frac{\text{effective\_writes}}{\text{age}}$$
* **Stability Tiers:**
  * **`HOT`:** $\text{write\_rate} \ge \tau_{\text{hot}}$ (or under active quarantine) $\rightarrow$ skipped immediately.
  * **`WARM`:** Moderate write rate or $\text{age} < \text{min\_age\_for\_cold}$ $\rightarrow$ queued at low priority.
  * **`COLD`:** $\text{write\_rate} \le \tau_{\text{cold}}$, $\text{age} \ge \text{min\_age\_for\_cold}$, $\text{idle\_time} > \text{min\_idle}$ $\rightarrow$ queued at highest priority.
* **COW Fault Feedback Loop:** When a merged page experiences a write fault in Module 5, Module 1 logs the fault and applies a quarantine window ($\Delta t_{\text{quarantine}}$) preventing rapid re-scanning.
* **Distinction in Reporting:** We distinguish **candidate reduction** ($\% \text{ of pages skipped}$) from **measured execution time reduction** (ms saved in M2).

---

### Module 2: Feature Extraction & MinHash + LSH Clustering Engine
* **Purpose:** Address Gap 1 (Binary Exactness) and Gap 4 (Unguided Delta Search) by grouping near-duplicate pages into similarity clusters at sub-quadratic cost.
* **Pipeline:**
  1. **Sub-block Chunking:** Divide each 4096-byte page into $M$ fixed-size blocks (default $B = 128\text{ bytes} \implies 32\text{ blocks}$).
  2. **MinHash Signatures:** Compute $K$ hash functions (default $K = 32$) over the block shingles to build a compact signature vector.
  3. **LSH Banding:** Partition signatures into $b$ bands of $r$ rows ($K = b \times r$, e.g., $b=8, r=4$). Hash each band into hash buckets.
  4. **Pairwise Verification & Union-Find:** Pages sharing at least one LSH bucket are evaluated via Jaccard similarity. Pairs meeting the threshold ($J \ge \tau_{\text{sim}}$, default $0.70$) are merged using a Union-Find disjoint set structure into clusters.
* **Algorithmic Complexity Note:** LSH reduces candidate comparisons in the *expected case* from $O(n^2)$ to $O(n)$ by restricting verification to pages sharing locality-sensitive buckets. Worst-case bucket collisions on identical pages remain $O(n^2)$ within that bucket.

---

### Module 3: Base-Delta Storage & Adaptive Patching Manager
* **Purpose:** Compress clusters of near-duplicate pages into a representative base page plus compact delta diffs.
* **Representative Base Selection:**
  * For small clusters ($|C| \le 16$): Compute pairwise Hamming/byte distance; choose medoid minimizing $\sum_{p \in C} \text{dist}(\text{base}, p)$.
  * For large clusters ($|C| > 16$): Use oldest/most stable page or random medoid sample to maintain linear complexity.
* **Delta Representation (Metadata-Aware):**
  * Raw XOR diff: $\Delta_{\text{raw}} = P_{\text{target}} \oplus P_{\text{base}}$.
  * Run-Length Encoding (RLE) of contiguous non-zero byte spans:
    $$\text{Delta Record} = \{\text{base\_id}: \text{uint32}, \text{num\_runs}: \text{uint16}, [(\text{offset}: \text{uint16}, \text{length}: \text{uint16}, \text{data}: \text{bytes}) \dots]\}$$
  * **Strict Accounting:** Delta size **must** include offset metadata (4 bytes per span), record header (6 bytes), and base pointer reference.
* **Adaptive Thresholding:**
  $$\text{if } \text{Size}(\text{Delta Record}) > \delta_{\text{max}} \times 4096 \implies \text{Store as full unmerged page}$$

---

### Module 4: Computational Acceleration Benchmark
* **Purpose:** Quantify the computational efficiency of delta generation and reconstruction.
* **Evaluation Scope:**
  * **Baseline:** Pure Python scalar byte loops.
  * **Vectorized:** NumPy uint64-vectorized XOR and non-zero span detection.
  * **Optional Native Extension:** C-extension via `ctypes`/`clang` (optional exploratory bonus, not a requirement for project success).
* **Metrics:** Processing latency ($\mu\text{s}$ per page), throughput ($\text{MB/s}$), and speedup factor.

---

### Module 5: Simulated COW Fault Handler & Reconstruction Engine
* **Purpose:** Prove correctness and measure the latency of restoring delta-compressed pages upon write access.
* **Page Table State Machine:**
  * Maps $\text{page\_id} \to (\text{State}, \text{Payload})$, where $\text{State} \in \{\text{FULL\_PAGE}, \text{BASE\_PAGE}, \text{DELTA\_PAGE}\}$.
* **Simulated Write Path:**
  1. Process attempts write to $\text{page\_id}$.
  2. If $\text{DELTA\_PAGE}$: Reconstruct full page $\hat{P} = P_{\text{base}} \oplus \text{Apply}(\text{Delta})$.
  3. Verify byte-for-byte exact equality: $\hat{P} == P_{\text{original}}$.
  4. Allocate new simulated private page object, apply write modifications, update Page Table to $\text{FULL\_PAGE}$.
  5. Trigger Module 1 COW feedback penalty.
* **Metrics:** Reconstruction latency distribution (Mean, Median, P95, P99) and reconstruction success rate ($100\%$).

---

## 5. Evaluation Methodology & Metrics

### 5.1 Storage Model
We evaluate **Simulated Logical Memory Savings** using rigorous metadata accounting:

$$\text{Baseline Storage} = N \times 4096 \text{ bytes}$$

$$\text{AMDEC Storage} = \sum \text{Size}(P_{\text{base}}) + \sum \text{Size}(P_{\text{unique}}) + \sum \text{Size}(P_{\text{unmerged}}) + \sum \text{Size}(\text{Delta Records}) + \text{Page Table Overhead}$$

$$\text{Logical Memory Savings (\%)} = \frac{\text{Baseline Storage} - \text{AMDEC Storage}}{\text{Baseline Storage}} \times 100$$

$$\text{Compression Ratio} = \frac{\text{Baseline Storage}}{\text{AMDEC Storage}}$$

### 5.2 Official Metric Table

| Metric Category | Metric Name | Formal Definition | Unit |
|---|---|---|---|
| **Storage** | Logical Memory Savings | $(\text{Baseline} - \text{AMDEC}) / \text{Baseline} \times 100$ | $\%$ |
| **Storage** | Compression Ratio | $\text{Baseline} / \text{AMDEC}$ | ratio ($X:1$) |
| **Storage** | Average Delta Size | Mean byte footprint of stored delta records (incl. metadata) | Bytes |
| **Sampling (M1)** | Candidate Reduction | $1 - (\text{Candidates Forwarded to M2} / \text{Total Pages})$ | $\%$ |
| **Sampling (M1)** | Tier Distribution | Proportions of HOT, WARM, and COLD pages | $\%$ |
| **Clustering (M2)** | Cluster Precision | True Near-Duplicates Clustered / Total Clustered Pairs | ratio $[0, 1]$ |
| **Clustering (M2)** | Cluster Recall | Identified Similar Pairs / Ground-Truth Similar Pairs | ratio $[0, 1]$ |
| **Clustering (M2)** | Cluster Size Distribution | Mean, Median, and Max cluster cardinality | pages/cluster |
| **Latency (M5)** | Reconstruction Latency | Wall-clock time to reconstruct $P_{\text{base}} \oplus \text{Delta}$ | $\mu\text{s}$ |
| **Latency (M5)** | Tail Latency | 95th (P95) and 99th (P99) percentile reconstruction latency | $\mu\text{s}$ |
| **Runtime** | Component Execution Time | Wall-clock time for M1, M2, M3, and M5 | $\text{ms}$ / seconds |

### 5.3 Workload Parameter Sweeps
Rather than testing a single fixed scenario, AMDEC is evaluated across multi-dimensional parameter sweeps:
1. **Similarity Composition Sweep:** Near-duplicate proportion varied from $20\% \to 40\% \to 60\% \to 80\%$.
2. **Mutation / Divergence Sweep:** Mutation rate varied from $0.1\% \to 1\% \to 5\% \to 10\% \to 20\% \to 30\%$ byte modifications.
3. **Exact Duplicate Sweep:** Exact matches varied from $0\% \to 7\% \to 15\% \to 30\%$.
4. **Scalability Sweep:** Workload scaled from $N = 10,000 \to 25,000 \to 50,000 \to 100,000$ pages.
5. **Write Traffic Skew Sweep:** Uniform random writes vs. Zipfian write distributions ($\alpha = 1.0 \to 1.5$).

### 5.4 Baseline Comparison Framework

> [!NOTE]
> Published systems (KSM, UKSM) operated in Linux kernel space. Direct wall-clock runtime comparisons with our Python userspace simulator would be unscientific. Comparisons are strictly conceptual and based on reported literature figures.

| System | Execution Environment | Evaluation Source | Comparison Scope |
|---|---|---|---|
| **KSM** (Linux Mainline) | Linux Kernel Space | Published Literature (Arcangeli et al., 2009) | Exact-match memory savings, CPU scan overhead |
| **UKSM** | Linux Kernel Space | Published Literature (Xia et al., 2018) | Adaptive region scan efficiency, exact sharing |
| **Difference Engine** | Xen Hypervisor | Published Literature (Gupta et al., 2008) | Sub-page delta savings, unguided search costs |
| **AMDEC (Ours)** | macOS/Linux Userspace Simulator | Measured Experimental Runs (This Work) | Near-duplicate clustering savings, candidate reduction, reconstruction latency |

---

## 6. Required Ablation Studies

To isolate and prove the scientific contribution of each architectural component, three ablation experiments are mandatory:

```
[Ablation 1: Module 1 Sampling Value]
   AMDEC (Full Pipeline)  vs.  AMDEC (M1 Disabled — Full Uniform Scan)
   → Measure: Candidate count to M2, M2 clustering execution time, false COW breaks.

[Ablation 2: Adaptive Delta Thresholding]
   AMDEC (Adaptive Threshold)  vs.  AMDEC (Fixed Delta Threshold / No Fallback)
   → Measure: Logical memory savings, tail delta sizes, reconstruction latency.

[Ablation 3: LSH vs. Brute-Force Similarity Search]
   AMDEC (MinHash + LSH)  vs.  Brute-Force Pairwise Jaccard Search (on N=1,000–5,000)
   → Measure: Total pair comparisons, clustering wall-clock time, cluster recall.
```

---

## 7. Canonical Repository Structure

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

## 8. Phased Implementation Roadmap

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7
 Setup      Workload     Module 2    Module 3    Module 5    M4 & Full   Analysis    Paper
 & Scope    & M1         (LSH)       (Delta)     (COW)       Evaluation  & Plots     Drafting
 (Wk 0)     (Wk 1-2)     (Wk 3-4)    (Wk 5)      (Wk 6)      (Wk 7)      (Wk 8)      (Wk 9-10)
                            ▲                                               ▲
                            │                                               │
                        REVIEW 2                                        REVIEW 3
                       CHECKPOINT                                      CHECKPOINT
```

### Phase Details

* **Phase 0 — Repository & Scope Setup (Week 0):** Python project layout, virtualenv, dependencies (`requirements.txt`), git branching, and invention disclosure record.
* **Phase 1 — Synthetic Workload & Module 1 (Weeks 1–2):** Page data model, realistic anonymous memory generator (80/7/13 composition), Zipfian write simulator, stability tiering (`HOT`/`WARM`/`COLD`), priority queue, COW break quarantine feedback, unit tests.
* **Phase 2 — Module 2: MinHash + LSH Clustering (Weeks 3–4):** 128B chunking, MinHash signatures, LSH banding, candidate pair generation, Jaccard thresholding, Union-Find clustering, M1 $\to$ M2 integration. *(Review 2 Target)*
* **Phase 3 — Module 3: Base-Delta Storage (Week 5):** Medoid base selection, XOR + RLE diff encoding, metadata-aware delta size accounting, adaptive size thresholding, reconstruction fidelity tests.
* **Phase 4 — Module 5: Simulated COW & Reconstruction (Week 6):** Simulated page table, write-fault handling, byte-for-byte correctness verification, reconstruction latency profiling (P95, P99).
* **Phase 5 — Module 4 & Full Evaluation (Week 7):** NumPy-vectorized diff benchmark, parameter sweeps (mutation rate, similarity composition, scale up to 100K pages), ablation experiments (M1 ablation, Delta ablation, LSH ablation).
* **Phase 6 — Statistical Analysis & Visualization (Week 8):** CSV export aggregation, publication-grade matplotlib/seaborn plots, confidence interval calculations, literature comparison tables.
* **Phase 7 — Paper Writing & Submission QA (Weeks 9–10):** Manuscript drafting (Methodology $\to$ Evaluation $\to$ Related Work $\to$ Intro $\to$ Abstract), plagiarism check ($<10\%$), AI-content check ($<5\%$), 3-scenario demo script, backup demo recording.

---

## 9. Review Checkpoints

### ⚑ Review 2 Checkpoint (Target: End of Week 4)
**Required Demonstrable Deliverables:**
1. **Module 1 & Module 2 Fully Operational:**
   - Synthetic workload generation with ground-truth labels.
   - Module 1 stability tiering filtering out hot pages.
   - Module 2 MinHash-LSH pipeline producing validated similarity clusters.
2. **Integration Pipeline:** Candidate pages from M1 stream directly into M2.
3. **Quantitative Evidence:**
   - Candidate reduction percentage ($\% \text{ pages skipped}$).
   - LSH bucket collision and cluster size distribution.
   - Cluster precision on synthetic ground truth.
4. **Automated Unit Tests:** Passing test suite for Workload, M1, and M2.
5. **Live Demo:** Running script generating workload $\to$ M1 filtering $\to$ M2 clustering $\to$ inspecting resulting clusters.

---

### ⚑ Review 3 Checkpoint (Final Project Review)
**Required Demonstrable Deliverables:**
1. **Complete End-to-End Pipeline:** All modules (M1 $\to$ M2 $\to$ M3 $\to$ M4 $\to$ M5) executed via a single reproducible script (`experiments/run_full_eval.py`).
2. **Comprehensive Experimental Results:**
   - Logical memory savings across similarity and mutation sweeps.
   - Reconstruction latency profile (Mean, Median, P95, P99).
   - Module 4 acceleration speedup (scalar vs NumPy vectorized).
   - Full ablation study suite (M1, Delta, LSH).
   - Conceptual comparison table against literature baselines.
3. **Correctness Verification:** $100\%$ byte-for-byte round-trip reconstruction passing automated tests.
4. **Academic Manuscript:** Complete paper formatted to IEEE/Springer guidelines with all figures, equations, and citations.
5. **Demonstration Assets:** Live 3-scenario demo + pre-recorded backup video.

---

## 10. Research Limitations & Defensive Positioning

To ensure academic rigor during defense and peer review, the following limitations are explicitly acknowledged:

1. **Simulation Scope:** AMDEC is a userspace simulation designed to evaluate representations and clustering algorithms; it does not replace an in-kernel virtual memory manager or measure kernel TLB/page-table translation overheads.
2. **Execution Timing:** Absolute wall-clock timings reflect Python userspace performance and must not be conflated with kernel-space C/assembly latencies.
3. **Synthetic Workloads:** While our synthetic workloads accurately model published anonymous memory characteristics (up to 86% near-duplicates, Zipfian access skew), they represent controlled synthetic benchmarks rather than live production VM traces.
4. **Hardware Offloading:** Hardware acceleration results (e.g., Intel DSA) are analyzed as software-vectorized analogies (NumPy) rather than bare-metal silicon measurements.

---

*Last Updated: August 2026 | AMDEC Project Specification | Design Science Research (DSR)*
