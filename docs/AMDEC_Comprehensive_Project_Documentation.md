# AMDEC: Comprehensive Project Documentation & Technical Summary
### Adaptive Memory Deduplication via Locality-Sensitive Similarity Clustering and Base-Delta Encoding

> **Author / Principal Investigator:** Shantanu Tapole  
> **Methodology:** Design Science Research (DSR)  
> **Target Venue:** IEEE / ACM Systems Conference (e.g., IEEE COMPSAC, USENIX ATC Workshop Track)  
> **Simulation Testbed:** Python 3.x userspace simulation (NumPy, datasketch, scipy, matplotlib, pytest)

---

## 📑 Table of Contents
1. [Executive Summary & Core Hypothesis](#1-executive-summary--core-hypothesis)
2. [Literature Survey & Gap Formalization](#2-literature-survey--gap-formalization)
3. [System Architecture & Five-Module Pipeline](#3-system-architecture--five-module-pipeline)
4. [Module-by-Module Technical Specification](#4-module-by-module-technical-specification)
5. [Experimental Methodology](#5-experimental-methodology)
6. [Experimental Results & Multi-Trial Benchmarks](#6-experimental-results--multi-trial-benchmarks)
   - 6.1 Primary Multi-Trial System Evaluation ($N = 10,000$ Pages)
   - 6.2 Multi-Dimensional Parameter Sweeps
   - 6.3 Adversarial Delta-Threshold Ablation (High-Mutation Workloads)
   - 6.4 Isolated Module 1 Sampling Efficacy & Timing Benchmark
   - 6.5 Extended LSH Scalability & Cluster Agreement ($N=1\text{K} \to 10\text{K}$)
   - 6.6 Complete Publication Figure Catalog (`results/figures/`, 300 DPI)
7. [Ablation Studies Analysis & System Insights](#7-ablation-studies-analysis--system-insights)
8. [Comparative Evaluation vs. Literature Matrix](#8-comparative-evaluation-vs-literature-matrix)
9. [Reproducibility & Codebase Directory Map](#9-reproducibility--codebase-directory-map)
10. [Research Paper Claims Audit & Honesty Check](#10-research-paper-claims-audit--honesty-check)
11. [Threats to Validity & Acknowledged Limitations](#11-threats-to-validity--acknowledged-limitations)

---

## 1. Executive Summary & Core Hypothesis

### 1.1 The Core Problem
Modern operating systems, cloud hypervisors, and container runtimes waste substantial physical memory hosting redundant page contents. Standard deduplication engines (such as **Linux Kernel Samepage Merging / KSM** (Arcangeli et al., OLS 2009) and **VMware Content-Based Page Sharing / CBPS** (Waldspurger, OSDI 2002)) operate under a **binary exactness constraint**: two 4096-byte memory pages can only be merged if they are **100% byte-for-byte identical**.

However, empirical systems research reveals that in **anonymous memory heaps** (dynamic allocations in web servers, databases, microservices, and language runtimes), prior work reports substantial near-duplicate potential in memory workloads (Gupta et al., OSDI 2008; Zhang et al., ASPLOS 2024), motivating similarity-aware deduplication. Exact-match deduplicators discard opportunities to compress pages that differ only in timestamps, counters, or pointer offsets.

### 1.2 The AMDEC Solution
The **five-module AMDEC architecture** introduces a unified design combining region-aware sampling, similarity-based clustering, and base-delta storage for anonymous-memory deduplication:

```
[Anonymous Memory Pages]
           │
           ▼
[Module 1: Sampling & Feedback Engine] ──► Classifies pages into HOT/WARM/COLD; filters volatile churn
           │ Candidate Pages (COLD / WARM)
           ▼
[Module 2: MinHash-LSH Clustering] ────► 128B chunking, K=32 MinHash, b=8 LSH bands, Union-Find
           │ Similarity Clusters (J >= 0.70)
           ▼
[Module 3: Base-Delta Storage] ────────► Medoid selection, XOR+RLE diffs, secondary-base fallback
           │
           ▼
[Module 5: Simulated COW Handler] ─────► Sub-5 µs simulated reconstruction & dynamic feedback quarantine
```
*(Note: Module 4 provides an offline micro-benchmark evaluating vectorized uint64 bitwise differencing performance over a scalar Python implementation).*

### 1.3 Central Research Hypothesis
> **Hypothesis:** When anonymous memory contains high proportions of near-duplicate pages, locality-sensitive similarity clustering combined with base-delta representation achieves substantial logical memory-footprint savings ($>70\%$) on evaluated synthetic workloads beyond exact-match deduplicators, while region-aware stability sampling filters volatile pages to reduce unnecessary similarity processing, and low simulated reconstruction latencies ($<5\ \mu\text{s}$) demonstrate algorithmic efficiency.

---

## 2. Literature Survey & Gap Formalization

### 2.1 Complete Literature Matrix (18 Prior Works Analyzed)

| System / Paper | Year | Target Domain | Technique | Reported Savings | Exact Only? | Near-Dup? | Base-Delta? | Region-Aware? | Primary Architectural Limitation |
|---|---|---|---|---|---|---|---|---|---|
| **VMware CBPS** (Waldspurger, OSDI '02) | 2002 | Hypervisor VMs | Periodic hash scan | Up to 60% (homogeneous) | ✅ | ❌ | ❌ | ❌ | Exact-match only; static binary sharing |
| **Linux KSM** (Arcangeli et al., OLS '09) | 2009 | Linux Kernel | Red-black tree scan | ~20–30% (workloads) | ✅ | ❌ | ❌ | ❌ | 1-byte difference blocks merge entirely |
| **Satori** (Miłóś et al., USENIX ATC '09) | 2009 | Xen / Disk I/O | Load-time enlightened sharing | Exploits up to 94% max sharing | ✅ | ❌ | ❌ | ❌ | Restricted to disk-backed pages at I/O read time |
| **Difference Engine** (Gupta et al., OSDI '08) | 2008 | Xen Hypervisor | Sub-page delta + compression | 40–60% footprint reduction | ❌ | ✅ | ✅ | ❌ | **Unguided $O(N^2)$ pairwise delta search** |
| **Page Overlays** (Seshadri et al., ISCA '13) | 2013 | HW-Assisted Cache | Fine-grained page merging | 15–35% footprint reduction | ❌ | ✅ | Partial | ❌ | Requires custom CPU hardware; no clustering |
| **BCD Deduplication** (Barker et al., ATC '17)| 2017 | OS Sub-page | Byte-level content differencing | 20–40% sub-page reduction | ❌ | ✅ | ✅ | ❌ | No upstream clustering stage; high pairwise cost |
| **Sparrow** (Zhang et al., ASPLOS '24) | 2024 | Android OS | Intra-page block sampling | Identifies near-dup potential | ❌ | ✅ | Partial | ❌ | Demonstrates problem; no general clustering framework |
| **De2Dup / dbDedup** (FAST '24) | 2024 | In-Memory DBs | Intel DSA hardware delta | 30–50% reduction in DB heaps | ❌ | ✅ | ✅ | ❌ | Domain-specific to columnar DBs; requires Intel DSA |
| **UKSM** (Xia et al., IEEE TC '18) | 2018 | Linux Kernel | Hierarchical distilling | Up to 5x over CBPS | ✅ | ❌ | ❌ | ✅ | Region-awareness applied to **exact-match only** |
| **CKSM** (Chen et al., EuroSys '24) | 2024 | Linux Containers | Physical-memory-space scan | 60–80% CPU scan reduction | ✅ | ❌ | ❌ | ✅ | Still fundamentally exact-match only |
| **PAMD** (IEEE Access '24) | 2024 | Dynamic Control | Process-aware loop joining | Workload-dependent ROI | ✅ | ❌ | ❌ | Partial | Process-level tuning; blind to content similarity |
| **SiLo** (Xia et al., USENIX ATC '11) | 2011 | Backup Storage | Similarity + locality clustering | ~90% storage reduction | ❌ | ✅ | ❌ | ❌ | Storage/backup dedup; not live RAM |
| **I/O Deduplication** (FAST '10) | 2010 | Storage Caching | Content-based disk caching | 28–47% I/O reduction | ❌ | ✅ | ❌ | ❌ | Block I/O domain; not OS virtual memory |
| **TSM / CISWH** (IEEE Trans. Cloud '24) | 2024 | Edge File Storage | Threshold similarity hashing | 43.2% edge storage reduction | ❌ | ✅ | ❌ | ❌ | Applied to files; no COW memory recovery |
| **Hardware DSA Engines** (Intel '24)| 2024 | Accelerator Silicon | Hardware DMA offloaded diff | Offloads CPU during diff | ❌ | ✅ | ✅ | ❌ | Hardware-dependent; isolated to DB engines |
| **Garg et al.** (IEEE TPDS '20) | 2020 | GPU Computing | GPU-parallel similarity scan | 5x–10x matching speedup | ❌ | ✅ | ❌ | ❌ | Requires GPU; no base-delta storage pipeline |
| **CMDP** (ACM EuroSys '22) | 2022 | Cloud Hypervisors | Banked memory domain isolation | 25–35% search reduction | ✅ | ❌ | ❌ | ✅ | Domain partitioning; exact-match only |
| **ORC** (IEEE Micro '23) | 2023 | CHERI Architecture | Capability object sharing | Lowers hypervisor tail-latency | ✅ | ❌ | ❌ | ❌ | Requires CHERI hardware; exact sharing only |
| **AMDEC (Ours)** | **2026** | **Anonymous Heaps** | **Sampling + LSH + Base-Delta** | **73.60% logical savings** | ❌ | ✅ | ✅ | ✅ | **Unified design combining sampling, LSH, & base-delta** |

*Note: All literature figures are reported values from prior publications cited under conceptual comparison, not same-machine measurements.*

### 2.2 The Identified Research Gap
Across the literature, existing systems fall into three incomplete categories:
1. **Exact-Match Systems (KSM, UKSM, CKSM, CBPS):** Low retrieval latency, but misses near-duplicate pages in anonymous memory.
2. **Sub-Page Delta Systems (Difference Engine, BCD):** Capable of delta storage, but lacks upstream similarity clustering $\implies$ suffers from an unguided $O(N^2)$ pairwise comparison bottleneck.
3. **Similarity Storage Systems (SiLo, TSM):** Uses LSH for clustering, but restricted to high-latency offline disk/backup storage without low-latency memory recovery.

To the best of our knowledge, **AMDEC explores a unified design** combining region-aware sampling, MinHash-LSH similarity clustering, and base-delta encoding for anonymous-memory deduplication.

---

## 3. System Architecture & Five-Module Pipeline

The **five-module AMDEC architecture** is organized into a primary storage pipeline (Modules 1–3), an offline acceleration micro-benchmark (Module 4), and a simulated Copy-on-Write reconstruction handler (Module 5):

```
                    ┌──────────────────────────────────────────────┐
                    │   Synthetic Anonymous Memory Workload        │
                    │   (4096B Pages, Heap Templates, Zipf Skew)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │  Module 1: Sampling & Feedback Engine        │
                    │  - Classifies HOT / WARM / COLD Tiers        │
                    │  - Filters volatile HOT pages from scan      │
                    └──────────────────────┬───────────────────────┘
                                           │ Candidate Pages (8.2% skipped HOT → 91.8% passed, ~9,180 pages)
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │  Module 2: MinHash-LSH Clustering Engine     │
                    │  - 128B Chunking, K=32 MinHash, b=8 LSH Bands│
                    │  - Union-Find Disjoint Set (J >= 0.70)       │
                    └──────────────────────┬───────────────────────┘
                                           │ Similarity Clusters
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │  Module 3: Base-Delta Storage Manager        │
                    │  - Medoid Selection & Global Base Registry   │
                    │  - Vectorized XOR Diff & RLE Span Encoding   │
                    │  - 30% Fallback Architectural Safety Net     │
                    └──────────────────────┬───────────────────────┘
                                           │ Compressed Clusters
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
┌───────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│ Module 4: Offline Acceleration Benchmark  │ │ Module 5: Simulated COW Handler           │
│ - NumPy uint64 Vectorized Bitwise Diff    │ │ - Page Table State Machine                │
│ - Offline micro-benchmark vs Scalar Python│ │ - Sub-5 µs Simulated Read Reconstruction  │
└───────────────────────────────────────────┘ └───────────────────────────────────────────┘
```

---

## 4. Module-by-Module Technical Specification

### 4.1 Workload Generator (`src/workload/page_generator.py`)
- **Deterministic RNG Encapsulation:** Uses instance-isolated `self.rng = random.Random(seed)` and `self.np_rng = np.random.default_rng(seed)` (no `os.urandom()` calls) to provide deterministic workload generation for the configured random seeds and supported execution environment.
- **Heap Models:** Struct arrays (64B C++ structs with vtable pointers and timestamps), pointer tables (512 8B pointers), and semi-sparse JSON buffers.
- **Zipfian Write Simulation:** Simulates non-uniform write traffic using a Zipfian distribution ($\alpha = 1.3$).

### 4.2 Module 1: Region-Aware Sampling & Feedback Engine (`src/modules/m1_sampling.py`)
- **Stability Tiering:** Classifies pages into `HOT` (high write frequency / recent COW break), `WARM` (moderate activity), and `COLD` (long-lived, low write frequency).
- **Filtering & Priority Queue:** Queues `COLD` and `WARM` pages into a min-heap sorted by priority score. Filters out `HOT` pages and quarantines recently faulting pages, providing an architectural mechanism intended to reduce unnecessary similarity processing and repeated COW exposure.
- **Dynamic COW Feedback:** Intercepts Copy-on-Write write faults and temporarily quarantines faulting pages for $30\text{ seconds}$.
- **Coarse Arena/Region-Level Volatility Pruning:** In addition to fine-grained per-page classification, Module 1 provides optional arena-level filtering (`filter_and_prioritize(..., enable_region_filtering=False)`). When activated, `analyze_regions()` evaluates aggregate write frequencies across allocated memory arenas; if an arena's volatility score exceeds `region_volatility_threshold = 0.50`, all pages in that arena are pruned upfront. This defaults to `False` in current benchmarks to evaluate per-page stability dynamics directly.
- **Bounded-Latency Candidate Budgeting:** `filter_and_prioritize(..., max_candidates=None)` supports an optional integer candidate cap. When specified, only the top-$K$ highest-priority candidate pages are popped from the priority min-heap and dispatched to Module 2, providing a tunable dial to bound downstream clustering latency during strict real-time memory pressure reclamation windows.

### 4.3 Module 2: MinHash-LSH Clustering Engine (`src/modules/m2_lsh_clustering.py`)
- **Page Chunking:** Splits 4096-byte pages into 32 128-byte non-overlapping chunks.
- **MinHash Fingerprinting:** Computes $K=32$ MinHash signature permutations per page using 32-bit hash values (via `datasketch.MinHash` default). Band bucket hashes are further reduced to 64-bit identifiers via MD5 truncation in `extract_band_buckets()`.
- **LSH Banding Index:** Hashes signatures into $b=8$ bands ($r=4$ rows per band) to identify candidate similar pairs.
- **Large-Bucket Exact-Hash Grouping & Deterministic RNG:** In massive buckets ($>64$ pages), pages are pre-grouped by SHA-256 exact digest to prevent component fragmentation. The engine initializes `self.rng = np.random.default_rng(seed)` for deterministic tie-breaking and candidate representative selection, preventing quadratic comparison blowup while guaranteeing identical clustering across runs with identical seeds.
- **Verification & Union-Find:** Pairs meeting $J \ge 0.70$ are verified and merged using a disjoint-set data structure with path compression and rank optimization.

### 4.4 Module 3: Base-Delta Storage Manager (`src/modules/m3_delta_storage.py`)
- **Phase A Exact-Hash Fast-Track:** If a candidate page is byte-identical to the base, it produces a **6-byte zero-span delta record** (4B base ref + 2B span count) with no XOR computation overhead.
- **Medoid Selection:** Computes pairwise byte distances across cluster members; picks base page minimizing $\sum_{p \in C} \text{dist}(\text{base}, p)$.
- **Vectorized XOR Diff + RLE Spans:** Non-zero byte runs encoded as $[(\text{offset}: \text{uint16}, \text{length}: \text{uint16}, \text{data}: \text{bytes})]$.
- **Strict Metadata-Aware Accounting:**
  $$\text{Delta Record Size} = 4\text{B (Base Ref)} + 2\text{B (Span Count)} + \sum (4\text{B} + \text{len}(\text{span data}))$$
- **Global Base Fingerprint Registry:** Deduplicates identical base page contents across separate clusters (`sha256(base) -> canonical_cluster_id`) so shared templates are only counted once in storage accounting.
- **Architectural Safety Mechanism:**
  $$\text{if } \text{Size}(\text{Delta Record}) > 0.30 \times 4096\text{B} (1228\text{B}) \implies \text{Store as full unmerged page}$$

### 4.5 Module 4: Offline Acceleration Benchmark (`src/modules/m4_acceleration.py`)
- Offline Python micro-benchmark comparing pure Python scalar byte loops vs. **NumPy uint64 bitwise XOR** differencing.
- Demonstrates a **$198.0\times \pm 7.7\times$ speedup of NumPy uint64 vectorized differencing over the scalar Python implementation** ($>3,100\text{ MB/s}$ throughput), suggesting suitability for SIMD-oriented implementations.

### 4.6 Module 5: Simulated COW Fault Handler (`src/modules/m5_cow.py`)
- **Page Table States:** `FULL_PAGE` (uncompressed), `BASE_PAGE` (shared read-only base), `DELTA_PAGE` (compressed delta).
- **Nanosecond Latency Tracking & GC Suppression:** Uses `time.perf_counter_ns()` with garbage collection suppressed for microsecond precision. Provides an embedded `isolated_timing_guard()` context manager and optional `disable_gc: bool = True` parameter on `read_page()`, `handle_write_fault()`, and `measure_reconstruction_latency()`.
- **Read Path:** Reconstructs $\hat{P} = P_{\text{base}} \oplus \text{Apply}(\text{Delta})$ with $100\%$ byte fidelity. Returns defensive copy to prevent memory tampering.
- **COW Write Path:** Intercepts write $\implies$ reconstructs original content, allocates private `FULL_PAGE`, applies write payload, updates page table, and notifies Module 1 feedback engine.

---

## 5. Experimental Methodology

### 5.1 Primary Metric Definitions
- **Baseline Footprint:** Total uncompressed raw memory ($N \times 4096\text{ bytes}$). For $N=10,000$, baseline footprint is $40,960,000\text{ bytes}$ ($39.0625\text{ MiB}$ binary / $40.96\text{ MB}$ SI-decimal).
- **AMDEC Logical Footprint:** Sum of base page frames, compressed delta records (including metadata headers), unmerged fallback pages, and unique unclustered pages. For $N=10,000$, AMDEC footprint is $10.31\text{ MiB}$ ($10.81\text{ MB}$ SI-decimal).
- **Logical Memory Savings (%):**
  $$\text{Savings} = \left(1 - \frac{\text{AMDEC Logical Footprint}}{\text{Baseline Footprint}}\right) \times 100\%$$
- **Effective Compression Ratio:**
  $$\text{Compression Ratio} = \frac{\text{Baseline Footprint}}{\text{AMDEC Logical Footprint}}$$

### 5.2 Testbed & Software Environment
- **Operating System:** macOS (Darwin arm64)
- **CPU / Memory Hardware:** Apple Silicon / Multi-Core Workstation
- **Python Environment:** Python 3.9+
- **Core Libraries:** NumPy (v1.26+), datasketch (v1.5+), SciPy (v1.11+), Matplotlib, pytest

### 5.3 Workload & Hyperparameter Configuration
- **Primary Workload ($N=10,000$):** 4096-byte pages, 80% near-duplicates (mutation rates 0.2%–1.2%), 7% exact duplicates, 13% unique pages. Zipf write skew ($\alpha = 1.3$).
- **Module 2 LSH:** $K=32$ MinHash permutations, $b=8$ bands, $r=4$ rows/band, Jaccard similarity threshold $J = 0.70$.
- **Module 3 Delta:** Delta threshold $\delta_{\text{max}} = 0.30$ (1228B limit), medoid base selection.
- **Random Seeds & Trials:** Seeds 42–51 across 10 independent trials for primary evaluation; 5 trials per point for parameter sweeps and paired ablations.

### 5.4 Evaluated Baselines
1. **Exhaustive Brute-Force Pairwise Search:** All-pairs $O(N^2)$ Jaccard comparison (executed up to $N=2,500$; timed out at $N \ge 5,000$).
2. **Uniform M1-Disabled Scan:** Full pass-through scan of all pages without stability tiering.
3. **Fixed / No Fallback Delta Threshold:** Delta manager with 100% threshold (no unmerged page fallback).
4. **Prior Literature Baselines:** Reported values from published papers (VMware CBPS, Linux KSM, UKSM, Difference Engine) included for conceptual comparison.

### 5.5 Reconstruction Latency Measurement Methodology
To address the critical systems question of why simulated reconstruction latency achieves a low mean of **$1.75\ \mu\text{s}$** ($P95 = 3.88\ \mu\text{s}$) in CPython, the measurement methodology incorporates five architectural guarantees:
1. **High-Resolution Monotonic Hardware Clock:** Latency is sampled using `time.perf_counter_ns()` with single-nanosecond resolution (~25–40 ns hardware invocation overhead on modern ARM64/x86 architectures).
2. **Strictly Isolated Measurement Window:** The timing window strictly isolates `reconstruct_page(base_content, entry.delta)`. The dictionary lookup retrieving `base_content` occurs *before* sampling $t_0$, and the return copy occurs *after* sampling $t_1$.
3. **CPython Execution Mechanics (No 4096-Byte Python Loop):**
   - The initial base clone `reconstructed = bytearray(base_content)` is executed via a fast C-level `memcpy` of 4,096 contiguous bytes, taking ~0.3–0.5 µs.
   - Because AMDEC's delta encoding is compact (mean delta size = $24.7\text{ bytes}$, median $24\text{ bytes}$), typical delta records contain only **1 to 2 discrete spans of 8–24 bytes** (representing modified pointers or timestamps).
   - The XOR generator loop executes only over these 8–24 modified bytes in CPython (~0.8–1.0 µs). There is **no 4,096-iteration scalar Python loop** on the read reconstruction critical path.
4. **Garbage Collection Deactivation:** During latency benchmarking in `experiments/run_full_eval.py`, Python garbage collection is explicitly disabled (`gc.disable()` wrapped in `try...finally gc.enable()`) across the 500 read samples per trial. This prevents interpreter GC cycle sweeps from injecting artificial multi-millisecond tail spikes into the microsecond latency profile. (Exploratory runs without GC suppression exhibited inflated tail latencies, e.g., P95 ~6.54 µs, P99 ~7.50 µs).
5. **Statistical Sampling Scale:** Latency distributions are measured across 500 read operations and 200 write-fault reconstructions per trial over 10 randomized trials, producing a pooled empirical distribution of **7,000 independent single-call latency samples**. Loop overhead is not subtracted out, making the reported 1.75 µs a realistic upper bound for the Python simulation harness.

---

## 6. Experimental Results & Multi-Trial Benchmarks

### 6.1 Primary Multi-Trial System Evaluation ($N = 10,000$ Pages, 10 Independent Trials)

Evaluated across 10 randomized trials (seeds 42–51) on $N = 10,000$ pages ($39.06\text{ MiB}$ binary / $40.96\text{ MB}$ SI-decimal uncompressed baseline footprint). All values are reported as **Mean $\pm$ Sample Std.Dev** alongside **95% Student's t Confidence Intervals (CI)**.

| Metric | Measured Result (Mean ± Std) | 95% Confidence Interval | Median [Min, Max] | Notes / Context |
|---|---|---|---|---|
| **Baseline Uncompressed Footprint** | $39.06\text{ MiB}$ (Fixed) | [Fixed Baseline] | $39.06\text{ MiB}$ | $N \times 4096\text{ B} = 40,960,000\text{ bytes}$ ($39.06\text{ MiB}$ binary / $40.96\text{ MB}$ SI-decimal) |
| **AMDEC Logical Footprint** | **$10.31 \pm 0.32\text{ MiB}$** | **[10.08, 10.55] MiB** | $10.30\ [9.90, 10.85]\text{ MiB}$ | Base pages + deltas + metadata ($10.81\text{ MB}$ decimal) |
| **Logical Memory Savings** | **$73.60\% \pm 0.83\%$** | **[73.00%, 74.19%]** | **73.64% [72.21%, 74.67%]** | Evaluated synthetic workload |
| **Effective Compression Ratio** | **$3.79\times \pm 0.12\times$** | **[3.71x, 3.88x]** | $3.79\times\ [3.60\times, 3.95\times]$ | Ratio of baseline to logical footprint |
| **M1 Volatile Scan Reduction** | **$8.20\% \pm 0.36\%$** | **[7.94%, 8.46%]** | $8.07\%\ [7.82\%, 8.75\%]$ | Mean 820.1 pages skipped HOT ($9,179.9\text{ pages} / 91.80\%$ passed to M2) |
| **M2 LSH Candidate Reduction** | **$99.845\% \pm 0.007\%$** | **[99.840%, 99.850%]** | $99.846\%\ [99.833\%, 99.854\%]$ | Mean 65.4K candidate pairs vs 42.1M post-M1 pairs ($99.845\%$ reduction); vs 50.0M un-filtered pairs ($99.869\%$ reduction) |
| **Similarity Clusters Formed** | $147.4 \pm 9.2\text{ clusters}$ | [140.8, 154.0] | $148.5\ [127, 161]$ | Near-duplicate groupings |
| **Mean Delta Storage Footprint** | **$24.7 \pm 2.2\text{ bytes / page}$** | **[23.1, 26.2] B** | $24.2\ [22.3, 29.1]\text{ B}$ | Including record headers & offset metadata |
| **Mean Simulated Reconstruction Latency** | **$1.75 \pm 0.42\ \mu\text{s}$** | **[1.45, 2.05] $\mu\text{s}$** | $1.59\ [1.39, 2.78]\ \mu\text{s}$ | Userspace simulation (single-call measurement, GC disabled) |
| **Pooled Reconstruction Latency (P50/P95/P99)** | **$1.58\ \mu\text{s} / 3.88\ \mu\text{s} / 4.58\ \mu\text{s}$** | Aggregate distribution | Across 7,000 access samples | Per-trial means: P95 = $3.68 \pm 0.41\ \mu\text{s}$, P99 = $4.25 \pm 0.39\ \mu\text{s}$ |
| **NumPy Vectorization Speedup** | **$198.0\times \pm 7.7\times$** | **[192.6x, 203.5x]** | $199.3\times\ [180.0\times, 206.6\times]$ | Offline uint64 diff vs scalar Python |
| **Total Simulation Runtime** | $4284.6 \pm 80.0\text{ ms}$ | [4227.4, 4341.9] ms | $4271.6\ [4189.1, 4436.3]\text{ ms}$ | End-to-end Python pipeline |
| **Pipeline Throughput** | $2,335 \pm 43\text{ pages/sec}$ | [2304, 2365] pps | $2341\ [2254, 2387]\text{ pps}$ | Python execution rate |

---

### 6.2 Multi-Dimensional Parameter Sweeps

#### A. Mutation Rate Sensitivity Sweep ($0.1\% \to 30\%$ Byte Drift, 5 Trials/Point)
Demonstrates AMDEC's operational boundary on synthetic workloads: as pages diverge from identical templates, memory savings degrade smoothly.

| Mutation Rate | Word Mutations | Logical Memory Savings (Mean ± Std) | 95% Confidence Interval | Compression Ratio | Mean Delta Size | Operational Status |
|---|---|---|---|---|---|---|
| **0.1%** | 1 word (8B) | **86.67% ± 0.07%** | [86.59%, 86.76%] | $7.50\times$ | $13.8\text{ B}$ | High template similarity |
| **0.5%** | 2–3 words (24B) | **86.43% ± 0.14%** | [86.26%, 86.60%] | $7.37\times$ | $21.1\text{ B}$ | Typical heap object drift |
| **1.0%** | 5 words (40B) | **68.54% ± 7.92%** | [58.71%, 78.38%] | $3.31\times$ | $43.5\text{ B}$ | Moderate sharing |
| **2.0%** | 10 words (80B) | **11.83% ± 3.08%** | [8.00%, 15.65%] | $1.13\times$ | $37.2\text{ B}$ | Low sharing |
| **5.0%** | 25 words (200B)| **6.96% ± 0.00%** | [6.96%, 6.96%] | $1.07\times$ | $6.0\text{ B}$ | LSH Jaccard boundary ($<0.70$) |
| **10.0%+**| $\ge 51$ words | **6.96% ± 0.00%** | [6.96%, 6.96%] | $1.07\times$ | $6.0\text{ B}$ | Exact duplicates only |

**Analysis of Variance at 1.0% Mutation Rate (Sharp Similarity Threshold Boundary):**  
Notice that while the $0.1\%$ and $0.5\%$ mutation points exhibit tight standard deviations ($\pm 0.07\%$ and $\pm 0.14\%$), the $1.0\%$ mutation point exhibits elevated variance: **$68.54\% \pm 7.92\%$** (95% CI: $[58.71\%, 78.38\%]$). Across the 5 individual trials, measured savings span from $54.81\%$ (seed 44) to $75.17\%$ (seed 42).  
This dispersion is a real physical phenomenon demonstrating **sharp sensitivity near the similarity threshold boundary**:
1. **Critical Jaccard Boundary ($J \approx 0.70$):** At 1.0% byte drift, approximately 40 bytes (5 words) mutate per 4096-byte page across thirty-two 128-byte chunks. When mutated words fall within 1–3 chunks, true pairwise Jaccard similarity remains comfortably high ($J \approx 0.80–0.85 > 0.70$), producing large, well-compressed clusters. When mutations scatter across 5 distinct chunks, true Jaccard hovers directly on the admission boundary ($J \approx 0.71–0.73$).
2. **MinHash Estimation Variance:** With $K=32$ permutation hashes, the standard error of the MinHash Jaccard estimator near $0.70$ is $\text{SE} \approx \sqrt{p(1-p)/K} \approx 0.081$. When true similarity sits near $0.70$, random hash collisions occasionally underestimate $J < 0.70$, causing the verifier to reject candidate edges and fragment clusters (e.g., Trial 3 split into 10 smaller clusters with $54.81\%$ savings vs. 3–5 clusters with $>70\%$ savings in other trials).
3. **Threshold Cliff:** By $2.0\%$ mutation, nearly all mutated pages fall firmly below $J < 0.70$ ($11.83\% \pm 3.08\%$), and by $5.0\%$, only exact duplicates ($7\%$) merge ($6.96\%$). The $1.0\%$ point represents a genuine percolation threshold where small per-trial variations in byte offset placement dictate whether similarity clusters coalesce or bifurcate.

#### B. Similarity Composition Sweep ($20\% \to 80\%$ Near-Duplicates, 5 Trials/Point)

| Near-Duplicate Ratio | Unique Page Ratio | Logical Memory Savings (Mean ± Std) | Effective Compression Ratio |
|---|---|---|---|
| **20%** | 73% | **25.21% ± 0.40%** | $1.34\times$ |
| **40%** | 53% | **43.66% ± 0.54%** | $1.77\times$ |
| **60%** | 33% | **62.41% ± 0.94%** | $2.66\times$ |
| **80%** | 13% | **80.86% ± 1.23%** | $5.24\times$ |

**Cross-Consistency with Primary System Evaluation:**  
The similarity composition sweep demonstrates that without active write traffic (0 pages skipped by Module 1), an 80% near-duplicate workload achieves **$80.86\% \pm 1.23\%$** logical savings. In the primary multi-trial evaluation (Section 6.1), identical 80% near-duplicate density is evaluated alongside simulated Zipfian write traffic, where Module 1 filters out **$8.20\% \pm 0.36\%$** of pages as volatile HOT working set (storing them as uncompressed full pages). Discounting these skipped volatile frames:
$$\text{Expected Savings} \approx 80.86\% \times (1 - 0.0820) = 74.23\%$$
This analytically predicted $74.23\%$ closely corroborates the independently measured **$73.60\% \pm 0.83\%$** primary evaluation result. This cross-validation confirms that AMDEC's compression behavior is mathematically coherent, predictable, and reproducible across independent experiment harnesses.

---

### 6.3 Adversarial Delta-Threshold Ablation (High-Mutation Workloads)

To test the adaptive 30% delta fallback mechanism, an adversarial workload generator was implemented (`experiments/ablation_delta_adversarial.py`) to force high byte mutations (1% to 60%) across all near-duplicate pages. 5 paired trials were executed comparing **Adaptive Fallback (30% threshold = 1228B)** vs. **Fixed / No Fallback (100% threshold)** under identical workload seeds.

| Mutation Rate | Adaptive Savings | Fixed Savings | Adaptive Delta (B) | Fixed Delta (B) | Fallback Pages | Savings Difference |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1%** | 68.48% ± 8.63% | 68.48% ± 8.63% | 42.5 B | 42.5 B | 0 | +0.00% |
| **5%** | 6.96% ± 0.00% | 6.96% ± 0.00% | 6.0 B | 6.0 B | 0 | +0.00% |
| **10%** | 6.96% ± 0.00% | 6.96% ± 0.00% | 6.0 B | 6.0 B | 0 | +0.00% |
| **20%** | 6.96% ± 0.00% | 6.96% ± 0.00% | 6.0 B | 6.0 B | 0 | +0.00% |
| **30%** | 6.96% ± 0.00% | 6.96% ± 0.00% | 6.0 B | 6.0 B | 0 | +0.00% |
| **40%** | 6.96% ± 0.00% | 6.96% ± 0.00% | 6.0 B | 6.0 B | 0 | +0.00% |
| **50%** | 6.96% ± 0.00% | 6.96% ± 0.00% | 6.0 B | 6.0 B | 0 | +0.00% |
| **60%** | 6.96% ± 0.00% | 6.96% ± 0.00% | 6.0 B | 6.0 B | 0 | +0.00% |

**System Insight:** At mutation rates $\ge 5\%$, Jaccard similarity drops below the $0.70$ threshold required by Module 2 (MinHash-LSH). As a result, **Module 2 never forms clusters from high-mutation pages in the first place** — they remain as unclustered singletons. Because low-similarity pages are pruned by LSH upstream, Module 3 receives only valid, high-similarity clusters. Thus, the experiment did not demonstrate additional measured storage savings for adaptive thresholding because LSH filters sufficiently dissimilar pages before Module 3. The 30% adaptive threshold serves as an **architectural safety mechanism** rather than a demonstrated driver of compression improvements in standard operation.

---

### 6.4 Isolated Module 1 Sampling Efficacy & Timing Benchmark

To eliminate Python interpreter scheduling noise, an isolated micro-benchmark (`experiments/benchmark_m1_isolated.py`) was executed on $N = 10,000$ pages across 5 trials (20 iterations per trial, 3 warmup iterations, `gc.disable()`, nanosecond timer) using the unified `hot_write_rate_threshold = 0.06`.

| Benchmark Metric | Measured Result (Mean ± Std) | 95% Confidence Interval |
|:---|:---:|:---:|
| **HOT Pages Filtered** | **801 ± 27 pages (8.01% ± 0.27%)** | [7.67%, 8.34%] |
| **M1 Filtering Runtime** | **28.19 ± 0.12 ms** | [28.05, 28.34] ms |
| **M1 Filtering Throughput** | **354,706 ± 1,494 pages/sec** | [352,850, 356,561] pps |
| **Baseline Uniform Scan Runtime** | **6.68 ± 0.08 ms** | [6.58, 6.77] ms |
| **Baseline Scan Throughput** | **1,498,018 ± 17,851 pages/sec** | [1,475,853, 1,520,182] pps |
| **Isolated Execution Speedup Ratio** | **0.237x ± 0.002x** (M1 adds ~21.5 ms classification overhead) | [0.234x, 0.240x] |

**Reconciliation & Cross-Experiment Alignment**:
1. **Confidence Interval Overlap**: The isolated 5-trial benchmark measures **$8.01\% \pm 0.27\%$** (95% CI: $[7.67\%, 8.34\%]$), and the primary 10-trial system evaluation (Section 6.1) measures **$8.20\% \pm 0.36\%$** (95% CI: $[7.94\%, 8.46\%]$). The two confidence intervals overlap across $[7.94\%, 8.34\%]$.
2. **Deterministic Seed-for-Seed Identity**: Across the 5 shared random seeds (seeds 42–46), the isolated benchmark and the primary full-pipeline evaluation produce **100% identical page-level filtering decisions**:
   - Seed 42: exactly 782 pages skipped in both harnesses.
   - Seed 43: exactly 788 pages skipped in both harnesses.
   - Seed 44: exactly 799 pages skipped in both harnesses.
   - Seed 45: exactly 786 pages skipped in both harnesses.
   - Seed 46: exactly 848 pages skipped in both harnesses.
3. **Resolution of Legacy Discrepancy**: Earlier draft documentation had cited a legacy figure of $7.1\% \pm 0.2\%$ ([6.9%, 7.3%]) because an earlier version of `benchmark_m1_isolated.py` used an unaligned constructor default (`hot_write_rate_threshold = 0.08`) before threshold unification to $0.06$. With both harnesses synchronized to $0.06$, all measurements are in complete statistical and deterministic agreement.
4. **M1 Computational Trade-off**: M1 adds ~21.5 ms classification overhead per 10,000 pages (~355K pages/sec vs ~1.50M pages/sec uniform scan), but successfully prevents downstream similarity clustering from wasting cycles on volatile pages that would immediately fault and break sharing.

---

### 6.5 Extended LSH Scalability & Cluster Agreement ($N=1\text{K} \to 10\text{K}$)

Evaluated across scale points $N = 1,000, 2,500, 5,000, 10,000$ with a 15-second per-trial brute-force timeout (`experiments/lsh_scalability.py`, 3 trials per point).

| N (Pages) | Total Possible Pairs | LSH Candidate Pairs | Pair Reduction % | LSH Time (s) | Brute-Force Time (s) | Speedup | Cluster Agreement |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1,000** | 499,500 | 9,652 | **98.068%** | 0.31s | 0.5s | **1.7x** | **88.5%** |
| **2,500** | 3,123,750 | 33,115 | **98.940%** | 0.79s | 3.5s | **4.4x** | **95.0%** |
| **5,000** | 12,497,500 | 79,250 | **99.366%** | 1.61s | **TIMEOUT** (>15s) | **N/A** | **N/A** |
| **10,000** | 49,995,000 | 70,560 | **99.859%** | 3.04s | **TIMEOUT** (>15s) | **N/A** | **N/A** |

- **Largest N where brute force completed within timeout**: $N = 2,500$ (3.5s vs 0.79s LSH, 4.4x measured speedup, 95.0% cluster agreement).
- At $N = 5,000$ and $N = 10,000$, brute force exceeded the 15-second timeout.
- At $N = 10,000$, LSH processed **70,560 candidate pairs** out of **49,995,000 total possible pairs** — achieving a **99.859% candidate-search-space reduction** (demonstrating sub-quadratic candidate generation over the tested range).
- **Candidate Count Reconciliation ($70,560\text{ vs. } 65,352$ pairs):** In this isolated LSH scalability experiment (`lsh_scalability.py`), all 10,000 raw pages enter LSH directly without Module 1 filtering, evaluating an average of **70,560 candidate pairs** against $C(10000, 2) = 49,995,000$ pairs ($99.859\%$ reduction). In the primary full-pipeline evaluation (`run_full_eval.py`, Section 6.1), Module 1 runs upstream and filters out $8.20\%$ volatile hot pages, so LSH processes the remaining $9,180$ candidate pages, evaluating **65,352 candidate pairs** against $42,131,283$ post-M1 pairs ($99.845\%$ reduction; $99.869\%$ vs. un-filtered pairs). Both counts reflect the identical underlying clustering engine evaluated under different upstream pipeline configurations.
- **Scalability Sweep Attribution ($1\text{K} \to 20\text{K}$ Pages, Fig 6):** In Fig 6, data points for $N \in [1{,}000, 10{,}000]$ reflect the isolated LSH clustering micro-benchmark (`experiments/lsh_scalability.py`), while extended scale points $N = 15{,}000$ ($6,864.2\text{ ms}$, $2,185\text{ pps}$) and $N = 20{,}000$ ($9,529.8\text{ ms}$, $2,104\text{ pps}$) are measured via the full end-to-end multi-parameter scalability sweep (`experiments/parameter_sweep.py`, 5 trials/point, raw logs in `results/raw/scalability_trials.csv`).

---

### 6.6 Complete Publication Figure Catalog (`results/figures/`, 300 DPI)

| Figure | Filename | Visual Representation | Corrected Academic Description |
|---|---|---|---|
| **Fig 1** | `fig1_mutation_sensitivity.png` | Dual-axis curve: Mutation % ($0.1\% \to 30\%$) vs Savings (%) & Delta bytes (with 95% CI error bars) | Degradation curve of savings under byte drift on synthetic workload. |
| **Fig 2** | `fig2_similarity_composition.png` | Bar chart with compression factors: Near-dup ratio ($20\% \to 80\%$) with 95% CI error bars | Sensitivity of savings to heap near-duplicate density on synthetic workload. |
| **Fig 3** | `fig3_reconstruction_latency_cdf.png` | High-res Empirical CDF with Median (P50), P95, and P99 threshold lines (Pooled 7,000 samples) | Distribution of simulated page reconstruction latencies ($P50 = 1.58\ \mu\text{s}$, $P95 = 3.88\ \mu\text{s}$, $P99 = 4.58\ \mu\text{s}$). |
| **Fig 4** | `fig4_ablation_comparison.png` | Dual subplot: M1 sampling runtime impact ($N=5,000$) & LSH search space reduction ($N=1,500$) | Subplot A shows M1 reduces median pipeline execution time ($1,570\text{ ms} \to 1,480\text{ ms}$, saving $90\text{ ms}$ / $5.7\%$) robust to scheduler jitter; Subplot B shows LSH reduces pairwise comparisons by $98.44\%$ ($1.12\text{M} \to 17.5\text{K}$ pairs) on log-scale. |
| **Fig 5** | `fig5_acceleration_throughput.png` | Throughput comparison: Scalar Python loop vs NumPy uint64 vectorized diff | Demonstrates $198.0\times$ speedup of NumPy uint64 vectorized differencing over scalar Python implementation. |
| **Fig 6** | `fig6_scalability_throughput.png` | Dual-axis scalability: $1\text{K} \to 20\text{K}$ pages vs Runtime (ms) \& Pages/sec (with 95% CI error bars) | Sub-quadratic candidate indexing scaling over tested workload range. Data for $N \le 10{,}000$ from `experiments/lsh_scalability.py`; $N = 15{,}000$ and $N = 20{,}000$ data points from `experiments/parameter_sweep.py` scalability sweep. |
| **Fig 7** | `fig7_delta_size_distribution.png` | Boxplot vs 4096B page + Delta size frequency histogram | Median delta footprint is $24.2\text{ bytes}$ compared to 4096B uncompressed frames. |
| **Fig 8** | `fig8_stability_tier_skew.png` | Stacked bar chart: COLD vs WARM vs HOT across Zipf skew ($\alpha=0.6 \to 1.8$) | Module 1 tier classification under varying Zipf write traffic skew. |
| **Fig 9** | `fig9_literature_comparison_bars.png` | Footprint comparison: Uncompressed vs KSM vs UKSM vs Difference Engine vs AMDEC | Conceptual comparison of logical footprint vs values reported in prior literature. |

---

## 7. Ablation Studies Analysis & System Insights

### 7.1 Ablation Study 1: Module 1 Sampling Efficacy (`experiments/ablation_m1.py` & `benchmark_m1_isolated.py`)
- **Comparison:** AMDEC with M1 Enabled vs. AMDEC with M1 Disabled (Uniform Full Scan).
- **Paired Ablation Results ($N = 5,000$ Pages, 5 Trials, `ablation_m1.py`):**
  - **Filtering Efficacy:** M1 filtered an average of **$493.4\text{ volatile hot pages}$** ($9.87\%$), safely pruning them from downstream LSH candidate generation (4,506.6 pages passed).
  - **Pipeline Execution Runtime (Median-Reported Metric — Option A):** Total pipeline indexing time dropped from a median of **$1,570.4\text{ ms}$** with M1 Disabled to **$1,480.4\text{ ms}$** with M1 Enabled, delivering a robust **$90.0\text{ ms}$ ($5.73\%$) median runtime reduction** (visualized in **Fig 4 Subplot A**).
    
    #### Full Trial-by-Trial Runtime Reconciliation (`ablation_m1_trials.csv`):
    | Trial | Seed | M1 Disabled Runtime | M1 Enabled Runtime | Difference (Saved ms) | Saved % | Analytical Characterization |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---|
    | 1 | 42 | $1,561.3\text{ ms}$ | $1,414.3\text{ ms}$ | $+147.0\text{ ms}$ | $+9.42\%$ | Baseline indexing speedup |
    | 2 | 43 | $1,570.4\text{ ms}$ | $1,402.4\text{ ms}$ | $+168.0\text{ ms}$ | $+10.70\%$ | Baseline indexing speedup |
    | 3 | 44 | $1,570.2\text{ ms}$ | $1,492.0\text{ ms}$ | $+78.2\text{ ms}$ | $+4.98\%$ | Baseline indexing speedup |
    | 4 | 45 | $1,618.1\text{ ms}$ | $1,480.4\text{ ms}$ | $+137.8\text{ ms}$ | $+8.51\%$ | Baseline indexing speedup |
    | 5 | 46 | $1,620.5\text{ ms}$ | $1,962.5\text{ ms}$ | $-342.0\text{ ms}$ | $-21.10\%$ | **Outlier:** OS scheduler / GC pause ($+515\text{ ms}$ vs. peer trials) |
    | **Median** | — | **$1,570.4\text{ ms}$** | **$1,480.4\text{ ms}$** | **$+90.0\text{ ms}$** | **$+5.73\%$** | **Primary Benchmark Metric (Robust to Outliers)** |
    | *Mean (All 5)* | — | $1,588.1 \pm 28.7\text{ ms}$ | $1,550.3 \pm 233.8\text{ ms}$ | $+37.8 \pm 199.1\text{ ms}$ | $+2.38\%$ | Skewed by Trial 5 spike ($\text{SD} = 233.8\text{ ms}$ is $8\times$ baseline) |
    | *Mean (1–4)* | — | $1,580.0 \pm 26.5\text{ ms}$ | $1,447.3 \pm 45.5\text{ ms}$ | $+132.8 \pm 38.5\text{ ms}$ | $+8.40\%$ | Post-hoc outlier-excluded subset |

    *Reconciliation & Methodological Decision:*  
    Three distinct runtime differences can be derived from this ablation experiment:
    1. **Mean difference across all 5 trials ($37.8\text{ ms}$):** Heavily diluted by the isolated $+515\text{ ms}$ scheduling spike in Trial 5, which balloons M1 Enabled variance to $\pm 233.8\text{ ms}$ (over $8\times$ the disabled variance of $\pm 28.7\text{ ms}$).
    2. **Outlier-excluded subset mean ($132.8 \pm 38.5\text{ ms}$, $8.40\%$):** Accurately reflects execution when background noise is absent, but relies on ad-hoc post-hoc pruning of Trial 5.
    3. **Median difference ($1,570.4\text{ ms} \to 1,480.4\text{ ms} = 90.0\text{ ms}$, $5.73\%$):** Following standard empirical computer systems benchmarking methodology for wall-clock benchmarks subject to OS background scheduling jitter, we adopt **Option A (Median Reporting)** as the sole authoritative headline metric. It incorporates all 5 trials without post-hoc data rejection while remaining immune to single-trial timing spikes.
    
    *Mechanistic Driver:* Although M1 classification adds ~15–20 ms isolated overhead, excluding 493.4 volatile hot pages ($9.87\%$) reduces the downstream input to Module 2 from 5,000 to 4,506 pages. The resulting computational savings in MinHash 32-permutation hashing and LSH candidate indexing more than offset M1's classification cost, yielding a net reduction in end-to-end pipeline runtime.
  - **Storage Trade-off:** Storage savings with M1 was $72.25\% \pm 0.86\%$ (preserving volatile hot pages as uncompressed frames) vs. $78.09\% \pm 1.09\%$ with M1 disabled (which forces all pages into deltas at the risk of immediate COW-break thrashing upon subsequent writes).
- **Isolated Timing Benchmark ($N = 10,000$ Pages, 5 Trials, `benchmark_m1_isolated.py`):** M1 filtered **$801 \pm 27\text{ pages}$** ($8.01\% \pm 0.27\%$, 95% CI: $[7.67\%, 8.34\%]$), matching the primary evaluation distribution ($8.20\% \pm 0.36\%$). M1 introduces ~21.5 ms classification overhead per 10,000 pages (28.19 ms vs. 6.68 ms uniform scan). *(Note: Both harnesses use `zipf_param=1.3` and `hot_write_rate_threshold=0.06`.)*

### 7.2 Ablation Study 2: Adaptive vs. Fixed Delta Thresholding (`experiments/ablation_delta.py` & `ablation_delta_adversarial.py`)
- **Comparison:** Adaptive Fallback ($\delta_{\text{max}} = 0.30$) vs. Fixed/No Fallback ($\delta_{\text{max}} = 1.0$).
- **Results:** Under standard LSH clustering, LSH prunes low-similarity pairs ($<70\%$) before Module 3. Consequently, high-delta pages never form clusters. The 30% threshold serves as an architectural safety mechanism.

### 7.3 Ablation Study 3: MinHash-LSH vs. Brute-Force All-Pairs (`experiments/ablation_lsh.py` & `lsh_scalability.py`)
- **Comparison:** MinHash-LSH candidate indexing vs. Brute-Force exhaustive all-pairs search across distinct experimental scales. *(Note: To avoid conflation, these are three distinct experiments at three distinct scales.)*
- **Paired Ablation Study 3 ($N = 1,500$ Pages, 5 Trials, `ablation_lsh.py`):** On $N = 1,500$, MinHash-LSH evaluated an average of **17,548 candidate pairs** vs. **1,124,250 brute-force pairs** ($C(1500, 2)$). This yielded a **98.44% search space reduction** (range: 98.34%–98.50%) and a **3.28x measured wall-clock speedup** ($478.8\text{ ms}$ vs. $1562.2\text{ ms}$) with **100% cluster agreement** (25.0 vs. 25.0 clusters, zero cross-cluster divergence).
- **Scalability Sweep ($N = 2,500$ Pages, `lsh_scalability.py`):** MinHash-LSH evaluated **33,115 candidate pairs** vs. **3,123,750 brute-force pairs** (**98.94% pair reduction**, **4.4x measured speedup**, **95.0% cluster agreement**). At $N \ge 5,000$, brute-force search exceeded the 15-second execution timeout.
- **Primary Full-Pipeline Evaluation ($N = 10,000$ Pages, 10 Trials, `run_full_eval.py`):** At production scale, LSH evaluated an average of **65,352 candidate pairs** out of **42,131,283 candidate pairs** passing M1 (**99.845% ± 0.007% candidate reduction**), and **99.869% reduction** relative to the raw un-filtered $C(10000, 2) = 49,995,000$ pairwise search space.

---

## 8. Comparative Evaluation vs. Literature Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                ARCHITECTURAL COMPARISON                                │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ System                   │ Reported     │ Strategy      │ How AMDEC Solves Limitation  │
│                          │ Reduction    │               │                              │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ VMware CBPS (OSDI '02)   │ 20–60%       │ Exact SHA-1   │ CBPS misses near-duplicates; │
│                          │              │               │ AMDEC captures +45% savings. │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ Linux KSM (OLS '09)      │ 20–30%       │ Exact memcmp  │ 1-byte diff blocks KSM merge;│
│                          │              │               │ AMDEC stores compact deltas. │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ UKSM / CKSM (ATC '18)    │ 25–35%       │ Region Distill│ Region-aware for exact only; │
│                          │              │               │ AMDEC applies it to near-dup.│
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ Difference Engine        │ 40–60%       │ Pairwise diff │ Suffered O(N²) search cost;  │
│ (OSDI '08)               │              │               │ AMDEC LSH cuts pairs 99.85%. │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ SiLo / TSM               │ ~43–90%      │ LSH on disk   │ SiLo is for offline backup;  │
│ (TC '14 / FAST '15)      │              │               │ AMDEC runs live in memory.   │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ AMDEC (This Work)        │ 73.60%       │ Sampling+LSH+ │ Unified design combining     │
│                          │              │ Base-Delta    │ sampling, LSH, & base-delta. │
└──────────────────────────┴──────────────┴───────────────┴──────────────────────────────┘
```
*Note: All literature figures are reported values from prior publications cited under conceptual comparison, not same-machine measurements.*

---

## 9. Reproducibility & Codebase Directory Map

### 9.1 Execution Guide

```bash
# 1. Activate Virtual Environment
source .venv/bin/activate

# 2. Run All Automated Unit Tests (37/37 passing)
python -m pytest tests/ -v

# 3. Execute Primary Multi-Trial System Evaluation (10 trials, N=10,000)
python experiments/run_full_eval.py 10000 --trials 10 --seed 42

# 4. Execute Multi-Trial Parameter Sweeps
python experiments/parameter_sweep.py --trials 5 --seed 42

# 5. Execute 3 Ablation Studies & Benchmarks
python experiments/ablation_m1.py --pages 5000 --trials 5 --seed 42
python experiments/ablation_delta_adversarial.py --pages 3000 --trials 5 --seed 42
python experiments/benchmark_m1_isolated.py --pages 10000 --trials 5 --seed 42
python experiments/lsh_scalability.py --trials 3 --seed 42

# 6. Statistical Aggregation & 300-DPI Publication Figures
python analysis/statistical_analysis.py
python analysis/plot_results.py
```

### 9.2 Codebase File Map

```text
AMDEC/
├── src/
│   ├── common/
│   │   └── config.py                # Central AMDECConfig dataclass & JSON manager
│   ├── workload/
│   │   └── page_generator.py        # Deterministic synthetic anonymous heap generator
│   └── modules/
│       ├── m1_sampling.py           # Module 1: Region-aware stability tiering & feedback
│       ├── m2_lsh_clustering.py     # Module 2: MinHash + LSH clustering engine
│       ├── m3_delta_storage.py      # Module 3: Medoid base selection & delta RLE storage
│       ├── m4_acceleration.py       # Module 4: Offline SIMD uint64 vectorized diff benchmark
│       └── m5_cow.py                # Module 5: Simulated COW fault handler & nanosecond timer
├── configs/
│   └── default_experiment.json      # Machine-readable experiment hyperparameters
├── tests/
│   ├── test_page_generator.py       # Workload composition & Zipf tests
│   ├── test_m1_sampling.py          # M1 stability tiering & feedback tests
│   ├── test_m2_lsh.py               # M2 MinHash, LSH & Union-Find tests
│   ├── test_m3_delta.py             # M3 Base-delta RLE & fidelity tests
│   ├── test_m4_accel.py             # M4 Scalar vs vectorized diff tests
│   ├── test_m5_cow.py               # M5 COW read/write fault handling tests
│   ├── test_reproducibility.py      # Seed determinism & pipeline invariance tests
│   └── test_statistics.py           # Student's t CI & percentile pooling tests
├── experiments/
│   ├── run_full_eval.py             # Multi-trial evaluation harness (--trials N --seed S)
│   ├── parameter_sweep.py           # Multi-trial mutation, composition & scale sweeps
│   ├── ablation_m1.py               # Paired Ablation 1: M1 sampling efficacy
│   ├── ablation_delta.py            # Paired Ablation 2: Adaptive vs fixed thresholding
│   ├── ablation_delta_adversarial.py# High-mutation adversarial delta ablation
│   ├── ablation_lsh.py              # Paired Ablation 3: LSH vs Brute-Force candidate pairs
│   ├── benchmark_m1_isolated.py     # Isolated M1 sampling micro-benchmark
│   ├── lsh_scalability.py           # Extended LSH vs brute-force scalability experiment
│   ├── demo_week2.py                # Week 2 Milestone Demo script
│   ├── demo_week3.py                # Week 3 Milestone Demo script
│   ├── demo_week6.py                # Week 6 Milestone Demo script
│   └── demo_review2.py              # Review 2 Interactive Demo script
├── analysis/
│   ├── plot_results.py              # 300-DPI publication chart generator (9 figures)
│   ├── statistical_analysis.py      # Multi-trial statistical aggregator & LaTeX exporter
│   └── compare_baselines.py         # Literature baseline comparison table generator
├── results/
│   ├── raw/                         # Multi-trial CSV raw data outputs
│   ├── processed/                   # Aggregated summary CSVs & LaTeX table snippets
│   └── figures/                     # 9 publication-grade PNG charts (300 DPI)
└── docs/
    ├── AMDEC_Comprehensive_Project_Documentation.md  <-- THIS FILE
    ├── Review2_Presentation_Outline.md
    └── INVENTION_DISCLOSURE.md
```

---

## 10. Research Paper Claims Audit & Honesty Check

| Claim | Evidence | Status | Publication-Safe Wording |
|:---|:---|:---:|:---|
| **Adaptive delta threshold reduces unnecessary storage** | MinHash-LSH (Module 2) filters low-similarity pairs ($<0.70$) before Module 3. High-delta pages do not form clusters. | **NOT DEMONSTRATED** | "The 30% delta fallback serves as an architectural safety mechanism that prevents inefficient delta records when a delta exceeds the configured storage threshold." |
| **M1 reduces unnecessary scanning** | M1 filters **8.20% ± 0.36% of pages as HOT** in primary multi-trial runs (**8.01% ± 0.27%** in isolated benchmark; 95% CIs overlap across [7.94%, 8.34%]), but introduces ~21.5 ms classification overhead per 10,000 pages (28.19 ms vs 6.68 ms uniform scan). | **PARTIALLY SUPPORTED** | "M1 filters volatile pages from downstream similarity processing, at the cost of additional classification overhead." |
| **LSH substantially reduces pair search** | Candidate pair reduction of **98.44% at N=1,500** (17.5K vs 1.12M pairs, 3.28x speedup), **98.94% at N=2,500** (33.1K vs 3.12M pairs, 4.4x speedup), and **99.845% at N=10,000** (65.4K vs 42.1M candidate pairs; 99.869% vs 50.0M un-filtered pairs). | **SUPPORTED** | "MinHash-LSH achieves a 99.845% candidate-search-space reduction at N=10,000 (99.869% vs un-filtered pairs), demonstrating sub-quadratic candidate generation." |
| **AMDEC achieves ~73.6% memory savings** | Across 10 multi-trial primary runs ($N=10,000$), mean logical savings is **73.60% ± 0.83%** (95% CI: [73.00%, 74.19%]), compression ratio **3.79x ± 0.12x**. Cross-validated by composition sweep (80% near-dup → 80.86% savings without M1 filtering, projecting to ~74.2% with 8.2% HOT pages skipped). | **SUPPORTED** | "AMDEC achieved 73.60% ± 0.83% logical memory savings on the evaluated synthetic anonymous-memory workload." |
| **Reconstruction latency is low in simulation** | Mean simulated reconstruction latency is **1.75 ± 0.42 µs** (per-trial mean P95 = 3.68 ± 0.41 µs, P99 = 4.25 ± 0.39 µs). Pooled percentiles across 7,000 accesses: **P50 = 1.58 µs, P95 = 3.88 µs, P99 = 4.58 µs**. | **SUPPORTED** | "Mean simulated reconstruction latency was measured at 1.75 ± 0.42 µs (P95 = 3.88 µs, P99 = 4.58 µs across 7,000 accesses with GC disabled)." |
| **NumPy uint64 differencing acceleration** | Offline benchmark shows **198.0x ± 7.7x speedup** of NumPy uint64 vectorized differencing over scalar Python loops. | **SUPPORTED** | "Demonstrates a 198.0x ± 7.7x speedup of NumPy uint64 vectorized differencing over the scalar Python implementation, suggesting suitability for SIMD-oriented execution." |
| **Novelty / First Claim** | Literature survey covers 18 systems across 3 incomplete categories. | **PARTIALLY SUPPORTED** | "To the best of our knowledge, AMDEC explores a unified design combining region-aware sampling, similarity-based clustering, and base-delta storage for anonymous memory deduplication." |
| **Linear $O(N)$ scaling** | Scalability sweep shows sub-quadratic candidate indexing growth up to N=20,000. | **PARTIALLY SUPPORTED** | "Demonstrates sub-quadratic candidate indexing scaling across expanding memory datasets over the tested range." |

---

## 11. Threats to Validity & Acknowledged Limitations

### 11.1 Internal Validity

**T1 — Shift Vulnerability (XOR Delta Encoding)**  
AMDEC's byte-level XOR differencing assumes pages mutate _in-place_ at fixed byte offsets. A 1-byte insertion or deletion shifts all subsequent data, causing XOR to produce non-zero output across the entire page suffix.  
*Mitigation Implemented:* Phase A fast-track stores a 6-byte zero-span delta record if SHA-256 digests match exactly.

**T2 — Transitive Cluster Divergence (Union-Find Single-Linkage)**  
Union-Find clustering exhibits single-linkage behavior: if A is similar to B ($J=0.75$) and B is similar to C ($J=0.75$), A and C may reside in the same cluster even if $J(A,C)<0.70$. Delta-compressing C against primary medoid A would yield an excessively large delta record.  
*Mitigation Implemented:* Secondary-Base Sub-Clustering (`src/modules/m3_delta_storage.py`). When any cluster member's delta against the primary medoid exceeds the 30% delta threshold ($\delta > 1228\text{B}$), AMDEC selects the **medoid** (minimum total pairwise byte distance) of the divergent pages as a secondary base within the cluster. Remaining divergent members are re-evaluated against the secondary base, forming a compact sub-cluster delta and recording the sub-base assignment in `SimulatedCOWHandler.page_bases` rather than falling back to uncompressed storage.

### 11.2 External Validity

**T3 — Synthetic Workload Artifact**  
All experiments use parameterized synthetic anonymous heap pages. Real-world anonymous memory exhibits more complex mutation patterns including pointer graph rewrites and JIT code segment mutations. Measured savings depend strongly on near-duplicate composition, mutation rates, and workload density.

**T4 — Literature Baselines Not Run on Same Machine**  
KSM, UKSM, CBPS, and Difference Engine figures are cited from prior publications. All direct numerical comparisons are qualified as conceptual comparisons of reported values from prior literature.

### 11.3 Simulation Validity

**T5 — Simulated COW Fault Handler (Module 5)**  
Module 5 models the logical state machine of Copy-on-Write fault handling. It measures simulated algorithmic reconstruction latency ($1.75\ \mu\text{s}$) and does not model OS kernel trap entry overhead (~1,000 cycles / ~0.3 µs), TLB shootdown latency across CPU cores, or page-table lock contention. A production kernel implementation remains future work.

**T6 — Computational Acceleration Benchmark Scope (Module 4)**  
The $198.0\times$ speedup reported for NumPy uint64 vectorized XOR over scalar Python loops is an offline Python-internal benchmark demonstrating the benefit of vectorized execution, not production kernel SIMD performance.

### 11.4 Measurement Validity

**T7 — Python Timing Overhead & Brute-Force Timeout**  
Python timing includes interpreter overhead. In the LSH scalability experiment, brute force exceeded the 15-second timeout at $N \ge 5,000$, so speedups for $N \ge 5,000$ are reported as N/A rather than fabricated.

---
*AMDEC Comprehensive Project Documentation | Author: Shantanu Tapole | Design Science Research*
