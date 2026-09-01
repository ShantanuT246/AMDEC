# AMDEC: Comprehensive Project Documentation & Technical Summary
### Adaptive Memory Deduplication via Locality-Sensitive Similarity Clustering and Base-Delta Encoding

> **Author / Principal Investigator:** Shantanu Tapole  
> **Methodology:** Design Science Research (DSR)  
> **Target Venue:** IEEE / ACM Systems Conference (e.g., IEEE COMPSAC, USENIX ATC Workshop Track)  
> **Simulation Testbed:** Python 3.x userspace simulation (NumPy, datasketch, scipy, pandas, matplotlib, pytest)

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
Modern operating systems, cloud hypervisors, and container runtimes waste substantial physical memory hosting redundant page contents. Standard deduplication engines (such as **Linux Kernel Samepage Merging / KSM** [CITATION REQUIRED] and **VMware Content-Based Page Sharing / CBPS** [CITATION REQUIRED]) operate under a **binary exactness constraint**: two 4096-byte memory pages can only be merged if they are **100% byte-for-byte identical**.

However, empirical systems research reveals that in **anonymous memory heaps** (dynamic allocations in web servers, databases, microservices, and language runtimes):
- Less than **$7\%$ of pages are exact duplicates**.
- Up to **$86\%$ of pages are near-duplicates** (sharing identical structural templates, object vtables, or text formats, but differing only in timestamps, counters, or pointer offsets).

Consequently, exact-match deduplicators discard over **$80\%$ of potential memory reduction opportunities**.

### 1.2 The AMDEC Solution
The **five-module AMDEC architecture** introduces a unified pipeline combining region-aware sampling, similarity-based clustering, and base-delta storage to capture near-duplicate anonymous memory pages without unguided $O(N^2)$ pairwise searching:

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
[Module 3: Base-Delta Storage] ────────► Medoid selection, XOR+RLE diffs, 30% fallback safety net
           │
           ▼
[Module 5: Simulated COW Handler] ─────► Sub-5 µs simulated reconstruction & dynamic feedback quarantine
```
*(Note: Module 4 provides an offline micro-benchmark evaluating vectorized uint64 bitwise differencing overhead).*

### 1.3 Central Research Hypothesis
> **Hypothesis:** When anonymous memory contains high proportions of near-duplicate pages, locality-sensitive similarity clustering combined with base-delta representation achieves substantial logical memory reduction ($>70\%$) beyond exact-match deduplicators, while region-aware stability sampling filters volatile pages to prevent Copy-on-Write thrashing, and low simulated reconstruction latencies ($<5\ \mu\text{s}$) demonstrate algorithmic efficiency.

---

## 2. Literature Survey & Gap Formalization

### 2.1 Complete Literature Matrix (18 Prior Works Analyzed)

| System / Paper | Year | Target Domain | Technique | Reported Savings | Exact Only? | Near-Dup? | Base-Delta? | Region-Aware? | Primary Architectural Limitation |
|---|---|---|---|---|---|---|---|---|---|
| **VMware CBPS** | 2002 | Hypervisor VMs | Periodic hash scan | Up to 60% (homogeneous) | ✅ | ❌ | ❌ | ❌ | Exact-match only; static binary sharing |
| **Linux KSM** | 2009 | Linux Kernel | Red-black tree scan | ~20–30% (workloads) | ✅ | ❌ | ❌ | ❌ | 1-byte difference blocks merge entirely |
| **Satori** | 2009 | Xen / Disk I/O | Load-time enlightened sharing | Exploits up to 94% max sharing | ✅ | ❌ | ❌ | ❌ | Restricted to disk-backed pages at I/O read time |
| **Difference Engine** | 2008 | Xen Hypervisor | Sub-page delta + compression | 40–60% footprint reduction | ❌ | ✅ | ✅ | ❌ | **Unguided $O(N^2)$ pairwise delta search** |
| **Page Overlays** | 2013 | HW-Assisted Cache | Fine-grained page merging | 15–35% footprint reduction | ❌ | ✅ | Partial | ❌ | Requires custom CPU hardware; no clustering |
| **BCD Deduplication** | 2017 | OS Sub-page | Byte-level content differencing | 20–40% sub-page reduction | ❌ | ✅ | ✅ | ❌ | No upstream clustering stage; high pairwise cost |
| **Sparrow** | 2024 | Android OS | Intra-page block sampling | Identifies 86.1% near-dup potential | ❌ | ✅ | Partial | ❌ | Demonstrates the problem; no general clustering framework |
| **De2Dup / dbDedup** | 2024 | In-Memory DBs | Intel DSA hardware delta | 30–50% reduction in DB heaps | ❌ | ✅ | ✅ | ❌ | Domain-specific to columnar DBs; requires Intel DSA |
| **UKSM** | 2018 | Linux Kernel | Hierarchical distilling | Up to 5x over CBPS | ✅ | ❌ | ❌ | ✅ | Region-awareness applied to **exact-match only** |
| **CKSM** | 2024 | Linux Containers | Physical-memory-space scan | 60–80% CPU scan reduction | ✅ | ❌ | ❌ | ✅ | Still fundamentally exact-match only |
| **PAMD** | 2024 | Dynamic Control | Process-aware loop joining | Workload-dependent ROI | ✅ | ❌ | ❌ | Partial | Process-level tuning; blind to content similarity |
| **SiLo** | 2011 | Backup Storage | Similarity + locality clustering | ~90% storage reduction | ❌ | ✅ | ❌ | ❌ | Storage/backup dedup; not live RAM |
| **I/O Deduplication** | 2010 | Storage Caching | Content-based disk caching | 28–47% I/O reduction | ❌ | ✅ | ❌ | ❌ | Block I/O domain; not OS virtual memory |
| **TSM / CISWH** | 2024 | Edge File Storage | Threshold similarity hashing | 43.2% edge storage reduction | ❌ | ✅ | ❌ | ❌ | Applied to files; no COW memory recovery |
| **Hardware DSA Engines**| 2024 | Accelerator Silicon | Hardware DMA offloaded diff | Offloads CPU during diff | ❌ | ✅ | ✅ | ❌ | Hardware-dependent; isolated to DB engines |
| **Garg et al.** | 2020 | GPU Computing | GPU-parallel similarity scan | 5x–10x matching speedup | ❌ | ✅ | ❌ | ❌ | Requires GPU; no base-delta storage pipeline |
| **CMDP** | 2022 | Cloud Hypervisors | Banked memory domain isolation | 25–35% search reduction | ✅ | ❌ | ❌ | ✅ | Domain partitioning; exact-match only |
| **ORC** | 2023 | CHERI Architecture | Capability object sharing | Lowers hypervisor tail-latency | ✅ | ❌ | ❌ | ❌ | Requires CHERI hardware; exact sharing only |
| **AMDEC (Ours)** | **2026** | **Anonymous Heaps** | **Sampling + LSH + Base-Delta** | **73.60% logical savings** | ❌ | ✅ | ✅ | ✅ | **Unified design combining sampling, LSH, and base-delta** |

*Note: Baseline savings figures are reported values from prior literature cited under conceptual comparison, not same-machine measurements.*

### 2.2 The Identified Research Gap
Across the literature, existing systems fall into three incomplete categories:
1. **Exact-Match Systems (KSM, UKSM, CKSM, CBPS):** Low retrieval latency, but misses up to $86\%$ of near-duplicates in anonymous memory.
2. **Sub-Page Delta Systems (Difference Engine, BCD):** Capable of delta storage, but lacks upstream similarity clustering $\implies$ suffers from an unguided $O(N^2)$ pairwise comparison bottleneck.
3. **Similarity Storage Systems (SiLo, TSM):** Uses LSH for clustering, but restricted to high-latency offline disk/backup storage without low-latency memory recovery.

To the best of our knowledge, **AMDEC explores a unified design** combining region-aware sampling, MinHash-LSH similarity clustering, and base-delta encoding for anonymous memory deduplication.

---

## 3. System Architecture & Five-Module Pipeline

The **five-module AMDEC architecture** is organized into a primary storage pipeline (Modules 1–3), an acceleration benchmark (Module 4), and a simulated Copy-on-Write reconstruction handler (Module 5):

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
                                           │ Candidate Pages (COLD / WARM)
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
│ Module 4: Acceleration Benchmark          │ │ Module 5: Simulated COW Handler           │
│ - NumPy uint64 Vectorized Bitwise Diff    │ │ - Page Table State Machine                │
│ - Micro-benchmark vs Scalar Python        │ │ - Sub-5 µs Simulated Read Reconstruction  │
└───────────────────────────────────────────┘ └───────────────────────────────────────────┘
```

---

## 4. Module-by-Module Technical Specification

### 4.1 Workload Generator (`src/workload/page_generator.py`)
- **Deterministic RNG Encapsulation:** Uses instance-isolated `self.rng = random.Random(seed)` and `self.np_rng = np.random.default_rng(seed)` (no `os.urandom()` calls) to guarantee $100\%$ bitwise deterministic page generation across runs.
- **Realistic Heap Models:** Struct arrays (64B C++ structs with vtable pointers and timestamps), pointer tables (512 8B pointers), and semi-sparse JSON buffers.
- **Zipfian Write Simulation:** Simulates non-uniform write traffic using a Zipfian distribution ($\alpha = 1.3$).

### 4.2 Module 1: Region-Aware Sampling & Feedback Engine (`src/modules/m1_sampling.py`)
- **Stability Tiering:** Classifies pages into `HOT` (high write frequency / recent COW break), `WARM` (moderate activity), and `COLD` (long-lived, low write frequency).
- **Filtering & Priority Queue:** Queues `COLD` and `WARM` pages into a min-heap sorted by priority score. Filters out `HOT` pages to prevent downstream LSH pair generation and COW thrashing.
- **Dynamic COW Feedback:** Intercepts Copy-on-Write write faults and temporarily quarantines faulting pages for $30\text{ seconds}$.

### 4.3 Module 2: MinHash-LSH Clustering Engine (`src/modules/m2_lsh_clustering.py`)
- **Page Chunking:** Splits 4096-byte pages into 32 128-byte non-overlapping chunks.
- **MinHash Fingerprinting:** Computes $K=32$ 64-bit MinHash signature permutations per page.
- **LSH Banding Index:** Hashes signatures into $b=8$ bands ($r=4$ rows per band) to identify candidate similar pairs.
- **Large-Bucket Exact-Hash Grouping:** In massive buckets ($>64$ pages), pages are pre-grouped by SHA-256 exact digest to prevent component fragmentation while spanning representatives maintain candidate pair generation.
- **Verification & Union-Find:** Pairs meeting $J \ge 0.70$ are verified and merged using a disjoint-set data structure with path compression and rank optimization.

### 4.4 Module 3: Base-Delta Storage Manager (`src/modules/m3_delta_storage.py`)
- **Phase A Exact-Hash Fast-Track:** If a candidate page is byte-identical to the base, it produces a **6-byte zero-span delta record** (4B base ref + 2B span count) with no XOR computation overhead.
- **Medoid Selection:** Computes pairwise byte distances across cluster members; picks base page minimizing $\sum_{p \in C} \text{dist}(\text{base}, p)$.
- **Vectorized XOR Diff + RLE Spans:** Non-zero byte runs encoded as $[(\text{offset}: \text{uint16}, \text{length}: \text{uint16}, \text{data}: \text{bytes})]$.
- **Strict Metadata-Aware Accounting:**
  $$\text{Delta Record Size} = 4\text{B (Base Ref)} + 2\text{B (Span Count)} + \sum (4\text{B} + \text{len}(\text{span data}))$$
- **Global Base Fingerprint Registry:** Deduplicates identical base page contents across separate clusters (`sha256(base) -> canonical_cluster_id`) so shared templates are only counted once in storage accounting.
- **30% Fallback Safety Net:**
  $$\text{if } \text{Size}(\text{Delta Record}) > 0.30 \times 4096\text{B} (1228\text{B}) \implies \text{Store as full unmerged page}$$

### 4.5 Module 4: Computational Acceleration Benchmark (`src/modules/m4_acceleration.py`)
- Compares pure Python scalar byte loops vs. **NumPy uint64 bitwise XOR** differencing.
- Demonstrates an offline **$198.0\times \pm 7.7\times$ speedup** of NumPy uint64 vectorization over scalar Python loops ($>3,100\text{ MB/s}$ throughput), showing the suitability of vectorized execution for delta generation.

### 4.6 Module 5: Simulated COW Fault Handler (`src/modules/m5_cow.py`)
- **Page Table States:** `FULL_PAGE` (uncompressed), `BASE_PAGE` (shared read-only base), `DELTA_PAGE` (compressed delta).
- **Nanosecond Latency Tracking:** Uses `time.perf_counter_ns()` with GC disabled during benchmarking for microsecond precision.
- **Read Path:** Reconstructs $\hat{P} = P_{\text{base}} \oplus \text{Apply}(\text{Delta})$ with $100\%$ byte fidelity.
- **COW Write Path:** Intercepts write $\implies$ reconstructs original content, allocates private `FULL_PAGE`, applies write payload, updates page table, and notifies Module 1 feedback engine.

---

## 5. Experimental Methodology

### 5.1 Primary Metric Definitions
- **Baseline Footprint:** Total uncompressed raw memory ($N \times 4096\text{ bytes}$).
- **AMDEC Logical Footprint:** Sum of base page frames, compressed delta records (including metadata headers), unmerged fallback pages, and unique unclustered pages.
- **Logical Memory Savings (%):**
  $$\text{Savings} = \left(1 - \frac{\text{AMDEC Logical Footprint}}{\text{Baseline Footprint}}\right) \times 100\%$$
- **Effective Compression Ratio:**
  $$\text{Compression Ratio} = \frac{\text{Baseline Footprint}}{\text{AMDEC Logical Footprint}}$$

### 5.2 Testbed & Software Environment
- **Operating System:** macOS (Darwin arm64) [TO BE VERIFIED FOR KERNEL BUILD]
- **CPU / Memory Hardware:** Apple Silicon / Host Workstation Architecture [TO BE VERIFIED]
- **Python Environment:** Python 3.9.6
- **Core Libraries:** NumPy (v1.26+), datasketch (v1.5+), SciPy (v1.11+), Pandas, Matplotlib, pytest

### 5.3 Workload & Hyperparameter Configuration
- **Primary Workload ($N=10,000$):** 4096-byte pages, 80% near-duplicates (mutation rates 0.2%–1.2%), 7% exact duplicates, 13% unique pages. Zipf write skew ($\alpha = 1.3$).
- **Module 2 LSH:** $K=32$ MinHash permutations, $b=8$ bands, $r=4$ rows/band, Jaccard similarity threshold $J = 0.70$.
- **Module 3 Delta:** Adaptive threshold $\delta_{\text{max}} = 0.30$ (1228B limit), medoid base selection.
- **Random Seeds & Trials:** Seeds 42–51 across 10 independent trials for primary evaluation; 5 trials per point for parameter sweeps and paired ablations.

### 5.4 Evaluated Baselines
1. **Exhaustive Brute-Force Pairwise Search:** All-pairs $O(N^2)$ Jaccard comparison (executed up to $N=2,500$; timed out at $N \ge 5,000$).
2. **Uniform M1-Disabled Scan:** Full pass-through scan of all pages without stability tiering.
3. **Fixed / No Fallback Delta Threshold:** Delta manager with 100% threshold (no unmerged page fallback).
4. **Prior Literature Baselines:** Reported values from published papers (VMware CBPS, Linux KSM, UKSM, Difference Engine) included for conceptual comparison.

---

## 6. Experimental Results & Multi-Trial Benchmarks

### 6.1 Primary Multi-Trial System Evaluation ($N = 10,000$ Pages, 10 Independent Trials)

Evaluated across 10 randomized trials (seeds 42–51) on $N = 10,000$ pages ($39.06\text{ MB}$ uncompressed). All values are reported as **Mean $\pm$ Sample Std.Dev** alongside **95% Student's t Confidence Intervals (CI)**.

| Metric | Measured Result (Mean ± Std) | 95% Confidence Interval | Median [Min, Max] | Significance / Notes |
|---|---|---|---|---|
| **Baseline Uncompressed Footprint** | $39.06\text{ MB}$ (Fixed) | [Fixed Baseline] | $39.06\text{ MB}$ | Raw unshared allocation |
| **AMDEC Logical Footprint** | **$10.31 \pm 0.32\text{ MB}$** | **[10.08, 10.55] MB** | $10.30\ [9.90, 10.85]$ | Base pages + deltas + metadata |
| **Logical Memory Savings** | **$73.60\% \pm 0.83\%$** | **[73.00%, 74.19%]** | **73.64% [72.21%, 74.67%]** | **$3.79\times \pm 0.12\times$ effective compression** |
| **M1 Volatile Scan Reduction** | **$8.20\% \pm 0.36\%$** | **[7.94%, 8.46%]** | $8.07\%\ [7.82\%, 8.75\%]$ | Skips volatile HOT working set |
| **M2 LSH Candidate Reduction** | **$99.845\% \pm 0.007\%$** | **[99.840%, 99.850%]** | $99.846\%\ [99.833\%, 99.854\%]$ | Evaluated ~70.5K pairs vs 50M brute-force |
| **Similarity Clusters Formed** | $147.4 \pm 9.2\text{ clusters}$ | [140.8, 154.0] | $148.5\ [127, 161]$ | Near-duplicate groupings |
| **Mean Delta Storage Footprint** | **$24.7 \pm 2.2\text{ bytes / page}$** | **[23.1, 26.2] B** | $24.2\ [22.3, 29.1]\text{ B}$ | Including record headers & offset metadata |
| **Mean Simulated Reconstruction Latency** | **$1.75 \pm 0.42\ \mu\text{s}$** | **[1.45, 2.05] $\mu\text{s}$** | $1.59\ [1.39, 2.78]\ \mu\text{s}$ | Low on-demand retrieval penalty |
| **Pooled Reconstruction Latency (P50/P95/P99)** | **$1.58\ \mu\text{s} / 3.88\ \mu\text{s} / 4.58\ \mu\text{s}$** | Aggregate distribution | Across 7,000 access samples | Bounded tail latency ($<10\ \mu\text{s}$) |
| **NumPy Vectorization Speedup** | **$198.0\times \pm 7.7\times$** | **[192.6x, 203.5x]** | $199.3\times\ [180.0\times, 206.6\times]$ | uint64 vectorized diff vs scalar Python |
| **Total Simulation Runtime** | $4284.6 \pm 80.0\text{ ms}$ | [4227.4, 4341.9] ms | $4271.6\ [4189.1, 4436.3]\text{ ms}$ | End-to-end Python pipeline |
| **Pipeline Throughput** | $2,335 \pm 43\text{ pages/sec}$ | [2304, 2365] pps | $2341\ [2254, 2387]\text{ pps}$ | Sustained processing rate |

---

### 6.2 Multi-Dimensional Parameter Sweeps

#### A. Mutation Rate Sensitivity Sweep ($0.1\% \to 30\%$ Byte Drift, 5 Trials/Point)
Demonstrates AMDEC's operational boundary on synthetic workloads: as pages diverge from identical templates, memory savings degrade smoothly.

| Mutation Rate | Word Mutations | Logical Memory Savings (Mean ± Std) | 95% Confidence Interval | Compression Ratio | Mean Delta Size | Operational Status |
|---|---|---|---|---|---|---|
| **0.1%** | 1 word (8B) | **86.67% ± 0.07%** | [86.59%, 86.76%] | $7.50\times$ | $18.2\text{ B}$ | Optimal sharing |
| **0.5%** | 2–3 words (24B) | **86.43% ± 0.14%** | [86.26%, 86.60%] | $7.37\times$ | $24.0\text{ B}$ | Typical heap object drift |
| **1.0%** | 5 words (40B) | **68.54% ± 7.92%** | [58.71%, 78.38%] | $3.18\times$ | $38.4\text{ B}$ | High sharing |
| **2.0%** | 10 words (80B) | **11.83% ± 3.08%** | [8.00%, 15.65%] | $1.13\times$ | $68.1\text{ B}$ | Selective sharing |
| **5.0%** | 25 words (200B)| **6.96% ± 0.00%** | [6.96%, 6.96%] | $1.07\times$ | — | LSH Jaccard boundary ($<0.70$) |
| **10.0%+**| $\ge 51$ words | **6.96% ± 0.00%** | [6.96%, 6.96%] | $1.07\times$ | — | Exact duplicates only |

#### B. Similarity Composition Sweep ($20\% \to 80\%$ Near-Duplicates, 5 Trials/Point)

| Near-Duplicate Ratio | Unique Page Ratio | Logical Memory Savings (Mean ± Std) | Effective Compression Ratio |
|---|---|---|---|
| **20%** | 73% | **25.21% ± 0.40%** | $1.34\times$ |
| **40%** | 53% | **43.66% ± 0.54%** | $1.77\times$ |
| **60%** | 33% | **62.41% ± 0.94%** | $2.66\times$ |
| **80%** | 13% | **80.86% ± 1.23%** | $5.24\times$ |

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

**System Insight:** At mutation rates $\ge 5\%$, Jaccard similarity drops below the $0.70$ threshold required by Module 2 (MinHash-LSH). As a result, **Module 2 never forms clusters from high-mutation pages in the first place** — they remain as unclustered singletons. Because low-similarity pages are pruned by LSH upstream, Module 3 receives only valid, high-similarity clusters. Thus, the experiment did not demonstrate additional measured storage savings for adaptive thresholding because LSH filters sufficiently dissimilar pages before Module 3. The 30% adaptive threshold serves as an **architectural safety mechanism** rather than an active driver of measured compression in standard operation.

---

### 6.4 Isolated Module 1 Sampling Efficacy & Timing Benchmark

To eliminate Python interpreter scheduling noise, an isolated micro-benchmark (`experiments/benchmark_m1_isolated.py`) was executed on $N = 10,000$ pages across 5 trials (20 iterations per trial, 3 warmup iterations, `gc.disable()`, nanosecond timer).

| Benchmark Metric | Measured Result (Mean ± Std) | 95% Confidence Interval |
|:---|:---:|:---:|
| **HOT Pages Filtered** | 710 ± 15 pages (**7.1% ± 0.2%**) | [6.9%, 7.3%] |
| **M1 Filtering Runtime** | **32.77 ± 1.94 ms** | [31.07, 34.47] ms |
| **M1 Filtering Throughput** | **306,006 pages/sec** | [290,388, 321,624] pps |
| **Baseline Uniform Scan Runtime** | **7.33 ± 0.60 ms** | [6.80, 7.86] ms |
| **Baseline Scan Throughput** | **1,371,909 pages/sec** | [1,279,150, 1,464,668] pps |
| **Isolated Execution Speedup Ratio** | **0.224x ± 0.016x** (M1 is computationally heavier) | [0.210x, 0.238x] |

**Clarification of M1 Efficacy**:
1. **Pages Filtered**: M1 filters **7.1% of pages as HOT** (710 volatile pages) from downstream similarity processing.
2. **M1 Computational Cost**: M1 introduces an additional classification overhead of ~25 ms per full scan (~306K pages/sec vs ~1.37M pages/sec uniform scan).
3. **Pipeline Function**: M1's purpose is **reducing downstream candidate generation, avoiding processing of volatile pages, and supporting COW-feedback quarantine**, rather than making the initial scan pass faster.

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
- At $N = 10,000$, LSH processed **70,560 candidate pairs** out of **49,995,000 total possible pairs** — achieving a **99.859% candidate-search-space reduction** (demonstrating sub-quadratic candidate generation under the evaluated workload).

---

### 6.6 Complete Publication Figure Catalog (`results/figures/`, 300 DPI)

| Figure | Filename | Visual Representation | Corrected Academic Description |
|---|---|---|---|
| **Fig 1** | `fig1_mutation_sensitivity.png` | Dual-axis curve: Mutation % ($0.1\% \to 30\%$) vs Savings (%) & Delta bytes (with 95% CI error bars) | Degradation curve of savings under byte drift on synthetic workload. |
| **Fig 2** | `fig2_similarity_composition.png` | Bar chart with compression factors: Near-dup ratio ($20\% \to 80\%$) with 95% CI error bars | Demonstrates sensitivity of savings to heap near-duplicate density. |
| **Fig 3** | `fig3_reconstruction_latency_cdf.png` | High-res Empirical CDF with Median (P50), P95, and P99 threshold lines (Pooled 7,000 samples) | Distribution of simulated page reconstruction latencies ($P50 = 1.58\ \mu\text{s}$, $P95 = 3.88\ \mu\text{s}$, $P99 = 4.58\ \mu\text{s}$). |
| **Fig 4** | `fig4_ablation_comparison.png` | Dual subplot: M1 candidate filtering & LSH search space reduction | Shows M1 filters 7.1% volatile pages and LSH skips 98.1%–99.8% of candidate pairs. |
| **Fig 5** | `fig5_acceleration_throughput.png` | Throughput comparison: Scalar Python loop vs NumPy uint64 vectorized diff | Demonstrates $198.0\times$ speedup of NumPy uint64 vectorization over scalar Python implementation. |
| **Fig 6** | `fig6_scalability_throughput.png` | Dual-axis scalability: $1\text{K} \to 20\text{K}$ pages vs Runtime (ms) & Pages/sec (with 95% CI error bars) | Sub-quadratic candidate indexing scaling across expanding memory datasets. |
| **Fig 7** | `fig7_delta_size_distribution.png` | Boxplot vs 4096B page + Delta size frequency histogram | Median delta footprint is $24.2\text{ bytes}$ compared to 4096B uncompressed frames. |
| **Fig 8** | `fig8_stability_tier_skew.png` | Stacked bar chart: COLD vs WARM vs HOT across Zipf skew ($\alpha=0.6 \to 1.8$) | Module 1 tier classification under varying Zipf write traffic skew. |
| **Fig 9** | `fig9_literature_comparison_bars.png` | Footprint comparison: Uncompressed vs KSM vs UKSM vs Difference Engine vs AMDEC | Conceptual comparison of logical footprint vs values reported in prior literature. |

---

## 7. Ablation Studies Analysis & System Insights

### 7.1 Ablation Study 1: Module 1 Sampling Efficacy (`experiments/ablation_m1.py` & `benchmark_m1_isolated.py`)
- **Comparison:** AMDEC with M1 Enabled vs. AMDEC with M1 Disabled (Uniform Full Scan).
- **Results:** M1 filtered **$710\text{ volatile hot pages}$** ($7.1\%$), excluding them from downstream LSH pair generation and preventing immediate COW-break thrashing. M1 introduces ~25ms classification overhead per full scan.

### 7.2 Ablation Study 2: Adaptive vs. Fixed Delta Thresholding (`experiments/ablation_delta.py` & `ablation_delta_adversarial.py`)
- **Comparison:** Adaptive Fallback ($\delta_{\text{max}} = 0.30$) vs. Fixed/No Fallback ($\delta_{\text{max}} = 1.0$).
- **Results:** Under standard LSH clustering, LSH prunes low-similarity pairs ($<70\%$) before Module 3. Consequently, high-delta pages never form clusters. The 30% threshold serves as an architectural safety mechanism.

### 7.3 Ablation Study 3: MinHash-LSH vs. Brute-Force All-Pairs (`experiments/ablation_lsh.py` & `lsh_scalability.py`)
- **Comparison:** MinHash-LSH candidate indexing vs. Brute-Force exhaustive all-pairs search.
- **Results:** On $N = 2,500$, MinHash-LSH evaluated **33,115 candidate pairs** vs. **3,123,750 brute-force pairs** (**98.94% pair reduction**, **4.4x measured speedup**, **95.0% cluster agreement**). At $N = 10,000$, LSH achieved a **99.859% candidate-search-space reduction**.

---

## 8. Comparative Evaluation vs. Literature Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                ARCHITECTURAL COMPARISON                                │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ System                   │ Reported     │ Strategy      │ How AMDEC Solves Limitation  │
│                          │ Reduction    │               │                              │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ VMware CBPS (2002)       │ 20–60%       │ Exact SHA-1   │ CBPS misses near-duplicates; │
│                          │              │               │ AMDEC captures +45% savings. │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ Linux KSM (2009)         │ 20–30%       │ Exact memcmp  │ 1-byte diff blocks KSM merge;│
│                          │              │               │ AMDEC stores compact deltas. │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ UKSM (2018) / CKSM (2024)│ 25–35%       │ Region Distill│ Region-aware for exact only; │
│                          │              │               │ AMDEC applies it to near-dup.│
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ Difference Engine (2008) │ 40–60%       │ Pairwise diff │ Suffered O(N²) search cost;  │
│                          │              │               │ AMDEC LSH cuts pairs 99.85%. │
├──────────────────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ SiLo (2011) / TSM (2024) │ ~43–90%      │ LSH on disk   │ SiLo is for offline backup;  │
│                          │              │               │ AMDEC runs live in memory.   │
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

# 2. Run All Automated Unit Tests (36/36 passing)
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
│       ├── m4_acceleration.py       # Module 4: SIMD uint64 vectorized diff benchmark
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
│   ├── benchmark_m1_isolated.py     # Isolated M1 sampling micro-benchmark
│   └── lsh_scalability.py           # Extended LSH vs brute-force scalability experiment
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

| Claim | Status | Evidence | Safe Paper Wording |
|:---|:---:|:---|:---|
| **Adaptive delta threshold reduces unnecessary storage** | **NOT DEMONSTRATED** | MinHash-LSH (Module 2) filters low-similarity pairs ($<0.70$) before Module 3. High-delta pages do not form clusters. | "The 30% delta fallback serves as an architectural safety mechanism that prevents inefficient delta records when a delta exceeds the configured storage threshold." |
| **M1 reduces unnecessary scanning** | **PARTIALLY SUPPORTED** | M1 filters **7.1% of pages as HOT** from downstream LSH clustering, but M1 classification adds ~25ms overhead (32.77 ms vs 7.33 ms uniform scan). | "M1 filters volatile pages from downstream similarity processing, at the cost of additional classification overhead." |
| **LSH substantially reduces pair search** | **SUPPORTED** | Candidate pair reduction of **98.07% at N=1,000** to **99.859% at N=10,000** (70.5K vs 50M pairs), 4.4x measured speedup at N=2,500. | "MinHash-LSH achieves a 99.859% candidate-search-space reduction at N=10,000, demonstrating sub-quadratic candidate generation." |
| **AMDEC achieves ~73.6% memory savings** | **SUPPORTED** | Across 10 multi-trial primary runs ($N=10,000$), mean logical savings is **73.60% ± 0.83%** (95% CI: [73.00%, 74.19%]), compression ratio **3.79x ± 0.12x**. | "AMDEC achieved 73.60% ± 0.83% logical memory savings on the evaluated synthetic anonymous-memory workload." |
| **Reconstruction latency is low in simulation** | **SUPPORTED** | Mean simulated reconstruction latency is **1.75 ± 0.42 µs**. Pooled percentiles across 7,000 accesses: **P50 = 1.58 µs, P95 = 3.88 µs, P99 = 4.58 µs**. | "Mean simulated reconstruction latency was measured at 1.75 ± 0.42 µs (P95 = 3.88 µs)." |
| **NumPy uint64 differencing acceleration** | **SUPPORTED** | Offline benchmark shows **198.0x ± 7.7x speedup** of NumPy uint64 vectorized differencing over scalar Python loops. | "Demonstrates a 198.0x ± 7.7x speedup of NumPy uint64 vectorized differencing over the scalar Python implementation, suggesting suitability for SIMD-oriented execution." |
| **Novelty / First Claim** | **PARTIALLY SUPPORTED** | Literature survey covers 18 systems across 3 incomplete categories. | "To the best of our knowledge, AMDEC explores a unified design combining region-aware sampling, similarity-based clustering, and base-delta storage for anonymous memory deduplication." |
| **Linear $O(N)$ scaling** | **PARTIALLY SUPPORTED** | Scalability sweep shows linear growth in candidate indexing runtime up to N=20,000. | "Demonstrates sub-quadratic candidate indexing scaling across expanding memory datasets." |

---

## 11. Threats to Validity & Acknowledged Limitations

### 11.1 Internal Validity

**T1 — Shift Vulnerability (XOR Delta Encoding)**  
AMDEC's byte-level XOR differencing assumes pages mutate _in-place_ at fixed byte offsets. A 1-byte insertion or deletion shifts all subsequent data, causing XOR to produce non-zero output across the entire page suffix.  
*Mitigation Implemented:* Phase A fast-track stores a 6-byte zero-span delta record if SHA-256 digests match exactly.

**T2 — Transitive Cluster Divergence (Union-Find Single-Linkage)**  
Union-Find clustering exhibits single-linkage behavior: if A is similar to B ($J=0.75$) and B is similar to C ($J=0.75$), A and C may reside in the same cluster with $J(A,C)=0.50$.  
*Mitigation Implemented:* Secondary-base fallback optimizes delta assignment.

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
