# AMDEC
### Adaptive Memory Deduplication using Similarity Clustering

[![Build](https://img.shields.io/badge/build-CMake-blue)]()
[![Language](https://img.shields.io/badge/core-C%2B%2B17-orange)]()
[![Analysis](https://img.shields.io/badge/analysis-Python-yellow)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**AMDEC** is a userspace simulation of an adaptive memory deduplication engine that replaces rigid byte-for-byte exact matching (as used in Linux KSM) with **similarity-based clustering** and **base-delta compression**. It targets the large fraction of anonymous memory pages in modern workloads — cloud, container, and mobile — that are *near-duplicates* rather than exact copies, and which existing deduplication engines discard entirely.

This repository contains the full research artifact: simulation engine, evaluation suite, and result analysis pipeline, developed as part of a research paper (and optional patent filing) submission.

---

## Table of Contents

- [Problem & Motivation](#problem--motivation)
- [Core Idea](#core-idea)
- [System Architecture](#system-architecture)
- [Module Reference](#module-reference)
- [Project Structure](#project-structure)
- [Language & Tooling](#language--tooling)
- [Getting Started](#getting-started)
- [Running Experiments](#running-experiments)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Roadmap & Phases](#project-roadmap--phases)
- [Review Checkpoints](#review-checkpoints)
- [Risk Register](#risk-register)
- [Documentation](#documentation)
- [Citing This Work](#citing-this-work)
- [License](#license)

---

## Problem & Motivation

Modern operating systems reduce RAM usage via memory deduplication — most notably Linux's **Kernel Same-page Merging (KSM)** — by finding memory pages that are byte-for-byte identical and merging them into a single shared, Copy-on-Write (COW) physical page.

This works well for static, read-only content (shared libraries, disk-backed caches). It fails for **anonymous dynamic memory** — application heaps, stack allocations, runtime state — which dominates memory usage in cloud, container, and mobile workloads. A single differing byte (a timestamp, a pointer, a counter) is enough to block an exact-match engine from merging two otherwise near-identical pages.

Empirically: **over 50% of memory in modern systems is anonymous**, **fewer than 7% of anonymous pages are byte-identical**, yet **up to 86% share 30–90% of their content** with other pages in memory. Exact-match deduplication captures almost none of this — leaving gigabytes of recoverable memory unused across cloud servers, containers, and mobile devices.

**AMDEC addresses this gap** by detecting structural similarity instead of requiring exact identity, and compressing near-duplicate pages into compact base-delta representations.

---

## Core Idea

```
TRADITIONAL EXACT MATCHING (KSM/CBPS):
Page A: [ AAA...AAA ] ──▶ Hash A
Page B: [ AAA...AAA ] ──▶ Hash A ──▶ Match!    ──▶ Merged (COW, shared physical page)
Page C: [ AAA...AAB ] ──▶ Hash C ──▶ Mismatch! ──▶ NOT merged (4KB wasted)

AMDEC — ADAPTIVE SIMILARITY CLUSTERING:
Page A: [ AAA...AAA ] ──┐
Page C: [ AAA...AAB ] ──┴──▶ Similarity Cluster ──▶ Base Page A + Delta Patch C (1 byte)
```

Two mechanisms operate together:
1. **Exact duplicates** within a cluster are still merged via standard COW, exactly like KSM.
2. **Near-duplicates** are *not* merged into one shared page — they're replaced with a small **delta patch** referencing a **base page**. On read, the page is reconstructed as `base + delta`. On write, a COW-style fault triggers reconstruction, allocation of a new private page, and a page table update.

The saving comes from replacing a 4096-byte page with a delta patch that is often only tens to a few hundred bytes — not from physical page sharing, since the content genuinely differs.

---

## System Architecture

```
                    ┌─────────────────────────────────┐
                    │   Simulated Anonymous Pages      │
                    │  (synthetic heap-like workload)  │
                    └────────────────┬──────────────────┘
                                     │
                    ┌────────────────▼──────────────────┐
                    │ M1: Region-Aware                   │
                    │ Sampling & Feedback Engine         │
                    │ → filters WHICH pages to scan      │
                    └────────────────┬──────────────────┘
                                     │ candidate pages
                    ┌────────────────▼──────────────────┐
                    │ M2: Feature Extraction             │
                    │ & LSH Clustering Engine            │
                    │ → groups similar pages             │
                    └────────────────┬──────────────────┘
                                     │ similarity clusters
                    ┌────────────────▼──────────────────┐
                    │ M3: Base-Delta Storage             │
                    │ & Adaptive Patching Manager        │
                    │ → compresses cluster to base+delta │
                    └────────────────┬──────────────────┘
                                     │
                    ┌────────────────▼──────────────────┐
                    │ M4: Acceleration Benchmark         │
                    │ (scalar vs. AVX2 vectorized diff)  │
                    └────────────────┬──────────────────┘
                                     │
                    ┌────────────────▼──────────────────┐
                    │ M5: Simulated COW Fault Handler    │
                    │ & Reconstruction Engine            │
                    └────────────────────────────────────┘
```

Each module maps directly to a research gap and project objective:

| Module | Objective | Gap Addressed |
|---|---|---|
| M1 — Sampling & Feedback | Objective 3 | Gap 2 (CPU overhead), Gap 3 (semantic ignorance) |
| M2 — LSH Clustering | Objective 1 | Gap 1 (binary exactness), Gap 4 (unguided delta search) |
| M3 — Base-Delta Storage | Objective 2 | Gap 4 (inefficient patching without clustering) |
| M4 — Acceleration Benchmark | Supports Objective 1/2 | Gap 2 (CPU overhead) |
| M5 — COW Simulation | Ties system together | Correctness & safety proof |
| All combined | Objective 4 (30–50% RAM recovery) | — |

---

## Module Reference

### M1 — Region-Aware Sampling & Feedback Engine

**Purpose:** decide *which* pages are worth scanning before spending CPU on similarity checks, keeping continuous background operation cheap.

**How it works:**
- Each simulated page carries metadata: `alloc_time`, `write_count`, and a derived `write_rate`.
- Write behavior is modeled with a Zipf-distributed workload — a small fraction of pages are "hot" (frequently rewritten), most are "cold" (long-lived, stable).
- Pages are classified into three tiers:
  - **HOT** — high write rate → skipped entirely (merging would be immediately invalidated)
  - **WARM** — moderate activity → low scan priority
  - **COLD** — long-lived, stable → high scan priority
- Only `WARM`/`COLD` pages are pushed into a priority queue and passed to M2.

**Input:** page set with write metadata → **Output:** prioritized scan queue
**Complexity:** O(n)

---

### M2 — Feature Extraction & LSH Clustering Engine

**Purpose:** find near-duplicate pages without full pairwise byte comparison (which is O(n²) and infeasible for realistic page counts).

**How it works:**
1. **Chunking** — each 4096-byte page is split into fixed-size sub-blocks (default 128B, tunable).
2. **MinHash signature** — K hash functions (default 32, via xxHash/MurmurHash3) are applied per block; the minimum hash value per hash function across all blocks forms the page's signature.
3. **LSH banding** — the K-length signature is split into `b` bands; pages sharing a hash value in *any* band land in the same bucket, turning global comparison into O(n) bucket lookups.
4. **Cluster assembly** — within a bucket, a union-find structure merges candidate pairs; a Jaccard similarity threshold (`J`, default 0.7) filters out weak matches.

**Input:** candidate pages from M1 → **Output:** similarity clusters
**Complexity:** O(n·k·s) — n pages, k hash functions, s sub-blocks per page (expected-case; degrades toward pairwise only if bucket distribution is highly uneven)

---

### M3 — Base-Delta Storage & Adaptive Patching Manager

**Purpose:** compress each cluster down to one full base page plus compact deltas for its members.

**How it works:**
- **Base selection:** the page with lowest mean byte-distance to its cluster peers becomes the base (fallback: oldest/most-stable page, cheaper to compute).
- **Delta encoding:** XOR the candidate page against the base, then run-length-encode the runs of zero (unchanged) bytes — a lightweight alternative to full `bsdiff`.
- **Adaptive thresholding:** if the resulting delta exceeds `δ_max` (default 30% of page size) or the page's write frequency is too high, the system falls back to standard compression or leaves the page unmerged — trading memory savings for reconstruction latency only when it's worth it.
- **Page table:** an in-memory map from page ID to a tagged record (`BasePage` or `DeltaPage(base_ref, patch)`).

**Input:** similarity clusters from M2 → **Output:** compressed page store
**Complexity:** O(c·p) — c = cluster size, p = page size in bytes

---

### M4 — Acceleration Benchmark

**Purpose:** demonstrate that the CPU-heaviest step (byte-diffing) can be meaningfully accelerated, addressing the same CPU-overhead gap as M1 from a different angle.

**How it works:**
- A scalar (naive loop) diff implementation is benchmarked against an **AVX2-vectorized** XOR-diff implementation using Google Benchmark.
- Results are reported as measured throughput (MB/s), not extrapolated — any claims about dedicated hardware accelerators (e.g., Intel DSA) are explicitly labeled as *projected*, not measured, since real accelerator hardware is out of scope for this project.

**Input:** delta workload → **Output:** throughput metrics (scalar vs. SIMD)

---

### M5 — Simulated COW Fault Handler & Reconstruction Engine

**Purpose:** prove correctness and safety — that a delta-stored page can always be reconstructed exactly, and that writes are handled safely without kernel access.

**How it works:**
- A `simulateWrite(PageId)` call models a process writing to a page:
  1. Look up whether the page is `merged` (exact COW) or `delta-stored`.
  2. If delta-stored, reconstruct the full page via `decodeDelta(base, delta)` — timed with `<chrono>::steady_clock`.
  3. Allocate a new "private" page object; copy reconstructed data in.
  4. Update the simulated page table.
- This produces a genuine, measured **reconstruction latency** without any kernel or OS-level hook.
- *(Optional realism boost, not required):* trigger real page-protection faults via `mprotect`/`SIGSEGV` on actual memory for a more "live" demo — still entirely in userspace.

**Input:** write event + merged/delta page → **Output:** reconstructed private page
**Complexity:** O(δ) — δ = delta size

---

## Project Structure

```
amdec-sim/
├── CMakeLists.txt                      # Top-level CMake build
├── src/
│   ├── common/
│   │   ├── page.hpp                    # Page struct: 4096B content + metadata
│   │   └── types.hpp                   # Shared typedefs (PageId, Tier, etc.)
│   ├── workload/
│   │   └── page_generator.{hpp,cpp}    # Synthetic heap workload; Zipf write distribution
│   ├── m1_sampling/
│   │   └── sampler.{hpp,cpp}           # Tiering logic + priority queue
│   ├── m2_lsh/
│   │   ├── minhash.hpp                 # MinHash signature (hot path)
│   │   └── lsh_index.{hpp,cpp}         # LSH banding + bucket table + union-find
│   ├── m3_delta/
│   │   ├── base_selector.hpp
│   │   ├── delta_codec.hpp             # XOR-diff + RLE (hot path, raw arrays)
│   │   └── delta_store.cpp             # Adaptive thresholding + page table
│   ├── m4_accel/
│   │   ├── bench_scalar.cpp
│   │   └── bench_avx2.cpp              # AVX2 vectorized diff (Google Benchmark)
│   ├── m5_cow/
│   │   └── cow_handler.{hpp,cpp}       # Write-fault simulation + reconstruction timing
│   ├── simulator.cpp                   # Discrete-event harness wiring all modules
│   └── metrics.cpp                     # Metric collection → CSV export
├── experiments/
│   ├── run_full_eval.cpp
│   ├── ablation_m1.cpp
│   ├── ablation_adaptive.cpp
│   └── parameter_sweep.cpp
├── tests/                              # Catch2/GoogleTest unit + integration tests
├── analysis/                           # Python — reads CSV only, no algorithm logic
│   ├── plot_results.py
│   ├── compare_baselines.py
│   └── ablation_analysis.py
├── results/raw/                        # Timestamped CSV output from C++ binaries
├── third_party/                        # xxHash, Catch2, Google Benchmark
├── docs/
│   ├── INVENTION_DISCLOSURE.md
│   └── architecture.drawio
├── paper/                              # LaTeX source
├── references/                         # PDFs + BibTeX
├── requirements.txt
└── README.md
```

---

## Language & Tooling

| Layer | Language | Responsibility |
|---|---|---|
| Core engine (M1, M2, M3, M5) | **C++17** | Simulation logic, clustering, storage, reconstruction — classes + STL for structure, raw arrays/`memcpy` in hot paths |
| Acceleration (M4) | **C++17 + AVX2** | SIMD-vectorized diff benchmark |
| Analysis & plotting | **Python** | Reads CSV output only — `pandas`, `numpy`, `matplotlib`, `seaborn` |

**Rule of thumb:** if it touches a memory page, a hash, a delta, or a timer — C++. If it touches a chart, a table, or a number in the paper — Python. No algorithmic logic lives in Python.

**Key dependencies:**

| Tool | Purpose |
|---|---|
| CMake ≥ 3.20 | Build system |
| clang++ / AppleClang | Compiler (Xcode Command Line Tools) |
| xxHash / MurmurHash3 | Hot-path hashing for MinHash |
| Catch2 or GoogleTest | Unit/integration testing |
| Google Benchmark | M4 throughput measurement |
| AVX2 intrinsics (`<immintrin.h>`) | Vectorized diff |
| pandas, numpy, matplotlib, seaborn, scipy | Result analysis and figures |

---

## Getting Started

```bash
# Clone and enter the repo
git clone <repo-url> amdec-sim && cd amdec-sim

# Install build prerequisites (macOS)
xcode-select --install
brew install cmake

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run unit tests
ctest --test-dir build

# Set up Python analysis environment
pip install -r requirements.txt --break-system-packages
```

---

## Running Experiments

The evaluation pipeline is designed to be reproducible from two commands:

```bash
./build/experiments/run_full_eval --output results/raw/eval_$(date +%Y%m%d_%H%M%S).csv
python analysis/plot_results.py results/raw/eval_<timestamp>.csv
```

| Experiment | Command | Output |
|---|---|---|
| Full pipeline evaluation | `./build/experiments/run_full_eval` | All primary metrics → CSV |
| Parameter sweep | `./build/experiments/parameter_sweep` | Per-parameter CSV rows → Python heatmaps |
| Ablation: M1 disabled | `./build/experiments/ablation_m1` | CPU overhead spike (proves M1's value) |
| Ablation: fixed threshold | `./build/experiments/ablation_adaptive` | Memory savings drop or latency spike |
| Acceleration benchmark | `./build/src/m4_accel/bench` | Scalar vs. AVX2 throughput (Google Benchmark JSON) |

---

## Evaluation Metrics

| Metric | Formula | Unit | Target |
|---|---|---|---|
| Memory savings | `(S_base − S_system) / S_base × 100` | % | ≥ 30% |
| CPU overhead | `(T_scan + T_cluster + T_delta) / T_total × 100` | % | ≤ 5% |
| Reconstruction latency | `mean(T_reconstruct)` per page | µs | ≤ 10 µs |
| Scan efficiency | `pages_merged / pages_scanned` | ratio | > 0.6 |
| Cluster precision | `1 − false_positive_rate` | ratio | > 0.9 |

**Baselines:** KSM and UKSM figures are used strictly as **literature-cited comparisons**, not same-machine benchmarks — this is stated explicitly in the paper's Evaluation section and in every results table.

**Tunable parameters** (defined before experimentation to avoid post-hoc parameter fishing):

| Parameter | Symbol | Default | Sweep Range |
|---|---|---|---|
| Block size for MinHash | `B` | 128B | 64, 128, 256 |
| Number of hash functions | `K` | 32 | 16, 32, 64 |
| LSH bands | `b` | 8 | 4, 8, 16 |
| Jaccard threshold | `J` | 0.7 | 0.5, 0.7, 0.9 |
| Hot-page write-rate threshold | `τ_hot` | 0.1 | 0.05, 0.1, 0.2 |
| Max delta size ratio | `δ_max` | 30% | 20%, 30%, 40% |

---

## Project Roadmap & Phases

DSR (Design Science Research) methodology across 10 weeks:

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5  →  Phase 6
 Setup      Survey     Design      Implement    Evaluate    Write       Publish
 & Scope    & Gap       & Arch      & Test       & Results   Paper       & File
 (Wk 0)   (Wk 1)     (Wk 1–2)   (Wk 2–6)    (Wk 7)      (Wk 8)     (Wk 9–10)
```

### Phase 0 — Project Initialization & Scoping (Pre-Week 1)
GitHub repo setup, branch strategy, issue tracking, and reference archive. Includes writing and date-stamping the **Invention Disclosure Document (IDD)** — critical if pursuing a patent, since it establishes earliest evidence of invention. Also confirms scope (userspace simulation only, literature-cited baselines only) up front.

### Phase 1 — Literature Survey & Gap Formalization (Week 1)
Consolidates the literature review into a structured matrix (paper, technique, exact/near-dup/base-delta capability, limitation), numbers each identified gap, and links each to the module that addresses it. Produces the finalized three-part Contributions Statement that anchors both the paper's Introduction and any patent's independent claims.

### Phase 2 — System Design & Architecture (Weeks 1–2)
Finalizes the five-module architecture diagram, documents formal inputs/outputs/complexity per module, defines the full tunable parameter space, and pre-registers all evaluation metrics and targets — done *before* implementation to prevent post-hoc metric selection.

### Phase 3 — Implementation (Weeks 2–6)
Week-by-week build of the C++ core engine:
- **Week 2:** CMake skeleton, synthetic workload generator, Module 1 (sampling/tiering)
- **Week 3:** Module 2 (MinHash + LSH clustering)
- **Week 4:** M1+M2 integration, preliminary metrics, **Review 2 preparation**
- **Week 5:** Module 3 (base-delta storage, adaptive thresholding)
- **Week 6:** Module 5 (COW simulation) + full pipeline integration

### Phase 4 — Experimentation & Evaluation (Week 7)
Runs the full experiment suite (full evaluation, parameter sweep, ablations, acceleration benchmark), exports all results as versioned CSVs, generates publication-quality figures, and compiles the comparison table against cited KSM/UKSM baselines with explicit disclosure of the comparison's literature-based nature.

### Phase 5 — Paper Writing (Week 8)
Written in dependency order — Methodology → Implementation → Results → Related Work → Introduction → Conclusion → Abstract (Introduction and Abstract are written last, once the actual contribution and results are known). Includes a structure, claims-integrity, and language checklist before moving to formatting.

### Phase 6 — Quality Assurance, Submission & Patent Filing (Weeks 9–10)
Pre-submission quality gates (plagiarism <10%, AI-content <5%, citation completeness, figure resolution, page limits), Review 3 demo preparation (3 live scenarios + backup video), venue selection, and — if pursuing IP protection — the provisional patent filing path, which must occur **before** public disclosure of the paper.

---

## Review Checkpoints

### Review 2 — End of Week 4 (~60% Complete)
- Modules 1 & 2 implemented, producing clusters on synthetic data
- Preliminary metrics: candidate reduction ratio, cluster size distribution
- Finalized architecture diagram
- GitHub commit history showing progressive, dated builds
- Slides: problem → architecture → live M1+M2 demo → preliminary numbers

### Review 3 — End of Week 10 (Final)
- All 5 modules implemented and integrated
- Full evaluation suite completed (all 5 metrics)
- Ablation study results
- Comparison table vs. KSM/UKSM literature baselines
- Paper drafted in IEEE/Springer format, all sections complete
- Plagiarism < 10%, AI-content < 5%
- Live demo (3 scenarios) + backup video ready
- Target venue selected and submission formatted

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LSH produces too many false positives | Medium | High | Tune Jaccard threshold `J`; add secondary exact-hash verification for small clusters |
| Memory savings < 30% target | Medium | High | Increase similarity ratio in synthetic workload; verify against the paper's 86% near-dup claim |
| Delta sizes exceed `δ_max` too often | Low | Medium | Re-tune `δ_max`; consider a proper patch format (e.g., bsdiff-style edit list) |
| Reconstruction latency > 10 µs | Low | High | Profile with Instruments/`perf`; move delta decode to the AVX2 path |
| AVX2 compilation fails on target Mac | Low | Medium | Gate with `-DENABLE_AVX2=OFF` fallback; test scalar path in CI |
| C++ undefined behavior in hot paths | Medium | Medium | Run under AddressSanitizer/UBSan during development; bounds-check in debug builds |
| Live demo fails during Review 2/3 | Medium | Medium | Pre-run binaries, pipe output to terminal; keep CSV + static Python plots as fallback |
| Paper plagiarism > 10% | Low | High | Write Methodology and Results sections entirely from scratch |
| Venue rejection | Medium | Low | Pre-submit to arXiv; keep 2 backup venues identified |

---

## Documentation

- [`docs/INVENTION_DISCLOSURE.md`](docs/INVENTION_DISCLOSURE.md) — dated novelty disclosure (patent evidentiary record)
- [`docs/architecture.drawio`](docs/architecture.drawio) — editable architecture diagram source
- [`references/`](references/) — cited papers (PDF) and BibTeX entries

---

## Citing This Work

```bibtex
@inproceedings{amdec2026,
  title     = {Adaptive Memory Deduplication via Locality-Sensitive Similarity Clustering and Base-Delta Encoding},
  author    = {Shantanu Tapole},
  year      = {2026}
}
```

---

## License

MIT (or Apache 2.0 — finalize at repo creation). See [`LICENSE`](LICENSE).

---

> **Note on scope:** AMDEC is a userspace simulation, not a kernel or hypervisor implementation. All KSM/UKSM comparisons are drawn from published literature figures, not same-hardware benchmarks — this is stated explicitly throughout the paper and this repository, in line with academic reporting standards.