# AMDEC — End-to-End Research & Patent Project Roadmap
### Adaptive Memory Deduplication using Similarity Clustering

> **Methodology:** Design Science Research (DSR)  
> **Platform:** macOS, pure userspace simulation  
> **Engine (Modules 1–3, 5):** C++ (STL + raw arrays in hot paths, xxHash/MurmurHash3, `<chrono>`, CMake, Catch2/GoogleTest)  
> **Acceleration (Module 4):** C++ with AVX2 SIMD intrinsics + Google Benchmark  
> **Analysis & Plotting:** Python only (reads CSV output — no algorithm logic)  
> **Target Output:** Peer-reviewed research paper (Scopus/UGC-CARE indexed) + optional patent filing  
> **Timeline:** ~10 weeks (adjust start date to your calendar)

---

## 📋 Quick Overview

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5  →  Phase 6
 Setup      Survey     Design      Implement    Evaluate    Write       Publish
 & Scope    & Gap       & Arch      & Test       & Results   Paper       & File
 (Wk 0)   (Wk 1)     (Wk 1–2)   (Wk 2–6)    (Wk 7)      (Wk 8)     (Wk 9–10)
                           ↑               ↑
                      Review 2         Review 3
                      Checkpoint       Checkpoint
```

---

## Phase 0 — Project Initialization & Scoping
### Duration: Pre-Week 1 (Day 1–3)

This phase ensures you start the project with a clean, traceable foundation — essential for patent disclosure and academic reproducibility.

### 0.1 Administrative Setup

| Task | Owner | Status |
|---|---|---|
| Create GitHub repository (`amdec-sim`) with MIT/Apache 2.0 license | You | `[ ]` |
| Set up branch strategy: `main`, `dev`, `experiments/*`, `paper/*` | You | `[ ]` |
| Create project wiki / README with problem statement | You | `[ ]` |
| Set up issue tracker: one issue per module/deliverable | You | `[ ]` |
| Archive all reference papers in `/references/` folder (PDF + BibTeX) | You | `[ ]` |

### 0.2 Novelty & IP Disclosure (Critical for Patent Path)

> [!IMPORTANT]
> Before writing a single line of code, document your novelty claim in writing. This timestamp is your earliest evidence of invention for a provisional patent.

- [ ] Write a **1-page Invention Disclosure Document (IDD)**:
  - What is the problem?
  - What prior art exists? (your gap table from Review 1)
  - What is the specific novel contribution? (LSH clustering + base-delta + region-aware sampling — combined, not individually)
  - What would it look like deployed in an OS or hypervisor?
- [ ] Date-stamp and store the IDD in the repo as `docs/INVENTION_DISCLOSURE.md`
- [ ] If pursuing a patent: submit IDD to your institution's Technology Transfer Office (TTO) or file a **provisional patent application** (12-month priority window)

### 0.3 Ethical & Scope Declarations

- [ ] Confirm scope: userspace simulation only — no kernel modification, no proprietary tools, no private data
- [ ] Confirm that all cited baselines (KSM/UKSM) will be used only as **literature-cited figures**, not same-machine benchmarks (this must be disclosed explicitly in the paper's Evaluation section)
- [ ] Identify target venue (see Phase 6) early so you know the page limit, format, and review criteria from day one

---

## Phase 1 — Literature Survey & Gap Formalization
### Duration: Week 1 (parallel with Phase 0)

> **DSR Step:** Problem Identification

This phase should already be substantially complete from Review 1. The goal here is to crystallize the gaps into a format that feeds directly into your Contributions section.

### 1.1 Survey Consolidation

- [ ] Organize all reviewed papers into a structured **Literature Matrix**:

| Paper | Year | Technique | Memory Savings | Exact-only? | Near-dup? | Base-Delta? | Limitation |
|---|---|---|---|---|---|---|---|
| KSM (Linux) | 2009 | Hash merge | ~20% | ✅ | ❌ | ❌ | Only exact duplicates |
| UKSM | 2012 | Adaptive scan | ~25% | ✅ | ❌ | ❌ | Still exact-only |
| SiLo | 2014 | LSH for dedup | High | ❌ | ✅ | ❌ | Storage, not RAM |
| ... | | | | | | | |

- [ ] Identify and number all gaps (at minimum Gap 1–4 from your existing doc)
- [ ] For each gap, write a one-sentence **Addressed By:** statement linking it to a specific module

### 1.2 Contributions Statement (Drives Paper Section 1 & Patent Claims)

Write these three bullet points precisely — they will anchor your paper's Introduction and your patent's independent claims:

> [!NOTE]
> **Contribution 1 (Algorithmic):** A region-aware, feedback-driven page sampling engine that classifies anonymous memory pages into stability tiers, reducing unnecessary scan overhead compared to full-scan approaches.
>
> **Contribution 2 (Algorithmic):** An LSH-based similarity clustering pipeline that identifies near-duplicate memory pages at sub-quadratic cost, enabling deduplication beyond exact-copy matching.
>
> **Contribution 3 (Systems):** A base-delta storage manager with adaptive thresholding and simulated COW reconstruction, demonstrating safe, low-latency retrieval of delta-compressed pages.

---

## Phase 2 — System Design & Architecture
### Duration: Week 1–2 (parallel with implementation setup)

> **DSR Step:** Design & Development (Architecture)

### 2.1 Architecture Documentation

- [ ] Finalize system architecture diagram (draw.io / Excalidraw / LaTeX TikZ for paper)
- [ ] Document all 5 modules with formal inputs, outputs, and complexity

| Module | Input | Output | Complexity | Addresses |
|---|---|---|---|---|
| M1: Region-Aware Sampling | Page set with write metadata | Prioritized scan queue | O(n) | Gap 2, Gap 3 |
| M2: LSH Clustering | Candidate pages | Similarity clusters | O(n·k) k=num_hashes | Gap 1, Gap 4 |
| M3: Base-Delta Storage | Clusters | Compressed page store | O(c·p) c=cluster size | Gap 4 |
| M4: Acceleration Benchmark | Delta workload | Throughput metrics | — | Gap 2 |
| M5: COW Simulation | Write event + merged page | Reconstructed private page | O(delta) | Correctness proof |

### 2.2 Parameter Space Definition

Define all tunable parameters **before** experiments — this prevents parameter-fishing after the fact (a common reviewer concern):

| Parameter | Symbol | Default | Range to Sweep |
|---|---|---|---|
| Block size for MinHash | `B` | 128B | 64, 128, 256 |
| Number of hash functions | `K` | 32 | 16, 32, 64 |
| LSH bands | `b` | 8 | 4, 8, 16 |
| Jaccard threshold | `J` | 0.7 | 0.5, 0.7, 0.9 |
| Hot-page write-rate threshold | `τ_hot` | 0.1 | 0.05, 0.1, 0.2 |
| Max delta size ratio | `δ_max` | 30% | 20%, 30%, 40% |

### 2.3 Evaluation Design (Pre-register your metrics)

> [!IMPORTANT]
> Define metrics and baselines **before** coding. This is a methodological requirement that reviewers check for. It is also required for patent claims (you cannot claim something you have not defined).

**Primary Metrics:**

| Metric | Formula | Unit | Target |
|---|---|---|---|
| Memory savings | `(S_base - S_system) / S_base x 100` | % | >= 30% |
| CPU overhead | `T_scan+cluster+delta / T_total x 100` | % | <= 5% |
| Reconstruction latency | `mean(T_reconstruct)` per page | µs | <= 10 µs |
| Scan efficiency | `pages_merged / pages_scanned` | ratio | > 0.6 |
| Cluster precision | `1 - false_positive_rate` | ratio | > 0.9 |

**Baselines (from literature):**
- KSM: ~20% memory savings, documented CPU overhead from original paper
- UKSM: ~25–30% memory savings, lower overhead

---

## Phase 3 — Implementation
### Duration: Weeks 2–6

> **DSR Step:** Design & Development (Artifact)

### 3.1 Language & Tooling Split

| Layer | Language | What it does | Key tools |
|---|---|---|---|
| **Core engine** | C++ | Modules 1, 2, 3, 5 — page simulation, sampling/tiering, LSH clustering, base-delta storage, COW reconstruction | STL (`unordered_map`, `vector`, `priority_queue`) for bookkeeping; raw arrays + `memcpy` in hot paths (byte-diff, hash computation) |
| **Hashing** | C++ | MinHash signature generation | xxHash or MurmurHash3 (header-only, zero-dependency) |
| **Timing** | C++ | All latency/overhead measurements | `<chrono>` `steady_clock` |
| **Acceleration** | C++ | Module 4 SIMD benchmark | AVX2 intrinsics for vectorized XOR-diff; Google Benchmark for throughput |
| **Build** | — | Project build system | CMake (≥ 3.20) |
| **Testing** | C++ | Unit + integration tests | Catch2 or GoogleTest |
| **CSV output** | C++ | Export all results | `std::ofstream` — raw CSV, no plotting |
| **Analysis** | Python | Reads CSV, builds tables & graphs | `pandas`, `numpy`, `matplotlib`, `seaborn` |

> [!NOTE]
> `datasketch` is **not used** — LSH is implemented natively in C++. Python has zero algorithm logic; it only post-processes output the C++ binary writes to disk.

### 3.2 Project File Structure

```
amdec-sim/
├── CMakeLists.txt                      # Top-level CMake build
├── src/
│   ├── CMakeLists.txt
│   ├── common/
│   │   ├── page.hpp                    # Page struct: 4096B content + metadata
│   │   └── types.hpp                   # Shared typedefs (PageId, Tier, etc.)
│   ├── workload/
│   │   ├── page_generator.hpp
│   │   └── page_generator.cpp          # Synthetic heap workload; Zipf write distribution
│   ├── m1_sampling/
│   │   ├── sampler.hpp
│   │   └── sampler.cpp                 # Tiering logic + priority queue (STL)
│   ├── m2_lsh/
│   │   ├── minhash.hpp                 # MinHash signature; xxHash/MurmurHash3 hot path
│   │   ├── lsh_index.hpp
│   │   └── lsh_index.cpp               # LSH banding + bucket table + union-find
│   ├── m3_delta/
│   │   ├── base_selector.hpp
│   │   ├── delta_codec.hpp             # XOR-diff + RLE — raw arrays, no STL in hot path
│   │   └── delta_store.cpp             # Adaptive thresholding + page table
│   ├── m4_accel/
│   │   ├── bench_scalar.cpp            # Baseline scalar diff
│   │   └── bench_avx2.cpp              # AVX2 vectorized diff (Google Benchmark)
│   ├── m5_cow/
│   │   ├── cow_handler.hpp
│   │   └── cow_handler.cpp             # Write-fault simulation + reconstruction timing
│   ├── simulator.cpp                   # Discrete-event harness wiring all modules
│   └── metrics.cpp                     # Metric collection → CSV export
├── experiments/
│   ├── run_full_eval.cpp               # Main evaluation binary
│   ├── ablation_m1.cpp                 # M1 disabled
│   ├── ablation_adaptive.cpp           # Fixed vs. adaptive threshold
│   └── parameter_sweep.cpp             # Grid search over parameter space
├── tests/
│   ├── test_m1_sampling.cpp
│   ├── test_m2_lsh.cpp
│   ├── test_m3_delta.cpp               # Round-trip: reconstruct(base, delta) == original
│   └── test_m5_cow.cpp
├── analysis/                           # Python downstream — reads CSV only
│   ├── plot_results.py                 # matplotlib/seaborn: all paper figures
│   ├── compare_baselines.py            # Build KSM/UKSM comparison table (pandas)
│   └── ablation_analysis.py           # Post-hoc ablation stats (numpy)
├── results/
│   └── raw/                            # Timestamped CSVs written by C++ binaries
├── third_party/
│   ├── xxhash/                         # xxHash header-only
│   ├── catch2/                         # or googletest
│   └── benchmark/                      # Google Benchmark
├── docs/
│   ├── INVENTION_DISCLOSURE.md
│   └── architecture.drawio
├── paper/                              # LaTeX source
├── references/                         # PDFs + BibTeX
├── requirements.txt                    # Python: pandas numpy matplotlib seaborn scipy
└── README.md
```

### 3.3 Week-by-Week Implementation Tasks

#### Week 2 — CMake Skeleton + Workload Generator + Module 1

- [ ] Scaffold CMake project: top-level `CMakeLists.txt`, subdirectory targets, add `third_party/` (xxHash header, Catch2 via FetchContent)
- [ ] Implement `page.hpp` — `Page` struct: `uint8_t content[4096]`, `uint64_t alloc_time`, `uint32_t write_count`, `float write_rate`, `Tier tier`
- [ ] Implement `page_generator.cpp`:
  - Generate N = 10,000–100,000 pages with Zipf-distributed write activity
  - Configurable similarity ratio (80% near-duplicate, 7% exact, 13% unique) — mutate a base template at random byte offsets
- [ ] Implement `m1_sampling/sampler.cpp` — compute `write_rate = write_count / age`; classify into `HOT / WARM / COLD`; push cold/warm into `std::priority_queue`
- [ ] Write `tests/test_m1_sampling.cpp` (Catch2) — `REQUIRE` that hot pages are never enqueued; cold pages rank above warm
- [ ] **Commit milestone:** `feat: CMake build + M1 tiering — hot/warm/cold classification passing Catch2 tests`

#### Week 3 — Module 2 (LSH Clustering)

- [ ] Implement `m2_lsh/minhash.hpp`:
  - `computeSignature(const uint8_t* page, size_t K)` — split page into `B`-byte blocks; apply K xxHash seeds per block; take per-position minimum across blocks
  - **Hot path rule:** raw pointer arithmetic + xxHash; no `std::vector` allocation inside this function
- [ ] Implement `m2_lsh/lsh_index.cpp`:
  - Band splitting: group K hashes into `b` bands of `K/b` hashes each
  - Bucket table: `std::unordered_map<size_t, std::vector<PageId>>` — hash each band → bucket
  - Union-Find: merge pages sharing ≥ 1 bucket; apply Jaccard threshold `J` to filter false-positive pairs
- [ ] Validate on 10 hand-crafted pages (3 near-duplicate triplets + 1 unrelated) — inspect cluster output manually
- [ ] Write `tests/test_m2_lsh.cpp` — assert Jaccard ≥ 0.7 pages cluster together; Jaccard < 0.3 pages don't
- [ ] **Commit milestone:** `feat: M2 LSH clustering — correct clusters on synthetic heap data`

#### Week 4 — M1+M2 Integration + Review 2 Prep

- [ ] Wire `simulator.cpp`: generator → M1 sampler → M2 LSH index → output clusters to stdout/CSV
- [ ] Run on N = 50,000 pages; collect preliminary metrics: candidate reduction ratio, cluster size distribution, false-positive rate
- [ ] Export preliminary results to `results/raw/prelim_<timestamp>.csv`
- [ ] Run `analysis/plot_results.py` on preliminary CSV — generate cluster size histogram for Review 2 slides
- [ ] Prepare **Review 2 materials** (see checkpoint below)
- [ ] Begin `m3_delta/base_selector.hpp` — base selection interface

> **REVIEW 2 CHECKPOINT — Target: End of Week 4**

#### Week 5 — Module 3 (Base-Delta Storage)

- [ ] Implement `m3_delta/base_selector` — choose page with minimum mean byte-distance to cluster peers (or oldest/most-stable as fast fallback)
- [ ] Implement `m3_delta/delta_codec.hpp` — **hot path: raw arrays only**:
  ```cpp
  // XOR two pages into a diff buffer, then RLE-encode runs of zero bytes
  size_t encodeDelta(const uint8_t* base, const uint8_t* page, uint8_t* out, size_t len);
  size_t decodeDelta(const uint8_t* base, const uint8_t* delta, uint8_t* out, size_t len);
  ```
- [ ] Implement adaptive thresholding: if `delta_size / PAGE_SIZE > δ_max`, skip merging this page (leave uncompressed)
- [ ] Implement in-memory page table: `std::unordered_map<PageId, PageRecord>` where `PageRecord` is a tagged union (`BasePage` or `DeltaPage`)
- [ ] Write `tests/test_m3_delta.cpp` — `REQUIRE(decodeDelta(base, encodeDelta(base, page)) == page)` for 1000 random page pairs

#### Week 6 — Module 5 + Full Pipeline Integration

- [ ] Implement `m5_cow/cow_handler.cpp`:
  - `simulateWrite(PageId id)` — look up page record; if delta-stored, call `decodeDelta`; time with `<chrono> steady_clock`; allocate new private page object; update page table
- [ ] Wire full pipeline in `simulator.cpp`: generator → M1 → M2 → M3 → M5 → write metrics to CSV
- [ ] Integration test (`tests/test_m5_cow.cpp`): write to a merged page; assert reconstructed bytes equal original; assert `simulateWrite` latency < 50 µs (loose bound for CI)
- [ ] **Commit milestone:** `feat: full C++ pipeline — generate → sample → cluster → compress → reconstruct`

---

## Phase 4 — Experimentation & Evaluation
### Duration: Week 7

> **DSR Step:** Demonstration & Evaluation

This phase must be **reproducible from a single pair of commands**:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/experiments/run_full_eval --output results/raw/eval_$(date +%Y%m%d_%H%M%S).csv
python analysis/plot_results.py results/raw/eval_<timestamp>.csv
```
Reviewers and patent examiners may request reproduction.

### 4.1 Experiment Suite

| Experiment | Binary / Script | Outputs |
|---|---|---|
| Full pipeline evaluation | `./build/experiments/run_full_eval` | All 5 primary metrics → CSV |
| Parameter sweep | `./build/experiments/parameter_sweep` | Per-parameter CSV rows; Python plots heatmaps |
| Ablation: M1 disabled | `./build/experiments/ablation_m1` | CPU overhead spike (proves M1's value) → CSV |
| Ablation: Fixed threshold | `./build/experiments/ablation_adaptive` | Memory savings drop OR latency spike → CSV |
| Acceleration benchmark | `./build/src/m4_accel/bench` (Google Benchmark) | Scalar vs. AVX2 throughput in MB/s → JSON → Python bar chart |
| Python analysis | `analysis/plot_results.py` | All paper figures from CSVs; no algorithm logic |

### 4.2 Results Documentation

- [ ] Export all results to `results/raw/*.csv` (never overwrite — version with timestamps)
- [ ] Generate publication-quality figures:
  - Memory savings over time (line chart)
  - CPU overhead breakdown (stacked bar)
  - Reconstruction latency CDF
  - Cluster size distribution (histogram)
  - Ablation comparison table
  - Acceleration benchmark bar chart
- [ ] Compile comparison table: your system vs. KSM vs. UKSM (literature baselines)
- [ ] Document unexpected results honestly — reviewers respect acknowledged limitations

### 4.3 Baseline Validation

- [ ] Pull exact figures from cited KSM and UKSM papers
- [ ] Create a BibTeX entry for each source used in the comparison table
- [ ] Add explicit footnote in paper: *"KSM and UKSM figures cited from [ref] — not reproduced on the same hardware"*

---

## Phase 5 — Paper Writing
### Duration: Week 8

> **DSR Step:** Communication

Write sections in this order — **not** Introduction first:

### 5.1 Section Writing Order & Page Budget (IEEE Two-Column)

| Order | Section | Pages | Content Source |
|---|---|---|---|
| 1 | Methodology / System Design | 2–3 pp | Phase 2 design docs |
| 2 | Implementation | 0.5–1 pp | Phase 3 code overview |
| 3 | Evaluation / Results | 2–3 pp | Phase 4 outputs |
| 4 | Related Work | 1–1.5 pp | Phase 1 literature matrix |
| 5 | Introduction | 1 pp | Written last — after you know results |
| 6 | Conclusion & Future Work | 0.5–1 pp | — |
| 7 | Abstract | 150–250 words | Written very last |

### 5.2 Paper Quality Checklist

**Structure:**
- [ ] Each section traces back to a stated gap and contribution
- [ ] All figures are numbered, captioned, and referenced in body text
- [ ] All tables have headers and units
- [ ] Equations are numbered; symbols are defined on first use

**Claims integrity:**
- [ ] No claim is made without supporting experimental data or a cited reference
- [ ] Literature-baseline comparisons are explicitly disclosed as such
- [ ] Scope of simulation is clearly stated in Methodology (not buried in a footnote)

**Language:**
- [ ] Abstract answers: What is the problem? What did you do? What were the results? Why does it matter?
- [ ] Introduction ends with a clear **"In summary, the contributions of this paper are:"** bullet list
- [ ] Every section has a transition sentence leading to the next

### 5.3 Formatting

- [ ] Use IEEE two-column template (conferences) or Springer LNCS (journals)
- [ ] All figures at 300 DPI minimum
- [ ] BibTeX references — no manual formatting
- [ ] Consistent notation throughout

---

## Phase 6 — Quality Assurance, Submission & Patent Filing
### Duration: Weeks 9–10

> **DSR Step:** Communication (submission + IP)

### 6.1 Pre-Submission Quality Gates

| Check | Tool | Threshold | Action if Failed |
|---|---|---|---|
| Plagiarism | iThenticate / Turnitin | < 10% overall, < 3% per source | Rewrite flagged passages in your own words |
| AI-content detection | GPTZero / Originality.ai | < 5% | Manually rewrite AI-assisted sections |
| Grammar & style | Grammarly Premium | 0 critical errors | Fix all flagged errors |
| Citation completeness | Reference manager | All in-text citations have BibTeX entry | Cross-check every `\cite{}` |
| Figure resolution | Preview / ImageMagick | >= 300 DPI | Re-export all charts |
| Page limit | PDF word count | Within venue limit | Cut Related Work or consolidate results |

### 6.2 Demo Preparation (Review 3)

- [ ] Prepare **3 live demo scenarios**:
  1. **Normal case:** 80% near-duplicate heap → show clustering output + memory savings
  2. **Stress case:** 100% unique pages → graceful fallback, minimal overhead
  3. **Edge case:** large delta exceeds `δ_max` → adaptive fallback path activates
- [ ] Record backup demo video (screen recording with text overlays)
- [ ] Write demo script: what you click, what numbers to point to, what you say
- [ ] **Dry run the demo 48 hours before Review 3**

### 6.3 Venue Selection

| Priority | Type | Examples | Timeline |
|---|---|---|---|
| 1st | Scopus-indexed conference (Q2/Q3) | IEEE COMPSAC, ACM SAC, USENIX ATC (workshop track) | 3–6 months review |
| 2nd | UGC-CARE listed journal | IJCA, IJERT (Group 1) | 2–4 months review |
| 3rd | arXiv preprint | arxiv.org/cs.OS | Immediate |

> [!TIP]
> Submit to arXiv first (free, immediate). This timestamps your ideas publicly — useful for patent priority AND ensures your work is citeable while waiting for peer review.

### 6.4 Patent Filing Path (Optional but Recommended)

> [!IMPORTANT]
> **Do not publish the paper before filing a provisional patent** if you intend to pursue patent protection. Most jurisdictions allow a 12-month grace period after first public disclosure, but some international jurisdictions (e.g., Europe) have strict novelty requirements — prior public disclosure can invalidate your patent claims.

| Step | When | Who |
|---|---|---|
| Invention Disclosure Document (IDD) | Phase 0 (already done) | You + supervisor |
| Prior art search | After Phase 1 | You + TTO |
| Provisional patent application | Before paper submission | TTO / patent attorney |
| Full patent application | Within 12 months of provisional | Patent attorney |

**Patentable claims your system supports:**
1. A method for classifying anonymous memory pages into stability tiers using write-frequency heuristics
2. A system for near-duplicate memory page detection using MinHash-LSH with configurable banding parameters
3. A base-delta storage mechanism with adaptive thresholding for compressed anonymous memory deduplication

---

## Review Checkpoint Summary

### REVIEW 2 — End of Week 4 (~60% Complete)

| Deliverable | Status |
|---|---|
| Modules 1 & 2 implemented and producing clusters on synthetic data | `[ ]` |
| Preliminary metrics: candidate reduction ratio, cluster size distribution | `[ ]` |
| Architecture diagram finalized (draw.io or equivalent) | `[ ]` |
| GitHub commit history showing progressive, dated builds | `[ ]` |
| Slides: problem → architecture → live M1+M2 demo → preliminary numbers | `[ ]` |

**Review 2 presentation structure:**
1. Problem recap (1 slide) — the 4 gaps, your 3 contributions
2. System architecture (1 slide) — the 5-module diagram
3. Live demo: run C++ binary → pipe cluster output to terminal; show Python-generated cluster size histogram
4. Preliminary results (1–2 slides): candidate reduction ratio, cluster size histogram from `analysis/plot_results.py`
5. Phase C–F plan (1 slide): what's next and when

---

### REVIEW 3 — End of Week 10 (Final)

| Deliverable | Status |
|---|---|
| All 5 modules implemented and integrated | `[ ]` |
| Full evaluation suite completed (all 5 metrics) | `[ ]` |
| Ablation study results | `[ ]` |
| Comparison table vs. KSM/UKSM literature baselines | `[ ]` |
| Paper drafted in IEEE/Springer format, all sections complete | `[ ]` |
| Plagiarism < 10%, AI-content < 5% | `[ ]` |
| Live demo (3 scenarios) + backup video ready | `[ ]` |
| Target venue selected and submission formatted | `[ ]` |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LSH produces too many false positives | Medium | High | Tune Jaccard threshold `J`; add secondary exact-hash verification step for small clusters |
| Memory savings < 30% target | Medium | High | Increase similarity ratio in synthetic workload; verify workload matches paper's 86% near-dup claim |
| Delta sizes too large — most pages exceed `δ_max` | Low | Medium | Re-tune `δ_max`; switch from XOR+RLE to a proper patch format (e.g., bsdiff-style edit list) |
| Reconstruction latency too high (> 10 µs) | Low | High | Profile with Instruments or `perf`; move delta decode to AVX2 SIMD path |
| CMake/AVX2 compilation fails on target Mac | Low | Medium | Gate AVX2 with `cmake -DENABLE_AVX2=OFF` fallback; test on CI with scalar path |
| C++ UB in hot path (raw pointer arithmetic) | Medium | Medium | Run under AddressSanitizer (`-fsanitize=address,undefined`) during development; add bounds checks in debug builds |
| Review 2 live demo fails | Medium | Medium | Pre-run binary + pipe output to terminal; have CSV + Python plots ready as static fallback |
| Paper plagiarism > 10% | Low | High | Write methodology and results sections entirely from scratch in your own words |
| Venue rejection | Medium | Low | Pre-submit to arXiv; have 2 backup venues identified |

---

## Key Dependencies & Tooling

### C++ (Core Engine)

| Tool | Purpose | How to get |
|---|---|---|
| **CMake ≥ 3.20** | Build system | `brew install cmake` |
| **clang++ / AppleClang** | C++ compiler (ships with Xcode CLT) | `xcode-select --install` |
| **xxHash** | MinHash hot-path hashing (header-only) | `third_party/xxhash/` (git submodule or FetchContent) |
| **MurmurHash3** | Alternative to xxHash (header-only) | Copy `MurmurHash3.h/.cpp` into `third_party/` |
| **Catch2** | Unit + integration testing | CMake FetchContent |
| **Google Benchmark** | Module 4 throughput measurement | CMake FetchContent |
| **AVX2 intrinsics** | Vectorized XOR-diff in Module 4 | Built-in via `<immintrin.h>`; enable with `-mavx2` in CMake |

### Python (Analysis Only — no algorithm logic)

| Library | Purpose | Install |
|---|---|---|
| `pandas` | Load CSV results, build comparison tables | `pip install pandas` |
| `numpy` | Post-hoc statistics (mean/stddev, ablation deltas) | `pip install numpy` |
| `matplotlib` / `seaborn` | All paper figures | `pip install matplotlib seaborn` |
| `scipy` | Curve fitting, significance tests | `pip install scipy` |

### Documentation & Submission

| Tool | Purpose | Install |
|---|---|---|
| draw.io / Excalidraw | Architecture diagrams | Web-based, free |
| Overleaf / LaTeX | Paper typesetting | Overleaf (web) or `brew install --cask mactex` |
| iThenticate | Plagiarism checking | Institutional license |

---

## Gantt-Style Phase Timeline

```
Week →    0    1    2    3    4    5    6    7    8    9   10
          |    |    |    |    |    |    |    |    |    |    |
Phase 0   ████ █                                              Setup & IP Disclosure
Phase 1        ████                                           Literature & Gaps
Phase 2        ████ ██                                        Design & Architecture
Phase 3 (M1)        ████                                      Implement Module 1
Phase 3 (M2)             ████ ██                              Implement Module 2
Phase 3 (M3)                  ████ ██                         Implement Module 3
Phase 3 (M5)                       ████                       Implement Module 5
Phase 3 (M4)                            ████                  Module 4 Accel.
Phase 4                                  ████                 Evaluation
Phase 5                                       ████            Paper Writing
Phase 6                                            ████ ████  QA, Demo, Submit
                             |                   |
                         Review 2            Review 3
                      (Wk 4 end)          (Wk 10 end)
```

---

*Last updated: 2026-08-30 | Project: AMDEC | Methodology: DSR | Engine: C++ (CMake/Catch2/xxHash/AVX2) | Analysis: Python (pandas/numpy/matplotlib)*
