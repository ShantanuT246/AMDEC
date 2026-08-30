# Adaptive Memory Deduplication using Similarity Clustering
## Methodology, Technical Architecture & Project Roadmap

**Platform: macOS, pure userspace simulation (Python + optional C). No VM, no kernel access, no Linux dependency.**

---

## 1. Research Methodology

This project follows a **Design Science Research (DSR)** approach — standard for systems/OS-level research:

| Phase | What it means here |
|---|---|
| 1. Problem Identification | Done — your gap analysis (Review 1) |
| 2. Design & Development | Build the artifact — clustering + delta engine (Reviews 2–3) |
| 3. Demonstration | Run it against generated memory workloads |
| 4. Evaluation | Compare against KSM/UKSM published baselines using memory savings, CPU overhead, latency |
| 5. Communication | Paper write-up (Review 3) |

**Why simulation, and why that's legitimate:** your novelty — similarity clustering + base-delta storage — is an **algorithmic contribution**. It can be fully built, run, and evaluated in a userspace simulator without kernel access. This is exactly how most of the papers in your own literature review validate new algorithms before (if ever) they reach production kernel code. A well-evaluated simulator with honest, rigorous metrics is a complete research artifact — not a shortcut.

**What this means concretely:** no `/proc`, no KSM execution, no kernel module, no VM. Memory pages are represented as Python/C objects; write behavior, COW faults, and page access patterns are modeled through a discrete-event simulator; KSM/UKSM comparisons are made against **published figures from literature**, stated explicitly as such in your paper (not same-machine benchmarks).

---

## 2. System Architecture

```
                    ┌─────────────────────────────────┐
                    │   Simulated Anonymous Pages      │
                    │  (synthetic heap-like workload)  │
                    └────────────────┬──────────────────┘
                                     │
                    ┌────────────────▼──────────────────┐
                    │ MODULE 1: Region-Aware             │
                    │ Sampling & Feedback Engine         │
                    │ → filters WHICH pages to scan      │
                    └────────────────┬──────────────────┘
                                     │ candidate pages
                    ┌────────────────▼──────────────────┐
                    │ MODULE 2: Feature Extraction       │
                    │ & LSH Clustering Engine            │
                    │ → groups similar pages             │
                    └────────────────┬──────────────────┘
                                     │ similarity clusters
                    ┌────────────────▼──────────────────┐
                    │ MODULE 3: Base-Delta Storage       │
                    │ & Adaptive Patching Manager        │
                    │ → compresses cluster to base+delta │
                    └────────────────┬──────────────────┘
                                     │
                    ┌────────────────▼──────────────────┐
                    │ MODULE 4: Acceleration Benchmark   │
                    │ (vectorized diff, simulated DSA)   │
                    └────────────────┬──────────────────┘
                                     │
                    ┌────────────────▼──────────────────┐
                    │ MODULE 5: Simulated COW Fault      │
                    │ Handler & Reconstruction Engine    │
                    └────────────────────────────────────┘
```

Each module maps 1:1 to one of your four objectives — this makes your "objectives achieved" table for Review 3 straightforward to fill in.

| Module | Objective | Gap addressed |
|---|---|---|
| 1. Sampling & Feedback | Objective 3 | Gap 2 (CPU overhead), Gap 3 (semantic ignorance) |
| 2. LSH Clustering | Objective 1 | Gap 1 (binary exactness), Gap 4 (unguided delta search) |
| 3. Base-Delta Storage | Objective 2 | Gap 4 (inefficient patching) |
| 4. Acceleration Benchmark | Supports Objective 1/2 | Gap 2 (CPU overhead) |
| 5. COW Simulation | Ties system together | Demonstrates correctness/safety |
| All combined | Objective 4 (30–50% RAM recovery) | — |

---

## 3. Module-Level Design

### Module 1 — Region-Aware Sampling & Feedback Engine

**Purpose:** decide *which* pages are worth scanning before spending CPU on similarity checks. This is what keeps overhead low and directly answers Gap 2.

**Technique:**
- Generate synthetic page objects with attributes: `alloc_time`, `last_write_time`, `write_count`, `content` (4096 bytes).
- Model write behavior with a configurable distribution — a fraction of pages get "touched" every simulated tick, following a Zipf or exponential distribution (mirrors real heaps: a few pages are hot, most are cold).
- Classify pages into **stability tiers**: `hot` (frequently written → skip), `warm` (moderate → low-priority scan), `cold` (long-lived, rarely modified → high-priority scan).
- Maintain a priority queue; only `cold`/`warm` pages flow into Module 2.

**Pseudocode:**
```python
for page in tracked_pages:
    age = current_time - page.alloc_time
    write_rate = page.write_count / age
    if write_rate > THRESHOLD_HOT:
        tier = "hot"       # skip
    elif age > MIN_AGE and write_rate < THRESHOLD_COLD:
        tier = "cold"      # high priority
    else:
        tier = "warm"      # low priority
    scan_queue.enqueue(page, priority=tier)
```

---

### Module 2 — Feature Extraction & LSH Clustering Engine

**Purpose:** replace exact-hash comparison with a similarity metric that groups near-duplicate pages without pairwise byte comparison (O(n²) and infeasible at scale).

**Pipeline:**
1. **Chunking** — split each 4096-byte page into fixed-size sub-blocks (64B–128B; tune experimentally).
2. **Feature generation** — compute a MinHash signature per page from its sub-blocks.
3. **LSH banding** — pages sharing enough minhash bands land in the same bucket → candidate pairs, turning O(n²) comparison into O(n) bucket lookups.
4. **Cluster assignment** — within a bucket, use a Jaccard similarity threshold + union-find to finalize clusters.

**Why LSH:** it's the established tool for large-scale near-duplicate detection (used in SiLo, which you already cite) — this is your strongest technical link back to Category 4 of your literature review.

**Pseudocode:**
```python
def extract_signature(page_bytes, block_size=128, num_hashes=32):
    blocks = chunk(page_bytes, block_size)
    return [minhash(b, num_hashes) for b in blocks]

def lsh_buckets(signature, bands=8):
    return [hash(band) for band in split(signature, bands)]

for page in scan_queue:
    sig = extract_signature(page.content)
    for bucket_id in lsh_buckets(sig):
        LSH_table[bucket_id].add(page.id)

clusters = union_find_merge(LSH_table)
```

---

### Module 3 — Base-Delta Storage & Adaptive Patching Manager

**Purpose:** within each cluster, pick a representative page and store the rest as compact diffs.

**Technique:**
- **Base selection:** page with lowest average byte-distance to others in cluster (or simplest: oldest/most stable page).
- **Delta computation:** lightweight byte-level diff — XOR + run-length encoding of differing byte ranges (cheaper than full `bsdiff`).
- **Adaptive thresholding:**
```python
if delta_size > MAX_DELTA_SIZE or page.write_frequency > REWRITE_THRESHOLD:
    fallback_to_compression_or_leave_unmerged(page)
else:
    store_as(base_ref=cluster.base_id, delta=delta_patch)
```
- **Reconstruction on read:** apply delta to base on access — time this, it's a key metric proving you're not trading memory for unusable latency.

---

### Module 4 — Acceleration Benchmark

**Purpose:** demonstrate the CPU cost of delta computation can be reduced, addressing Gap 2 further.

**Realistic scope:** you won't have real Intel DSA hardware. Handle this honestly:
- Benchmark delta computation with plain Python loops vs. **NumPy-vectorized XOR-diff** (a real, measurable speedup you can produce on your Mac).
- Optionally extrapolate expected DSA/hardware-accelerator gains from published throughput specs, clearly labeled in your paper as a **projected, not measured**, figure.
- If you want a stronger claim: implement the diff step in C (compiles natively via `clang`/Xcode Command Line Tools) and compare against the Python version — a legitimate, real "acceleration" result.

---

### Module 5 — Simulated COW Fault Handler & Reconstruction Engine

**Purpose:** demonstrate correctness — prove merged/delta-stored pages behave safely under writes.

**Technique:**
- Model "a process writes to page X" as a function call in your test harness. On trigger:
  1. Look up whether page X is `merged` (exact COW) or `delta-stored`.
  2. If delta-stored, reconstruct full page from `base + delta` — time this (real, measurable work).
  3. Allocate a new "private" page object, copy reconstructed data in.
  4. Update your simulated page table (a dict mapping page ID → storage location).
- This gives you a genuine, measurable **reconstruction latency** metric without kernel access.
- **Optional realism boost:** use Python's `mmap` module or C's `mprotect`/`SIGSEGV` handler (both work natively on macOS) to trigger *real* page-protection faults on actual memory pages in your test harness, caught via a signal handler. This makes your Review 3 demo look "live" without needing kernel modification.

---

## 4. Evaluation Methodology

Decide this **before** coding, so metrics are principled rather than fitted to whatever results you get.

**Baselines:** standard KSM and UKSM — using **published figures from the literature you already cite**, explicitly stated as a literature-baseline comparison, not a same-machine benchmark.

**Datasets:**
- **Primary:** synthetic anonymous-heap workloads — generate many similar structs/objects with small randomized fields (pointers, timestamps, counters), mirroring the paper's own claim about anonymous memory (up to 86% near-duplicate, <7% exact match). Fully reproducible, and the standard approach absent live kernel access.
- **Optional supporting figure (not your core dataset):** macOS-native tools — `vm_stat` (system-wide VM stats) and `vmmap <pid>` (per-process memory breakdown) — for a real motivating screenshot in your Introduction showing anonymous memory's real-world footprint. Not used for algorithm evaluation, since Mac tools don't expose raw page bytes the way `/proc` does on Linux.

**Metrics:**

| Metric | How to measure |
|---|---|
| Memory saved (%) | (baseline size − your system size) / baseline size |
| CPU overhead | scanning + clustering + delta time as % of total runtime |
| Reconstruction latency | time to rebuild a page on simulated write-fault |
| Scan efficiency | pages scanned vs. pages actually merged (precision of Module 1) |
| Cluster quality | avg. cluster size, false-positive rate (dissimilar pages wrongly clustered) |

**Ablation study (panels specifically look for this):**
- Disable Module 1 (region-aware filtering) → show CPU overhead spikes.
- Use a fixed threshold instead of adaptive thresholding → show memory savings drop or latency spikes.
- This proves each component earns its place, not just the system as a whole.

---

## 5. Tooling

- **Core language:** Python — fast to build, `datasketch` library gives you MinHash+LSH without implementing from scratch, runs natively on macOS with zero setup.
- **Performance module (optional):** C for the delta-diff step, compiled via `clang` (ships with Xcode Command Line Tools) — gives you a real "before/after" acceleration number for Module 4.
- **Visualization:** matplotlib/seaborn for cluster-size distributions, memory-saved-over-time graphs, CPU overhead comparison charts vs. literature-cited KSM/UKSM numbers.
- **Version control:** GitHub from day one — commit history is your proof of progressive work for both reviews.

---

## 6. Project Roadmap

Assuming **~8–10 weeks** from now to Review 3 (adjust dates to your actual calendar — the structure holds regardless of exact length).

### Phase A — Setup & Core Pipeline (Weeks 1–2) → builds toward Review 2
- [ ] Set up Python project structure + GitHub repo
- [ ] Build the synthetic page generator (Module 1 input): realistic heap-like data with controllable similarity ratios
- [ ] Implement Module 1 — sampling/tiering logic
- [ ] Write unit tests for Module 1 (does it correctly skip hot pages, prioritize cold pages?)

### Phase B — Clustering Engine (Weeks 3–4) → Review 2 target: 60% complete here
- [ ] Implement Module 2 — chunking, MinHash signature generation
- [ ] Integrate `datasketch` LSH indexing
- [ ] Implement union-find cluster assembly
- [ ] Validate: run on synthetic data, confirm clusters actually group similar pages (sanity-check manually on a few clusters)
- [ ] **Milestone: Modules 1+2 producing real cluster output on test data — this is your 60% checkpoint**
- [ ] Prepare architecture diagram (draw.io) for Review 2 presentation
- [ ] Prepare Review 2 slides: problem recap → architecture → live demo of clustering on sample data → preliminary numbers (candidate reduction ratio, cluster size distribution)

### Phase C — Base-Delta Storage (Weeks 5–6)
- [ ] Implement Module 3 — base selection, delta computation (XOR + RLE)
- [ ] Implement adaptive thresholding logic
- [ ] Implement Module 5 — simulated COW fault handler + reconstruction, with latency timing
- [ ] Integration test: full pipeline end-to-end (generate pages → sample → cluster → base-delta → simulate writes → measure reconstruction)

### Phase D — Acceleration & Evaluation (Week 7)
- [ ] Implement Module 4 — NumPy-vectorized diff benchmark (and C version if time allows)
- [ ] Run full evaluation suite: memory saved, CPU overhead, reconstruction latency, scan efficiency, cluster quality
- [ ] Run ablation studies (Module 1 disabled, fixed vs. adaptive threshold)
- [ ] Generate all result graphs/tables
- [ ] Gather published KSM/UKSM baseline numbers from your cited papers for comparison table

### Phase E — Paper Writing (Week 8)
- [ ] Write Methodology section (you already have this doc as a base)
- [ ] Write Results & Discussion from Phase D output
- [ ] Condense your Review 1 literature review into a Related Work section
- [ ] Write Introduction and Abstract (last, once everything else is done)
- [ ] Select target venue (Scopus/UGC-CARE indexed conference or journal matching your scope)
- [ ] Format to venue's template (IEEE two-column / Springer LNCS)

### Phase F — Polish & Submission (Weeks 9–10) → Review 3
- [ ] Run plagiarism check (<10%) — rewrite any flagged sections in your own words
- [ ] Run AI-content check (<5%) — manually rewrite anything AI-assisted to remove uniform sentence rhythm
- [ ] Record backup demo video (in case live demo fails)
- [ ] Prepare 2–3 live demo scenarios + 1 edge case
- [ ] Final read-through: does every section trace back to your novelty claim?
- [ ] Submit to chosen venue / finalize Review 3 report and presentation

---

## 7. Review Checkpoint Mapping

| Review 2 (60%) | Review 3 (Final) |
|---|---|
| Module 1 + 2 implemented, producing clusters on synthetic data | Module 3 + 5 fully functional (base-delta + reconstruction) |
| Architecture diagram finalized | Module 4 benchmarked |
| Preliminary results: candidate reduction ratio, cluster size stats | Full evaluation vs. KSM/UKSM baselines, all 5 metrics |
| GitHub commit history showing progressive build | Ablation study results |
| | Paper drafted in IEEE/Springer format, plagiarism <10%, AI-content <5% |
| | Live demo + backup video ready |
