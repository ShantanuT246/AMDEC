# AMDEC — Review 2 Presentation Package
### Adaptive Memory Deduplication via Locality-Sensitive Similarity Clustering

---

## 🎯 Slide 1: Title & Problem Context
- **Title:** AMDEC: Adaptive Memory Deduplication using Locality-Sensitive Similarity Clustering
- **Presenter:** Shantanu Tapole
- **Core Problem:** Standard OS/Hypervisor page-sharing engines (KSM, UKSM) only merge **100% byte-identical** pages.
- **The Gap:** In anonymous heap memory (VMs, microservices, container heaps), up to **86% of pages are near-duplicate** (sharing structures, templates, but differing by timestamps, pointers, or counters) while $<7\%$ are exact matches. Exact dedup leaves ~80% of potential savings unexploited.

---

## 🏗️ Slide 2: 5-Module Architecture

```
[Synthetic Anonymous Workload] (80% near-dup, 7% exact, 13% unique; Zipf write traffic)
               │
               ▼
[Module 1: Region-Aware Sampling Engine] ──► Skips HOT/Volatile pages, penalizes COW breaks
               │ Prioritized Candidate Queue
               ▼
[Module 2: Feature Extraction & LSH Clustering] ──► 128B chunking, MinHash (K=32), LSH (b=8), Union-Find
               │ Similarity Clusters (J >= 0.70)
               ▼
[Module 3: Base-Delta Storage Manager] ──► Medoid base selection, XOR + RLE diffs, adaptive threshold
               │
               ▼
[Module 4: Computational Acceleration] ──► NumPy vectorized vs. scalar benchmarks
               │
               ▼
[Module 5: Simulated COW Reconstruction] ──► 100% round-trip fidelity, latency profiling
```

---

## 📊 Slide 3: Review 2 Technical Deliverables & Experimental Evidence

### 1. Workload & Module 1 (Sampling Efficacy)
- **Workload:** 10,000 pages (39.1 MB memory pool).
- **Write Skew:** Zipfian distribution ($\alpha = 1.3$).
- **Candidate Filtering:** Module 1 safely filters 8.2% volatile hot pages (~820 pages skipped, 91.8% passed), saving CPU scanning time.

### 2. Module 2 (MinHash-LSH Clustering Efficacy)
- **Search Space Reduction:** Eliminates **99.85%** of pairwise comparisons (evaluates ~65.4K candidate pairs instead of 42.1M post-M1 possible pairs; 99.87% vs 50.0M un-filtered pairs).
- **Clustering Throughput:** $>3,300\text{ pages/second}$ in Python.
- **Cluster Quality:** **98.9% Ground-Truth Cluster Purity** (zero false-positive cross-cluster pollution).

### 3. Module 3 (Base-Delta Compression Efficacy)
- **Logical Memory Savings:** **$73.6\%$** overall memory reduction ($3.79\times$ compression) across the full pipeline ($80.9\%$ on near-duplicate clusters).
- **Reconstruction Fidelity:** $100\%$ byte-for-byte exact equality verified.
- **Automated Tests:** **37/37 passing tests** in `pytest`.

---

## 🚀 Slide 4: Live Demo Script (3-Step Walkthrough)
1. Run `python experiments/demo_review2.py`
2. **Step 1:** Show ground-truth workload composition (80% near-duplicate, 7% exact, 13% unique).
3. **Step 2:** Show Module 1 stability tiering & candidate queue output.
4. **Step 3:** Show Module 2 LSH cluster assembly, cluster size histogram, and Module 3 base-delta storage savings.

---

## 🗺️ Slide 5: Roadmap to Review 3 (Weeks 5–10)
- **Weeks 5–6:** Module 5 Simulated COW Fault Handler & Reconstruction latency profiling (P95, P99).
- **Week 7:** Module 4 Vectorized acceleration benchmarks, multi-parameter sweeps (mutation rate vs savings), and 3 ablation studies.
- **Week 8:** Statistical analysis, confidence intervals, publication-grade figures.
- **Weeks 9–10:** IEEE paper drafting, plagiarism QA ($<10\%$), final Review 3 defense.
