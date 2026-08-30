# AMDEC: Adaptive Memory Deduplication via Locality-Sensitive Similarity Clustering and Base-Delta Encoding

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

AMDEC is an algorithmic research and evaluation simulation artifact designed to investigate similarity-based deduplication for anonymous memory pages. It combines region-aware stability sampling, MinHash Locality-Sensitive Hashing (LSH) clustering, and compact base-delta storage representations.

---

## 🎯 Central Research Hypothesis

> When anonymous memory heaps contain high proportions of near-duplicate pages (differing only by timestamps, counters, or pointer offsets), similarity clustering combined with base-delta representation achieves meaningful logical memory reduction beyond exact-match deduplicators (e.g., KSM/UKSM), while adaptive sampling prevents CPU scanning overhead on volatile pages.

---

## 🏗️ Architecture Overview

```
[Synthetic Anonymous Workload] (80% near-dup, 7% exact, 13% unique; Zipf writes)
               │
               ▼
[M1: Region-Aware Sampling Engine] (HOT / WARM / COLD tiering, COW feedback)
               │ Candidate Pages
               ▼
[M2: MinHash + LSH Clustering] (128B chunking, K=32 MinHash, b=8 LSH bands, Union-Find)
               │ Similarity Clusters
               ▼
[M3: Base-Delta Storage Manager] (Medoid selection, XOR+RLE diff, metadata-aware size)
               │
               ▼
[M4: Computational Acceleration] (NumPy vectorized vs. scalar diff benchmarks)
               │
               ▼
[M5: Simulated COW Reconstruction] (Simulated page table, 100% round-trip fidelity, latency profiling)
```

---

## 🧪 Scientific Scope & Distinctions

1. **Pure Userspace Simulation:** Runs entirely in Python 3.x (with NumPy, datasketch, scipy) on macOS/Linux with zero kernel modifications or root privileges.
2. **Strict Metric Separation:**
   - **Simulated Logical Memory Savings:** Algorithmic footprint of base pages, unique pages, delta payloads, and metadata overhead.
   - **Python Process RSS:** Memory consumed by the Python runtime (not claimed as deduplication savings).
   - **Simulation Runtime:** Wall-clock timings of the Python simulator on the test machine.
   - **Baseline Comparisons:** KSM/UKSM cited strictly from published peer-reviewed literature.

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated Tests
```bash
pytest tests/ -v
```

### 3. Run Week 2 Workload & Sampling Demo
```bash
python experiments/demo_week2.py
```

---
