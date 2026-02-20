# FKCD Benchmark Suite

This benchmark framework evaluates the performance of:

1. **FKCD (Exact Mode)**
2. **FKCD (Neighbor-Restricted Approximation)**
3. **Louvain (python-louvain)**
4. **Leiden (igraph + leidenalg)**

on synthetic graphs with *planted ground-truth communities*.

The evaluation focuses on:

- **Modularity (Q)**
- **Normalized Mutual Information (NMI)**
- **Runtime (seconds)**

---

# Overview

The benchmarking harness:

- Generates **LFR benchmark graphs**
- Computes community detection results using each method
- Measures detection accuracy (NMI)
- Measures modularity
- Measures runtime
- Produces plots and structured result files

The LFR benchmark generator is provided by NetworkX:

https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.LFR_benchmark_graph.html

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate