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
```

Install required dependencies:

```bash
pip install networkx numpy matplotlib scipy scikit-learn
```

Optional (recommended for full comparison):

```bash
pip install python-louvain
```

For Leiden (recommended via conda):

```bash
conda install -c conda-forge python-igraph leidenalg
```

---

# Usage

Basic run:

```bash
python run_benchmarks.py
```

Custom parameters:

```bash
python run_benchmarks.py   --sizes 500 1000 2000   --mus 0.1 0.3   --repeats 3   --workers 4
```

---

# Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--sizes` | Node sizes for LFR graphs | 500 1000 |
| `--mus` | Mixing parameters (community separation strength) | 0.1 0.3 |
| `--repeats` | Number of repetitions per config | 2 |
| `--workers` | Workers for FKCD WERW stage | 1 |
| `--rho-neigh` | Random walk trials (FKCD neigh) | 200 |
| `--rho-exact` | Random walk trials (FKCD exact) | 400 |
| `--min-community` | Initial minimum community size | 10 |
| `--no-exact` | Skip FKCD exact mode | False |
| `--no-neigh` | Skip FKCD neigh mode | False |
| `--no-louvain` | Skip Louvain | False |
| `--no-leiden` | Skip Leiden | False |

---

# Output

Results are saved in:

```
bench_output/
```

Generated files:

- `bench_results.json` — full raw results
- `bench_results.csv` — tabular version
- `bench_time.png` — runtime comparison
- `bench_nmi.png` — NMI comparison
- `bench_modularity.png` — modularity comparison

---

# Metrics

## Modularity (Q)

Measures structural quality of detected communities.

Higher is better.

## Normalized Mutual Information (NMI)

Measures agreement between detected communities and ground truth.

Range: [0,1]

- 1 → perfect recovery
- 0 → no agreement

---

# Experimental Design

The benchmark uses the **LFR synthetic graph model**, which:

- Generates scale-free degree distributions
- Generates power-law community size distributions
- Allows control over community mixing via parameter μ

Low μ → stronger community structure  
High μ → weaker community structure  

---

# Interpretation Guidelines

When evaluating results:

- Compare **NMI** to assess detection accuracy
- Compare **Modularity** to assess structural quality
- Compare **Runtime** to evaluate scalability
- Compare FKCD exact vs neighbor-restricted tradeoffs

Expected behavior:

- FKCD exact should match or slightly exceed FKCD neigh in accuracy
- FKCD neigh should be significantly faster
- Leiden typically performs very well on NMI
- Louvain is fast but may have lower accuracy in difficult regimes

---

# Reproducibility

All experiments are deterministic with fixed random seeds.

To increase robustness:

```bash
--repeats 5
```

---

# Extending the Benchmark

You may extend this framework to:

- Real-world datasets
- Larger synthetic graphs (10K+ nodes)
- Additional metrics (ARI, Adjusted MI)
- Memory usage profiling
- Stability analysis across seeds

---

# Citation

If you use FKCD in your research, please cite:

De Meo, P., Ferrara, E., Fiumara, G., & Provetti, A. (2011).  
*Generalized Louvain method for community detection in large networks.*

See repository root for BibTeX entries.
