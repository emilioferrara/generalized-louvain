# Generalized Louvain Method for Community Detection in Large Networks
**Fast κ-path Community Detection (FKCD)**  
Scalable and research-grade implementation

---

## Overview

This repository contains an integrated, production-ready implementation of the **Fast κ-path Community Detection (FKCD)** framework.

The pipeline consists of:

1. **WERW-Kpath edge centrality**
2. **Edge proximity computation**
3. **Weighted graph construction**
4. **Community detection via Generalized Louvain / modularity maximization**

Two proximity computation modes are provided:

- `exact` — full pairwise proximity (as defined in the original formulation)
- `neigh` — neighbor-restricted approximation (faster, scalable variant)

Optional Numba acceleration is supported.

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (recommended for performance):

```bash
pip install scipy
pip install numba
pip install python-louvain
```

---

## Command-Line Usage

```bash
python fkcd_integrated.py [OPTIONS]
```

### Required Argument (if not using demo)

```bash
--input PATH_TO_EDGE_LIST
```

If omitted, the script runs on the Zachary Karate Club demo graph.

---

## Core Parameters

### Random Walk Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--kappa` | Maximum walk length | 5 |
| `--rho` | Number of walk trials | m−1 (edges−1) |
| `--workers` | Parallel workers | auto-detected |
| `--seed` | Random seed | 42 |

---

## Proximity Modes

### 1️⃣ Exact Mode (default)

```bash
--proximity-mode exact
```

Implements:

r_uv = sqrt( sum_k (L(u,k) - L(k,v))^2 / d(k) )

- Full pairwise sum over all nodes
- Vectorized via sparse matrix batching
- Requires `scipy`
- Computationally expensive for large graphs

Recommended for:
- Reproducibility
- Small to medium graphs
- Academic validation

---

### 2️⃣ Neighbor-Restricted Mode (Fast Approximation)

```bash
--proximity-mode neigh
```

Computes proximity only over:

k in N(u) ∪ N(v)

Advantages:
- Dramatically faster
- Scales to 100K–1M+ nodes
- Empirically similar community structure
- Matches practical approximation strategies

Recommended for:
- Large-scale networks
- Performance benchmarking
- Applied research

---

## Numba Acceleration

Enable strict Numba usage:

```bash
--use-numba
```

Behavior:

- If Numba is installed → uses JIT acceleration
- If not installed → raises an error
- Without flag → uses Numpy fallback if Numba unavailable

---

## Community Detection

By default:

- Uses `python-louvain` if available
- Falls back to NetworkX greedy modularity otherwise

To prefer Louvain:

```bash
--prefer-louvain
```

---

# Academic References

## Required Citation

De Meo, P., Ferrara, E., Fiumara, G., & Provetti, A. (2011).  
*Generalized Louvain method for community detection in large networks.*  
Proceedings of the 11th International Conference on Intelligent Systems Design and Applications (ISDA), IEEE, pp. 88–93.

### BibTeX

```bibtex
@inproceedings{demeo2011generalized,
  title        = {Generalized Louvain method for community detection in large networks},
  author       = {De Meo, Pasquale and Ferrara, Emilio and Fiumara, Giacomo and Provetti, Alessandro},
  booktitle    = {2011 11th International Conference on Intelligent Systems Design and Applications},
  pages        = {88--93},
  year         = {2011},
  organization = {IEEE},
  doi          = {10.1109/ISDA.2011.6121636}
}
```

---

## Related Work — Edge Centrality

De Meo, P., Ferrara, E., Fiumara, G., & Ricciardello, A. (2012).  
*A novel measure of edge centrality in social networks.*  
Knowledge-Based Systems, 30, 136–150.

### BibTeX

```bibtex
@article{demeo2012novel,
  title   = {A novel measure of edge centrality in social networks},
  author  = {De Meo, Pasquale and Ferrara, Emilio and Fiumara, Giacomo and Ricciardello, Antonio},
  journal = {Knowledge-Based Systems},
  volume  = {30},
  pages   = {136--150},
  year    = {2012},
  doi     = {10.1016/j.knosys.2012.01.007},
  publisher = {Elsevier}
}
```

---

## Related Work — Mixing Local and Global Information

De Meo, P., Ferrara, E., Fiumara, G., & Provetti, A. (2014).  
*Mixing local and global information for community detection in large networks.*  
Journal of Computer and System Sciences, 80(1), 72–87.

### BibTeX

```bibtex
@article{demeo2014mixing,
  title   = {Mixing local and global information for community detection in large networks},
  author  = {De Meo, Pasquale and Ferrara, Emilio and Fiumara, Giacomo and Provetti, Alessandro},
  journal = {Journal of Computer and System Sciences},
  volume  = {80},
  number  = {1},
  pages   = {72--87},
  year    = {2014},
  doi     = {10.1016/j.jcss.2013.07.012},
  publisher = {Elsevier}
}
```

---

## 📄 Papers

Full-text PDFs of the original research contributions are included in this repository:

- **ISDA 2011**  
  *Generalized Louvain Method for Community Detection in Large Networks*  
  [`paper/ISDA2011_Generalized_Louvain.pdf`](paper/ISDA2011_Generalized_Louvain.pdf)

- **Knowledge-Based Systems 2012**  
  *A Novel Measure of Edge Centrality in Social Networks*  
  [`paper/KBS2012_Kappa_Path_Edge_Centrality.pdf`](paper/KBS2012_Kappa_Path_Edge_Centrality.pdf)

- **Journal of Computer and System Sciences 2014**  
  *Mixing Local and Global Information for Community Detection in Large Networks*  
  [`paper/JCSS2014_Mixing_Local_Global.pdf`](paper/JCSS2014_Mixing_Local_Global.pdf)


---


## License

MIT License

---

## Reproducibility

See `/benchmarks` for scripts to reproduce runtime experiments.

---

## Contributing

Pull requests and reproducibility reports are welcome.
