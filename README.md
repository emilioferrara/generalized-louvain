# FKCD Integrated  
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

## Batch Size (Exact Mode)

```bash
--batch-size 512
```

Controls memory/speed tradeoff during exact proximity computation.

Larger = faster but higher memory usage.

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

## Output

```bash
--output-partition output.json
```

Produces:

```json
{
  "method": "...",
  "modularity": 0.42,
  "partition": { "node": community_id }
}
```

---

## Example Runs

### Fast Large-Scale Mode

```bash
python fkcd_integrated.py   --input graph.edgelist   --proximity-mode neigh   --workers 8   --rho 1000
```

---

### Exact Academic Mode

```bash
python fkcd_integrated.py   --input graph.edgelist   --proximity-mode exact   --batch-size 512   --rho 500
```

---

## Performance Notes

| Graph Size | Recommended Mode |
|------------|------------------|
| < 10K nodes | `exact` |
| 10K–100K | `neigh` |
| 100K–1M+ | `neigh` + numba + Louvain |

Exact mode complexity is approximately:

O(|E| · |V|)

Neighbor-restricted mode reduces this substantially in sparse networks.

---

# Academic References

## Required Citation

De Meo, P., Ferrara, E., Fiumara, G., & Provetti, A. (2011).  
Generalized Louvain method for community detection in large networks.  
11th International Conference on Intelligent Systems Design and Applications (ISDA), IEEE, pp. 88–93.

## Related Work

De Meo, P., Ferrara, E., Fiumara, G., & Ricciardello, A. (2012).  
A novel measure of edge centrality in social networks.  
Knowledge-Based Systems, 30, 136–150.

De Meo, P., Ferrara, E., Fiumara, G., & Provetti, A. (2014).  
Mixing local and global information for community detection in large networks.  
Journal of Computer and System Sciences, 80(1), 72–87.

---

## License

MIT License

---

## Reproducibility

See `/benchmarks` for scripts to reproduce runtime experiments.

---

## Contributing

Pull requests and reproducibility reports are welcome.
