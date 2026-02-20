# FKCD Integrated

**Fast κ-path Community Detection (FKCD)** — integrated, production-ready implementation
suitable for research and large-scale experimentation.

This repository contains an optimized implementation of the FKCD pipeline (WERW-Kpath → proximity → community detection)
with two proximity computation modes (exact pairwise and neighbor-restricted approximation) and optional Numba acceleration.

---

## Contents

- `fkcd_integrated.py` — main CLI script (single-file implementation).
- `requirements.txt` — recommended Python dependencies.
- `pyproject.toml` — packaging metadata (PEP 621) and optional build info.
- `CITATION.cff` — citation metadata for academic reuse.
- `docs/` — documentation skeleton.
- `.github/workflows/` — CI templates (unit tests / lint / basic benchmarking).
- `benchmarks/` — scripts to reproduce timing and scaling experiments.
- `examples/` — usage examples and small sample graphs.
- `LICENSE` — MIT license.

---

## Quick start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the integrated FKCD pipeline (neighbor-restricted approximation, fast):

```bash
python fkcd_integrated.py --input examples/small_test_graph.edgelist --proximity-mode neigh --workers 8 --rho 1000
```

For *exact* proximity (requires `scipy`):

```bash
python fkcd_integrated.py --input examples/small_test_graph.edgelist --proximity-mode exact --batch-size 512
```

## Academic use / citation

If you use this code in published research, please cite the repository (CITATION.cff) and the accompanying paper (if any). See `CITATION.cff` for machine-readable citation metadata.

## Reproducible benchmarks

See `benchmarks/README.md` for instructions to reproduce the timing and scaling experiments on graphs of different sizes.

---

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for guidelines on testing, style, and pull requests.

