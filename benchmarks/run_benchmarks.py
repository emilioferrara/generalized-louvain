#!/usr/bin/env python3
"""
run_benchmarks.py (with modularity plotting integrated)

Benchmarking harness to compare:
 - FKCD exact
 - FKCD neighbor-restricted (approx)
 - Louvain (python-louvain)
 - Leiden (igraph + leidenalg)

Outputs (in outdir, default ./bench_output):
  - bench_results.json
  - bench_results.csv
  - bench_time.png
  - bench_nmi.png
  - bench_modularity.png   <-- NEW (grouped bar chart mean ± std)
"""
import argparse
import json
import time
import traceback
from pathlib import Path
import csv
import sys
import math
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Optional imports (handled below)
try:
    from sklearn.metrics import normalized_mutual_info_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except Exception:
    HAS_LOUVAIN = False

try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except Exception:
    HAS_LEIDEN = False

# Try importing functions from fkcd_integrated.py (must be in same folder)
try:
    import fkcd_integrated as fkcd_mod
    HAS_FKCD = True
except Exception:
    import importlib.util
    fkcd_path = Path(__file__).parent / "../fkcd_integrated.py"
    if fkcd_path.exists():
        spec = importlib.util.spec_from_file_location("fkcd_integrated", str(fkcd_path))
        fkcd_mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(fkcd_mod)
            sys.modules["fkcd_integrated"] = fkcd_mod
            HAS_FKCD = True
        except Exception:
            print("Failed to import fkcd_integrated.py:")
            traceback.print_exc()
            HAS_FKCD = False
    else:
        HAS_FKCD = False

if not HAS_FKCD:
    raise RuntimeError("fkcd_integrated.py not found or failed to import. Place it in the same folder and retry.")

# Extract FKCD functions
werw_kpath_parallel = getattr(fkcd_mod, "werw_kpath_parallel", None)
proximity_neighbor_restricted = getattr(fkcd_mod, "proximity_neighbor_restricted", None)
proximity_exact = getattr(fkcd_mod, "proximity_exact", None)
detect_communities = getattr(fkcd_mod, "detect_communities", None)
nx_to_csr = getattr(fkcd_mod, "nx_to_csr", None)
csr_neighbors = getattr(fkcd_mod, "csr_neighbors", None)
_werw_worker = getattr(fkcd_mod, "_werw_worker", None)

if any(f is None for f in [werw_kpath_parallel, proximity_neighbor_restricted, detect_communities, nx_to_csr]):
    print("Warning: Some FKCD internals were not found in fkcd_integrated.py. Make sure the file exports werw_kpath_parallel, proximity_neighbor_restricted, detect_communities, nx_to_csr.")
    # still continue; some features may be unavailable

# helpers
def part_to_labels(part, nodes):
    return [part.get(n, -1) for n in nodes]

def compute_nmi(true_labels, pred_labels):
    if not HAS_SKLEARN:
        return None
    try:
        return float(normalized_mutual_info_score(true_labels, pred_labels))
    except Exception:
        return None

# Renamed wrappers to avoid name collisions
def run_louvain_method(G):
    if not HAS_LOUVAIN:
        return None, (None, None)
    t0 = time.perf_counter()
    part = community_louvain.best_partition(G, weight='weight')
    t1 = time.perf_counter()
    # modularity
    try:
        from networkx.algorithms.community.quality import modularity
        comms = {}
        for n,c in part.items():
            comms.setdefault(c, []).append(n)
        Q = modularity(G, list(comms.values()), weight='weight')
    except Exception:
        Q = None
    return part, (t1 - t0, Q)

def run_leiden_method(G):
    if not HAS_LEIDEN:
        return None, (None, None)
    mapping = {n: i for i,n in enumerate(G.nodes())}
    edges = [(mapping[u], mapping[v]) for u,v in G.edges()]
    g = ig.Graph(n=len(mapping), edges=edges, directed=False)
    # weights
    weights = [G[u][v].get('weight', 1.0) for u,v in G.edges()]
    try:
        g.es['weight'] = weights
    except Exception:
        pass
    t0 = time.perf_counter()
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight')
    t1 = time.perf_counter()
    part = {}
    # convert igraph vertex indices back to original nodes
    inv_map = {v:k for k,v in mapping.items()}
    for cid, cluster in enumerate(partition):
        for v in cluster:
            node = inv_map[int(v)]
            part[node] = cid
    # modularity
    try:
        from networkx.algorithms.community.quality import modularity
        comms = {}
        for n,cid in part.items():
            comms.setdefault(cid, []).append(n)
        Q = modularity(G, list(comms.values()), weight='weight')
    except Exception:
        Q = None
    return part, (t1 - t0, Q)

def run_fkcd_neigh(G, rho=200, kappa=5, workers=1, seed=123):
    if werw_kpath_parallel is None or proximity_neighbor_restricted is None:
        raise RuntimeError("FKCD neigh internals not available in fkcd_integrated.py")
    A, node_to_idx, idx_to_node = nx_to_csr(G)
    t0 = time.perf_counter()
    omega, edges, edge_index = werw_kpath_parallel(A, kappa=kappa, rho=rho, workers=workers, seed=seed)
    t1 = time.perf_counter()
    werw_time = t1 - t0
    t0 = time.perf_counter()
    proximities = proximity_neighbor_restricted(A, omega, edges, edge_index, {i: max(1, A.indptr[i+1]-A.indptr[i]) for i in range(A.shape[0])})
    t1 = time.perf_counter()
    prox_time = t1 - t0
    nodes_order = list(G.nodes())
    H = nx.Graph()
    H.add_nodes_from(nodes_order)
    for eidx,(u,v) in enumerate(edges):
        u_orig = nodes_order[u]; v_orig = nodes_order[v]
        H.add_edge(u_orig, v_orig, weight=float(proximities.get(eidx, 0.0)))
    part, method = detect_communities(H, prefer_louvain=True)
    try:
        from networkx.algorithms.community.quality import modularity
        comms = {}
        for n,cid in part.items():
            comms.setdefault(cid, []).append(n)
        Q = modularity(H, list(comms.values()), weight='weight')
    except Exception:
        Q = None
    return part, (werw_time + prox_time, Q)

def run_fkcd_exact(G, rho=400, kappa=5, workers=1, seed=123, batch_size=256):
    if proximity_exact is None:
        return None, (None, None)
    A, node_to_idx, idx_to_node = nx_to_csr(G)
    t0 = time.perf_counter()
    omega, edges, edge_index = werw_kpath_parallel(A, kappa=kappa, rho=rho, workers=workers, seed=seed)
    t1 = time.perf_counter()
    werw_time = t1 - t0
    t0 = time.perf_counter()
    proximities = proximity_exact(A, omega, edges, edge_index, {i: max(1, A.indptr[i+1]-A.indptr[i]) for i in range(A.shape[0])}, batch_size=batch_size)
    t1 = time.perf_counter()
    prox_time = t1 - t0
    nodes_order = list(G.nodes())
    H = nx.Graph()
    H.add_nodes_from(nodes_order)
    for eidx,(u,v) in enumerate(edges):
        H.add_edge(nodes_order[u], nodes_order[v], weight=float(proximities.get(eidx, 0.0)))
    part, method = detect_communities(H, prefer_louvain=True)
    try:
        from networkx.algorithms.community.quality import modularity
        comms = {}
        for n,cid in part.items():
            comms.setdefault(cid, []).append(n)
        Q = modularity(H, list(comms.values()), weight='weight')
    except Exception:
        Q = None
    return part, (werw_time + prox_time, Q)

def generate_lfr_with_fallback(n, mu, avg_degree=8, min_comm=10, tau1=3.0, tau2=1.5, seed=42, max_attempts=5):
    """
    Try to generate an LFR graph; if it fails because min_community is too large,
    progressively reduce min_community until success or until floor (4).
    """
    min_c = min_comm
    for attempt in range(max_attempts):
        try:
            G = nx.generators.community.LFR_benchmark_graph(n, tau1=tau1, tau2=tau2, mu=mu,
                                                            average_degree=avg_degree, min_community=min_c, seed=seed+attempt)
            # success
            return G, min_c
        except Exception as e:
            msg = str(e)
            # try to detect cause and reduce min_community
            min_c = max(4, int(min_c * 0.75))
            print(f"LFR gen attempt {attempt} failed (min_community now {min_c}): {msg}")
            continue
    # last attempt
    try:
        G = nx.generators.community.LFR_benchmark_graph(n, tau1=tau1, tau2=tau2, mu=mu,
                                                        average_degree=avg_degree, min_community=min_c, seed=seed+max_attempts)
        return G, min_c
    except Exception as e:
        raise RuntimeError(f"LFR generation failed after retries: {e}")

def run_single_experiment(n, mu, avg_degree=8, min_community=10, repeats=1,
                          run_exact=True, run_neigh=True, run_louvain=True, run_leiden=True,
                          rho_neigh=200, rho_exact=400, workers=1, seed_base=123):
    recs = []
    for r in range(repeats):
        seed = seed_base + r
        print(f"[exp] n={n} mu={mu} repeat={r} seed={seed}")
        # generate LFR graph with fallback
        try:
            G, used_min_comm = generate_lfr_with_fallback(n, mu, avg_degree=avg_degree, min_comm=min_community, seed=seed)
            print(f"Generated LFR graph with min_community={used_min_comm}")
        except Exception as e:
            print("Failed to generate LFR graph:", e)
            traceback.print_exc()
            continue

        # Build ground truth labels
        communities = {}
        for node, data in G.nodes(data=True):
            comms = data.get('community')
            if isinstance(comms, set):
                # choose deterministic element
                communities[node] = sorted(list(comms))[0]
            else:
                communities[node] = comms
        unique_comms = sorted(list({communities[n] for n in G.nodes()}))
        comm_to_cid = {c:i for i,c in enumerate(unique_comms)}
        true_part = {n: comm_to_cid[communities[n]] for n in G.nodes()}
        nodes_order = list(G.nodes())
        true_labels = part_to_labels(true_part, nodes_order)

        # Louvain
        louv_time = louv_Q = louv_nmi = None
        if run_louvain and HAS_LOUVAIN:
            try:
                part, (t,LQ) = run_louvain_method(G)
                louv_time = t; louv_Q = LQ
                louv_labels = part_to_labels(part, nodes_order)
                louv_nmi = compute_nmi(true_labels, louv_labels)
            except Exception:
                print("Louvain run failed:")
                traceback.print_exc()

        # Leiden
        leiden_time = leiden_Q = leiden_nmi = None
        if run_leiden and HAS_LEIDEN:
            try:
                part, (t,LQ) = run_leiden_method(G)
                leiden_time = t; leiden_Q = LQ
                leiden_labels = part_to_labels(part, nodes_order)
                leiden_nmi = compute_nmi(true_labels, leiden_labels)
            except Exception:
                print("Leiden run failed:")
                traceback.print_exc()

        # FKCD neigh
        fkcd_time = fkcd_Q = fkcd_nmi = None
        if run_neigh:
            try:
                part, (t,FQ) = run_fkcd_neigh(G, rho=rho_neigh, kappa=5, workers=workers, seed=seed)
                fkcd_time = t; fkcd_Q = FQ
                fkcd_labels = part_to_labels(part, nodes_order)
                fkcd_nmi = compute_nmi(true_labels, fkcd_labels)
            except Exception:
                print("FKCD (neigh) failed:")
                traceback.print_exc()

        # FKCD exact (only if available and n reasonable)
        fkcd_exact_time = fkcd_exact_Q = fkcd_exact_nmi = None
        if run_exact and proximity_exact is not None:
            if n <= 1000:
                try:
                    part, (t,FQ) = run_fkcd_exact(G, rho=rho_exact, kappa=5, workers=workers, seed=seed, batch_size=128)
                    fkcd_exact_time = t; fkcd_exact_Q = FQ
                    fkcd_exact_labels = part_to_labels(part, nodes_order)
                    fkcd_exact_nmi = compute_nmi(true_labels, fkcd_exact_labels)
                except Exception:
                    print("FKCD exact failed:")
                    traceback.print_exc()
            else:
                print("Skipping FKCD exact for n >", 1000)

        rec = {
            "n": n, "mu": mu, "repeat": r, "seed": seed,
            "louv_time": louv_time, "louv_Q": louv_Q, "louv_nmi": louv_nmi,
            "leiden_time": leiden_time, "leiden_Q": leiden_Q, "leiden_nmi": leiden_nmi,
            "fkcd_time": fkcd_time, "fkcd_Q": fkcd_Q, "fkcd_nmi": fkcd_nmi,
            "fkcd_exact_time": fkcd_exact_time, "fkcd_exact_Q": fkcd_exact_Q, "fkcd_exact_nmi": fkcd_exact_nmi,
            "has_louvain": HAS_LOUVAIN, "has_leiden": HAS_LEIDEN, "has_sklearn": HAS_SKLEARN
        }
        recs.append(rec)
    return recs

def aggregate_and_plot(all_results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bench_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    # CSV
    keys = sorted(list({k for r in all_results for k in r.keys()}))
    with open(out_dir / "bench_results.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in keys})
    # Plotting - runtime and NMI (existing)
    labels = [f"{r['n']},mu={r['mu']},rep={r['repeat']}" for r in all_results]
    ind = np.arange(len(all_results))
    def extract(key):
        return [ (r.get(key) if r.get(key) is not None else np.nan) for r in all_results ]

    plt.figure(figsize=(10,5))
    # plot only if values exist
    louv_time_vals = extract("louv_time")
    if any([not math.isnan(x) for x in louv_time_vals if x is not None]):
        plt.plot(ind, louv_time_vals, marker='o', label='Louvain time')
    leiden_time_vals = extract("leiden_time")
    if any([not math.isnan(x) for x in leiden_time_vals if x is not None]):
        plt.plot(ind, leiden_time_vals, marker='s', label='Leiden time')
    plt.plot(ind, extract("fkcd_time"), marker='^', label='FKCD (neigh) time')
    plt.plot(ind, extract("fkcd_exact_time"), marker='x', label='FKCD (exact) time')
    plt.xticks(ind, labels, rotation=45, ha='right')
    plt.ylabel("time (s)")
    plt.title("Runtime comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bench_time.png")
    plt.close()

    plt.figure(figsize=(10,5))
    louv_nmi_vals = extract("louv_nmi")
    if any([not math.isnan(x) for x in louv_nmi_vals if x is not None]):
        plt.plot(ind, louv_nmi_vals, marker='o', label='Louvain NMI')
    leiden_nmi_vals = extract("leiden_nmi")
    if any([not math.isnan(x) for x in leiden_nmi_vals if x is not None]):
        plt.plot(ind, leiden_nmi_vals, marker='s', label='Leiden NMI')
    plt.plot(ind, extract("fkcd_nmi"), marker='^', label='FKCD (neigh) NMI')
    plt.plot(ind, extract("fkcd_exact_nmi"), marker='x', label='FKCD (exact) NMI')
    plt.xticks(ind, labels, rotation=45, ha='right')
    plt.ylabel("NMI")
    plt.title("NMI vs ground truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bench_nmi.png")
    plt.close()

    # NEW: modularity grouped bar chart (mean ± std across repeats) per (n,mu)
    # Group records by (n,mu)
    groups = defaultdict(list)
    for r in all_results:
        key = (int(r.get("n", -1)), float(r.get("mu", -1.0)))
        groups[key].append(r)
    group_keys = sorted(groups.keys())
    if len(group_keys) == 0:
        print("No results to plot modularity.")
        return

    # method -> key in records
    method_keys = {
        "Louvain": "louv_Q",
        "Leiden": "leiden_Q",
        "FKCD (neigh)": "fkcd_Q",
        "FKCD (exact)": "fkcd_exact_Q"
    }
    labels_mod = [f"n={n},μ={mu}" for (n,mu) in group_keys]
    means = {name: [] for name in method_keys}
    stds  = {name: [] for name in method_keys}

    for key in group_keys:
        recs = groups[key]
        for name, mk in method_keys.items():
            vals = []
            for r in recs:
                v = r.get(mk)
                if v is None or v == "":
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            if len(vals) == 0:
                means[name].append(np.nan)
                stds[name].append(0.0)
            else:
                means[name].append(np.mean(vals))
                stds[name].append(np.std(vals))

    # Plot grouped bar chart
    x = np.arange(len(group_keys))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8, len(group_keys)*1.2), 6))
    offset = -1.5 * width
    colors = ['C0','C1','C2','C3']
    for i, name in enumerate(method_keys.keys()):
        y = np.array(means[name])
        err = np.array(stds[name])
        # Only plot if any non-nan
        if np.all(np.isnan(y)):
            continue
        ax.bar(x + offset + i*width, y, width, yerr=err, label=name, capsize=4, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels_mod, rotation=45, ha='right')
    ax.set_ylabel("Modularity (Q)")
    ax.set_title("Modularity by method (mean ± std across repeats)")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "bench_modularity.png")
    plt.close()
    print("Saved modularity plot to", out_dir / "bench_modularity.png")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[500,1000], help="node sizes to test")
    p.add_argument("--mus", type=float, nargs="+", default=[0.1,0.3], help="LFR mixing parameters")
    p.add_argument("--repeats", type=int, default=2, help="repeats per config")
    p.add_argument("--outdir", type=str, default="bench_output", help="output folder")
    p.add_argument("--no-exact", action="store_true", help="skip FKCD exact (useful for large graphs)")
    p.add_argument("--no-neigh", action="store_true", help="skip FKCD neigh")
    p.add_argument("--no-louvain", action="store_true", help="skip Louvain")
    p.add_argument("--no-leiden", action="store_true", help="skip Leiden")
    p.add_argument("--rho-neigh", type=int, default=200, help="WERW rho for neigh mode")
    p.add_argument("--rho-exact", type=int, default=400, help="WERW rho for exact mode")
    p.add_argument("--workers", type=int, default=1, help="workers simulation (FKCD WERW inner)")
    p.add_argument("--min-community", type=int, default=10, help="initial min_community for LFR (will fallback on failure)")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    all_results = []
    run_exact = not args.no_exact
    run_neigh = not args.no_neigh
    run_louvain_flag = not args.no_louvain
    run_leiden_flag = not args.no_leiden

    print("Environment: HAS_FKCD", HAS_FKCD, "HAS_LOUVAIN", HAS_LOUVAIN, "HAS_LEIDEN", HAS_LEIDEN, "HAS_SKLEARN", HAS_SKLEARN)
    for n in args.sizes:
        for mu in args.mus:
            recs = run_single_experiment(n, mu, repeats=args.repeats, min_community=args.min_community,
                                         run_exact=run_exact, run_neigh=run_neigh,
                                         run_louvain=run_louvain_flag, run_leiden=run_leiden_flag,
                                         rho_neigh=args.rho_neigh, rho_exact=args.rho_exact,
                                         workers=args.workers)
            all_results.extend(recs)
            # save intermediate after every config
            aggregate_and_plot(all_results, out_dir)
    # final save
    aggregate_and_plot(all_results, out_dir)
    print("All done. Results in", out_dir)

if __name__ == "__main__":
    main()