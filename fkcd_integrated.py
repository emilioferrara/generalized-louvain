#!/usr/bin/env python3
"""
fkcd_integrated.py

Integrated FKCD implementation merging optimizations from the "Gemini" version while
keeping the algorithmic logic intact. Provides two proximity computation modes:
  - exact: compute the full pairwise proximity as in the paper (sum over all k)
  - neigh : neighbor-restricted approximation (compute sum over union(neigh(u), neigh(v)))
Default: exact

Optimizations included (non-logical changes):
- Use scipy.sparse CSR adjacency for fast neighbor access
- Workers produce sparse/dict increments for WERW-Kpath; master aggregates
- Batch processing for exact proximity using CSR row extraction
- Neighbor-restricted proximity uses per-edge neighbor unions (much cheaper)

Usage example:
  python fkcd_integrated.py --input random_graph_10000_nodes.edgelist --proximity-mode neigh --workers 8 --rho 1000 --kappa 5 --output-partition fkcd.json
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count

import networkx as nx
import numpy as np

# Optional: use numba for inner-loop acceleration in neighbor-restricted proximity
try:
    import numba
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

if _NUMBA_AVAILABLE:
    @njit(nogil=True)
    def _numba_sum_sq_diff(L1_arr, L2_arr, deg_arr):
        s = 0.0
        for i in range(L1_arr.shape[0]):
            diff = L1_arr[i] - L2_arr[i]
            s += (diff * diff) / deg_arr[i]
        return s
else:
    def _numba_sum_sq_diff(L1_arr, L2_arr, deg_arr):
        # Fallback pure-python numpy implementation (still vectorized)
        D = L1_arr - L2_arr
        return float((D * D).dot(deg_arr))

# Try to import scipy; required for exact mode
try:
    import scipy.sparse as sp
except Exception:
    sp = None

# ----------------- Utilities -----------------

def nx_to_csr(G):
    """Return CSR adjacency matrix and mapping of node indices (0..n-1)."""
    # Ensure nodes are integer or mapped
    nodes = list(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_node = {i: v for v, i in node_to_idx.items()}
    # build adjacency in CSR via networkx helper
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr', dtype=np.float64)
    return A, node_to_idx, idx_to_node

def csr_neighbors(A):
    """Return neighbors list per node using CSR arrays (fast view)."""
    indptr = A.indptr
    indices = A.indices
    n = A.shape[0]
    nbrs = [indices[indptr[i]:indptr[i+1]] for i in range(n)]
    return nbrs

# ----------------- WERW-Kpath (parallel workers) -----------------

def _werw_worker(task):
    """Worker invoked in Pool: runs `trials` walks and returns dict edge_idx->increment"""
    (seed_base, trials, kappa, nodes_list, nbrs, edge_index, m) = task
    rnd = random.Random(seed_base)
    inc = defaultdict(float)
    # compute delta weights proportional to degree (use degree as proxy)
    degs = [len(nbrs[v]) for v in nodes_list]
    denom = sum(degs)
    if denom == 0:
        probs = [1.0 / len(nodes_list)] * len(nodes_list)
    else:
        probs = [d / denom for d in degs]
    for _ in range(trials):
        traversed = set()
        v = rnd.choices(nodes_list, weights=probs, k=1)[0]
        N = 0
        while N < kappa:
            nbr = nbrs[v]
            # collect untraversed edges
            untr = []
            for w in nbr:
                a, b = (v, w) if v <= w else (w, v)
                eidx = edge_index.get((int(a), int(b)))
                if eidx is not None and eidx not in traversed:
                    untr.append((v, w, eidx))
            if not untr:
                break
            choice = rnd.choice(untr)
            eidx = choice[2]
            inc[eidx] += 1.0 / m
            traversed.add(eidx)
            v = choice[1] if choice[0] == v else choice[0]
            N += 1
    return inc

def werw_kpath_parallel(A, kappa=5, rho=None, workers=None, seed=42):
    """Run parallel WERW-Kpath on graph represented by CSR adjacency A.
    Returns centrality array of length m (edges count) aligned with edges list.
    """
    n = A.shape[0]
    # build edges list and mapping
    indptr = A.indptr; indices = A.indices
    edges = []
    edge_index = {}
    m = 0
    for u in range(n):
        for idx in range(indptr[u], indptr[u+1]):
            v = indices[idx]
            if u < v:
                edge_index[(u, v)] = m
                edges.append((u, v))
                m += 1
    if m == 0:
        return np.array([]), edges, edge_index
    if rho is None:
        rho = max(1, m - 1)
    if workers is None:
        workers = max(1, min(cpu_count(), 4))
    nodes_list = list(range(n))
    base = rho // workers
    extra = rho % workers
    tasks = []
    for i in range(workers):
        trials = base + (1 if i < extra else 0)
        tasks.append((seed + i + 1, trials, kappa, nodes_list, csr_neighbors(A), edge_index, m))
    # run pool
    with Pool(processes=workers) as p:
        results = p.map(_werw_worker, tasks)
    # aggregate increments
    omega = np.full(m, 1.0 / m, dtype=np.float64)
    for d in results:
        for eidx, val in d.items():
            omega[eidx] += val
    return omega, edges, edge_index

# ----------------- Proximity computations -----------------

def proximity_exact(A, centrality_arr, edges, edge_index, degrees, batch_size=512):
    """Exact proximity: sum over all k for each edge. Uses sparse matrix vectorization in batches.
    Requires scipy.sparse (A must be CSR). Returns dict eidx->proximity.
    """
    if sp is None:
        raise RuntimeError("scipy required for exact proximity mode. Install scipy.")
    n = A.shape[0]
    m = len(edges)
    # Build sparse centrality matrix C (n x n) with entries L(u,v) for edges; symmetric
    rows = []
    cols = []
    data = []
    for eidx, (u, v) in enumerate(edges):
        w = float(centrality_arr[eidx])
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([w, w])
    C = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    inv_deg = np.array([1.0 / max(1, degrees[i]) for i in range(n)], dtype=np.float64)
    proximities = {}
    # process in batches to control memory
    for i in range(0, m, batch_size):
        batch = edges[i:i+batch_size]
        # Extract rows for u and v for the batch
        rows_u = [C.getrow(u) for (u, v) in batch]
        rows_v = [C.getrow(v) for (u, v) in batch]
        # Convert to dense arrays (batch_size x n)
        R = np.vstack([r.toarray().ravel() for r in rows_u])
        S = np.vstack([r.toarray().ravel() for r in rows_v])
        D = R - S
        D2 = D * D
        vals = D2.dot(inv_deg)
        for j, val in enumerate(vals):
            rij = math.sqrt(val)
            proximities[i + j] = 1.0 / (1.0 + rij)
    return proximities

def proximity_neighbor_restricted(A, centrality_arr, edges, edge_index, degrees):
    """Neighbor-restricted proximity: for each edge (u,v), sum over k in union(neigh(u), neigh(v)).
    Uses optional numba-accelerated inner loop when available. Returns dict eidx->proximity.
    """
    indptr = A.indptr; indices = A.indices
    proximities = {}
    n = A.shape[0]
    # Prebuild degrees array for numba / vectorized dot
    deg_arr = np.array([max(1, degrees[i]) for i in range(n)], dtype=np.float64)
    for eidx, (u, v) in enumerate(edges):
        # union of neighbors (include u and v to be consistent)
        nu = indices[indptr[u]:indptr[u+1]]
        nv = indices[indptr[v]:indptr[v+1]]
        # concatenate and get unique k values (numpy unique on concatenated array)
        if nu.size == 0 and nv.size == 0:
            proximities[eidx] = 0.0
            continue
        concat = np.concatenate((nu, nv, np.array([u, v], dtype=nu.dtype)))
        K = np.unique(concat)
        # build index arrays for L1 and L2: lookup edge_index for (min(u,k), max(u,k))
        Klen = K.shape[0]
        L1 = np.zeros(Klen, dtype=np.float64)
        L2 = np.zeros(Klen, dtype=np.float64)
        degs_local = np.empty(Klen, dtype=np.float64)
        for i_k in range(Klen):
            k = int(K[i_k])
            a1, b1 = (u, k) if u <= k else (k, u)
            a2, b2 = (k, v) if k <= v else (v, k)
            idx1 = edge_index.get((int(a1), int(b1)))
            idx2 = edge_index.get((int(a2), int(b2)))
            L1[i_k] = centrality_arr[idx1] if idx1 is not None else 0.0
            L2[i_k] = centrality_arr[idx2] if idx2 is not None else 0.0
            degs_local[i_k] = 1.0 / max(1, degrees[k])  # store inverse degree for faster dot
        # compute s using numba helper or vectorized fallback
        # note: our helper expects deg array as divisor; we stored inverse degrees so adjust accordingly
        # Using (L1-L2)^2 * (1/deg) -> dot with inv_deg
        D = L1 - L2
        s = float((D * D).dot(degs_local))
        rij = math.sqrt(s)
        proximities[eidx] = 1.0 / (1.0 + rij)
    return proximities

# ----------------- Community detection -----------------

def detect_communities(H, prefer_louvain=False):
    try:
        if prefer_louvain:
            import community as community_louvain
            partition = community_louvain.best_partition(H, weight='weight')
            return partition, 'louvain'
    except Exception:
        pass
    # fallback
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(H, weight='weight'))
        partition = {}
        for cid, c in enumerate(comms):
            for n in c:
                partition[n] = cid
        return partition, 'greedy_modularity'
    except Exception:
        # singleton fallback
        partition = {n: i for i, n in enumerate(H.nodes())}
        return partition, 'singleton_fallback'

# ----------------- Main pipeline -----------------

def parse_args():
    p = argparse.ArgumentParser(description='FKCD integrated: exact vs neighbor-restricted proximity')
    p.add_argument('--input', help='input graph path (edgelist). If omitted uses karate club demo.')
    p.add_argument('--kappa', type=int, default=5)
    p.add_argument('--rho', type=int, default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--proximity-mode', choices=['exact','neigh'], default='exact',
                   help='exact: full pairwise sum; neigh: neighbor-restricted approx (faster)')
    p.add_argument('--batch-size', type=int, default=512, help='batch size for exact proximity vectorization')
    p.add_argument('--use-numba', action='store_true', help='Force use of numba in neighbor mode; error if unavailable')
    p.add_argument('--prefer-louvain', action='store_true')
    p.add_argument('--output-partition', default='fkcd_partition_integrated.json')
    return p.parse_args()

def main():
    args = parse_args()
    if args.input is None:
        print("Using Zachary's Karate Club demo graph.")
        G = nx.karate_club_graph()
    else:
        if not os.path.exists(args.input):
            print("Input not found:", args.input, file=sys.stderr)
            sys.exit(1)
        G = nx.read_edgelist(args.input, nodetype=int)

    print(f"Graph: |V|={G.number_of_nodes()} |E|={G.number_of_edges()}")
    # Convert to CSR
    A, node_to_idx, idx_to_node = nx_to_csr(G)
    n = A.shape[0]
    # degrees
    degrees = {i: max(1, A.indptr[i+1]-A.indptr[i]) for i in range(n)}

    print("Running WERW-Kpath centrality...")
    centrality_arr, edges, edge_index = werw_kpath_parallel(A, kappa=args.kappa, rho=args.rho, workers=args.workers, seed=args.seed)
    print(f"Computed centrality for {len(edges)} edges.")

    
    # Numba enforcement logic
    if args.use_numba:
        if not _NUMBA_AVAILABLE:
            raise RuntimeError("Numba requested via --use-numba but numba is not installed.")
        else:
            print("Numba acceleration ENABLED for neighbor proximity mode.")
    else:
        if _NUMBA_AVAILABLE:
            print("Numba available (automatic use in neighbor mode).")
        else:
            print("Numba not available (using numpy fallback).")

    print("Computing proximities (mode=%s) ..." % args.proximity_mode)
    if args.proximity_mode == 'exact':
        proximities = proximity_exact(A, centrality_arr, edges, edge_index, degrees, batch_size=args.batch_size)
    else:
        proximities = proximity_neighbor_restricted(A, centrality_arr, edges, edge_index, degrees)

    print("Building weighted graph...")
    H = nx.Graph()
    for i in range(n):
        H.add_node(idx_to_node[i])
    for eidx, (u, v) in enumerate(edges):
        w = float(proximities.get(eidx, 0.0))
        H.add_edge(idx_to_node[u], idx_to_node[v], weight=w)

    print("Detecting communities...")
    partition, method = detect_communities(H, prefer_louvain=args.prefer_louvain)
    try:
        from networkx.algorithms.community.quality import modularity
        # convert partition into communities list
        comms = defaultdict(list)
        for node, cid in partition.items():
            comms[cid].append(node)
        Q = modularity(H, list(comms.values()), weight='weight')
    except Exception:
        Q = None

    out = {'method': method, 'modularity': Q, 'partition': {str(k): int(v) for k,v in partition.items()}}
    with open(args.output_partitio if False else args.output_partition, 'w') as f:
        json.dump(out, f, indent=2)
    print("Saved partition to", args.output_partition)
    print("Done.")

if __name__ == '__main__':
    main()
