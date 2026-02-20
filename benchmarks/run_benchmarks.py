#!/usr/bin/env python3
# run_benchmarks.py - simple runner for fkcd_integrated timings on small graphs
import subprocess, time, json, os
graphs = ['examples/small_test_graph.edgelist']
out = []
for g in graphs:
    cmd = ['python','fkcd_integrated.py','--input',g,'--proximity-mode','neigh','--rho','200','--workers','1']
    t0 = time.time()
    subprocess.check_call(cmd)
    t1 = time.time()
    out.append({'graph':g,'time_s':t1-t0})
with open('benchmarks/results.json','w') as f:
    json.dump(out,f,indent=2)
print('Benchmarks complete, results in benchmarks/results.json')
