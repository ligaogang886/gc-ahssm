# -*- coding: utf-8 -*-
"""Latency, throughput and memory of the estimator, measured per layer.

Runs the complete pipeline over a directory of trajectories and reports, for every
layer, the per-observation latency (mean / median / p95 / p99), the end-to-end
latency and throughput, and process memory.  Hardware is detected at run time and
written to the report, so the figures can be reproduced on another machine.

Usage:  python experiments/runtime_benchmark.py --input <dir> --graph <file> [--limit N]
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import gcahssm as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='layer-wise latency, throughput and memory')
    ap.add_argument('--input', nargs='+', required=True,
                    help='one or more directories holding estimator output records')
    ap.add_argument('--graph', required=True, help='network JSON used by the run')
    ap.add_argument('--out', required=True, help='path of the text report to write')
    ap.add_argument('--limit', type=int, default=0, help='optional cap on the number of trajectories')
    args = ap.parse_args()

    # ---------- 硬件/环境 ----------
    env = {
        'python': sys.version.split()[0],
        'machine': platform.machine(),
        'cpu': platform.processor(),
        'cores': os.cpu_count(),
    }
    print('== environment ==')
    for k, v in env.items():
        print(f'  {k}: {v}', flush=True)

    import psutil
    proc = psutil.Process()

    # ---------- 图加载计时 + tracemalloc 峰值 (单独, 之后关闭避免拖慢) ----------
    import tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    G = R.load_graph(args.graph)
    t_graph = time.perf_counter() - t0
    _, peak_graph = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'\n== network load: {t_graph:.2f}s | tracemalloc peak {peak_graph/1e6:.1f} MB '
          f'(network plus its anchor KD-tree)', flush=True)

    # ---------- 全量逐轨迹分层计时 ----------
    import glob
    files = []
    for d in args.input:
        files.extend(sorted(glob.glob(os.path.join(d, '*.json'))))
    if args.limit:
        files = files[:args.limit]
    print(f'\n== per-trajectory layer timing: {len(files)} trajectories ==', flush=True)

    rows = []
    T = {'clean': [], 'kf': [], 'emis': [], 'vit': [], 'fus': []}   # 每轨迹层耗时 s
    rss_max = 0
    t_pipe0 = time.perf_counter()
    t_read_all = 0.0
    for i, f in enumerate(files):
        tr = time.perf_counter()
        d = json.load(open(f, encoding='utf-8'))
        rec = d[0] if isinstance(d, list) else d
        data = rec['data']
        t_read_all += time.perf_counter() - tr
        flio = rec.get('flio', 'A')
        n = len(data)

        tc = time.perf_counter()
        R.clean_missing_obs(data)
        dt_clean = time.perf_counter() - tc
        tc = time.perf_counter()
        R.step31_kf(data)
        dt_kf = time.perf_counter() - tc
        tc = time.perf_counter()
        R.step33_emission(data, G)
        dt_emis = time.perf_counter() - tc
        tc = time.perf_counter()
        R.step34_viterbi(data, G)
        dt_vit = time.perf_counter() - tc
        tc = time.perf_counter()
        R.step35_fusion(data, G)
        dt_fus = time.perf_counter() - tc

        t_e2e = dt_clean + dt_kf + dt_emis + dt_vit + dt_fus
        T['clean'].append(dt_clean); T['kf'].append(dt_kf)
        T['emis'].append(dt_emis); T['vit'].append(dt_vit); T['fus'].append(dt_fus)
        rss = proc.memory_info().rss
        if rss > rss_max:
            rss_max = rss
        rows.append(dict(day=os.path.basename(os.path.dirname(f)), file=os.path.basename(f),
                         flio=flio, n=n, clean=dt_clean, kf=dt_kf, emis=dt_emis,
                         vit=dt_vit, fus=dt_fus, e2e=t_e2e, rss=rss))
        if (i + 1) % 500 == 0 or i == len(files) - 1:
            el = time.perf_counter() - t_pipe0
            print(f'  progress {i+1}/{len(files)} | elapsed {el:.0f}s | peak RSS {rss_max/1e6:.0f} MB', flush=True)
    t_pipe = time.perf_counter() - t_pipe0

    # ---------- 汇总统计 ----------
    N = sum(r['n'] for r in rows)
    N_tr = len(rows)
    L = []
    L.append('=' * 88)
    L.append('Real-time performance of the estimator (layer-wise)')
    L.append(f'trajectories: {N_tr:,} | scored observations: {N:,} | '
             f'mean length {N/max(N_tr,1):.0f} points')
    L.append(f'hardware: {env["cpu"]} | single-threaded Python {env["python"]} | cores={env["cores"]}')
    L.append('')
    L.append('[1] end to end (clean -> kinematic filter -> likelihood -> inference -> refinement)'
             ', file reading excluded)')
    t_all = sum(r['e2e'] for r in rows)
    L.append(f'  total pipeline time {t_all:.1f}s (file reading {t_read_all:.1f}s excluded) | '
             f'throughput {N/t_all:,.0f} obs/s | mean {t_all/N*1000:.3f} ms/obs')
    L.append('')
    L.append('[2] per-layer latency (layer time per trajectory divided by its length)')
    L.append(f'  {"layer":<20}{"mean ms/obs":>13}{"median":>10}{"p95":>10}{"p99":>10}{"max":>10}{"share":>8}')
    total_per_obs = sum(np.sum(T[k]) for k in T) / N
    for k, name in [('clean', 'preprocessing'), ('kf', 'continuous motion'),
                    ('emis', 'observation likelihood'), ('vit', 'temporal inference'),
                    ('fus', 'topology-guided refinement')]:
        per = np.array([r[k] / r['n'] * 1000 for r in rows])
        frac = np.sum(T[k]) / sum(np.sum(T[k2]) for k2 in T) * 100
        L.append(f'  {name:<20}{per.mean():>13.3f}{np.median(per):>10.3f}'
                 f'{np.percentile(per, 95):>10.3f}{np.percentile(per, 99):>10.3f}{per.max():>10.3f}{frac:>7.1f}%')
    L.append('')
    e2e_per = np.array([r['e2e'] / r['n'] * 1000 for r in rows])
    L.append('[3] end-to-end latency per observation (aggregated per trajectory)')
    L.append(f'  mean={e2e_per.mean():.3f} ms | median={np.median(e2e_per):.3f} ms | '
             f'p95={np.percentile(e2e_per, 95):.3f} ms | p99={np.percentile(e2e_per, 99):.3f} ms | '
             f'max={e2e_per.max():.3f} ms')
    L.append(f'  margin against a one-second observation cadence: '
             f'{1000/e2e_per.mean():,.0f}x (mean) | {1000/np.percentile(e2e_per,99):,.0f}x (p99)')
    L.append('')
    L.append('[4] cost against trajectory length (linearity of the complexity)')
    ns = np.array([r['n'] for r in rows]); ts = np.array([r['e2e'] for r in rows])
    slope = np.polyfit(ns, ts, 1)
    corr = np.corrcoef(ns, ts)[0, 1]
    L.append(f'  least-squares slope={slope[0]*1000:.4f} ms/obs | R={corr:.4f} '
             f'(proportionality indicates linear cost in the number of points)')
    L.append('')
    L.append('[5] memory')
    L.append('  network structure (tracemalloc peak): {:.1f} MB'.format(peak_graph/1e6))
    L.append('  process RSS peak (trajectory records plus network, '
             'processed one at a time): {:.1f} MB'.format(rss_max/1e6))
    L.append('')
    txt = '\n'.join(L)
    print('\n' + txt, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_txt = args.out
    with open(out_txt, 'w', encoding='utf-8') as fp:
        fp.write(txt + '\n')
    print(f'\nreport written: {out_txt}', flush=True)

    out_csv = os.path.splitext(args.out)[0] + '_per_track.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as fp:
        w = csv.DictWriter(fp, fieldnames=['day', 'file', 'flio', 'n', 'clean', 'kf', 'emis',
                                           'vit', 'fus', 'e2e', 'rss'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'per-trajectory detail: {out_csv}', flush=True)



if __name__ == '__main__':
    main()
