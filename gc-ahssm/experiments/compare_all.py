# -*- coding: utf-8 -*-
"""Seven-method comparison table.

Combines the four baselines carried by ``compare_baselines`` with the two graph-prior
sequence baselines carried by ``baselines_graph_priors`` and the proposed model, and
prints one table per movement group so that the ordering of the methods can be read
directly.

Every method is scored on the same records and with the same protocol as
``evaluate.py``.  The proposed model is read back from the ``fusion_*`` fields of an
estimator output record, while the six baselines are re-run from the measurements.

Usage
-----
    python experiments/compare_all.py --input out/xian --graph data/graphs/xian_graph.json
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import baselines_graph_priors as GB  # noqa: E402
import compare_baselines as CB  # noqa: E402
import gcahssm as R  # noqa: E402

FLIP_TH = 150.0
METRICS = ('MAE', 'RMSE', 'STD', 'P50', 'P90')
ORDER = ('CVM', 'EKF', 'T-HMM', 'Flight2Vec', 'M1', 'M2', 'GC-AHSSM')
LABELS = {'M1': 'HMM-MM', 'M2': 'KF-HMM'}


def _m(err):
    a = np.asarray(err, float)
    if a.size == 0:
        return (float('nan'),) * 5
    return (float(np.mean(np.abs(a))), float(np.sqrt(np.mean(a ** 2))),
            float(np.std(a)), float(np.median(np.abs(a))),
            float(np.percentile(np.abs(a), 90)))


def main():
    ap = argparse.ArgumentParser(description='seven-method comparison')
    ap.add_argument('--input', nargs='+', required=True,
                    help='one or more directories of estimator output records')
    ap.add_argument('--graph', required=True, help='network JSON')
    ap.add_argument('--limit', type=int, default=0, help='cap the records of each directory')
    args = ap.parse_args()

    G = R.load_graph(args.graph)
    TH = CB.thmm_prepare(G)

    agg = {m: {fl: {'x': [], 'z': [], 't': []} for fl in ('A', 'D')} for m in ORDER}
    n_rec = 0
    for d in args.input:
        picks = sorted(glob.glob(os.path.join(d, '*.json')))
        if args.limit:
            picks = picks[:args.limit]
        for f in picks:
            rec_ok = True
            try:
                CB_rec = CB.eval_track(f, TH)
                GB_rec = GB.eval_track(f, G)
            except Exception:                                      # noqa: BLE001
                rec_ok = False
            if not rec_ok or CB_rec is None or GB_rec is None:
                continue
            fl = CB_rec['flio']
            if fl not in ('A', 'D'):
                continue
            n_rec += 1
            # proposed model and the four baselines (masking identical to compare_baselines)
            for m in ('GC-AHSSM', 'CVM', 'EKF', 'T-HMM', 'Flight2Vec'):
                xv, zv, tv = CB.track_errors(CB_rec, m)
                g = agg[m][fl]
                g['x'].append(xv[~np.isnan(xv)])
                g['z'].append(zv[~np.isnan(zv)])
                g['t'].append(tv[~np.isnan(tv)])
            # the two graph-prior sequence baselines (masking identical to their module)
            for m in ('M1', 'M2'):
                xe, ze, te, phys_ok = GB_rec['out'][m]
                ok = phys_ok & ~np.isnan(xe) & ~np.isnan(ze) & ~np.isnan(te)
                g = agg[m][fl]
                g['x'].append(xe[ok]); g['z'].append(ze[ok]); g['t'].append(te[ok])
        print('[ok] %s' % d)

    print('=' * 96)
    print('seven-method comparison | records %d' % n_rec)
    print('protocol: position on all points including holding, heading above %g deg removed, '
          'physical mask kf_speed <= %g m/s' % (FLIP_TH, GB.V_MAX))
    print('=' * 96)
    for name, fl in (('Arrival', 'A'), ('Departure', 'D'), ('Overall', None)):
        ks = ('A', 'D') if fl is None else (fl,)
        print('\n### %s' % name)
        print('  %-14s %s' % ('method', ''.join('%12s' % m for m in METRICS)))
        for m in ORDER:
            ta = np.concatenate([np.concatenate(agg[m][k]['t']) for k in ks])
            xa = np.concatenate([np.concatenate(agg[m][k]['x']) for k in ks])
            za = np.concatenate([np.concatenate(agg[m][k]['z']) for k in ks])
            tk = ta[ta <= FLIP_TH]
            label = LABELS.get(m, m)
            print('  %-14s' % label + ''.join('%12s' % ('%.3f' % v) for v in _m(tk)) +
                  '   | position x' + ''.join('%10s' % ('%.3f' % v) for v in _m(xa)))
        print('  (heading metrics first, then position along x for reference)')


if __name__ == '__main__':
    main()
