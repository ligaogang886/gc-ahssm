# -*- coding: utf-8 -*-
"""Evaluation protocol used for every number reported in the paper.

Definitions
-----------
reference position   the measured position of the record itself (``properties.x`` /
                     ``properties.z``, airport local affine frame, metres)
reference heading    ``properties.refdir``, the direction of the network segment that
                     the measurement was matched to; it is produced by the estimator
                     as a by-product but is **only read here**, never during
                     inference
physical mask        ``properties.kf_speed <= V_MAX`` and a non-empty reference
                     heading, which removes the reporting outliers caused by position
                     jumps
position error       reference minus estimate, evaluated on every point of the record
                     including the holding points
heading error        angular difference between the refined heading and the reference
                     heading, with differences above ``FLIP_TH`` removed because they
                     correspond to the reversed-reference ambiguity of segments whose
                     direction is not observable from a single report
metrics              MAE, RMSE, STD, P50 and P90, all in the order of ``METRIC_ORDER``

Usage
-----
    python experiments/evaluate.py --input <results_dir> [<results_dir> ...]
"""
import argparse
import glob
import json
import os

import numpy as np

FLIP_TH = 150.0
V_MAX = 30.0
METRIC_ORDER = ('MAE', 'RMSE', 'STD', 'P50', 'P90')
GROUPS = (('Arrival', 'A'), ('Departure', 'D'), ('Overall', None))


def ae(a, b):
    d = abs(float(a) - float(b))
    return 360.0 - d if d > 180.0 else d


def ae_arr(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return np.where(d > 180.0, 360.0 - d, d)


def metrics(err):
    a = np.asarray(err, float)
    if a.size == 0:
        return (float('nan'),) * 5
    return (float(np.mean(np.abs(a))), float(np.sqrt(np.mean(a ** 2))),
            float(np.std(a)), float(np.median(np.abs(a))),
            float(np.percentile(np.abs(a), 90)))


def load_record(path):
    rec = json.load(open(path, encoding='utf-8'))
    return rec[0] if isinstance(rec, list) else rec


def column(data, key):
    return np.array([p['properties'].get(key) for p in data], float)


def track_errors(rec, x_key='fusion_x', z_key='fusion_z', dir_key='fusion_dir'):
    """Per-point errors of one record, already masked by the physical criterion."""
    data = rec['data']
    ref_x, ref_z = column(data, 'x'), column(data, 'z')
    ref_d = column(data, 'refdir')
    sp = column(data, 'kf_speed')
    phys = (sp <= V_MAX) & ~np.isnan(ref_d)
    ex, ez, hd = column(data, x_key), column(data, z_key), column(data, dir_key)
    ok_pos = phys & ~np.isnan(ex) & ~np.isnan(ez)
    ok_dir = phys & ~np.isnan(hd)
    xv = np.where(ok_pos, ex - ref_x, np.nan)
    zv = np.where(ok_pos, ez - ref_z, np.nan)
    tv = np.where(ok_dir, ae_arr(hd, ref_d), np.nan)
    return xv, zv, tv


def collect(paths, **kw):
    """Group the error arrays of many records by movement type."""
    agg = {fl: {'x': [], 'z': [], 't': [], 'n_flip': 0} for fl in ('A', 'D')}
    for p in paths:
        try:
            rec = load_record(p)
        except Exception:
            continue
        fl = rec.get('flio', 'A')
        if fl not in agg:
            continue
        xv, zv, tv = track_errors(rec, **kw)
        g = agg[fl]
        g['x'].append(xv[~np.isnan(xv)])
        g['z'].append(zv[~np.isnan(zv)])
        tvv = tv[~np.isnan(tv)]
        g['t'].append(tvv)
        g['n_flip'] += int((tvv > FLIP_TH).sum())
    return agg


def group_rows(agg):
    """[(group, n_position, n_heading, x5, z5, heading5, n_out_of_range), ...]"""
    rows = []
    for name, fl in GROUPS:
        ks = ('A', 'D') if fl is None else (fl,)
        xa = np.concatenate([np.concatenate(agg[k]['x']) for k in ks])
        za = np.concatenate([np.concatenate(agg[k]['z']) for k in ks])
        ta = np.concatenate([np.concatenate(agg[k]['t']) for k in ks])
        flip = sum(agg[k]['n_flip'] for k in ks)
        tk = ta[ta <= FLIP_TH]
        rows.append((name, int(len(xa)), int(len(tk)), metrics(xa), metrics(za),
                     metrics(tk), int(flip)))
    return rows


def fmt(rows):
    out = []
    for name, n_pos, n_dir, x5, z5, t5, flip in rows:
        out.append('  %-10s n_pos=%9d n_dir=%9d reversed=%7d' % (name, n_pos, n_dir, flip))
        for label, m5 in (('x      ', x5), ('z      ', z5), ('heading', t5)):
            out.append('      %s ' % label + ' '.join(
                '%s=%9.3f' % (k, v) for k, v in zip(METRIC_ORDER, m5)))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description='evaluation protocol of the paper')
    ap.add_argument('--input', nargs='+', required=True,
                    help='one or more directories of estimator output records')
    ap.add_argument('--x-key', default='fusion_x')
    ap.add_argument('--z-key', default='fusion_z')
    ap.add_argument('--dir-key', default='fusion_dir')
    args = ap.parse_args()

    paths = []
    for d in args.input:
        paths.extend(sorted(glob.glob(os.path.join(d, '*.json'))))
    paths = [p for p in paths if os.path.basename(p) != 'manifest.json'
             and not os.path.basename(p).endswith('.truth.json')]
    agg = collect(paths, x_key=args.x_key, z_key=args.z_key, dir_key=args.dir_key)
    print('protocol: position on all points including holding, heading with '
          'differences above %g deg removed, physical mask kf_speed <= %g m/s'
          % (FLIP_TH, V_MAX))
    print('records: %d' % len(paths))
    print(fmt(group_rows(agg)))


if __name__ == '__main__':
    main()
