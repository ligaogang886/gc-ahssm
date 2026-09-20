# -*- coding: utf-8 -*-
"""Build the sample trajectory bundle shipped in ``data/samples``.

The full surveillance archive is operationally sensitive and cannot be
redistributed.  This script documents how the small bundle that *is* shipped was
drawn from it and it re-creates that bundle byte for byte.

Selection rule
--------------
1. trajectories are split by movement type (arrival A / departure D);
2. inside each group the files are ordered by name and sampled on an evenly
   spaced grid, so the bundle spans short, medium and long records instead of
   taking the first files alphabetically;
3. every record is rewritten with the field subset that the estimator consumes
   plus a short operational header, which keeps the bundle small enough for a
   repository while preserving the exact numeric values of the observations.

Usage
-----
    python scripts/make_sample_bundle.py --src <traj_dir> --out data/samples/xian \
        --airport xian --per-class 6 --seed 20260508
"""
import argparse
import glob
import hashlib
import json
import os

# the fields the estimator itself reads (everything else is operational metadata
# that is dropped from the redistributable bundle)
POINT_KEYS = ('x', 'z', 'lon', 'lat', 'originLon', 'originLat',
              'angle_self', 'angle_graph', 'graph_match_dist', 'graph_seg_idx')
HEADER_KEYS = ('cfno', 'afn', 'flio', 'stno', 'dapn', 'rway', 'starttime', 'endtime')


def slim(rec):
    out = {k: rec[k] for k in HEADER_KEYS if k in rec}
    out['pointcount'] = len(rec['data'])
    data = []
    for item in rec['data']:
        props = item.get('properties', {})
        data.append({
            'time': item.get('time', ''),
            'properties': {k: props[k] for k in POINT_KEYS if k in props},
        })
    out['data'] = data
    return out


def pick(files, per_class, seed):
    """evenly spaced, deterministic pick inside one movement class"""
    files = sorted(files)
    if len(files) <= per_class:
        return files
    step = len(files) / float(per_class)
    return [files[min(len(files) - 1, int(i * step + step / 2.0))] for i in range(per_class)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--airport', required=True)
    ap.add_argument('--per-class', type=int, default=6, help='number of A and of D trajectories')
    ap.add_argument('--seed', type=int, default=20260508)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, '*.json')))
    groups = {'A': [], 'D': []}
    for f in files:
        rec = json.load(open(f, encoding='utf-8'))
        rec = rec[0] if isinstance(rec, list) else rec
        fl = rec.get('flio')
        if fl in groups:
            groups[fl].append(f)

    os.makedirs(args.out, exist_ok=True)
    manifest = {'airport': args.airport, 'source_directory': os.path.basename(args.src),
                'seed': args.seed, 'selection': 'evenly spaced by name inside each movement class',
                'note': 'redistributable subset - only the fields consumed by the estimator are kept',
                'files': []}
    total_pts = 0
    for fl in ('A', 'D'):
        for f in pick(groups[fl], args.per_class, args.seed):
            rec = json.load(open(f, encoding='utf-8'))
            rec = rec[0] if isinstance(rec, list) else rec
            body = slim(rec)
            name = os.path.basename(f)
            dst = os.path.join(args.out, name)
            with open(dst, 'w', encoding='utf-8') as fp:
                json.dump(body, fp, ensure_ascii=False, separators=(',', ':'))
            blob = open(dst, 'rb').read()
            total_pts += len(body['data'])
            manifest['files'].append({
                'file': name, 'movement': fl, 'points': len(body['data']),
                'bytes': len(blob), 'sha1': hashlib.sha1(blob).hexdigest()[:12]})

    manifest['n_files'] = len(manifest['files'])
    manifest['n_points'] = total_pts
    with open(os.path.join(args.out, 'manifest.json'), 'w', encoding='utf-8') as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    print('%s: %d files / %d points -> %s' % (args.airport, manifest['n_files'], total_pts, args.out))


if __name__ == '__main__':
    main()
