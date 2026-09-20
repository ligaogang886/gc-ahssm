# -*- coding: utf-8 -*-
"""Run the estimator over a directory of trajectory records.

The input directory holds one JSON record per trajectory in the schema described in
the section "Record format" of ``README.md``.  Records are processed independently, so the run is
deterministic and embarrassingly parallel at the file level.

Usage
-----
    python experiments/run_pipeline.py --input data/samples/xian \
        --graph data/graphs/xian_graph.json --output out/xian

The output record of a trajectory keeps its measurements and adds the fields written
by each layer (``kf_*``, ``emission_*``, ``temporal_*``, ``fusion_*``, ``refdir``),
which is what ``experiments/evaluate.py`` reads.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from gcahssm import load_graph, run_one, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='GC-AHSSM trajectory estimator')
    ap.add_argument('--input', required=True, help='directory of input records')
    ap.add_argument('--graph', required=True, help='network JSON')
    ap.add_argument('--output', required=True, help='directory of output records')
    ap.add_argument('--limit', type=int, default=0, help='optional cap on the number of records')
    ap.add_argument('--overwrite', action='store_true', help='recompute records that already exist')
    args = ap.parse_args()

    G = load_graph(args.graph)
    os.makedirs(args.output, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(args.input, '*.json'))
                   if not os.path.basename(f).endswith(('.truth.json',)) and
                   os.path.basename(f) != 'manifest.json')
    if args.limit:
        files = files[:args.limit]

    print('input : %s (%d records)' % (args.input, len(files)))
    print('graph : %s' % args.graph)
    print('output: %s' % args.output)

    t0 = time.time()
    done = skipped = failed = 0
    summaries = []
    for f in files:
        name = os.path.basename(f)
        out_f = os.path.join(args.output, name)
        if os.path.exists(out_f) and not args.overwrite:
            skipped += 1
            continue
        try:
            rec = json.load(open(f, encoding='utf-8'))
            rec = rec[0] if isinstance(rec, list) else rec
            run_one(rec['data'], G)
            with open(out_f, 'w', encoding='utf-8') as fp:
                json.dump(rec, fp, ensure_ascii=False)
            s = summarize(rec['data'])
            if s:
                summaries.append(s)
            done += 1
        except Exception as exc:                                   # noqa: BLE001
            failed += 1
            print('  failed %s: %s' % (name, exc))

    el = time.time() - t0
    print('\nprocessed %d | skipped %d | failed %d | %.1fs' % (done, skipped, failed, el))
    if summaries:
        n = float(len(summaries))
        print('mean absolute heading difference against the reference direction: %.2f deg'
              % (sum(s['mean'] for s in summaries) / n))
        print('median of the same quantity: %.2f deg'
              % (sum(s['median'] for s in summaries) / n))


if __name__ == '__main__':
    main()
