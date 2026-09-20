# -*- coding: utf-8 -*-
"""Build the topology-aware graph of an airport from its vector network.

The input is the ``*_airport_paths.json`` file produced by ``build_paths.py`` and the
output is the single graph document that every entry point of this release consumes.

Structure of the output
-----------------------
  meta                  counts (adjacency strict / space / total, components)
  paths                 the paths as they come from the vector network
  path_dict             pathname -> {start: node_id, end: node_id}
  nodes                 node_id -> {x, z, paths: []}  endpoints clustered within 1 m
  adjacency             union of the two adjacency sets, used by the transition model
  adjacency_strict      adjacency through a shared endpoint
  adjacency_space       links added where two paths cross and the extract does not snap
  transition_matrix     pathname -> {pathname: prob}  with a self transition of 0.7
  segments              pathname -> [{idx, a, b, len, dir}]  the raw links
  internal_points       pathname -> [[x, z, along, seg, t]]  anchors every 1 m

Usage
-----
    python scripts/build_graph.py --src <paths.json> --out <graph.json>
    --check compares the result field by field against an existing graph document,
    which is how the equivalence of two builds of the same network is demonstrated.
"""
import os
import sys
import json
import math
import argparse
import numpy as np

# ---- Defaults of the published network ----
NODE_R = 1.0          # 端点聚类半径(m)
STEP_M = 1.0          # 内部点间距(m)
STAY_PROB = 0.7       # 转移矩阵自环概率
SPACE_R = 30.0        # 空间交叉补边阈值(m)
MIN_CROSS_ANG = 20.0  # 补边最小夹角(°)


def path_dir(pn, segdict, cache):
    if pn in cache:
        return cache[pn]
    vx = vy = 0.0
    for s in segdict.get(pn, []):
        if s['len'] < 1e-6:
            continue
        vx += s['b'][0] - s['a'][0]
        vy += s['b'][1] - s['a'][1]
    ang = math.degrees(math.atan2(vy, vx)) % 360
    cache[pn] = ang
    return ang


def build(src_json, node_r=NODE_R, step_m=STEP_M, stay_prob=STAY_PROB,
          space_r=SPACE_R, min_cross_ang=MIN_CROSS_ANG, verbose=True):
    """返回 (out_dict, comps)"""
    def log(*a):
        if verbose:
            print(*a, flush=True)

    g = json.load(open(src_json, encoding='utf-8'))
    paths_raw = g['paths']
    log(f'输入: {len(paths_raw)} paths')

    # ---------- 0) segments(原始段几何) ----------
    segments = {}
    for p in paths_raw:
        pn = p['pathname']
        pts = [(float(pt['x']), float(pt['z'])) for pt in p['points']]
        segs = []
        for i in range(len(pts) - 1):
            ax, az = pts[i]
            bx, bz = pts[i + 1]
            L = math.hypot(bx - ax, bz - az)
            if L < 1e-6:
                continue
            d_ang = math.degrees(math.atan2(bz - az, bx - ax)) % 360
            segs.append({'idx': len(segs), 'a': [round(ax, 2), round(az, 2)],
                         'b': [round(bx, 2), round(bz, 2)],
                         'len': round(L, 2), 'dir': round(d_ang, 3)})
        segments[pn] = segs
    log(f'段总数: {sum(len(v) for v in segments.values())}')

    # ---------- 1) 端点坐标聚类成节点 ----------
    bucket = {}
    nodes = {}

    def get_node(x, z):
        key = (round(x), round(z))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nid in bucket.get((key[0] + dx, key[1] + dy), []):
                    nd = nodes[nid]
                    if math.hypot(nd['x'] - x, nd['z'] - z) <= node_r:
                        return nid
        nid = len(nodes)
        nodes[nid] = {'x': round(x, 2), 'z': round(z, 2), 'paths': []}
        bucket.setdefault(key, []).append(nid)
        return nid

    path_dict = {}
    for p in paths_raw:
        pn = p['pathname']
        pts = p['points']
        e0 = (float(pts[0]['x']), float(pts[0]['z']))
        e1 = (float(pts[-1]['x']), float(pts[-1]['z']))
        n0 = get_node(*e0)
        n1 = get_node(*e1)
        path_dict[pn] = {'start': n0, 'end': n1}
        nodes[n0]['paths'].append(pn)
        nodes[n1]['paths'].append(pn)
    log(f'端点节点: {len(nodes)}')

    # ---------- 2) adjacency_strict(共享节点连通) ----------
    adjacency_strict = {}
    for nd in nodes.values():
        plist = nd['paths']
        for p1 in plist:
            for p2 in plist:
                if p1 != p2:
                    adjacency_strict.setdefault(p1, set()).add(p2)
    adjacency_strict = {k: sorted(v) for k, v in adjacency_strict.items()}
    log(f'共享节点邻接: {sum(len(v) for v in adjacency_strict.values())} 条')

    # ---------- 3) adjacency_space(空间交叉补边) ----------
    from scipy.spatial import cKDTree
    all_pts = []
    all_owner = []
    for q in sorted(segments.keys()):
        for s in segments[q]:
            all_pts.append(s['a'])
            all_owner.append(q)
            all_pts.append(s['b'])
            all_owner.append(q)
    A = np.array(all_pts, dtype=float)
    T = cKDTree(A)
    pairs = T.query_pairs(r=space_r)
    pair_set = set()
    for (i, j) in pairs:
        a, b = all_owner[i], all_owner[j]
        if a != b:
            pair_set.add((a, b) if a < b else (b, a))
    dir_cache = {}
    space_pairs = set()
    for (a, b) in pair_set:
        if b in adjacency_strict.get(a, []):
            continue
        da = abs((path_dir(a, segments, dir_cache)
                  - path_dir(b, segments, dir_cache) + 180) % 360 - 180)
        if da >= min_cross_ang:
            space_pairs.add((a, b))
    adjacency_space = {}
    for (a, b) in space_pairs:
        adjacency_space.setdefault(a, set()).add(b)
        adjacency_space.setdefault(b, set()).add(a)
    adjacency_space = {k: sorted(v) for k, v in adjacency_space.items()}
    log(f'空间交叉补边(<= {space_r}m, 夹角>= {min_cross_ang}°): '
        f'{sum(len(v) for v in adjacency_space.values())} 条')

    # ---------- 4) adjacency_full(总) + transition_matrix ----------
    adjacency_full = {}
    for k in segments.keys():
        s = set(adjacency_strict.get(k, [])) | set(adjacency_space.get(k, []))
        adjacency_full[k] = sorted(s)
    log(f'总邻接: {sum(len(v) for v in adjacency_full.values())} 条')

    transition_matrix = {}
    for pn in path_dict:
        nbs = adjacency_full.get(pn, [])
        tm = {pn: stay_prob}
        if nbs:
            nb_p = (1.0 - stay_prob) / len(nbs)
            for nb in nbs:
                tm[nb] = nb_p
        transition_matrix[pn] = tm

    # ---------- 5) 1m 内部点 ----------
    internal_points = {}
    total_pts = 0
    for p in paths_raw:
        pn = p['pathname']
        pts = [(float(pt['x']), float(pt['z'])) for pt in p['points']]
        ipts = []
        along = 0.0
        for i in range(len(pts) - 1):
            ax, az = pts[i]
            bx, bz = pts[i + 1]
            L = math.hypot(bx - ax, bz - az)
            if L < 1e-6:
                continue
            nstep = max(1, int(round(L / step_m)))
            for s in range(nstep):
                tt = s / nstep
                ipts.append([round(ax + (bx - ax) * tt, 2),
                             round(az + (bz - az) * tt, 2),
                             round(along + tt * L, 2), i, round(tt, 3)])
            along += L
        internal_points[pn] = ipts
        total_pts += len(ipts)
    log(f'内部点总数({step_m}m): {total_pts}')

    # ---------- 6) 连通性检查 ----------
    from collections import deque
    visited = set()
    comps = []
    all_pn = list(path_dict.keys())
    for pn in all_pn:
        if pn in visited:
            continue
        q = deque([pn])
        visited.add(pn)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adjacency_full.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    log(f'连通分量: {len(comps)}, 最大 {len(comps[0])} path'
        f' ({"%.1f" % (len(comps[0])/len(all_pn)*100)}%), 孤立: '
        f'{sum(1 for c in comps if len(c)==1)}')

    out = {
        'meta': {
            'n_paths': len(paths_raw), 'n_nodes': len(nodes),
            'n_segments': sum(len(v) for v in segments.values()),
            'n_internal_points': total_pts,
            'internal_step_m': step_m,
            'adjacency_strict': sum(len(v) for v in adjacency_strict.values()),
            'adjacency_space': sum(len(v) for v in adjacency_space.values()),
            'adjacency_total': sum(len(v) for v in adjacency_full.values()),
            'space_R_m': space_r, 'min_cross_ang': min_cross_ang,
            'n_components': len(comps), 'max_component': len(comps[0]),
            'stay_prob': stay_prob, 'node_r_m': node_r,
        },
        'paths': paths_raw,
        'path_dict': path_dict,
        'nodes': nodes,
        'adjacency': adjacency_full,
        'adjacency_strict': adjacency_strict,
        'adjacency_space': adjacency_space,
        'transition_matrix': transition_matrix,
        'segments': segments,
        'internal_points': internal_points,
    }
    return out, comps


def compare(old_json, new_out):
    """与旧图逐字段比对"""
    old = json.load(open(old_json, encoding='utf-8'))
    ok = True
    print('\n===== reproduction check =====')
    om, nm = old['meta'], new_out['meta']
    for k in om:
        if k in ('source', 'stay_prob', 'node_r_m'):
            continue
        a, b = om[k], nm.get(k)
        flag = 'OK ' if a == b else 'DIFF'
        if a != b:
            ok = False
        print(f'  meta.{k:20s} old={a} new={b}  [{flag}]')

    def norm(v):
        """JSON 键统一为 str，便于与新生成对象（可能 int 键）比对"""
        if isinstance(v, dict):
            return {str(k): norm(x) for k, x in v.items()}
        if isinstance(v, list):
            return [norm(x) for x in v]
        return v

    # 逐条比对结构
    for key in ['paths', 'path_dict', 'nodes', 'adjacency', 'adjacency_strict',
                'adjacency_space', 'transition_matrix', 'segments', 'internal_points']:
        o, n = norm(old[key]), norm(new_out[key])
        if len(o) != len(n):
            print(f'  {key:20s} entry count DIFF old={len(o)} new={len(n)}')
            ok = False
            continue
        same = True
        if isinstance(o, list):
            for i, (x, y) in enumerate(zip(o, n)):
                if json.dumps(x, sort_keys=True, ensure_ascii=False) != \
                        json.dumps(y, sort_keys=True, ensure_ascii=False):
                    print(f'  {key}[{i}] content differs')
                    print(f'     old: {json.dumps(x, ensure_ascii=False)[:200]}')
                    print(f'     new: {json.dumps(y, ensure_ascii=False)[:200]}')
                    same = False
                    break
        else:
            for k in o:
                if k not in n or json.dumps(o[k], sort_keys=True, ensure_ascii=False) != \
                        json.dumps(n[k], sort_keys=True, ensure_ascii=False):
                    same = False
                    print(f'  {key}["{k}"] content differs')
                    print(f'     old: {json.dumps(o[k], ensure_ascii=False)[:200]}')
                    print(f'     new: {json.dumps(n.get(k), ensure_ascii=False)[:200]}')
                    break
        print(f'  {key:20s} entry count {len(o)}  content {"identical" if same else "differs at key level"}')
        if not same:
            ok = False
    print(f'\nreproduction verdict: {"identical" if ok else "differences found"}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='输入 <机场>_airport_paths.json')
    ap.add_argument('--out', required=True, help='output graph document')
    ap.add_argument('--node-r', type=float, default=NODE_R)
    ap.add_argument('--step', type=float, default=STEP_M)
    ap.add_argument('--stay', type=float, default=STAY_PROB)
    ap.add_argument('--space-r', type=float, default=SPACE_R)
    ap.add_argument('--min-cross-ang', type=float, default=MIN_CROSS_ANG)
    ap.add_argument('--check', help='复现校验：与该旧图文件逐字段比对')
    args = ap.parse_args()

    out, comps = build(args.src, args.node_r, args.step, args.stay,
                       args.space_r, args.min_cross_ang)

    if args.check:
        ok = compare(args.check, out)
        if not ok:
            print('reproduction failed, the file was not written')
            sys.exit(2)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, sort_keys=True)
    sz = os.path.getsize(args.out) / 1024 / 1024
    print(f'output: {args.out} ({sz:.1f} MB)')
    print('meta:', out['meta'])
    # 分量明细（前 8 大）
    print('connected components, largest eight:', [len(c) for c in comps[:8]])


if __name__ == '__main__':
    main()
