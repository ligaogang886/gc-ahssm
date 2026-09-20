# -*- coding: utf-8 -*-
"""Network layer: read the topology-aware directed surface network.

The network JSON is produced by ``scripts/build_graph.py`` from an OpenStreetMap
extract.  Every link carries a direction of travel and is partitioned into
segments whose anchor points are spaced by ``ANCHOR_STEP_M`` metres, which is what
makes the discrete state space of the estimator independent of the OSM geometry.
"""
import json
import math

import numpy as np

from .common import *

def load_graph(graph_json):
    data = json.load(open(graph_json, encoding='utf-8'))
    # ---- unified graph file branch (carries adjacency / transition_matrix / internal_points) ----
    if 'transition_matrix' in data and 'internal_points' in data:
        paths = data['paths']
        seg_pts = []; seg_path = []; seg_idx = []; seg_start = []; seg_end = []
        for p in paths:
            pn = p['pathname']
            for i in range(len(p['points']) - 1):
                a, b = p['points'][i], p['points'][i + 1]
                ax, az = float(a['x']), float(a['z'])
                bx, bz = float(b['x']), float(b['z'])
                if math.hypot(bx - ax, bz - az) < 1e-6:
                    continue
                seg_pts.append(((ax, az), (bx, bz)))
                seg_path.append(pn); seg_idx.append(i)
                seg_start.append(a.get('pointname')); seg_end.append(b.get('pointname'))
        S = np.array(seg_pts, dtype=float).reshape(-1, 2, 2)
        D = S[:, 1] - S[:, 0]
        DL = np.linalg.norm(D, axis=1)
        dir_f = np.degrees(np.arctan2(D[:, 1], D[:, 0])) % 360.0
        adjacency = data['adjacency']
        transition_matrix = data['transition_matrix']
        path_startend = {pn: (info['start'], info['end'])
                         for pn, info in data['path_dict'].items()}
        # 1m 内部点 KDTree(精匹配候选) + 归属表
        ip_all = []; ip_owner = []   # owner: (path 内段 idx 映射到全局段)
        seg_global = {}              # (pathname, seg_i) -> 全局段 idx
        for gi, (pn, si) in enumerate(zip(seg_path, seg_idx)):
            seg_global[(pn, si)] = gi
        ip_pn = []; ip_seg = []; ip_t = []
        for pn, plist in data['internal_points'].items():
            for (x, z, along, si, tt) in plist:
                ip_all.append((x, z))
                ip_owner.append((pn, si, tt))
        if ip_all:
            try:
                from scipy.spatial import cKDTree
                ip_tree = cKDTree(np.array(ip_all, dtype=float))
            except Exception:
                ip_tree = None
        else:
            ip_tree = None
        print(f'graph loaded: {len(paths)} paths, {len(S)} segments, '
              f'{sum(len(v) for v in adjacency.values())} directed links, '
              f'{len(ip_all)} one-metre anchors')
        return {
            'paths': paths, 'path_startend': path_startend,
            'adjacency': adjacency, 'transition_matrix': transition_matrix,
            'S': S, 'D': D, 'DL': DL, 'dir_f': dir_f,
            'seg_path': seg_path, 'seg_idx': seg_idx,
            'seg_start': seg_start, 'seg_end': seg_end,
            'ip_tree': ip_tree, 'ip_owner': ip_owner, 'seg_global': seg_global,
        }

    paths = data['paths']
    seg_pts = []        # (M,2,2)
    seg_path = []       # (M,) pathname
    seg_idx = []        # (M,) path 内段索引
    seg_start = []      # (M,) pointname
    seg_end = []
    path_startend = {}  # pathname -> (start, end)

    for p in paths:
        pts = p['points']
        path_startend[p['pathname']] = (p['startpoint'], p['endpoint'])
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ax, az = float(a['x']), float(a['z'])
            bx, bz = float(b['x']), float(b['z'])
            if math.hypot(bx - ax, bz - az) < 1e-6:
                continue
            seg_pts.append(((ax, az), (bx, bz)))
            seg_path.append(p['pathname'])
            seg_idx.append(i)
            seg_start.append(a['pointname'])
            seg_end.append(b['pointname'])

    S = np.array(seg_pts, dtype=float).reshape(-1, 2, 2)      # (M,2,2)
    D = S[:, 1] - S[:, 0]                                      # (M,2)
    DL = np.linalg.norm(D, axis=1)                             # (M,)
    dir_f = np.degrees(np.arctan2(D[:, 1], D[:, 0])) % 360.0   # (M,) 0-360

    # pathname 邻接: 端点坐标聚类(取整 0.5m, 跨 path 的 pointname 字符串不一致!)
    # + 空间邻接(path 端点间距离 <= 80m, 覆盖脱离道/OSM 捕捉误差)
    end_pts = {}
    for p in paths:
        pts = p['points']
        end_pts[p['pathname']] = [
            (float(pts[0]['x']), float(pts[0]['z'])),
            (float(pts[-1]['x']), float(pts[-1]['z']))]
    CR = 0.5
    node_to_paths = {}
    for pn, (e0, e1) in end_pts.items():
        for (ex, ez) in (e0, e1):
            key = (round(ex / CR) * CR, round(ez / CR) * CR)
            node_to_paths.setdefault(key, []).append(pn)
    adjacency = {}
    for node, conn in node_to_paths.items():
        for p1 in conn:
            for p2 in conn:
                if p1 == p2:
                    continue
                adjacency.setdefault(p1, set()).add(p2)
    # 空间邻接: 任一端点对距离 <= 80m
    SP = 80.0
    pn_list = sorted(end_pts.keys())
    arr = np.array([[e0[0], e0[1], e1[0], e1[1]] for e0, e1 in
                    (end_pts[pn] for pn in pn_list)], dtype=float)  # (P,4)
    P = len(pn_list)
    chunk = 250
    for i0 in range(0, P, chunk):
        i1 = min(i0 + chunk, P)
        blk = arr[i0:i1]                          # (b,4)
        A = blk[:, :2][:, None, :]                # 起点 (b,1,2)
        B = blk[:, 2:][:, None, :]
        A0 = arr[:, :2][None, :, :]               # (1,P,2)
        A1 = arr[:, 2:][None, :, :]
        d0 = np.sqrt(((A - A0) ** 2).sum(axis=2))  # (b,P)
        d1 = np.sqrt(((A - A1) ** 2).sum(axis=2))
        d2 = np.sqrt(((B - A0) ** 2).sum(axis=2))
        d3 = np.sqrt(((B - A1) ** 2).sum(axis=2))
        dmin = np.minimum(np.minimum(d0, d1), np.minimum(d2, d3))
        for k in range(i1 - i0):
            near = np.where(dmin[k] <= SP)[0]
            pn_i = pn_list[i0 + k]
            for j in near:
                pn_j = pn_list[j]
                if pn_i != pn_j:
                    adjacency.setdefault(pn_i, set()).add(pn_j)
                    adjacency.setdefault(pn_j, set()).add(pn_i)

    # transition_matrix: pathname 级
    transition_matrix = {}
    for pn in path_startend:
        neighbors = list(adjacency.get(pn, []))
        tm = {pn: STAY_PROB}
        remain = 1.0 - STAY_PROB
        if neighbors:
            nbp = remain / len(neighbors)
            for nb in neighbors:
                tm[nb] = nbp
        transition_matrix[pn] = tm

    print(f'graph loaded: {len(paths)} paths, {len(S)} segments, '
          f'{len(node_to_paths)} junctions, {sum(len(v) for v in adjacency.values())} directed links')
    return {
        'paths': paths,
        'path_startend': path_startend,
        'adjacency': adjacency,
        'transition_matrix': transition_matrix,
        'S': S, 'D': D, 'DL': DL, 'dir_f': dir_f,
        'seg_path': seg_path, 'seg_idx': seg_idx,
        'seg_start': seg_start, 'seg_end': seg_end,
    }
