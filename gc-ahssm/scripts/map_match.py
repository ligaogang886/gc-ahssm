# -*- coding: utf-8 -*-
"""
轨迹图匹配（Viterbi 地图匹配 + 双向方向判定）
解决密集路网区单点最近匹配异常的问题：
- 候选段: 每点取距离<CAND_MAX 的 Top-K 最近段
- Viterbi: 发射=点段距离; 转移=图连通性(共享端点)约束, 全局最优段序列
- 方向: 点在段上投影弧长时序 Δs 判定前进/后退 + angle_self 辅助 + 时序传播
- angle_graph = 段方向 × 行进方向; 距离>阈值 → None
写入 properties: angle_self / angle_graph / graph_match_dist / graph_seg_idx
用法:
  python map_match.py <轨迹数据目录> [--graph 路网json] [--th 100] [--workers 6]
"""
import os, sys, json, glob, math, time, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree

DEFAULT_GRAPH = os.path.join('data', 'graphs', 'xian_graph.json')
MATCH_THRESHOLD = 100.0   # 匹配有效距离上限
CAND_MAX = 150.0          # 候选段距离上限
TOP_K = 8                 # 候选段数量
NODE_R = 20.0             # 端点聚类半径(m)
SIGMA = 20.0              # 发射高斯 σ(m)
DIR_W = 2.0               # 线条方向一致性权重（发射项）
JUMP_PENALTY = 8.0        # 非连通转移惩罚
CHUNK_PTS = 64            # candidates() 的点分块大小 (限制 (M,N,2) 张量峰值内存)

_G = None

def angle_diff(a, b):
    return (a - b + 180) % 360 - 180

def load_graph(graph_path):
    data = json.load(open(graph_path, encoding='utf-8'))
    segs, dirs, prev_v, next_v, pl_ang = [], [], [], [], []
    for p in data['paths']:
        pts = [(pt['x'], pt['z']) for pt in p['points']]
        n_p = len(pts)
        # 每条折线的边方向单位向量
        vecs = []
        for i in range(n_p - 1):
            x1, z1 = pts[i]; x2, z2 = pts[i + 1]
            vx, vz = x2 - x1, z2 - z1
            vlen = math.hypot(vx, vz)
            if vlen < 1e-6:
                vecs.append((0.0, 0.0))
            else:
                vecs.append((vx / vlen, vz / vlen))
        # 每段在该路径内的局部走向（前后 3 条边的平均方向）—— 区分直线段与弯道段
        pl = []
        for i in range(n_p - 1):
            lo = max(0, i - 3); hi = min(n_p - 2, i + 3)
            sx = sum(vecs[k][0] for k in range(lo, hi + 1))
            sz = sum(vecs[k][1] for k in range(lo, hi + 1))
            sn = math.hypot(sx, sz)
            pl.append((sx / sn, sz / sn) if sn > 1e-9 else vecs[i])
        for i in range(n_p - 1):
            x1, z1 = pts[i]; x2, z2 = pts[i + 1]
            if math.hypot(x2 - x1, z2 - z1) < 1e-6:
                continue
            segs.append([(x1, z1), (x2, z2)])
            dirs.append(math.degrees(math.atan2(z2 - z1, x2 - x1)))
            pv = vecs[i - 1] if i > 0 else vecs[i]
            nv = vecs[i + 1] if i < n_p - 2 else vecs[i]
            prev_v.append(pv)
            next_v.append(nv)
            pl_ang.append(math.degrees(math.atan2(pl[i][1], pl[i][0])))
    S = np.array(segs, dtype=float).reshape(-1, 2, 2)
    D = np.array(dirs, dtype=float)
    L = np.linalg.norm(S[:, 1] - S[:, 0], axis=1)
    PV = np.array(prev_v, dtype=float)
    NV = np.array(next_v, dtype=float)
    PL = np.array(pl_ang, dtype=float)
    return build_adj(S), S, D, L, PV, NV, PL

def build_adj(S):
    """端点聚类成节点 → 段邻接表"""
    N = len(S)
    eps = np.vstack([S[:, 0], S[:, 1]])            # (2N, 2)
    tree = cKDTree(eps)
    pairs = tree.query_pairs(r=NODE_R)
    # 并查集
    parent = list(range(2 * N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for a, b in pairs:
        union(a, b)
    node_of = [find(i) for i in range(2 * N)]
    seg_nodes = [(node_of[2*i], node_of[2*i+1]) for i in range(N)]
    # 段邻接: 共享节点
    adj = [set() for _ in range(N)]
    node_segs = {}
    for i, (na, nb) in enumerate(seg_nodes):
        node_segs.setdefault(na, []).append(i)
        node_segs.setdefault(nb, []).append(i)
    for node, segs_ in node_segs.items():
        for i in segs_:
            for j in segs_:
                if i != j:
                    adj[i].add(j)
    return adj

def _init_global(g):
    global _G
    adj, S, D, L, PV, NV, PL = g
    _G = {'adj': adj, 'S': S, 'D': D, 'L': L, 'PV': PV, 'NV': NV, 'PL': PL,
          'a': S[:, 0][None, :, :],
          'd': S[:, 1] - S[:, 0],
          'd2': np.sum((S[:, 1] - S[:, 0]) ** 2, axis=1)}
    _G['d2'][_G['d2'] < 1e-12] = 1e-12

def seg_tangent(seg_idx, t):
    """
    图方向沿投影位置的曲率切线:
    投影参数 t∈[0,1] 靠近段起点时混合前邻边方向, 靠近终点时混合后邻边方向
    返回单位向量 (vx, vz)
    """
    own = _G['d'][seg_idx]
    own = own / np.linalg.norm(own)
    wp = max(0.0, 0.5 - t)          # 前邻边权重 (t→0 时 = 0.5)
    wn = max(0.0, t - 0.5)          # 后邻边权重 (t→1 时 = 0.5)
    v = (1 - wp - wn) * own + wp * _G['PV'][seg_idx] + wn * _G['NV'][seg_idx]
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return own
    return v / nv

def candidates(P):
    """每点 Top-K 候选段 (距离矩阵取小) → (cand_idx list, cand_dist list)

    按点分块计算: 整块写法会构造 (M, N, 2) 距离张量, N = 路网段数。西安 7469 段
    尚可, 重庆 14685 段 + 长轨迹 (M≈600) 时单进程峰值 >300 MB, 6 进程并发直接
    MemoryError。分块后峰值内存与轨迹长度解耦 (CHUNK_PTS × N), 逐点结果与整块
    计算逐位一致 (同一广播/归约顺序, 同 dtype)。
    """
    M = len(P)
    A, D, D2 = _G['a'], _G['d'], _G['d2']
    cand_idx, cand_dist = [], []
    for c0 in range(0, M, CHUNK_PTS):
        Pc = P[c0:c0 + CHUNK_PTS]
        v = Pc[:, None, :] - A
        t = np.clip(np.sum(v * D[None, :, :], axis=2) / D2[None, :], 0, 1)
        proj = A + t[:, :, None] * D[None, :, :]
        dist = np.linalg.norm(Pc[:, None, :] - proj, axis=2)   # (<=CHUNK_PTS, N)
        for i in range(len(Pc)):
            row = dist[i]
            k = min(TOP_K, int((row < CAND_MAX).sum()))
            if k == 0:
                cand_idx.append([]); cand_dist.append([])
                continue
            idxs = np.argpartition(row, k - 1)[:k]
            idxs = idxs[np.argsort(row[idxs])]
            cand_idx.append(list(idxs))
            cand_dist.append(list(row[idxs]))
    return cand_idx, cand_dist

def viterbi(cands, dists, win_ang):
    """Viterbi: 发射=距离高斯 + 路径局部走向一致性(|cos|, 无向走向匹配); 转移=图连通约束
    win_ang: 每点轨迹线条方向(度, ±20点窗口)，NaN 表示静止/无效(仅距离项)
    """
    m = len(cands)
    adj = _G['adj']
    PL = _G['PL']
    e = []
    for t in range(m):
        row = []
        for k in range(len(cands[t])):
            d = dists[t][k]
            ed = -d*d/(2*SIGMA*SIGMA)
            ev = 0.0
            if not math.isnan(win_ang[t]):
                # 走向一致性(取 |cos|): 轨迹线条 vs 候选段所在路径的局部走向
                diff = abs(angle_diff(win_ang[t], PL[cands[t][k]]))
                ev = DIR_W * abs(math.cos(math.radians(diff)))
            row.append(ed + ev)
        e.append(row)
    dp = [list(e[0])]
    bt = [None] * m
    for t in range(1, m):
        prev = dp[-1]
        cur = [0.0]*len(cands[t]); btrow = [0]*len(cands[t])
        for k in range(len(cands[t])):
            sk = cands[t][k]
            best = -1e18; bestj = 0
            for j in range(len(cands[t-1])):
                sj = cands[t-1][j]
                tr = 0.0 if (sj == sk or sk in adj[sj]) else -JUMP_PENALTY
                val = prev[j] + tr
                if val > best:
                    best = val; bestj = j
            cur[k] = e[t][k] + best
            btrow[k] = bestj
        dp.append(cur); bt[t] = btrow
    seq = [0]*m
    seq[m-1] = int(np.argmax(dp[-1]))
    for t in range(m-1, 0, -1):
        seq[t-1] = bt[t][seq[t]]
    return [cands[t][seq[t]] for t in range(m)]

def mapmatch(P, win_ang):
    """Viterbi 匹配（含线条方向一致性） → (best_seg (n,), best_dist (n,))"""
    n = len(P)
    cand_idx, cand_dist = candidates(P)
    best_seg = np.full(n, -1, dtype=int)
    best_dist = np.full(n, np.nan)
    i = 0
    while i < n:
        if not cand_idx[i]:
            i += 1
            continue
        j = i
        while j < n and cand_idx[j]:
            j += 1
        seq = viterbi(cand_idx[i:j], cand_dist[i:j], win_ang[i:j])
        for k, s in enumerate(seq):
            best_seg[i+k] = s
            cd = cand_dist[i+k]
            best_dist[i+k] = cd[cand_idx[i+k].index(s)]
        i = j
    return best_seg, best_dist

def compute_angle_graph(best_seg, best_dist, win_ang, n, P, selfs_clean):
    """
    angle_graph = 匹配段在投影位置的曲率切线方向 × 轨迹整体移动方向感
    - 每点投影到其匹配段(参数 t) → 沿图曲率切线方向(混合前后邻边)
    - 对连续同段 run: run 整体位移在切线方向上的投影确定行进方向(+/-)
    - run 位移过小/垂直 → 用 run 内线条方向(窗口±20点)中位数判定
    - 未知 run 前后传播
    - 后处理修正: 轨迹方向连续但 graph 翻转>90° 时翻回与前一方向一致
      (仅修正, 不禁止; 真实转向保留)
    """
    D = _G['D']; d = _G['d']; a = _G['a'][0]; d2 = _G['d2']
    # 每点投影参数 t
    tarr = np.full(n, 0.5)
    for i in range(n):
        sgi = best_seg[i]
        if sgi >= 0:
            v = P[i] - a[sgi]
            tarr[i] = float(np.clip(np.dot(v, d[sgi]) / d2[sgi], 0, 1))
    # 切线单位向量
    tan = np.zeros((n, 2))
    for i in range(n):
        sgi = best_seg[i]
        if sgi >= 0:
            tan[i] = seg_tangent(sgi, tarr[i])
    # 行进方向判定（run 分组）
    sign = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        sgi = best_seg[i]
        if sgi < 0:
            i += 1
            continue
        j = i
        while j < n and best_seg[j] == sgi:
            j += 1
        if best_dist[i] <= MATCH_THRESHOLD:
            i0 = max(0, i - 1); i1 = min(n - 1, j)
            vx = P[i1, 0] - P[i0, 0]
            vz = P[i1, 1] - P[i0, 1]
            mv = math.hypot(vx, vz)
            # 用 run 中心点的切线方向判定
            tk = tan[(i + j) // 2]
            if mv < 5.0:
                # 微动/静止(run 位移<5m, 排队等待/抖动): 方向不可靠, 置 0 由前后传播填充
                sg = 0
            elif abs(tk[0] * vx + tk[1] * vz) / max(mv, 1e-9) < 0.3:
                # 位移够大但投影模糊(接近垂直): 用 run 内线条方向中位数
                w = [win_ang[k] for k in range(i, j) if not math.isnan(win_ang[k])]
                if w:
                    med = float(np.median(w))
                    td = math.degrees(math.atan2(tk[1], tk[0]))
                    df = abs(angle_diff(med, td))
                    dr = abs(angle_diff(med, td + 180))
                    sg = 1 if df <= dr else -1
                else:
                    sg = 0
            else:
                sg = 1 if (tk[0] * vx + tk[1] * vz) > 0 else -1
            sign[i:j] = sg
        i = j
    # 传播未知 run
    last = 0
    for i in range(n):
        if sign[i] != 0:
            last = sign[i]
        elif last != 0 and best_seg[i] >= 0:
            sign[i] = last
    last = 0
    for i in range(n - 1, -1, -1):
        if sign[i] != 0:
            last = sign[i]
        elif last != 0 and best_seg[i] >= 0:
            sign[i] = last
    ang = np.full(n, np.nan)
    for i in range(n):
        if best_seg[i] < 0 or best_dist[i] > MATCH_THRESHOLD or sign[i] == 0:
            continue
        v = tan[i] * sign[i]
        a_ = math.degrees(math.atan2(v[1], v[0]))
        if a_ > 180: a_ -= 360
        elif a_ <= -180: a_ += 360
        ang[i] = a_

    # ---- 后处理修正: graph 翻转>90° 时, 比较 gc 与 cand(=gc+180) 谁更接近轨迹移动方向 ----
    # 移动方向 = 窗口 ±5 点 selfs_clean 中位数 (抗停止点噪声)
    # 只有 cand 更接近移动方向 且 与前一点更连续 时才翻转修正
    # → 真实转弯 (gc 接近移动方向) 保留; 错误翻转 (cand 接近移动方向) 修正; 不级联传播
    for i in range(1, n):
        gp, gc = ang[i - 1], ang[i]
        if gp is None or gc is None:
            continue
        if abs(angle_diff(gc, gp)) > 90:
            a0 = max(0, i - 5); b0 = min(n - 1, i + 5)
            ws = [selfs_clean[k] for k in range(a0, b0 + 1)
                  if not math.isnan(selfs_clean[k])]
            if len(ws) >= 2:
                med = float(np.median(ws))
                cand = gc + 180
                if cand > 180: cand -= 360
                elif cand <= -180: cand += 360
                if abs(angle_diff(cand, med)) < abs(angle_diff(gc, med)) \
                   and abs(angle_diff(cand, gp)) < abs(angle_diff(gc, gp)):
                    ang[i] = cand

    # ---- 最终整体方向对齐: 以轨迹整体移动方向(win_ang, ±20点窗口)为锚点 ----
    # 修复"整段反向"(run 内连续同向但整体与移动方向差 180°): 相邻点翻转检测抓不到,
    # 但 win_ang 能识别——|ang - win_ang| > 90° 说明该点图方向与移动方向相反 → 翻转 180°
    # 按 run 分组执行: 同段连续点用 run 内 win_ang 中位数为统一锚点, 整段一致翻转
    # 两个防误伤门槛:
    #  1) 方向一致性: run 内 win_ang 无向簇(映射到 [0,180)) ≥70% 一致才执行——
    #     转弯过渡/静止抖动区(方向分散)跳过, 保持 run 判定的 sign 结果
    #  2) 静止点(win_ang 全 NaN)不动
    i = 0
    while i < n:
        sgi = best_seg[i]
        if sgi < 0 or math.isnan(ang[i]):
            i += 1
            continue
        j = i
        while j < n and best_seg[j] == sgi:
            j += 1
        ws = [win_ang[k] for k in range(i, j) if not math.isnan(win_ang[k])]
        if len(ws) >= 2:
            # 无向化: 映射到 [0,180), 检查方向簇一致性
            ws180 = [(w + 180.0) % 180.0 for w in ws]
            med180 = float(np.median(ws180))
            def diff180(a, b):
                d = abs(a - b)
                return 180.0 - d if d > 90 else d
            frac = sum(1 for w in ws180 if diff180(w, med180) < 30.0) / len(ws180)
            if frac >= 0.7:
                med = float(np.median(ws))
                for k in range(i, j):
                    if math.isnan(ang[k]):
                        continue
                    if abs(angle_diff(ang[k], med)) > 90:
                        cand = ang[k] + 180
                        if cand > 180: cand -= 360
                        elif cand <= -180: cand += 360
                        ang[k] = cand
        i = j
    return ang

def process_file(f):
    try:
        traj = json.load(open(f, encoding='utf-8'))
    except Exception as e:
        return (os.path.basename(f), False, str(e))
    if isinstance(traj, list) and traj:
        data = traj[0].get('data', [])
    elif isinstance(traj, dict):
        data = traj.get('data', [])
    else:
        data = []
    if len(data) < 3:
        return (os.path.basename(f), False, 'too few points')
    # 断点续跑: 已含角度属性(angle_self + angle_graph)的文件视为已处理, 跳过
    # --force 时强制重算 (用于 angle_graph 算法更新后)
    if not os.environ.get('MAPMATCH_FORCE') and data \
       and 'angle_self' in data[0].get('properties', {}) \
       and 'angle_graph' in data[0].get('properties', {}):
        return (os.path.basename(f), True, 'skipped')

    coords = []
    for p in data:
        pr = p.get('properties', {})
        try:
            coords.append((float(pr['x']), float(pr['z'])))
        except (KeyError, ValueError):
            coords.append(None)
    pts = [c if c is not None else (0.0, 0.0) for c in coords]
    P = np.array(pts, dtype=float)
    n = len(data)

    ang_self = np.full(n, np.nan)
    for i in range(n):
        if i == 0:
            vx = pts[1][0]-pts[0][0]; vz = pts[1][1]-pts[0][1]
        elif i == n-1:
            vx = pts[n-1][0]-pts[n-2][0]; vz = pts[n-1][1]-pts[n-2][1]
        else:
            vx = pts[i+1][0]-pts[i-1][0]; vz = pts[i+1][1]-pts[i-1][1]
        ang_self[i] = math.degrees(math.atan2(vz, vx))
    # 静止点(位移<0.5m)的 self 不参与方向一致性匹配（原始值仍写入 JSON）
    selfs_clean = ang_self.copy()
    for i in range(n):
        if i == 0:
            vx = pts[1][0]-pts[0][0]; vz = pts[1][1]-pts[0][1]
        elif i == n-1:
            vx = pts[n-1][0]-pts[n-2][0]; vz = pts[n-1][1]-pts[n-2][1]
        else:
            vx = pts[i+1][0]-pts[i-1][0]; vz = pts[i+1][1]-pts[i-1][1]
        if math.hypot(vx, vz) < 0.5:
            selfs_clean[i] = np.nan

    # 轨迹线条方向（窗口 ±20 点位移，更强的直线性判断）—— 分岔口匹配与方向判定
    WIN = 20
    win_ang = np.full(n, np.nan)
    for i in range(n):
        a = max(0, i - WIN); b = min(n - 1, i + WIN)
        vx = pts[b][0] - pts[a][0]
        vz = pts[b][1] - pts[a][1]
        if math.hypot(vx, vz) >= 0.5:
            win_ang[i] = math.degrees(math.atan2(vz, vx))

    best_seg, best_dist = mapmatch(P, win_ang)
    ang_graph = compute_angle_graph(best_seg, best_dist, win_ang, n, P, selfs_clean)

    for i, p in enumerate(data):
        pr = p.get('properties', {})
        as_ = ang_self[i]
        pr['angle_self'] = None if math.isnan(as_) else round(float(as_), 4)
        ag = ang_graph[i]
        pr['angle_graph'] = None if math.isnan(ag) else round(float(ag), 4)
        pr['graph_match_dist'] = round(float(best_dist[i]), 2) if not np.isnan(best_dist[i]) else None
        pr['graph_seg_idx'] = int(best_seg[i]) if best_seg[i] >= 0 else None

    with open(f, 'w', encoding='utf-8') as fp:
        json.dump(traj, fp, ensure_ascii=False)
    return (os.path.basename(f), True, '')

def main():
    global MATCH_THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--graph', default=DEFAULT_GRAPH)
    ap.add_argument('--th', type=float, default=MATCH_THRESHOLD)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    MATCH_THRESHOLD = args.th

    print(f'loading the network and building the adjacency: {args.graph}', flush=True)
    adj, S, D, L, PV, NV, PL = load_graph(args.graph)
    print(f'network segments: {len(S)}', flush=True)
    # 支持目录或单文件
    if os.path.isfile(args.data_dir):
        files = [args.data_dir]
        print(f'single file mode: {files[0]}', flush=True)
    else:
        files = glob.glob(os.path.join(args.data_dir, '**', '*.json'), recursive=True)
        print(f'trajectories to process: {len(files)}', flush=True)
    t0 = time.time()
    ok = fail = 0
    if args.workers <= 1:
        # 单线程顺序模式（无子进程，避免卡顿）: 主进程直接处理
        _init_global((adj, S, D, L, PV, NV, PL))   # 单线程需手动初始化全局几何
        for fi, f in enumerate(files):
            try:
                name, s, err = process_file(f)
            except PermissionError:
                # 文件被占用(编辑器打开等): 跳过继续, 不中断整个任务
                print(f'  skipped (file in use): {os.path.basename(f)}', flush=True)
                continue
            if s:
                ok += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f'  failed {name}: {err}', flush=True)
            if (fi + 1) % 300 == 0:
                el = time.time() - t0
                rate = (fi + 1) / el
                eta = (len(files) - fi - 1) / rate
                print(f'progress {fi+1}/{len(files)} ({(fi+1)/len(files):.1%}), '
                      f'成功 {ok} 失败 {fail}, 速率 {rate:.1f}条/s, 预计剩余 {eta/60:.0f}min',
                      flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_global, initargs=((adj, S, D, L, PV, NV, PL),)) as ex:
            for name, s, err in ex.map(process_file, files):
                if s:
                    ok += 1
                else:
                    fail += 1
                    if fail <= 5:
                        print(f'  failed {name}: {err}', flush=True)
    print(f'done: succeeded {ok} / failed {fail}, elapsed {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
