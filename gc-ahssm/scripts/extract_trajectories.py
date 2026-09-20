# -*- coding: utf-8 -*-
"""
重庆江北轨迹提取 —— 对齐西安 "西安机场/轨迹数据/<YYYYMMDD>/<cfno>-<afn>-<flio>.json" 格式
================================================================================
上游数据形态与西安不同，因此这里的处理链路需要自己完成:
  主机场: 单航班轨迹 JSON（每航班一个文件, 已切分为单个运动段）
  重庆: 轨迹数据/<YYYYMMDD>/YYYYMMDDHHMMSS.db        (按分钟切分的原始监视库)

本脚本完成的转换:
  1. 逐库读取 track 表, 只用 flight_transform_x/z (经纬度) 重建轨迹点,
     按 flight_id 分组 (craft 会复用, flight_id 才唯一), 按 send_data_at 排序去重
  2. 机场区域过滤: 保留落在机场包络(含缓冲)内的轨迹, 剔除航路/外站目标
  3. 进出港判定: 沿用西安 finalize_xian_trajectories.py 的端点语义
     A(进港) = 前段跑道 + 后段机坪 ; D(离港) = 前段机坪 + 后段跑道
     阈值同样为 RW=35m / AP=100m / ST=30m
  4. 坐标: 经纬度 -> 重庆标定 6 参数仿射 -> x/z (与 chongqing_airport_paths.json 同系)
  5. 输出西安同构 JSON: [{"data":[{"time","properties":{originLon,originLat,x,z,...}}],
                          "cfno","afn","flio","rway","stno","starttime","endtime","pointcount","flid"}]

用法:
    python extract_trajectories.py --days 20250306:20250312 --jobs 6
    python extract_trajectories.py --days 20250306            # 单天试点
"""
import os
import sys
import json
import math
import time
import sqlite3
import argparse
import glob
import datetime as dt
import collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xml.etree.ElementTree as ET
from matplotlib.path import Path as MplPath

# The archive, the network and the output location are given on the command line;
# nothing about the operator-specific layout is baked into this file.
SRC_ROOT = None      # --src-root : nested directory of the daily surveillance databases
OUT_ROOT = None      # --out      : destination of the trajectory records
OSM = None           # --osm      : cleaned OSM extract, used for the field boundary
PATHS_JSON = None    # --paths    : network JSON, used for the terminal-point check

# 重庆标定仿射 (见 重庆机场/仿射变换_公式及参数.md)
AFF = (69219.734239, 73016.824547, -60472.614251, 83475.891532,
       -10045470.936244, 3701948.837999)

# 西安口径阈值(米)
SEG_RATIO = 0.4
SAMPLE = 2
RW, AP, ST = 35.0, 100.0, 30.0
MIN_PTS = 8            # 轨迹最少点数(西安口径, judge 里 len<8 直接丢)
BUF_M = 800.0          # 机场包络缓冲(米), 用于"机场区域外"过滤
SAMPLE_M = 0.25        # 距离查询点云的采样间距(米), 误差 <= SAMPLE_M/2
FIELD_M = 300.0        # 场内判定距离(米): 距任一机场要素超过它的点视为"机场区域外"
MIN_FIELD_PTS = 30     # 裁切后场内片段的最少点数(约 30s 滑行)
STILL_R = 30.0         # (保留) 旧半径判据阈值, 已被滑行速度裁切取代
STILL_K = 10           # 静止判定的窗口长度(点), 1Hz 等价于秒
STILL_WIN = 15.0       # 窗口位移阈值(米) -> 窗口平均速度阈值 1.5 m/s
MOVE_K = 10            # 滑行速度判定的窗口长度(点, 1Hz 等价秒)
MOVE_THR = 2.0         # 滑行速度阈值(m/s): 低于它的首/末段算机位推出/停稳, 裁掉
                       # 标定依据: 西安 D 类首端静止段 p50=11 点(无推出段),
                       # 重庆含 90~300 点 1~2 m/s 推出段; 阈值 2.0 时两侧口径一致
DWELL_MAX = 0.0        # (已废弃, 保留占位) 旧"停留保留秒数"参数, 由 MOVE_THR 裁切取代
MIN_DWELL_S = 60       # (已废弃) 旧静止段最小长度
GAP_MAX = 180          # 相邻定位报告间隔上限(秒); 超过视为目标失锁, 切开取最长段

_G = None
_GEO_CACHE = {}


# ---------------------------------------------------------------- 几何
def affine(lon, lat):
    a, b, c, d, tx, tz = AFF
    return a * lon + b * lat + tx, c * lon + d * lat + tz


def _load_geo():
    """把 OSM 的跑道/机坪/停机位转到 x/z 系 (与轨迹同系), 并给出机场包络

    距离查询用"段加密采样点云 + cKDTree"实现, 采样间距 SAMPLE_M;
    与精确线段距离的偏差 <= SAMPLE_M/2 (0.25m 采样 -> <=0.125m),
    相对 35/100/30 m 的判定阈值可忽略, 但换取 2~3 个数量级的提速。
    """
    if _GEO_CACHE:
        return _GEO_CACHE
    root = ET.parse(OSM).getroot()
    nd = {}
    for n in root.findall('node'):
        nd[n.get('id')] = (float(n.get('lat')), float(n.get('lon')))
    RS, AS, SS, POLY = [], [], [], []
    RW_NAME = []
    for w in root.findall('way'):
        t = {x.get('k'): x.get('v') for x in w.findall('tag')}
        aero = t.get('aeroway')
        if aero not in ('runway', 'apron', 'parking_position'):
            continue
        pts = [nd[x.get('ref')] for x in w.findall('nd') if x.get('ref') in nd]
        if len(pts) < 2:
            continue
        xy = np.array([affine(lon, lat) for lat, lon in pts], dtype=float)
        if aero == 'runway':
            RS.append(xy)
            RW_NAME.append(t.get('ref') or t.get('name') or '')
        elif aero == 'apron':
            AS.append(xy)
            if len(xy) >= 3:
                POLY.append(xy)
        else:
            SS.append(xy)

    def segs(arrs):
        if not arrs:
            return np.zeros((0, 2, 2))
        return np.vstack([np.stack([a[:-1], a[1:]], axis=1) for a in arrs])

    def cloud(arrs, sample_m=SAMPLE_M):
        """把折线密集采样成点云, 建 KD 树"""
        chunks = []
        for a in arrs:
            for i in range(len(a) - 1):
                p, q = a[i], a[i + 1]
                L = math.hypot(q[0] - p[0], q[1] - p[1])
                if L < 1e-9:
                    continue
                m = max(2, int(L / sample_m) + 1)
                tt = np.linspace(0.0, 1.0, m)
                chunks.append(p[None, :] + (q - p)[None, :] * tt[:, None])
        P = np.vstack(chunks) if chunks else np.zeros((0, 2))
        from scipy.spatial import cKDTree
        tree = cKDTree(P) if len(P) else None
        return tree, P

    RSa, ASa, SSa = segs(RS), segs(AS), segs(SS)
    allpts = np.vstack([a for a in (RS + AS + SS) if len(a)])
    bbox = (allpts[:, 0].min() - BUF_M, allpts[:, 0].max() + BUF_M,
            allpts[:, 1].min() - BUF_M, allpts[:, 1].max() + BUF_M)
    geo = {'RS': RSa, 'AS': ASa, 'SS': SSa, 'POLY': POLY, 'bbox': bbox,
           'n_poly': len(POLY), 'rw_arrs': RS, 'rw_names': RW_NAME}
    for k, arrs in (('trw', RS), ('tap', AS), ('tst', SS)):
        geo[k], geo[k + '_pts'] = cloud(arrs)
    geo['rw_tree'] = [cloud([a])[0] for a in RS]
    _GEO_CACHE.update(geo)
    return geo


def _tree_dist(tree, P):
    if tree is None:
        return np.full(len(P), np.inf)
    d, _ = tree.query(P)
    return d


def _min_dist(segs, P):
    """精确点到线段集最小距离(保留作为校验基线; 判定走 _tree_dist)"""
    if len(segs) == 0:
        return np.full(len(P), np.inf)
    a = segs[:, 0][None, :, :]
    d = segs[:, 1] - segs[:, 0]
    d2 = np.sum(d * d, axis=1)
    d2[d2 < 1e-12] = 1e-12
    out = np.full(len(P), np.inf)
    step = max(1, int(2.0e6 / max(1, len(segs))))
    for i in range(0, len(P), step):
        Q = P[i:i + step]
        v = Q[:, None, :] - a
        t = np.clip(np.sum(v * d[None, :, :], axis=2) / d2[None, :], 0, 1)
        proj = a + t[:, :, None] * d[None, :, :]
        out[i:i + step] = np.min(np.linalg.norm(Q[:, None, :] - proj, axis=2), axis=1)
    return out


def _init_worker(geo):
    global _G
    _G = geo
    _G['paths'] = [MplPath(p) for p in geo['POLY']] if geo['POLY'] else []


def _bbox_only(geo):
    """仅把包络传给读库子进程, 避免序列化整棵树"""
    return {'bbox': geo['bbox']}


def _sub_init(light):
    global _G
    _G = light


def _in_apron(P):
    if not _G['paths']:
        return np.zeros(len(P), dtype=bool)
    ins = np.zeros(len(P), dtype=bool)
    for pth in _G['paths']:
        ins |= pth.contains_points(P)
    return ins


# ---------------------------------------------------------------- 单库提取
def read_db(fn):
    """返回候选行: (fid, craft, fno, t, lon, lat, hdg, spd, ftype, site)"""
    con = sqlite3.connect('file:' + fn.replace(os.sep, '/') + '?mode=ro', uri=True)
    try:
        rows = list(con.execute(
            "select flight_id,craft,flight_no,send_data_at,flight_transform_x,"
            "flight_transform_z,flight_rotation,speed,flight_type,craft_site "
            "from track where flight_transform_x is not null and flight_transform_x<>'' "
            "and position_x is not null and position_x<>''"))
    finally:
        con.close()
    out = []
    x0, x1, z0, z1 = _G['bbox']
    for r in rows:
        try:
            lon = float(r[4]); lat = float(r[5])
        except (TypeError, ValueError):
            continue
        x, z = affine(lon, lat)
        if not (x0 <= x <= x1 and z0 <= z <= z1):
            continue
        try:
            hdg = float(r[6])
        except (TypeError, ValueError):
            hdg = float('nan')
        try:
            spd = float(r[7])
        except (TypeError, ValueError):
            spd = float('nan')
        out.append((r[0], r[1], r[2], int(r[3]), lon, lat, hdg, spd,
                    r[8], r[9] or ''))
    return out


def field_dist(P):
    """点到"机场要素"的最小距离 = min(跑道, 机坪, 停机位)"""
    return np.minimum(np.minimum(_tree_dist(_G['trw'], P),
                                 _tree_dist(_G['tap'], P)),
                      _tree_dist(_G['tst'], P))


def longest_field_run(P, thr=None):
    """返回距机场要素 < thr 的最长连续片段 [i0, i1) ; 全在外返回 None

    重庆原始监视流从进近一直录到离场爬升, 若直接套西安的"首段/末段"端点语义,
    离港轨迹的后 40% 会被场外爬升点淹没而使 D 类判定失败。因此先截出场内片段。
    """
    thr = FIELD_M if thr is None else thr
    m = field_dist(P) < thr
    i0 = i1 = 0
    s = None
    for i, v in enumerate(m):
        if v:
            if s is None:
                s = i
        elif s is not None:
            if i - s > i1 - i0:
                i0, i1 = s, i
            s = None
    if s is not None and len(m) - s > i1 - i0:
        i0, i1 = s, len(m)
    if i1 - i0 < 1:
        return None
    return i0, i1


def judge_ends(P):
    """西安口径的端点语义判定。
    返回 (flio, 明细) ; flio in {'A','D',None}
      A(进港): 前段在跑道 且 后段在机坪
      D(离港): 前段在机坪 且 后段在跑道
    """
    k = max(2, int(len(P) * SEG_RATIO))

    def stats(seg):
        seg = seg[::SAMPLE]
        dr = _tree_dist(_G['trw'], seg).min()
        dap = _tree_dist(_G['tap'], seg).min()
        dst = _tree_dist(_G['tst'], seg).min()
        inap = _in_apron(seg)
        near = (dap < AP) | inap | (dst < ST)
        return (dr < RW, bool(near.any()), dr, dap, dst)

    s0 = stats(P[:k])
    s1 = stats(P[-k:])
    if s0[0] and s1[1]:
        flio = 'A'
    elif s0[1] and s1[0]:
        flio = 'D'
    else:
        flio = None
    det = dict(front_rw=s0[0], front_ap=s0[1], back_rw=s1[0], back_ap=s1[1],
               front_drw=s0[2], front_dap=s0[3], front_dst=s0[4],
               back_drw=s1[2], back_dap=s1[3], back_dst=s1[4],
               disp_m=float(np.linalg.norm(P[-1] - P[0])))
    return flio, det


def infer_rway(P, flio):
    """用离跑道最近的那一端反查跑道名(对齐西安 json 的 rway 字段)"""
    trees = _G.get('rw_tree') or []
    names = _G.get('rw_names') or []
    if not trees:
        return ''
    k = max(2, int(len(P) * SEG_RATIO))
    end = P[:k] if flio == 'A' else P[-k:]
    best, bi = float('inf'), -1
    for i, tree in enumerate(trees):
        d = _tree_dist(tree, end).min()
        if d < best:
            best, bi = d, i
    if bi < 0 or best > 5.0 * RW:      # 离所有跑道都太远 -> 不给跑道名
        return ''
    return names[bi] if bi < len(names) else ''


def _still_run(P, from_head):
    """返回 [a, b): 首端(或末端)静止段区间(抗噪窗口判据)

    判据 = 连续 K 点窗口位移 < STILL_WIN(米), 等价于窗口平均速度 <
    STILL_WIN/K m/s。两种朴素判据都不可用:
      · "距首点半径": 机位上监视位置有 ±50m 级漂移, 半径 30m 判据实测
        把一条 487 点静止段只认出 20 点;
      · "逐点步长": 单点噪声跳变(2~5m)即打断, 实测只认出 115 点。
    窗口判据对缓漂移和单点跳变都不敏感, 对真实起步/停稳敏感。
    """
    n = len(P)
    if n < 2:
        return 0, n
    K, W = STILL_K, STILL_WIN
    if n <= K:
        return (0, n) if np.linalg.norm(P[-1] - P[0]) < W else (0, 0)
    if from_head:
        j = 0
        while j + K < n and np.linalg.norm(P[j + K] - P[j]) < W:
            j += 1
        return (0, min(n, j + K)) if j > 0 else (0, 0)
    j = n - 1
    while j - K >= 0 and np.linalg.norm(P[j - K] - P[j]) < W:
        j -= 1
    return (max(0, j + 1 - K), n) if j < n - 1 else (n, n)


def longest_active(T, gap_max=None):
    """取"定位报告连续"的最长片段 [i0, i1)

    重庆监视流在目标离开后仍会继续重发上一个位置(实测: 单条 flight_id 可以有
    53,401 秒位置逐位相同), 按整段入库会得到 15 小时长的僵尸轨迹。这里按
    相邻报告间隔切开, 只留最长的一段连续跟踪。
    """
    gap_max = GAP_MAX if gap_max is None else gap_max
    n = len(T)
    if n == 0:
        return None
    brk = np.where(np.diff(T) > gap_max)[0]
    starts = np.concatenate([[0], brk + 1])
    ends = np.concatenate([brk + 1, [n]])
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def resample_1hz(rs):
    """把原生报告重采样到 1Hz 均匀网格(位置线性插值, 其余字段最近邻带出)

    位置字段原生更新间隔实测中位 2s(1~5s), 逐秒报告里同一位置会重复多次
    (零阶保持台阶)。若直接按 1Hz 落库, 运动段约 60% 的相邻步是零位移,
    所有方法的窗口方向与差分会一起失真。这里对"位置更新事件"做线性插值
    再取 1Hz 网格, 与西安数据同节拍, KF 仍保持 dt=1 配置不变。
    """
    if len(rs) < 2:
        return rs
    t = np.array([r[3] for r in rs], dtype=float)
    lon = np.array([r[4] for r in rs], dtype=float)
    lat = np.array([r[5] for r in rs], dtype=float)
    if t[-1] - t[0] < 1.0:
        return rs
    chg = np.concatenate([[True], (np.abs(np.diff(lon)) > 1e-9) |
                          (np.abs(np.diff(lat)) > 1e-9)])
    tk, lk, ak = t[chg], lon[chg], lat[chg]
    grid = np.arange(t[0], t[-1] + 0.5, 1.0)
    j = np.clip(np.searchsorted(t, grid, side='right') - 1, 0, len(rs) - 1)
    glon = np.interp(grid, tk, lk)
    glat = np.interp(grid, tk, ak)
    return [(rs[j[i]][0], rs[j[i]][1], rs[j[i]][2], int(grid[i]),
             glon[i], glat[i], rs[j[i]][6], rs[j[i]][7], rs[j[i]][8],
             rs[j[i]][9]) for i in range(len(grid))]


def trim_dwell(P, flio, max_dwell=None):
    """把首/末端的机位静止段与低速推出/停稳段裁掉, 只保留滑行段

    西安口径实测(0508/0512): D 类首端静止段 p50=11 点, A 类末端 p50=88 点,
    即轨迹近似"起于滑行、止于停稳", 不含机位推出段。重庆原始监视流包含
    90~300 点的 1~2 m/s 推出段, 其方向由位置差分驱动、噪声极大:
    留它 D 类 θMAE=36.2°, 裁到 >=2 m/s 后 9.1° (同口径西安 13.4°)。
    对西安同口径回放几乎无影响(14.37 -> 13.39), 证明这就是西安既有切分口径。

    max_dwell 仅保留签名兼容, 不再使用。
    """
    n = len(P)
    K = MOVE_K
    if n < K + MIN_FIELD_PTS:
        return 0, n, 0
    v = np.array([np.linalg.norm(P[i + K] - P[i]) / K for i in range(n - K)])
    idx = np.where(v >= MOVE_THR)[0]
    if len(idx) == 0:
        return 0, n, 0
    i0 = int(idx[0])
    i1 = int(idx[-1]) + K
    return i0, i1, (i0 + (n - i1))


def worker_day(args):
    """处理一天: 并行读库 -> 按 flight_id 归并 -> 判定 -> 写 JSON"""
    day, jobs, verbose, limit = args
    files = sorted(glob.glob(os.path.join(SRC_ROOT, day, '*.db')))
    if limit:
        files = files[:limit]
    t0 = time.time()
    geo = _load_geo()
    rows = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_sub_init,
                             initargs=(_bbox_only(geo),)) as ex:
        for i, res in enumerate(ex.map(read_db, files, chunksize=8)):
            rows.extend(res)
    t_read = time.time() - t0

    # 按 flight_id 归并
    byf = collections.defaultdict(list)
    for r in rows:
        byf[r[0]].append(r)

    _init_worker(geo)
    outday = os.path.join(OUT_ROOT, day)
    os.makedirs(outday, exist_ok=True)

    stat = collections.Counter()
    used_name = set()
    objs = []
    dbg = []          # 被拒样本的明细(前 8 条)
    for fid, rs in byf.items():
        rs.sort(key=lambda x: x[3])
        # 跨库去重(同秒保留首条)
        ded, seen = [], set()
        for r in rs:
            if r[3] in seen:
                continue
            seen.add(r[3]); ded.append(r)
        rs = ded
        stat['by_fid'] += 1
        if len(rs) < MIN_PTS:
            stat['too_short'] += 1
            continue
        P = np.array([affine(r[4], r[5]) for r in rs])
        T = np.array([r[3] for r in rs], dtype=float)
        # 0) 切出"定位报告连续"的最长段(剔除位置停滞/目标失锁的僵尸段)
        act = longest_active(T)
        if act is None or act[1] - act[0] < MIN_PTS:
            stat['no_active'] += 1
            continue
        a0, a1 = act
        if a1 - a0 < len(rs):
            stat['cut_stale'] += len(rs) - (a1 - a0)
        P, rs, T = P[a0:a1], rs[a0:a1], T[a0:a1]
        # 1) 截出场内片段(剔除包络内但远离机场要素的点)
        seg = longest_field_run(P)
        if seg is None:
            stat['off_field'] += 1
            continue
        i0, i1 = seg
        if i1 - i0 < MIN_FIELD_PTS:
            stat['field_short'] += 1
            continue
        P, rs, T = P[i0:i1], rs[i0:i1], T[i0:i1]
        # 2) 端点语义判 A/D
        flio, det = judge_ends(P)
        if flio is None:
            if det['disp_m'] < 50.0:
                stat['static'] += 1      # 场内位移<50m = 停机位静止目标
            stat['reject_ends'] += 1
            if len(dbg) < 8 and det['disp_m'] >= 50.0:
                dbg.append((fid, len(rs), det))
            continue
        # 3) 裁掉机位停机段(每端保留 DWELL_MAX 秒), 使运动段占比与西安可比
        j0, j1, cut = trim_dwell(P, flio)
        if j1 - j0 < MIN_PTS:
            stat['dwell_all'] += 1
            continue
        P, rs = P[j0:j1], rs[j0:j1]
        stat['keep_' + flio] += 1
        stat['cut_pts'] += cut
        # 4) 重采样到 1Hz 均匀网格(与西安同节拍, KF 仍按 dt=1)
        rs = resample_1hz(rs)

        r0 = rs[0]
        craft, fno = r0[1], r0[2]
        rway = infer_rway(P, flio)
        # 文件名: <craft>-<flight_no>-<flio>.json, 冲突时加序号
        base = '%s-%s-%s' % (craft or fid, fno or fid, flio)
        outn = base
        j = 1
        while outn in used_name:
            j += 1; outn = '%s_%d' % (base, j)
        used_name.add(outn)

        data = []
        for r in rs:
            lon, lat, hdg, spd = r[4], r[5], r[6], r[7]
            x, z = affine(lon, lat)
            bj = (dt.datetime.fromtimestamp(r[3], dt.UTC) + dt.timedelta(hours=8))
            props = {
                'originLon': repr(lon), 'originLat': repr(lat),
                'lon': lon, 'lat': lat, 'x': round(x, 6), 'z': round(z, 6),
                'cfno': craft, 'afn': fno, 'flio': flio,
                'stno': r[9], 'speed': None if math.isnan(spd) else spd,
                'rotation': None if math.isnan(hdg) else hdg,
                'ftype': r[8], 'flid': fid, 'rway': rway,
                'dapn': '重庆', 'originTime': bj.strftime('%Y-%m-%d %H:%M:%S'),
            }
            data.append({'actp': 'Track', 'flag': 0, 'source': 'S',
                         'time': bj.strftime('%Y-%m-%d %H:%M:%S'),
                         'type': 'Plane', 'uuid': '%s-%d' % (fid, r[3]),
                         'properties': props})
        obj = [{
            'data': data, 'cfno': craft, 'afn': fno, 'flio': flio,
            'rway': rway, 'stno': r0[9], 'dapn': '重庆', 'flid': fid,
            'starttime': data[0]['time'], 'endtime': data[-1]['time'],
            'pointcount': len(data),
        }]
        objs.append((outn, obj))
        if verbose and len(objs) % 50 == 0:
            print(f'  {day} resolved {len(objs)} records...', flush=True)

    for outn, obj in objs:
        with open(os.path.join(outday, outn + '.json'), 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False)

    el = time.time() - t0
    print(f'[{day}] libraries {len(files)}  candidate rows {len(rows):,}  flight_id {stat["by_fid"]}  '
          f'保留 A={stat["keep_A"]} D={stat["keep_D"]}  '
          f'丢[全在场外 {stat["off_field"]} / 场内过短 {stat["field_short"]} / '
          f'静止 {stat["static"]} / 太短 {stat["too_short"]} / 端点不符 {stat["reject_ends"]} / '
          f'裁后过短 {stat["dwell_all"]} / 无活跃段 {stat["no_active"]}]  '
          f'裁掉[僵尸点 {stat["cut_stale"]:,} / 机位停留点 {stat["cut_pts"]:,}]  '
          f'耗时 {el:.0f}s (读 {t_read:.0f}s)', flush=True)
    for fid, n, d in dbg:
        print(f'    - rejected {fid} n={n} displacement {d["disp_m"]:.0f} m '
              f'前[drw={d["front_drw"]:.1f} dap={d["front_dap"]:.1f} dst={d["front_dst"]:.1f}] '
              f'后[drw={d["back_drw"]:.1f} dap={d["back_dap"]:.1f} dst={d["back_dst"]:.1f}] '
              f'前(跑道{d["front_rw"]} 机坪{d["front_ap"]}) 后(跑道{d["back_rw"]} 机坪{d["back_ap"]})',
              flush=True)
    return dict(day=day, files=len(files), rows=len(rows), fids=stat['by_fid'],
                A=stat['keep_A'], D=stat['keep_D'], short=stat['too_short'],
                ends=stat['reject_ends'], static=stat['static'],
                cut=stat['cut_pts'], sec=round(el, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', required=True, help='20250306 或 20250306:20250312')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0,
                    help='每天只处理前 N 个库(试点用; 会截断跨库轨迹, 数字不作准)')
    ap.add_argument('--out', default=None, help='输出根目录')
    ap.add_argument('--src-root', default=None, help='每日监视数据库所在目录')
    ap.add_argument('--osm', default=None, help='清洗后的 OSM 文件')
    ap.add_argument('--paths', default=None, help='路网 JSON')
    args = ap.parse_args()
    for name, value in (('OUT_ROOT', args.out), ('SRC_ROOT', args.src_root),
                        ('OSM', args.osm), ('PATHS_JSON', args.paths)):
        if value:
            globals()[name] = value
    for name in ('SRC_ROOT', 'OUT_ROOT', 'OSM', 'PATHS_JSON'):
        if globals()[name] is None:
            ap.error('--%s is required (or provide it through the default)' % name.lower().replace('_', '-'))

    if ':' in args.days:
        d0 = dt.date(int(args.days[:4]), int(args.days[4:6]), int(args.days[6:8]))
        d1 = dt.date(int(args.days.split(':')[1][:4]),
                     int(args.days.split(':')[1][4:6]),
                     int(args.days.split(':')[1][6:8]))
        days = []
        d = d0
        while d <= d1:
            days.append(d.strftime('%Y%m%d')); d += dt.timedelta(days=1)
    else:
        days = [args.days]
    days = [d for d in days if os.path.isdir(os.path.join(SRC_ROOT, d))]
    print('days to process:', days, ' workers', args.jobs, flush=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

    t0 = time.time()
    sums = []
    for day in days:                      # 逐天串行(跨库归并需整天数据)
        sums.append(worker_day((day, args.jobs, True, args.limit)))
    print('\n===== summary =====')
    ta = sum(s['A'] for s in sums); td = sum(s['D'] for s in sums)
    for s in sums:
        print(f"  {s['day']}  libraries {s['files']:5d}  candidate rows {s['rows']:10,d}  "
              f"A={s['A']:4d} D={s['D']:4d}  太短{s['short']:4d} 端点不符{s['ends']:4d}  {s['sec']:.0f}s")
    print(f'\ntotal arrival={ta}  departure={td}  all={ta+td}')
    print(f'output directory: {OUT_ROOT}')
    print(f'total elapsed {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
