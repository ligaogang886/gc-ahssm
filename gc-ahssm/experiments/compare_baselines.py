# -*- coding: utf-8 -*-
"""Four published baselines plus the proposed model, evaluated per trajectory.

cvm        constant-velocity interpolation of the measured positions
ekf        extended Kalman filter with a coordinated-turn model
thmm       grid-based HMM map matching with a transition prior
flight2vec motion-based sequence model

The four baselines are re-run from the raw measurements on every call, while the
proposed model is read back from ``fusion_*`` fields of an estimator output record,
which is what the paper table reports.  All five share one error mask so the
comparison is per-sample paired.
"""
import argparse
import glob
import math
import os
import time

import numpy as np

import gcahssm as R

FLIP_TH = 150.0        # heading differences above this are reversed-reference points
V_MAX = 30.0           # physical mask on the reported speed
METHODS = ('GC-AHSSM', 'CVM', 'EKF', 'T-HMM', 'Flight2Vec')


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

def norm_deg(x):
    return float(x) % 360.0

# =========================================================
# 观测窗口方向 (独立于图/refdir)
# =========================================================
def win_dir_from_pts(pts, win=10):
    n = len(pts)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    for i in range(n):
        a = max(0, i - win); b = min(n - 1, i + win)
        vx = pts[b, 0] - pts[a, 0]; vz = pts[b, 1] - pts[a, 1]
        if math.hypot(vx, vz) >= 0.5:
            out[i] = math.degrees(math.atan2(vz, vx)) % 360.0
    return out

# =========================================================
# 方法 1: CVM (差分法)
# =========================================================
def run_cvm(pts):
    n = len(pts)
    ex = np.full(n, np.nan); ez = np.full(n, np.nan); head = np.full(n, np.nan)
    if n < 3:
        return ex, ez, head
    # 原脚本: pred_i = x1 + (x1-x2), 其中 x2=coords[i-2], x1=coords[i-1]
    d = pts[1:] - pts[:-1]                    # d[j]=obs[j+1]-obs[j]
    v = d[:-1]                                # 点 i 使用位移 (i-2)->(i-1), 即 d[i-2]
    pred = pts[1:-1] + v                      # pred[i-2] = obs[i-1] + (obs[i-1]-obs[i-2])
    ex[2:] = pred[:, 0]; ez[2:] = pred[:, 1]
    head[2:] = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360.0
    return ex, ez, head

# =========================================================
# 方法 2: EKF
# =========================================================
def _fx_state(x):
    px, vx, pz, vz, at, an = x
    V = math.hypot(vx, vz)
    if V < 0.5:
        return np.array([vx, 0.0, vz, 0.0, 0.0, 0.0])
    ax = at * (vx / V) + an * (vz / V)
    az = at * (vz / V) - an * (vx / V)
    return np.array([vx, ax, vz, az, 0.0, 0.0])

def _jacobian(x):
    px, vx, pz, vz, at, an = x
    V2 = vx * vx + vz * vz
    V = math.sqrt(V2)
    F = np.zeros((6, 6))
    if V < 0.5:
        return F
    F[0, 1] = 1; F[2, 3] = 1
    f_vx_vx = vx * vx / (V2 ** 1.5); f_vz_vz = vz * vz / (V2 ** 1.5)
    f_vx_vz = vx * vz / (V2 ** 1.5)
    F[1, 1] = at * f_vx_vx + an * f_vx_vz
    F[1, 3] = at * f_vx_vz + an * f_vz_vz
    F[1, 4] = vx / V; F[1, 5] = vz / V
    F[3, 1] = at * f_vx_vz - an * f_vx_vx
    F[3, 3] = at * f_vz_vz - an * f_vx_vz
    F[3, 4] = vz / V; F[3, 5] = -vx / V
    return F

def _rk4(x, dt):
    k1 = _fx_state(x); k2 = _fx_state(x + 0.5 * dt * k1)
    k3 = _fx_state(x + 0.5 * dt * k2); k4 = _fx_state(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

def run_ekf(pts, DT=1.0):
    from scipy.linalg import expm
    n = len(pts)
    ex = np.full(n, np.nan); ez = np.full(n, np.nan); head = np.full(n, np.nan)
    if n < 2:
        return ex, ez, head
    d0 = pts[1] - pts[0]
    L = math.hypot(d0[0], d0[1])
    if L > 0.5:
        vx0, vz0 = d0[0] / L * 0.5, d0[1] / L * 0.5
    else:
        vx0, vz0 = 0.5, 0.0
    x_est = np.array([pts[0][0], vx0, pts[0][1], vz0, 0.0, 0.0])
    P = np.eye(6)
    Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01])
    Rm = np.diag([10, 10])
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    for i in range(1, n):
        F = _jacobian(x_est)
        Phi = expm(F * DT)
        x_pred = _rk4(x_est, DT)
        P_pred = Phi @ P @ Phi.T + Q
        Z = pts[i]
        S = H @ P_pred @ H.T + Rm
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_est = x_pred + K @ (Z - H @ x_pred)
        P = (np.eye(6) - K @ H) @ P_pred
        ex[i] = x_est[0]; ez[i] = x_est[2]
        sp = math.hypot(x_est[1], x_est[3])
        head[i] = 0.0 if sp <= 1e-6 else norm_deg(math.degrees(math.atan2(x_est[3], x_est[1])))
    return ex, ez, head

# =========================================================
# 方法 3: T-HMM — 拓扑路径贪心匹配 (路径级状态 + 粘滞转移 + 距离/方向发射)
# 发射: -0.05*min_dist_to_path - 0.2*ae(path_dir, obs_win_dir)
# =========================================================
def thmm_prepare(G):
    """预计算: 路径连续段索引 (load_graph 按 path 顺序 append, 段连续)、路径净方向"""
    S = G['S']; dir_f = G['dir_f']; seg_path = G['seg_path']
    pn_order = []; counts = []
    prev = None; c = 0
    for pn in seg_path:
        if pn != prev:
            if prev is not None:
                pn_order.append(prev); counts.append(c)
            prev = pn; c = 1
        else:
            c += 1
    pn_order.append(prev); counts.append(c)
    pstart = np.concatenate([[0], np.cumsum(counts)[:-1]])   # 每 path 首段全局索引
    paths_pts = {p['pathname']: p['points'] for p in G['paths']}
    net_dir = {}
    for pn in pn_order:
        pts = paths_pts[pn]
        ddx = float(pts[-1]['x']) - float(pts[0]['x'])
        ddz = float(pts[-1]['z']) - float(pts[0]['z'])
        net_dir[pn] = norm_deg(math.degrees(math.atan2(ddz, ddx))) if math.hypot(ddx, ddz) > 1e-6 else 0.0
    return {'S': S, 'dir_f': dir_f, 'pn_order': pn_order, 'counts': np.array(counts),
            'pstart': pstart, 'net_dir': net_dir,
            'seg_path': seg_path, 'seg_global_idx': np.arange(len(seg_path))}

def run_thmm(pts, wd, TH):
    """逐点贪心: emission(路径级) + 粘滞转移(+15 同 path / -5 异 path), argmax 选 path
    位置估计 = 选中路径最近段的投影点; 方向估计 = 该段几何方向, 按 obs_win_dir 取向
    (|seg_dir-wd|>90 → +180, 与 M1 同规则; 观测决定, 不读 refdir)"""
    n = len(pts)
    ex = np.full(n, np.nan); ez = np.full(n, np.nan); head = np.full(n, np.nan)
    if n == 0:
        return ex, ez, head
    S = TH['S']; dir_f = TH['dir_f']; Pn = TH['pn_order']; P = len(Pn)
    pstart = TH['pstart']; counts = TH['counts']
    net_dir = TH['net_dir']
    A = S[:, 0]; B = S[:, 1]; dv = B - A
    L2 = np.sum(dv * dv, axis=1); L2[L2 < 1e-12] = 1e-12
    prev_p = -1
    CH = 200
    neth = np.array([net_dir[p] for p in Pn])                 # (P,)
    for c0 in range(0, n, CH):
        pc = min(CH, n - c0)
        ptsc = pts[c0:c0 + pc]                                # (pc,2)
        v = ptsc[:, None, :] - A[None, :, :]                  # (pc,M,2)
        tt = np.sum(v * dv[None, :, :], axis=2) / L2[None, :]
        np.clip(tt, 0.0, 1.0, out=tt)
        proj = A[None, :, :] + tt[..., None] * dv[None, :, :]  # (pc,M,2)
        d2 = np.sum((ptsc[:, None, :] - proj) ** 2, axis=2)   # (pc,M)
        # 路径级最小距离: reduceat 沿连续 path 分组
        dmin_p = np.minimum.reduceat(d2, pstart, axis=1)      # (pc,P)
        wdc = wd[c0:c0 + pc]
        ang = np.zeros((pc, P), float)
        m_wd = ~np.isnan(wdc)
        if m_wd.any():
            ang[m_wd] = ae_arr(neth[None, :], wdc[m_wd][:, None])
        emission = -0.05 * dmin_p - 0.2 * ang                # (pc,P)
        picks = np.zeros(pc, int)
        for k in range(pc):
            total = emission[k].copy()
            if prev_p >= 0:
                total += np.where(np.arange(P) == prev_p, 15.0, -5.0)
            pk = int(np.argmax(total))
            picks[k] = pk
            prev_p = pk
        # 输出: 选中 path 内最近段 → 投影/方向
        for k in range(pc):
            pn_idx = picks[k]
            s0 = int(pstart[pn_idx]); cg = int(counts[pn_idx])
            seg = np.arange(s0, s0 + cg)
            sub = d2[k, seg]
            gi = seg[int(np.argmin(sub))]
            ex[c0 + k] = proj[k, gi, 0]; ez[c0 + k] = proj[k, gi, 1]
            sa = dir_f[gi]
            w = wdc[k]
            if not np.isnan(w) and ae(sa, w) > 90.0:
                sa = (sa + 180.0) % 360.0
            head[c0 + k] = sa
    return ex, ez, head

# =========================================================
# 方法 4: Flight2Vec — 窗口运动学外推 (剔除真值回退泄漏)
# =========================================================
def run_flight2vec(pts, window=5):
    n = len(pts)
    ex = np.full(n, np.nan); ez = np.full(n, np.nan); head = np.full(n, np.nan)
    if n < window + 1:
        return ex, ez, head
    d = np.diff(pts, axis=0)
    inst = np.full(n, np.nan)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360.0
    inst[0] = ang[0] if len(ang) else np.nan
    inst[1:] = ang
    spd = np.hypot(d[:, 0], d[:, 1])
    for i in range(n):
        if i >= window:
            wd_ = d[i - window:i]
            sp = spd[i - window:i]
            sp_pos = sp[sp > 0]
            avg_speed = float(np.mean(sp_pos)) if sp_pos.size else 0.0
            hc = inst[i - window:i]
            hcv = hc[~np.isnan(hc)]
            avg_head = float(np.mean(np.deg2rad(hcv))) if hcv.size else 0.0
            if avg_speed > 0:
                ex[i] = pts[i - 1, 0] + avg_speed * np.cos(avg_head)
                ez[i] = pts[i - 1, 1] + avg_speed * np.sin(avg_head)
            else:
                ex[i] = pts[i - 1, 0]; ez[i] = pts[i - 1, 1]
            head[i] = norm_deg(math.degrees(avg_head)) if hcv.size else 0.0
        else:
            ex[i] = pts[i, 0]; ez[i] = pts[i, 1]
            head[i] = 0.0 if np.isnan(inst[i]) else inst[i]
    return ex, ez, head

# =========================================================
# 单轨迹评估 (五方法同掩码)
# =========================================================
def eval_track(f, TH):
    try:
        rec = json.load(open(f, encoding='utf-8'))
        rec = rec[0] if isinstance(rec, list) else rec
    except Exception:
        return None
    flio = rec.get('flio', 'A')
    data = rec['data']
    n = len(data)
    pts = np.array([[float(p['properties']['x']), float(p['properties']['z'])] for p in data])
    ref_d = np.array([float(p['properties']['refdir']) if p['properties'].get('refdir') is not None else np.nan for p in data])
    sp = np.array([float(p['properties']['kf_speed']) if p['properties'].get('kf_speed') is not None else np.nan for p in data])
    phys_ok = (sp <= V_MAX) & ~np.isnan(ref_d)
    wd = win_dir_from_pts(pts)
    out = {}
    fd = np.array([p['properties'].get('fusion_dir') for p in data], float)
    fx = np.array([p['properties'].get('fusion_x') for p in data], float)
    fz = np.array([p['properties'].get('fusion_z') for p in data], float)
    out['GC-AHSSM'] = (fx, fz, fd)
    for mname, fn in (('CVM', run_cvm), ('EKF', run_ekf), ('Flight2Vec', run_flight2vec)):
        ex, ez, hd = fn(pts)
        out[mname] = (ex, ez, hd)
    ex, ez, hd = run_thmm(pts, wd, TH)
    out['T-HMM'] = (ex, ez, hd)
    return {'flio': flio, 'phys_ok': phys_ok, 'ref': pts, 'ref_d': ref_d, 'out': out}

def track_errors(r, mname):
    ex, ez, hd = r['out'][mname]
    ref = r['ref']; rd = r['ref_d']
    xe = ex - ref[:, 0]; ze = ez - ref[:, 1]
    okp = r['phys_ok']
    ok_pos = okp & ~np.isnan(ex) & ~np.isnan(ez)
    xv = np.where(ok_pos, xe, np.nan); zv = np.where(ok_pos, ze, np.nan)
    ok_d = okp & ~np.isnan(hd) & ~np.isnan(rd)
    tv = np.where(ok_d, ae_arr(hd, rd), np.nan)
    return xv, zv, tv

def aggregate(picks, TH):
    agg = {m: {fl: {'x': [], 'z': [], 't': [], 'flip': 0} for fl in ('A', 'D')}
           for m in METHODS}
    for f in picks:
        r = eval_track(f, TH)
        if r is None:
            continue
        fl = r['flio']
        for m in METHODS:
            xv, zv, tv = track_errors(r, m)
            g = agg[m][fl]
            g['x'].append(xv[~np.isnan(xv)])
            g['z'].append(zv[~np.isnan(zv)])
            tvv = tv[~np.isnan(tv)]
            g['t'].append(tvv)
            g['flip'] += int((tvv > FLIP_TH).sum())
    return agg

def group_rows(agg, m):
    rows = []
    for nm, fl in (('Arrival A', 'A'), ('Departure D', 'D'), ('Overall A+D', None)):
        ks = ('A', 'D') if fl is None else (fl,)
        xa = np.concatenate([np.concatenate(agg[m][k]['x']) for k in ks])
        za = np.concatenate([np.concatenate(agg[m][k]['z']) for k in ks])
        ta = np.concatenate([np.concatenate(agg[m][k]['t']) for k in ks])
        flip = sum(agg[m][k]['flip'] for k in ks)
        tk = ta[ta <= FLIP_TH]
        rows.append((nm, len(xa), len(tk), metrics(xa), metrics(za), metrics(tk), int(flip)))
    return rows

def pooled_arrays(agg):
    """每方法按 A/D 汇总误差数组 (供论文 CDF 图/速度带复用; θ 已剔>150)"""
    pool = {}
    for m in METHODS:
        xa = np.concatenate([np.concatenate(agg[m][k]['x']) for k in ('A', 'D')])
        za = np.concatenate([np.concatenate(agg[m][k]['z']) for k in ('A', 'D')])
        ta = np.concatenate([np.concatenate(agg[m][k]['t']) for k in ('A', 'D')])
        pool[m] = (xa, za, ta[ta <= FLIP_TH])
    return pool

def fmt_row(r):
    nm, np_, nd, x5, z5, t5, flip = r
    s = f'  {nm:<14} n_pos={np_:>9,} n_dir={nd:>9,} 剔翻转={flip:>6,}\n'
    for lab, m5 in (('x', x5), ('y', z5), ('θ', t5)):
        s += f'      {lab}: MAE={m5[0]:7.3f} RMSE={m5[1]:7.3f} STD={m5[2]:7.3f} P50={m5[3]:7.3f} P90={m5[4]:7.3f}\n'
    return s

# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description='proposed model against four baselines')
    ap.add_argument('--input', nargs='+', required=True,
                    help='one or more directories of estimator output records')
    ap.add_argument('--graph', required=True, help='network JSON')
    ap.add_argument('--limit', type=int, default=0, help='cap the records of each directory')
    args = ap.parse_args()

    G = R.load_graph(args.graph)
    TH = thmm_prepare(G)
    print('[grid prior baseline] paths %d, segments %d' % (len(TH['pn_order']), len(TH['S'])))

    agg = {m: {fl: {'x': [], 'z': [], 't': [], 'flip': 0} for fl in ('A', 'D')}
           for m in METHODS}
    n_rec = 0
    t0 = time.time()
    for d in args.input:
        picks = sorted(glob.glob(os.path.join(d, '*.json')))
        if args.limit:
            picks = picks[:args.limit]
        part = aggregate(picks, TH)
        n_rec += len(picks)
        for m in METHODS:
            for fl in ('A', 'D'):
                for k in ('x', 'z', 't'):
                    agg[m][fl][k].extend(part[m][fl][k])
                agg[m][fl]['flip'] += part[m][fl]['flip']
        print('[ok] %s: %d records (%.1f min)' % (d, len(picks), (time.time() - t0) / 60.0))

    print('=' * 88)
    print('protocol: position on all points including holding, heading above %g deg removed, '
          'physical mask kf_speed <= %g m/s' % (FLIP_TH, V_MAX))
    print('metrics : MAE = mean|e|, RMSE = sqrt(mean e^2), STD = std(e), P50 and P90 on |e|')
    print('records : %d' % n_rec)
    print('=' * 88)
    for m in METHODS:
        print('\n=== %s ===' % m)
        print(fmt_row(group_rows(agg, m)), end='')


if __name__ == '__main__':
    main()

