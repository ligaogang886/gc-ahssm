# -*- coding: utf-8 -*-
"""Two graph-prior sequence baselines.

m1  offline HMM map matching (distance emission, self loop plus adjacency transition)
m2  graph-free hybrid: constant-velocity filter, segment-wise emission over the whole
    network and an equal-weight fusion of motion and topology

Both are re-run from the raw measurements so that they share the error mask of the
proposed model.
"""
import argparse
import glob
import math
import os

import numpy as np

import gcahssm as R

EPS = 1e-12
FLIP_TH = 150.0
V_MAX = 30.0


def metrics(err):
    a = np.asarray(err, float)
    if a.size == 0:
        return (float('nan'),) * 5
    return (float(np.mean(np.abs(a))), float(np.sqrt(np.mean(a ** 2))),
            float(np.std(a)), float(np.median(np.abs(a))),
            float(np.percentile(np.abs(a), 90)))


def ae(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)

def win_dir(data, win=10):
    pts = np.array([[float(t['properties']['x']), float(t['properties']['z'])] for t in data])
    n = len(pts); out = [None]*n
    for i in range(n):
        a = max(0, i-win); b = min(n-1, i+win)
        vx = pts[b,0]-pts[a,0]; vz = pts[b,1]-pts[a,1]
        if math.hypot(vx, vz) >= 0.5:
            out[i] = math.degrees(math.atan2(vz, vx)) % 360
    return out

def _proj(pts, gi, G):
    a_pt, b_pt = G['S'][gi]
    vx = b_pt[0]-a_pt[0]; vz = b_pt[1]-a_pt[1]
    d2 = vx*vx+vz*vz
    if d2 < 1e-9: return a_pt
    t = max(0.0, min(1.0, ((pts[0]-a_pt[0])*vx + (pts[1]-a_pt[1])*vz)/d2))
    return (a_pt[0]+t*vx, a_pt[1]+t*vz)

# ---------------- M1: HMM map-matching (离线全程 Viterbi, 发射=仅距离, 转移=自环0.7+邻接0.3) ----
def run_m1(data, G):
    n = len(data)
    pts = np.array([[float(t['properties']['x']), float(t['properties']['z'])] for t in data])
    wd = win_dir(data)
    ip_tree = G['ip_tree']; n_ip = len(G['ip_owner'])
    KIP = 64; K1 = 16
    tm_adj = G['adjacency']
    sigma_d = R.SIGMA_DIST
    cands_t = []
    for i in range(n):
        dd, ii = ip_tree.query(pts[i], k=min(KIP, n_ip))
        dd = np.atleast_1d(dd); ii = np.atleast_1d(ii)
        seen = {}
        for ip_i, di in zip(ii, dd):
            pn_ip, si_ip, _ = G['ip_owner'][int(ip_i)]
            gi = G['seg_global'].get((pn_ip, si_ip))
            if gi is None or gi in seen: continue
            seen[gi] = float(di)
        cands_t.append(sorted(seen.items(), key=lambda kv: kv[1])[:K1])
    deltas = []; bps = []
    for i in range(n):
        cur = {}; bp_i = {}
        for gi, d in cands_t[i]:
            emit = math.log(math.exp(-d*d/(2*sigma_d*sigma_d)) + EPS)
            if i == 0:
                cur[gi] = emit; bp_i[gi] = None
            else:
                best = -1e18; bp_gi = None
                for pgi, pscore in deltas[-1].items():
                    if pgi == gi:
                        lt = math.log(0.7 + EPS)
                    elif gi in tm_adj.get(G['seg_path'][pgi], []):
                        lt = math.log(0.3*math.exp(-30.0/300.0) + EPS)
                    else:
                        continue
                    s = pscore + lt
                    if s > best: best = s; bp_gi = pgi
                cur[gi] = best + emit; bp_i[gi] = bp_gi
        deltas.append(cur); bps.append(bp_i)
    seq = [0]*n
    gi = max(deltas[-1].items(), key=lambda kv: kv[1])[0]
    seq[-1] = gi
    for i in range(n-2, -1, -1):
        gi = bps[i+1].get(seq[i+1])
        if gi is None or gi not in deltas[i]:
            gi = max(deltas[i].items(), key=lambda kv: kv[1])[0]
        seq[i] = gi
    head = [None]*n; px = [None]*n; pz = [None]*n
    for i in range(n):
        gi = seq[i]
        sx, sz = _proj(pts[i], gi, G)
        px[i] = sx; pz[i] = sz
        seg_ang = G['dir_f'][gi]
        if wd[i] is not None and ae(seg_ang, wd[i]) > 90:
            seg_ang = (seg_ang + 180) % 360
        head[i] = seg_ang
    return head, px, pz

# ---------------- M2: KF-HMM 无图 (3-1 KF + 全段emission + Viterbi自环0.8/均匀, 等权融合) --------
def run_m2(data, G):
    R.clean_missing_obs(data)
    R.step31_kf(data)
    M = len(G['S']); S = G['S']; dir_f = G['dir_f']; dir_b = (dir_f+180) % 360
    a_pts = S[:, 0]; d_vec = S[:, 1]-S[:, 0]; d2v = np.sum(d_vec*d_vec, axis=1)
    d2v[d2v < 1e-12] = 1e-12
    wd = win_dir(data)
    n = len(data); K = R.TOP_K
    emit_t = []
    kf_pts = []
    for it in data:
        pr = it['properties']
        kf_pts.append((float(pr['kf_x']), float(pr['kf_z']), float(pr.get('kf_dir', 0))))
    Pk = np.array([[p[0], p[1]] for p in kf_pts])
    for i in range(n):
        v = Pk[i][None, :] - a_pts
        tt = np.clip(np.sum(v * d_vec, axis=1) / d2v, 0, 1)
        proj = a_pts + tt[:, None] * d_vec
        dist = np.linalg.norm(Pk[i][None, :] - proj, axis=1)
        od = wd[i]
        if od is not None:
            ef = np.abs((od - dir_f + 180) % 360 - 180)
            eb = np.abs((od - dir_b + 180) % 360 - 180)
            ang = np.minimum(ef, eb)
        else:
            ang = np.zeros(M)
        emit = np.exp(-dist**2/(2*R.SIGMA_DIST**2)) * np.exp(-ang**2/(2*R.SIGMA_DIR**2))
        top = np.argpartition(-emit, K-1)[:K]
        top = top[np.argsort(-emit[top])]
        emit_t.append(list(zip(top.tolist(), emit[top].tolist())))
    # Viterbi: 转移 = 自环 0.8 + 0.2 均匀于其他候选 (无图)
    deltas = []; bps = []
    for i in range(n):
        cur = {}; bp_i = {}
        for gi, e in emit_t[i]:
            le = math.log(e + EPS)
            if i == 0:
                cur[gi] = le; bp_i[gi] = None
            else:
                L = len(emit_t[i-1])
                best = -1e18; bp_gi = None
                for pgi, pe in emit_t[i-1]:
                    lt = math.log(0.8 + EPS) if pgi == gi else math.log(0.2/max(L-1, 1) + EPS)
                    s = deltas[-1][pgi] + lt
                    if s > best: best = s; bp_gi = pgi
                cur[gi] = best + le; bp_i[gi] = bp_gi
        deltas.append(cur); bps.append(bp_i)
    seq = [0]*n
    gi = max(deltas[-1].items(), key=lambda kv: kv[1])[0]
    seq[-1] = gi
    for i in range(n-2, -1, -1):
        gi = bps[i+1].get(seq[i+1])
        if gi is None or gi not in deltas[i]:
            gi = max(deltas[i].items(), key=lambda kv: kv[1])[0]
        seq[i] = gi
    head = [None]*n; px = [None]*n; pz = [None]*n
    for i in range(n):
        gi = seq[i]
        sx, sz = _proj(Pk[i], gi, G)
        seg_ang = dir_f[gi]
        od = wd[i]
        if od is not None and ae(seg_ang, od) > 90:
            seg_ang = (seg_ang + 180) % 360
        kd = kf_pts[i][2]
        va = np.array([math.cos(math.radians(kd)), math.sin(math.radians(kd))])
        vb = np.array([math.cos(math.radians(seg_ang)), math.sin(math.radians(seg_ang))])
        vf = 0.5*va + 0.5*vb
        head[i] = math.degrees(math.atan2(vf[1], vf[0])) % 360
        px[i] = 0.5*Pk[i][0] + 0.5*sx
        pz[i] = 0.5*Pk[i][1] + 0.5*sz
    return head, px, pz

# ---------------- GC-AHSSM (定稿) ----------------
def run_gc(data, G):
    R.clean_missing_obs(data); R.step31_kf(data)
    R.step33_emission(data, G); R.step34_viterbi(data, G); R.step35_fusion(data, G)
    head = []; px = []; pz = []
    for it in data:
        pr = it['properties']
        head.append(pr.get('fusion_dir')); px.append(pr.get('fusion_x')); pz.append(pr.get('fusion_z'))
    return head, px, pz


# ---------------------------------------------------------------- aggregation
def group_rows(agg, m):
    """[(group, n_position, n_heading, x5, z5, heading5, n_reversed), ...]"""
    rows = []
    for name, fl in (('Arrival', 'A'), ('Departure', 'D'), ('Overall', None)):
        ks = ('A', 'D') if fl is None else (fl,)
        xa = np.concatenate([np.concatenate(agg[m][k]['x']) for k in ks])
        za = np.concatenate([np.concatenate(agg[m][k]['z']) for k in ks])
        ta = np.concatenate([np.concatenate(agg[m][k]['t']) for k in ks])
        flip = sum(agg[m][k]['flip'] for k in ks)
        rows.append((name, int(len(xa)), int(len(ta[ta <= FLIP_TH])),
                     metrics(xa), metrics(za), metrics(ta[ta <= FLIP_TH]), int(flip)))
    return rows


def fmt_row(r):
    name, n_pos, n_dir, x5, z5, t5, flip = r
    s = '  %-10s n_pos=%9d n_dir=%9d reversed=%7d\n' % (name, n_pos, n_dir, flip)
    for label, m5 in (('x      ', x5), ('z      ', z5), ('heading', t5)):
        s += '      %s ' % label + ' '.join('%s=%9.3f' % (k, v)
                                            for k, v in zip(('MAE', 'RMSE', 'STD', 'P50', 'P90'), m5)) + '\n'
    return s


def main():
    ap = argparse.ArgumentParser(description='graph-prior sequence baselines')
    ap.add_argument('--input', nargs='+', required=True,
                    help='one or more directories of estimator output records')
    ap.add_argument('--graph', required=True, help='network JSON')
    ap.add_argument('--limit', type=int, default=0, help='cap the records of each directory')
    args = ap.parse_args()

    G = R.load_graph(args.graph)
    methods = ('M1', 'M2', 'GC')
    agg = {m: {fl: {'x': [], 'z': [], 't': [], 'flip': 0} for fl in ('A', 'D')}
           for m in methods}
    n_rec = 0
    for d in args.input:
        picks = sorted(glob.glob(os.path.join(d, '*.json')))
        if args.limit:
            picks = picks[:args.limit]
        for f in picks:
            r = eval_track(f, G)
            if r is None:
                continue
            fl = r['flio']
            if fl not in agg['M1']:
                continue
            n_rec += 1
            for m in methods:
                xe, ze, te, phys_ok = r['out'][m]
                ok = phys_ok & ~np.isnan(xe) & ~np.isnan(ze) & ~np.isnan(te)
                g = agg[m][fl]
                g['x'].append(xe[ok]); g['z'].append(ze[ok]); g['t'].append(te[ok])
                g['flip'] += int((te[ok] > FLIP_TH).sum())
        print('[ok] %s: %d records' % (d, len(picks)))

    print('=' * 88)
    print('method M1 = HMM map matching, M2 = graph-free hybrid, GC = proposed model')
    print('protocol: position on all points including holding, heading above %g deg removed, '
          'physical mask kf_speed <= %g m/s' % (FLIP_TH, V_MAX))
    print('records : %d' % n_rec)
    print('=' * 88)
    for m in ('GC', 'M2', 'M1'):
        print('\n=== %s ===' % m)
        for row in group_rows(agg, m):
            print(fmt_row(row), end='')


if __name__ == '__main__':
    main()

