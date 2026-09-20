# -*- coding: utf-8 -*-
"""Observation-likelihood layer (Section III-C): topology-aware likelihood.

For every observation the network segments inside ``SIGMA_DIST`` are ranked with a
distance term and a directional term, and the best ``TOP_K`` survive.  The reported
direction of the observation is the local displacement direction of the measured
track, so the term never reads the reference orientation used for scoring.
"""
from ..common import *

def step33_emission(track_data, G):
    S = G['S']                      # (M,2,2)
    dir_f = G['dir_f']              # (M,)
    dir_b = (dir_f + 180) % 360     # (M,)
    a_pts = S[:, 0]                 # (M,2)
    d_vec = S[:, 1] - S[:, 0]       # (M,2)
    d2 = np.sum(d_vec * d_vec, axis=1)
    d2[d2 < 1e-12] = 1e-12

    M = len(S)
    # 预先计算整条轨迹的局部窗口运动方向（来自观测坐标 x/z, 独立于图, 抗停止点抖动）
    n_pts = len(track_data)
    obs_pts = np.array([[float(t['properties']['x']), float(t['properties']['z'])]
                        for t in track_data])
    WIN_OBS = 10
    win_ang_all = np.full(n_pts, np.nan)
    for i in range(n_pts):
        a = max(0, i - WIN_OBS); b = min(n_pts - 1, i + WIN_OBS)
        vx = obs_pts[b, 0] - obs_pts[a, 0]
        vz = obs_pts[b, 1] - obs_pts[a, 1]
        if math.hypot(vx, vz) >= 0.5:
            win_ang_all[i] = np.degrees(np.arctan2(vz, vx)) % 360.0
    # 静止点方向继承: 静止点无自身移动方向(位移<0.5m), 继承前一移动点的
    # 观测窗口方向(飞机静止时朝向延续)。独立于图, 不构成循环验证;
    # 用于静止点 emission 的方向约束, 避免纯距离匹配在交叉口/平行道选错段
    # (曾造成 B321A: Viterbi 选 N-520(309.1°) 而参考段方向 222.4°,
    #  86.7° 偏置非 180° 翻转, 方向消除无法修复, 只能修选段)。
    static_ang = np.copy(win_ang_all)
    last_dir = None
    for i in range(n_pts):
        if not math.isnan(static_ang[i]):
            last_dir = static_ang[i]
        else:
            static_ang[i] = last_dir
    # 轨迹开头即静止: 后向填充
    if math.isnan(static_ang[0]):
        first_dir = None
        for i in range(n_pts):
            if not math.isnan(static_ang[i]):
                first_dir = static_ang[i]
                break
        if first_dir is not None:
            for i in range(n_pts):
                if math.isnan(static_ang[i]):
                    static_ang[i] = first_dir
                else:
                    break

    for idx, item in enumerate(track_data):
        props = item['properties']
        x = float(props['kf_x'])
        z = float(props['kf_z'])
        kf_speed = float(props['kf_speed'])

        # 观测方向: 轨迹局部窗口运动方向（来自观测坐标, 独立于图）
        # —— 消除循环验证: 参考方向(refdir/图方向)不进入推断, 只用于评分
        obs_dir = win_ang_all[idx]
        if math.isnan(obs_dir):
            # 静止/微动点: 自身无方向, 用继承方向(前一移动点)约束选段
            inh = static_ang[idx]
            if inh is not None and not math.isnan(inh):
                obs_dir = inh
                direction_source = 'OBS_INHERIT'
            else:
                obs_dir = None
                direction_source = 'STATIC'
        else:
            direction_source = 'OBS_WIN'

        # ---- candidate segments: KD-tree lookup over the one-metre anchors, with a
        # ---- fallback that scores whole paths when the anchors are absent ----
        P = np.array([x, z])
        ip_tree = G.get('ip_tree')
        if ip_tree is not None:
            # 查最近 KIP 个 1m 内部点 → 去重得候选段(通常 1~20 段, 精匹配)
            KIP = 64
            n_ip = len(G['ip_owner'])
            kq = min(KIP, n_ip)
            _, ii = ip_tree.query(P, k=kq)
            cand_set = set()
            for ip_i in np.atleast_1d(ii):
                pn_ip, si_ip, _tt = G['ip_owner'][int(ip_i)]
                gi = G['seg_global'].get((pn_ip, si_ip))
                if gi is not None:
                    cand_set.add(gi)
            cand_arr = np.array(sorted(cand_set), dtype=int) if cand_set else np.arange(M)
        else:
            cand_arr = np.arange(M)

        # ---- 对候选段向量化投影距离 ----
        c = cand_arr
        sub_a = a_pts[c] if cand_arr.size else a_pts[:0]
        v = P[None, :] - sub_a                       # (C,2)
        t = np.clip(np.sum(v * d_vec[c], axis=1) / d2[c], 0, 1)
        proj = sub_a + t[:, None] * d_vec[c]
        dist = np.linalg.norm(P[None, :] - proj, axis=1)   # (C,)

        # ---- 向量化方向误差（取正反向较小者）----
        if obs_dir is not None:
            err_f = np.abs((obs_dir - dir_f[c] + 180) % 360 - 180)
            err_b = np.abs((obs_dir - dir_b[c] + 180) % 360 - 180)
            angle_err = np.minimum(err_f, err_b)
            seg_dir = np.where(err_f <= err_b, dir_f[c], dir_b[c])
            p_dir = gaussian(angle_err, SIGMA_DIR)
        else:
            angle_err = np.zeros(len(c))
            seg_dir = dir_f[c].copy()
            p_dir = np.ones(len(c))

        # ---- emission ----
        p_dist = gaussian(dist, SIGMA_DIST)
        emission = p_dist * p_dir

        # ---- Top-K ----
        k = min(TOP_K, len(c))
        if k == 0:
            props['emission_top_candidates'] = []
            continue
        top_local = np.argpartition(-emission, k - 1)[:k]
        top_local = top_local[np.argsort(-emission[top_local])]
        top_idx = c[top_local]   # 全局段索引

        candidates = []
        for si_local in top_local:
            si = int(c[si_local])   # 全局段索引
            candidates.append({
                'pathname': G['seg_path'][si],
                'segment_index': G['seg_idx'][si],
                'segment_start': G['seg_start'][si],
                'segment_end': G['seg_end'][si],
                'segment_dir': float(seg_dir[si_local]),
                'distance': float(dist[si_local]),
                'angle_error': float(angle_err[si_local]),
                'p_dist': float(p_dist[si_local]),
                'p_dir': float(p_dir[si_local]),
                'emission_probability': float(emission[si_local]),
            })
        props['observation_dir'] = float(obs_dir) if obs_dir is not None else None
        props['direction_source'] = direction_source
        # 观测窗口方向(±10点, 来自坐标, 独立于图) —— 供 3-5 层 topo_dir 的 180° 模糊消除
        # (不用 kf_dir: 静止点 kf_dir 乱跳会选反方向)
        props['obs_win_dir'] = float(win_ang_all[idx]) if not math.isnan(win_ang_all[idx]) else None
        props['emission_top_candidates'] = candidates
        best = candidates[0]
        props.update({
            'emission_best_path': best['pathname'],
            'emission_segment_index': int(best['segment_index']),
            'emission_dir': float(best['segment_dir']),
            'emission_distance': float(best['distance']),
            'emission_angle_error': float(best['angle_error']),
            'emission_p_dist': float(best['p_dist']),
            'emission_p_dir': float(best['p_dir']),
            'emission_probability': float(best['emission_probability']),
        })
    return track_data
