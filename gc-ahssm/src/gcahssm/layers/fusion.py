# -*- coding: utf-8 -*-
"""Refinement layer (Section III-E): topology-guided continuous refinement.

The segment sequence from the discrete layer is consumed as it is and never
re-selected, so the path accuracy of this layer is by construction the path accuracy
of the discrete layer.  The fused heading mixes the kinematic heading with the
geometric direction of the inferred segment with a speed-adaptive weight, and the
fused position is blended towards the orthogonal projection of the kinematic
solution onto that segment.
"""
from ..common import *

def fuse_dir_weighted(dirs, weights):
    v = np.array([0.0, 0.0])
    for d, w in zip(dirs, weights):
        r = np.radians(d)
        v += w * np.array([np.cos(r), np.sin(r)])
    ang = np.degrees(np.arctan2(v[1], v[0]))
    return normalize_degree(ang)


def project(px, pz, ax, az, bx, bz):
    abx, abz = bx - ax, bz - az
    apx, apz = px - ax, pz - az
    ab2 = abx * abx + abz * abz + 1e-6
    t = (apx * abx + apz * abz) / ab2
    t = max(0, min(1, t))
    return ax + t * abx, az + t * abz


def step35_fusion(track_data, G):
    paths = {p['pathname']: p for p in G['paths']}
    last_topo = None   # 前一移动点的 topo_dir（静止点方向传播）
    for i, item in enumerate(track_data):
        p = item['properties']
        kf_x = float(p['kf_x'])
        kf_z = float(p['kf_z'])
        kf_dir = float(p.get('kf_dir', 0))
        kf_speed = float(p.get('kf_speed', 0))
        innovation = float(p.get('innovation_residual', 0))

        cands = p['emission_top_candidates']
        if not cands:
            p['fusion_x'] = kf_x
            p['fusion_z'] = kf_z
            p['fusion_dir'] = kf_dir
            p['fusion_path'] = None
            p['fusion_motion_conf'] = 0.0
            p['fusion_topo_conf'] = 0.0
            continue

        dirs = []
        weights = []
        path_weights = {}
        for c in cands:
            w = c['emission_probability']
            dirs.append(c['segment_dir'])
            weights.append(w)
            path_weights[c['pathname']] = path_weights.get(c['pathname'], 0) + w

        # 投影路径用 Viterbi 解码结果(temporal)而非 emission 单点加权——两者必须一致,
        # 否则 temporal_segment_index 应用到错误路径导致投影错位(曾造成 52% 点错配)
        topo_path = p.get('temporal_best_path')
        if topo_path is None:
            topo_path = max(path_weights, key=path_weights.get)
        topo_prob = min(1.0, path_weights[topo_path] * TOPO_TEMP_SCALE)

        # 拓扑方向: 用 Viterbi 段的几何方向(图方向, 物理合理), 而非 emission 候选加权平均
        # (候选段在交叉口可能含不同/反向方向, 加权平均会互相抵消产生错误中间角)
        topo_dir = None
        if topo_path in paths:
            pts = paths[topo_path]['points']
            seg = p.get('temporal_segment_index', 0)
            if seg < len(pts) - 1:
                a_pt, b_pt = pts[seg], pts[seg + 1]
                dx = float(b_pt['x']) - float(a_pt['x'])
                dz = float(b_pt['z']) - float(a_pt['z'])
                if math.hypot(dx, dz) > 1e-6:
                    seg_ang = math.degrees(math.atan2(dz, dx)) % 360.0
                    # 消除 180° 模糊, 按运动状态分三种参考:
                    # 1) 正常速度移动点(>=3): 观测窗口方向(独立于图)
                    # 2) 低速移动点(<3, 轨迹聚集/排队): 观测窗口位移小、方向噪声
                    #    极大(实测 338/90/50/0/321° 乱跳), 会驱动 seg_ang 在 180°
                    #    边界反复翻转 → 融合"跳动一圈" → 改用时序连续方向 last_topo
                    # 3) 静止点(obs_win_dir=None): 坐标不动无朝向信息, 用图参考
                    #    refdir(=mapmatch 翻转对齐后的图方向)二值消除
                    ref_dir = p.get('obs_win_dir')
                    if ref_dir is not None and kf_speed >= LOW_SPEED_THRESHOLD_35:
                        if calc_angle_error(seg_ang, ref_dir) > 90:
                            seg_ang = (seg_ang + 180) % 360.0
                        topo_dir = seg_ang
                        # 仅当消除后方向与观测窗口方向接近(<45°, 可信)才更新传播源,
                        # 避免转弯/噪声点的错误方向被传播给后续低速/静止点
                        if calc_angle_error(topo_dir, ref_dir) < 45:
                            last_topo = topo_dir
                    elif ref_dir is not None:
                        # 低速移动点: 观测窗口位移小、方向噪声极大(实测 338/90/50/
                        # 0/321° 乱跳), 若用 obs_win_dir 消除会驱动 seg_ang 在 180°
                        # 边界反复翻转 → 融合"跳动一圈"。改用时序连续方向 last_topo
                        # 消除, 输出稳定贴合图段方向(21条实验 30.57→28.29° 最优)。
                        if last_topo is not None:
                            if calc_angle_error(seg_ang, last_topo) > 90:
                                topo_dir = (seg_ang + 180) % 360.0
                            else:
                                topo_dir = seg_ang
                        else:
                            topo_dir = seg_ang  # 无历史: 保持图段固有方向
                    else:
                        # 静止点: refdir 二值消除
                        rd = p.get('refdir')
                        if rd is not None and calc_angle_error(seg_ang, float(rd)) > 90:
                            topo_dir = (seg_ang + 180) % 360.0
                        else:
                            topo_dir = seg_ang
        if topo_dir is None:
            # 兜底: 候选加权平均
            topo_dir = fuse_dir_weighted(dirs, weights)

        motion_conf = gaussian(innovation, SIGMA_RESIDUAL)
        topo_conf = topo_prob

        if kf_speed < LOW_SPEED_THRESHOLD_35:
            # 静止点: kf_dir 由观测差分驱动, 噪声大(随机方向) → 完全信任拓扑方向
            alpha = 0.0
        else:
            alpha_base = motion_conf / (motion_conf + topo_conf + 1e-6)
            # 速度自适应: 低速(轨迹聚集/排队)时 KF 方向由观测差分驱动, 噪声大,
            # 提高图约束(topology)占比; 高速时 KF 可信, 恢复其权重。
            # 速度分箱实测: 0-1s: KF 115.3° vs 图 54.7°; 3-5: 11.3 vs 8.5;
            # 8-15: 2.0 vs 1.1(图优); 仅 15+: 3.1 vs 4.3(KF 优)
            speed_scale = kf_speed / (kf_speed + ALPHA_V_HALF)
            alpha = min(alpha_base, speed_scale)
            alpha = max(0.0, min(0.9, alpha))

        fusion_dir = fuse_dir_weighted([kf_dir, topo_dir], [alpha, 1 - alpha])

        fx_t, fz_t = kf_x, kf_z
        if topo_path in paths:
            pts = paths[topo_path]['points']
            seg = p.get('temporal_segment_index', 0)
            if seg < len(pts) - 1:
                a, b = pts[seg], pts[seg + 1]
                px, pz = project(kf_x, kf_z, float(a['x']), float(a['z']),
                                 float(b['x']), float(b['z']))
                fx_t = (1 - PROJECTION_BLEND) * kf_x + PROJECTION_BLEND * px
                fz_t = (1 - PROJECTION_BLEND) * kf_z + PROJECTION_BLEND * pz

        p['fusion_x'] = float(fx_t)
        p['fusion_z'] = float(fz_t)
        p['fusion_dir'] = float(fusion_dir)
        p['fusion_path'] = topo_path
        p['fusion_motion_conf'] = float(motion_conf)
        p['fusion_topo_conf'] = float(topo_conf)
        p['fusion_alpha'] = float(alpha)
        p['fusion_topo_dir'] = float(topo_dir)
    return track_data
