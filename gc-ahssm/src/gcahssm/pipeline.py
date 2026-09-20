# -*- coding: utf-8 -*-
"""Single-trajectory driver: the five layers in the order used in the paper."""
from .common import *
from .layers.kf import step31_kf
from .layers.emission import step33_emission
from .layers.viterbi import step34_viterbi
from .layers.fusion import step35_fusion

def clean_missing_obs(track_data):
    """原地剔除缺失观测点(originLon/originLat=0, 仿射后为原点, 属无效数据)"""
    i = 0
    removed = 0
    while i < len(track_data):
        pr = track_data[i].get('properties', {})
        ol, oa = pr.get('originLon'), pr.get('originLat')
        if ol is None or oa is None or \
           (abs(float(ol)) < 1e-9 and abs(float(oa)) < 1e-9):
            del track_data[i]
            removed += 1
        else:
            i += 1
    if removed:
        print(f'    dropped {removed} observations with a missing position', flush=True)
    return track_data


def clean_legacy_fields(track_data):
    """删除冗余/混淆的旧字段 (matched_* 坐标系错乱, 仅保留有效观测与图参考)"""
    drop_keys = ('matched_x', 'matched_z', 'matched_pathname', 'matched_pointname')
    for item in track_data:
        pr = item.get('properties', {})
        for k in drop_keys:
            pr.pop(k, None)
    return track_data


def run_one(track_data, G):
    clean_missing_obs(track_data)
    step31_kf(track_data)
    step33_emission(track_data, G)
    step34_viterbi(track_data, G)
    step35_fusion(track_data, G)
    clean_legacy_fields(track_data)
    return track_data


def summarize(track_data):
    """方向误差统计: fusion_dir vs refdir(图方向)"""
    errs = []
    matched = 0
    for item in track_data:
        p = item['properties']
        fd = p.get('fusion_dir')
        rd = p.get('refdir')
        if fd is not None and rd is not None:
            errs.append(calc_angle_error(fd, rd))
        if p.get('fusion_path') is not None:
            matched += 1
    if not errs:
        return None
    arr = np.array(errs)
    return {
        'n': len(errs),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        '<15°': float((arr < 15).mean()),
        '<35°': float((arr < 35).mean()),
        'matched_ratio': matched / len(track_data),
    }
