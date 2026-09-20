# -*- coding: utf-8 -*-
"""Discrete inference layer (Section III-D): temporal network inference.

A Viterbi pass over the candidate segments with a transition matrix that mixes the
topological adjacency of the network with a directional term, which yields the
segment sequence and the segment identifier of every observation.
"""
from ..common import *

def step34_viterbi(track_data, G):
    tm = G['transition_matrix']
    n = len(track_data)
    viterbi = []
    backpointer = []

    first = track_data[0]['properties']['emission_top_candidates'][:TOP_K]
    delta = {}
    bp = {}
    for i, c in enumerate(first):
        delta[i] = math.log(c['emission_probability'] + EPS)
        bp[i] = None
    viterbi.append(delta)
    backpointer.append(bp)

    for t in range(1, n):
        props = track_data[t]['properties']
        speed = float(props.get('kf_speed', 0))
        current = props['emission_top_candidates'][:TOP_K]
        previous = track_data[t - 1]['properties']['emission_top_candidates'][:TOP_K]
        prev_delta = viterbi[-1]
        cur_delta = {}
        cur_bp = {}

        for j, cj in enumerate(current):
            cur_path = cj['pathname']
            cur_seg = cj['segment_index']
            cur_dir = cj['segment_dir']
            emit_log = math.log(cj['emission_probability'] + EPS)
            best_score = -1e18
            best_prev = None

            for i, ci in enumerate(previous):
                if i not in prev_delta:
                    continue
                prev_path = ci['pathname']
                prev_seg = ci['segment_index']
                prev_dir = ci['segment_dir']

                trans = tm.get(prev_path, {}).get(cur_path, 0.0)
                if trans <= 0:
                    continue
                log_trans = math.log(trans + EPS)

                seg_penalty = 0.0
                if prev_path == cur_path and abs(cur_seg - prev_seg) <= 1:
                    seg_penalty += 0.5

                dir_penalty = log_gaussian(angle_diff(prev_dir, cur_dir), DIR_SIGMA)
                if speed < LOW_SPEED_THRESHOLD_34:
                    dir_penalty *= 0.5

                score = (prev_delta[i] + log_trans + emit_log
                         + seg_penalty + dir_penalty)
                if score > best_score:
                    best_score = score
                    best_prev = i

            if best_prev is not None:
                cur_delta[j] = best_score
                cur_bp[j] = best_prev

        if not cur_delta:
            # 转移断裂兜底: 用该点 emission 最大候选, 保持时序连续
            best_j = int(np.argmax([c['emission_probability'] for c in current]))
            cur_delta[best_j] = math.log(current[best_j]['emission_probability'] + EPS)
            cur_bp[best_j] = None

        viterbi.append(cur_delta)
        backpointer.append(cur_bp)

    # backtrack
    last = viterbi[-1]
    if not last:
        return track_data
    best_last = max(last, key=last.get)
    seq = [best_last]
    cur = best_last
    for t in range(n - 1, 0, -1):
        nxt = backpointer[t].get(cur)
        if nxt is None:
            # 兜底: 用该时刻 emission 最大候选, 保证回溯连续
            cands = track_data[t]['properties']['emission_top_candidates'][:TOP_K]
            nxt = int(np.argmax([c['emission_probability'] for c in cands]))
        cur = nxt
        seq.append(cur)
    seq.reverse()

    for t, idx in enumerate(seq):
        props = track_data[t]['properties']
        best = props['emission_top_candidates'][idx]
        props['temporal_best_path'] = best['pathname']
        props['temporal_segment_index'] = best['segment_index']
        props['temporal_dir'] = best['segment_dir']
        props['temporal_emission_probability'] = best['emission_probability']
        props['temporal_log_probability'] = float(viterbi[t].get(idx, 0))
    return track_data
