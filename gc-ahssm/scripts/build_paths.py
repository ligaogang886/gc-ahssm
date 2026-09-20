# -*- coding: utf-8 -*-
"""
将清洗后机场 OSM 构造为 InitMatrixPath 格式路网 JSON（通用）
用法:
    python build_paths_json.py <输入.osm> <输出.json> [选项]
选项:
    --a F --b F --c F --d F --tx F --tz F   仿射变换参数（重庆用户标定参数）
    --auto                                  自动: 以机场中心为原点等距投影
    --turn 5.0                              N/R 判定: 最大转向角阈值(度)，默认 5
说明:
- 图结构纳入: 跑道/滑行道/停止道 + 停机位线(parking_position) + 机坪轮廓(apron)
- 交叉点(被>=2 way引用 ∪ way首尾)处拆分 way，保留完整几何
- 端点命名 "Point (k)" 地理排序(北->南,西->东)编号
- 路径命名: 直线 N-i / 弯曲 R-j
- 内部点: 路径段完整节点序列，全局 Point0..PointN
- 坐标 float32; y=0(地面单层); roa=0
"""
import sys
import argparse
import xml.etree.ElementTree as ET
import math
import json
import struct
from collections import defaultdict

# 纳入图结构的线状要素: 跑道/滑行道/停止道 + 停机位线 + 机坪轮廓(机坪侧路径段)
GRAPH_AEROWAYS = {'taxiway', 'runway', 'stopway', 'parking_position', 'apron'}

def f32(v):
    return struct.unpack('<f', struct.pack('<f', v))[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('osm')
    ap.add_argument('out')
    ap.add_argument('--a', type=float)
    ap.add_argument('--b', type=float)
    ap.add_argument('--c', type=float)
    ap.add_argument('--d', type=float)
    ap.add_argument('--tx', type=float)
    ap.add_argument('--tz', type=float)
    ap.add_argument('--auto', action='store_true')
    ap.add_argument('--turn', type=float, default=5.0)
    args = ap.parse_args()

    # ---------- 读取 OSM ----------
    tree = ET.parse(args.osm)
    root = tree.getroot()
    node_xy = {}
    for n in root.findall('node'):
        node_xy[n.get('id')] = (float(n.get('lat')), float(n.get('lon')))

    way_records = []
    for w in root.findall('way'):
        tags = {t.get('k'): t.get('v') for t in w.findall('tag')}
        aero = tags.get('aeroway')
        if aero not in GRAPH_AEROWAYS:
            continue
        nids = [nd.get('ref') for nd in w.findall('nd')]
        nids = [n for n in nids if n in node_xy]
        if len(nids) < 2:
            continue
        way_records.append((aero, tags.get('ref', ''), nids))

    # ---------- 仿射参数 ----------
    lats = [v[0] for v in node_xy.values()]
    lons = [v[1] for v in node_xy.values()]
    lat0, lon0 = sum(lats)/len(lats), sum(lons)/len(lons)
    if args.auto:
        m_lon = 111320.0 * math.cos(math.radians(lat0))
        m_lat = 110540.0
        A, B, C, D = m_lon, 0.0, 0.0, m_lat
        TX = -m_lon * lon0
        TZ = -m_lat * lat0
        print(f'[auto] affine parameters: a={A:.6f} b={B} c={C} d={D} tx={TX:.6f} tz={TZ:.6f}')
    else:
        if None in (args.a, args.b, args.c, args.d, args.tx, args.tz):
            print('error: provide --a --b --c --d --tx --tz, or use --auto')
            sys.exit(1)
        A, B, C, D = args.a, args.b, args.c, args.d
        TX, TZ = args.tx, args.tz

    def aff(lon, lat):
        return (A * lon + B * lat + TX, C * lon + D * lat + TZ)

    # ---------- 交叉点判定 ----------
    node_ways = defaultdict(set)
    for i, (aero, ref, nids) in enumerate(way_records):
        for n in nids:
            node_ways[n].add(i)
    junctions = set()
    for nid, wset in node_ways.items():
        if len(wset) >= 2:
            junctions.add(nid)
    for aero, ref, nids in way_records:
        junctions.add(nids[0]); junctions.add(nids[-1])
        if nids[0] == nids[-1]:
            junctions.add(nids[0])

    # ---------- 拆分 ----------
    edges = []
    jdeg = defaultdict(int)
    for aero, ref, nids in way_records:
        seg = []
        for n in nids:
            seg.append(n)
            if n in junctions and len(seg) >= 2:
                if seg[0] in junctions and n in junctions:
                    edges.append((aero, ref, list(seg)))
                    jdeg[seg[0]] += 1; jdeg[n] += 1
                seg = [n]
        if len(seg) >= 2 and seg[0] in junctions and seg[-1] in junctions:
            edges.append((aero, ref, seg))
            jdeg[seg[0]] += 1; jdeg[seg[-1]] += 1

    # ---------- 端点编号（地理排序） ----------
    junction_order = sorted(junctions, key=lambda n: (-node_xy[n][0], node_xy[n][1]))
    jid2name = {n: f'Point ({k})' for k, n in enumerate(junction_order, 1)}

    # ---------- N/R 判定 ----------
    m_lat = 110540.0
    m_lon = 111320.0 * math.cos(math.radians(lat0))
    def max_turn_deg(nids):
        if len(nids) < 3:
            return 0.0
        mt = 0.0
        pts = [node_xy[n] for n in nids]
        for i in range(1, len(pts) - 1):
            la1, lo1 = pts[i-1]; la2, lo2 = pts[i]; la3, lo3 = pts[i+1]
            x1, y1 = (lo1-lon0)*m_lon, (la1-lat0)*m_lat
            x2, y2 = (lo2-lon0)*m_lon, (la2-lat0)*m_lat
            x3, y3 = (lo3-lon0)*m_lon, (la3-lat0)*m_lat
            v1 = (x2-x1, y2-y1); v2 = (x3-x2, y3-y2)
            n1 = math.hypot(*v1); n2 = math.hypot(*v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cosang = max(-1.0, min(1.0, (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)))
            mt = max(mt, math.degrees(math.acos(cosang)))
        return mt

    thr = args.turn
    n_paths = [e for e in edges if max_turn_deg(e[2]) < thr]
    r_paths = [e for e in edges if max_turn_deg(e[2]) >= thr]

    # ---------- 构造 ----------
    def build_path(aero, ref, nids, pname):
        pts_ll = [node_xy[n] for n in nids]
        length = 0.0
        for i in range(len(pts_ll) - 1):
            la1, lo1 = pts_ll[i]; la2, lo2 = pts_ll[i+1]
            length += math.hypot((lo2-lo1)*m_lon, (la2-la1)*m_lat)
        pts = []
        for n in nids:
            lat, lon = node_xy[n]
            ux, uz = aff(lon, lat)
            pts.append({'pointname': pname[0], 'x': round(f32(ux), 6),
                        'y': round(f32(0.0), 6), 'z': round(f32(uz), 6), 'roa': 0.0})
            pname[0] += 1
        return {'pathname': None, 'startpoint': jid2name[nids[0]],
                'endpoint': jid2name[nids[-1]],
                'pathLength': round(f32(length), 6), 'points': pts}

    paths = []
    g = [0]
    for aero, ref, nids in n_paths:
        p = build_path(aero, ref, nids, g)
        p['pathname'] = f'N-{len([x for x in paths]) + 1}'
        paths.append(p)
    n_count = len(paths)
    for aero, ref, nids in r_paths:
        p = build_path(aero, ref, nids, g)
        p['pathname'] = f'R-{len(paths) - n_count + 1}'
        paths.append(p)

    out = {'paths': paths}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---------- 统计 ----------
    import os, statistics
    print(f'output: {args.out}')
    print(f'file size: {os.path.getsize(args.out)/1024/1024:.2f} MB')
    print(f'paths: {len(paths)} (straight={n_count}, curved={len(paths)-n_count})')
    print(f'junctions: {len(junction_order)}, internal points: {g[0]}')
    print(f'total length: {sum(p["pathLength"] for p in paths):.0f} m')
    pts_per = [len(p['points']) for p in paths]
    print(f'points per path: mean {statistics.mean(pts_per):.1f}, median {statistics.median(pts_per)}, max {max(pts_per)}')
    print(f'centre after the affine map: ({aff(lon0, lat0)[0]:.0f}, {aff(lon0, lat0)[1]:.0f})')

if __name__ == '__main__':
    main()
