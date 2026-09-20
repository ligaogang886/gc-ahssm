# -*- coding: utf-8 -*-
"""
OSM 机场数据清洗（通用）：只保留机场 aeroway 要素
用法:
    python clean_airport_osm.py <输入.osm> <输出.osm>
保留:
- aeroway=* 的 way（runway/taxiway/parking_position/stopway/apron/jet_bridge/
  terminal/tower/aerodrome/hangar/concourse/...）
- aeroway multipolygon relation（自动检测）及其成员 way
- 带 aeroway 标签的节点（gate/holding_position/windsock/navigationaid/...）
剔除: highway/railway/building/landuse/公交线路/行政区划等
"""
import sys
import xml.etree.ElementTree as ET
import copy
from collections import Counter

def main():
    if len(sys.argv) < 3:
        print('usage: python clean_osm.py <input.osm> <output.osm>')
        sys.exit(1)
    SRC, DST = sys.argv[1], sys.argv[2]

    tree = ET.parse(SRC)
    root = tree.getroot()

    # 1. aeroway relation（multipolygon 且带 aeroway 标签）
    keep_rels = set()
    for r in root.findall('relation'):
        tags = {t.get('k'): t.get('v') for t in r.findall('tag')}
        if 'aeroway' in tags and tags.get('type') == 'multipolygon':
            keep_rels.add(r.get('id'))

    # 2. 保留的 way
    keep_ways = set()
    for w in root.findall('way'):
        tags = {t.get('k'): t.get('v') for t in w.findall('tag')}
        if 'aeroway' in tags:
            keep_ways.add(w.get('id'))
    for r in root.findall('relation'):
        if r.get('id') in keep_rels:
            for m in r.findall('member'):
                if m.get('type') == 'way':
                    keep_ways.add(m.get('ref'))

    # 3. 保留的节点
    keep_nodes = set()
    for w in root.findall('way'):
        if w.get('id') in keep_ways:
            for nd in w.findall('nd'):
                keep_nodes.add(nd.get('ref'))
    for n in root.findall('node'):
        tags = {t.get('k'): t.get('v') for t in n.findall('tag')}
        if 'aeroway' in tags:
            keep_nodes.add(n.get('id'))

    # 4. 重建
    new_root = ET.Element('osm', {
        'version': '0.6',
        'generator': 'cleaned-by-script: keep airport network only',
        'copyright': root.get('copyright', 'OpenStreetMap and contributors'),
        'attribution': root.get('attribution', 'http://www.openstreetmap.org/copyright'),
        'license': root.get('license', 'http://opendatacommons.org/licenses/odbl/1-0/'),
    })
    b = root.find('bounds')
    if b is not None:
        new_root.append(copy.deepcopy(b))
    nc = wc = rc = 0
    for n in root.findall('node'):
        if n.get('id') in keep_nodes:
            new_root.append(copy.deepcopy(n)); nc += 1
    for w in root.findall('way'):
        if w.get('id') in keep_ways:
            new_root.append(copy.deepcopy(w)); wc += 1
    for r in root.findall('relation'):
        if r.get('id') in keep_rels:
            new_root.append(copy.deepcopy(r)); rc += 1

    ET.indent(new_root, space=' ')
    ET.ElementTree(new_root).write(DST, encoding='utf-8', xml_declaration=True)

    aero = Counter()
    for w in new_root.findall('way'):
        tags = {t.get('k'): t.get('v') for t in w.findall('tag')}
        if 'aeroway' in tags:
            aero[tags['aeroway']] += 1
    print(f'output: {DST}')
    print(f'nodes {nc} / ways {wc} / relations {rc}')
    print('aeroway breakdown:', dict(aero))
    # 校验断链
    missing = set()
    for w in new_root.findall('way'):
        for nd in w.findall('nd'):
            if nd.get('ref') not in keep_nodes:
                missing.add(nd.get('ref'))
    print('missing node references:', len(missing))

if __name__ == '__main__':
    main()
