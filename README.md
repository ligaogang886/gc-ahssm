# GC-AHSSM — graph-constrained adaptive hybrid state-space model

Reference implementation of the estimator for online airport surface trajectory
estimation described in the accompanying paper, together with the map data it is
built on and a small bundle of sample records. The model couples a
constant-velocity motion layer with a topology-aware observation likelihood, a
temporal inference pass over the discrete network states and a topology-guided
refinement of position and heading. No parameter is tuned per airport: the network
is the only airport-specific input.

```
      measurements (1 Hz)                      topology-aware network
              │                                          │
      III-A continuous motion  ──►  III-C observation likelihood  ──►  III-D temporal inference
              │                                          │
              └──────────►  III-E topology-guided refinement  ◄───┘
                                     │
                    position + heading + segment sequence
```

## Contents

| Path | Contents |
| --- | --- |
| `src/gcahssm/` | the model: one module per layer, plus the network loader |
| `scripts/` | preparation chain: OSM cleaning, network construction, map matching, record extraction |
| `experiments/` | run the model, score it, compare against six baselines, measure latency and memory |
| `data/osm/` | cleaned OpenStreetMap extract of both airports (ODbL 1.0) |
| `data/graphs/` | vector network and topology-aware graph of both airports (ODbL 1.0) |
| `data/samples/` | six redacted records per airport, for format demonstration and smoke tests |

Two airports are included. The primary one is labelled `xian` and the second,
geographically independent one is labelled `chongqing`. Both were processed by the
same code with the same parameter values.

## Install

```bash
python -m venv .venv && . .venv/bin/activate     # or any Python >= 3.9 environment
pip install -r requirements.txt
```

NumPy and SciPy are the only requirements for the model and for every experiment
entry point. No GPU and no deep-learning framework is used anywhere in the chain.

## Quick start

Run the model over the shipped samples and score the output with the evaluation
protocol of the paper:

```bash
python experiments/run_pipeline.py --input data/samples/xian \
    --graph data/graphs/xian_graph.json --output out/xian

python experiments/evaluate.py --input out/xian
```

The second airport works the same way, with `chongqing` in place of `xian` and the
matching network file. On the two shipped bundles this gives an overall heading MAE
of 4.30 degrees at the primary airport and 4.61 degrees at the second one, with a
median heading difference below half a degree at both. These values are a smoke test
of the release and not a reproduction of the published tables, because the paper
evaluates the complete archive of each airport.

Compare the model against the baselines of the paper on a directory of records:

```bash
python experiments/compare_all.py --input out/xian --graph data/graphs/xian_graph.json
```

Measure latency, throughput and memory on your own machine:

```bash
python experiments/runtime_benchmark.py --input data/samples/xian \
    --graph data/graphs/xian_graph.json --out out/runtime_local.txt
```

## Record format

One JSON record per trajectory. The header carries the flight-level labels and the
`data` array carries the samples.

| Field | Meaning |
| --- | --- |
| `cfno`, `afn` | aircraft registration and flight number, used as labels only |
| `flio` | `A` for an arrival and `D` for a departure |
| `stno`, `rway` | stand and runway, used as labels only |
| `starttime`, `endtime`, `pointcount` | extent of the record |
| `data[i].time` | timestamp of the sample, one sample per second |
| `data[i].properties.x`, `.z` | measured position in metres in the airport local frame, `x` east and `z` north |
| `data[i].properties.lon`, `.lat` | the same position before the local affine map |
| `data[i].properties.originLon`, `.originLat` | origin of the local frame of this record |
| `data[i].properties.angle_self` | self-reported track angle in degrees |
| `data[i].properties.angle_graph` | reference direction in degrees, written by the preparation chain |
| `data[i].properties.graph_match_dist`, `.graph_seg_idx` | distance to the matched link and its index |

The estimator reads the position, the timestamp and the self-reported angle. The
direction of the matched link is used only to score a heading, never to estimate
one: `angle_graph` is absent on the first sample of every record, and where the field
is missing the reference direction falls back to `angle_self` and then to a constant,
so an input record without preparation fields can still be scored. All quantities that
the model itself writes are added to the output records under `properties`, so an
input record and an output record have the same shape.

## Map data

| Airport | paths | segments | one-metre anchors | components |
| --- | ---: | ---: | ---: | ---: |
| `xian`, primary | 2,010 | 7,469 | 153,668 | 2 |
| `chongqing`, second | 4,139 | 14,685 | 221,968 | 1 |

Both networks come from an OpenStreetMap extract of the airport and are distributed
under the ODbL 1.0. See `NOTICE` for the attribution that has to be preserved.

The chain that produces them from an extract is

```bash
python scripts/clean_osm.py data/osm/xian_airport.osm build/xian_clean.osm
python scripts/build_paths.py build/xian_clean.osm build/xian_paths.json --auto
python scripts/build_graph.py --src build/xian_paths.json --out build/xian_graph.json
```

`clean_osm.py` keeps the `aeroway` features only. `build_paths.py` splits the ways at
intersections, assigns a direction of travel to every link, partitions each link into
straight and curved paths and emits the connection points. `build_graph.py` adds the
adjacency, the discrete transition matrix, the segment partition and the one-metre
anchors.

The four files in `data/graphs/` are exactly what this chain produces. The primary
airport uses the equidistant projection that `--auto` derives from the centroid of the
extract. The second airport uses a calibrated affine map with `a` = 69219.734239,
`b` = 73016.824547, `c` = -60472.614251, `d` = 83475.891532, `tx` = -10045470.936244
and `tz` = 3701948.837999. With those values `sha1sum` of the regenerated files equals
`sha1sum` of the shipped ones, so the network in this repository is not a private
artefact that happens to be attached to public code.

## Recorded performance

Measured on the complete archive of the primary airport, which holds more than six
thousand trajectories over one week. Single thread on an 11th-generation Intel Core
i7-1165G7 at 2.80 GHz, 16 GB of RAM, no accelerator, CPython 3.12.4.

| Quantity | Value |
| --- | --- |
| end-to-end latency | 0.267 ms per observation, median 0.243 ms, p99 0.546 ms |
| throughput | 3,731 observations per second |
| margin against a 1 Hz measurement cadence | 3,750x |
| layer shares | likelihood 59.9 %, motion 20.1 %, temporal inference 11.4 %, refinement 8.3 %, preprocessing 0.2 % |
| steady-state memory | 177 MB resident, 222.5 MB peak, of which 87.6 MB is the network |
| cost growth | 0.287 ms per observation against record length, correlation 0.739 |

The measurement is single-threaded because the claim concerns the latency of one
causal update for one aircraft, and a multi-core run would shorten the wall clock of a
batch without changing that quantity. File reading is excluded, because that cost
belongs to the storage device rather than to the estimator.
`experiments/runtime_benchmark.py` prints the same description of the machine next to
its result, so a measurement taken elsewhere can be compared directly.

## What is not in this repository

The surface surveillance archive of neither airport can be redistributed, because it
is the property of the operator and is subject to its data-protection rules. What
stands in for it is `data/samples/`, six records per airport, re-serialised with the
operational metadata removed. The rule that selects them is in
`scripts/make_sample_bundle.py`, which also writes the manifest of the bundle.
`scripts/extract_trajectories.py` documents the three decisions that turn a
surveillance archive into records of this shape, so the release shows how the input of
the study was produced even though the input itself cannot be published.

The sample bundle is therefore a demonstration and a smoke test, not the data from
which the tables of the paper can be recomputed.

## Notes on the code

* `src/` is a relocation of the implementation that produced the reported numbers: the
  layers were moved into one module each without touching a line of algorithm code, so
  the released package and the study pipeline agree bit for bit. The equivalence was
  checked by running both on the same records and comparing every property of every
  sample, which gives a maximum absolute difference of zero over 715,559 values.
* `experiments/compare_baselines.py`, `experiments/baselines_graph_priors.py` and
  `experiments/runtime_benchmark.py` are the drivers of the study as well. Only their
  path handling was replaced by command-line arguments and their console reports were
  translated.
* Some inline comments and docstrings are in Chinese, the language of the authors'
  working notes, and they carry the derivation of a constant next to the constant
  itself. Every entry point, every console report and every document of this
  repository is in English.
* The estimator is deterministic. The same record gives the same output on any
  machine, so a difference between two runs is a difference in the input or in the
  network, never in the execution.

## Licence

The source code is released under the MIT licence. The OSM extracts and the two
networks are derivative databases of OpenStreetMap and are released under the Open
Database Licence version 1.0 with attribution to the OpenStreetMap contributors. The
sample records are provided as is and without warranty. See `LICENSE` and `NOTICE`.
