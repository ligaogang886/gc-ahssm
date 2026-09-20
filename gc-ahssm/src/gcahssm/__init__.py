# -*- coding: utf-8 -*-
"""GC-AHSSM - graph-constrained adaptive hybrid state-space model.

Reference implementation of the estimator used in the paper for online airport
surface trajectory estimation.  The package is organised along the layers of the
method:

===============  =====================================================
``layers.kf``    continuous motion layer (constant-velocity Kalman filter)
``layers.emission``  topology-aware observation likelihood
``layers.viterbi``   temporal inference over the discrete network states
``layers.fusion``    topology-guided continuous refinement
``graph``        topology-aware directed surface network
===============  =====================================================

Typical use::

    from gcahssm import load_graph, run_one
    G = load_graph('data/graphs/xian_graph.json')
    run_one(record['data'], G)

All model parameters are fixed once and are identical for every airport and for
every dataset.  No parameter is tuned per airport.
"""
from .common import (  # noqa: F401
    ALPHA_V_HALF, DIR_SIGMA, EPS, PROJECTION_BLEND, SIGMA_DIR, SIGMA_DIST,
    SIGMA_RESIDUAL, STAY_PROB, TOPO_TEMP_SCALE, TOP_K,
)
from .graph import load_graph  # noqa: F401
from .layers.emission import step33_emission  # noqa: F401
from .layers.fusion import step35_fusion  # noqa: F401
from .layers.kf import step31_kf  # noqa: F401
from .layers.viterbi import step34_viterbi  # noqa: F401
from .pipeline import (  # noqa: F401
    clean_legacy_fields, clean_missing_obs, run_one, summarize,
)

__version__ = '1.0.0'

__all__ = [
    'load_graph', 'run_one', 'summarize', 'clean_missing_obs', 'clean_legacy_fields',
    'step31_kf', 'step33_emission', 'step34_viterbi', 'step35_fusion',
    'SIGMA_DIST', 'SIGMA_DIR', 'TOP_K', 'DIR_SIGMA', 'STAY_PROB',
    'SIGMA_RESIDUAL', 'PROJECTION_BLEND', 'TOPO_TEMP_SCALE', 'ALPHA_V_HALF', 'EPS',
]
