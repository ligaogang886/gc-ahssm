# -*- coding: utf-8 -*-
"""Shared model parameters and angular utilities of GC-AHSSM.

Constants
---------
SIGMA_DIST / SIGMA_DIR    range and heading spread of the measurement likelihood
TOP_K                     number of network segments kept per observation
DIR_SIGMA                 directional term entering the discrete transition matrix
STAY_PROB                 self-transition probability of the discrete state
SIGMA_RESIDUAL / PROJECTION_BLEND / TOPO_TEMP_SCALE
                          residual gating and topology-guided refinement weights
ALPHA_V_HALF              half-speed point of the speed-adaptive fusion weight

All values are fixed once and reused for every airport and every dataset.
"""
import math

import numpy as np

# ---------------------------------------------------------------- parameters

SIGMA_DIST = 40.0
SIGMA_DIR = 50.0
TOP_K = 8
LOW_SPEED_THRESHOLD_33 = 5.0
LOW_SPEED_THRESHOLD_34 = 3.0
LOW_SPEED_THRESHOLD_35 = 3.0
DIR_SIGMA = 35.0
EPS = 1e-12
STAY_PROB = 0.7
SIGMA_RESIDUAL = 40.0
PROJECTION_BLEND = 0.75
TOPO_TEMP_SCALE = 20.0
ALPHA_V_HALF = 20.0  # 速度自适应 alpha 曲线半速点: kf_speed/(kf_speed+ALPHA_V_HALF)
                     # 21条实验: None(原)=30.66° / V=8=30.66 / V=12=30.61 / V=20=30.57(最优)

def normalize_degree(angle):
    angle = angle % 360
    if angle < 0:
        angle += 360
    return angle


def calc_angle_error(a, b):
    d = abs(a - b)
    return 360 - d if d > 180 else d


def angle_diff(a, b):
    d = abs(a - b)
    return min(d, 360 - d)


def gaussian(x, sigma):
    return np.exp(-(x * x) / (2 * sigma * sigma))


def log_gaussian(x, sigma):
    return -(x * x) / (2 * sigma * sigma + EPS)


def to_0_360(deg):
    """-180~180 -> 0~360"""
    if deg is None:
        return 0.0
    return float(deg) % 360.0
