# -*- coding: utf-8 -*-
"""Continuous layer (Section III-A): constant-velocity Kalman filter.

Per-observation prediction of ``[x, z, vx, vz]`` with a process-noise level that is
kept deliberately simple.  The filter supplies both the kinematic prior consumed by
the discrete layer and the speed series used by the physical mask at evaluation.
"""
from ..common import *

class KalmanFilter2D:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.X = np.zeros((4, 1))
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                           [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.P = np.eye(4) * 500
        self.Q = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0],
                           [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
        self.R = np.array([[5, 0], [0, 5]])
        self.y = np.zeros((2, 1))

    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        self.y = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ self.y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P


def step31_kf(track_data):
    """KF 连续运动估计, 写回 kf_* / innovation_residual"""
    kf = KalmanFilter2D(dt=1.0)
    # 首点用观测初始化状态, 避免首点速度爆炸
    p0 = track_data[0]['properties']
    kf.X = np.array([[float(p0['x'])], [float(p0['z'])], [0.0], [0.0]])
    for idx, item in enumerate(track_data):
        props = item['properties']
        x = float(props['x'])
        z = float(props['z'])
        kf.predict()
        kf.update([x, z])
        kf_x = float(kf.X[0, 0])
        kf_z = float(kf.X[1, 0])
        vx = float(kf.X[2, 0])
        vz = float(kf.X[3, 0])
        kf_dir = normalize_degree(np.degrees(np.arctan2(vz, vx)))
        kf_speed = math.sqrt(vx * vx + vz * vz)
        residual = math.sqrt(float(kf.y[0, 0]) ** 2 + float(kf.y[1, 0]) ** 2)
        # refdir: angle_graph(图方向) 优先, 空则 angle_self, 再则 0
        ag = props.get('angle_graph')
        as_ = props.get('angle_self')
        refdir = to_0_360(ag if ag is not None else (as_ if as_ is not None else 0))
        dir_error = calc_angle_error(kf_dir, refdir)
        props.update({
            'kf_x': kf_x, 'kf_z': kf_z,
            'kf_vx': vx, 'kf_vz': vz,
            'kf_speed': kf_speed, 'kf_dir': kf_dir,
            'refdir': refdir, 'kf_dir_error': dir_error,
            'innovation_residual': residual,
        })
    return track_data
