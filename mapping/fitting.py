import numpy as np
from scipy.optimize import least_squares
from mapping.signal_model import t2_signal, t1_ir_signal


def fit_t2_curve(TEs, signals, mask=None, bounds=(0, 2000)):
    N, H, W = signals.shape
    T2_map = np.zeros((H, W), dtype=np.float32)
    M0_map = np.zeros((H, W), dtype=np.float32)
    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    def residuals(params, t, y):
        M0, T2 = params
        return (M0 * np.exp(-t / T2) - y).ravel()

    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            y = signals[:, i, j]
            M0_init = float(np.max(y))
            T2_init = 100.0
            res = least_squares(
                residuals,
                x0=[M0_init, T2_init],
                args=(TEs, y),
                bounds=([0, bounds[0]], [np.inf, bounds[1]]),
            )
            M0_map[i, j], T2_map[i, j] = res.x
    return M0_map, T2_map


def fit_t1_ir(TIs, signals, mask=None, bounds=(1, 5000)):
    N, H, W = signals.shape
    T1_map = np.zeros((H, W), dtype=np.float32)
    M0_map = np.zeros((H, W), dtype=np.float32)
    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    def residuals(params, t, y):
        M0, T1 = params
        return (M0 * (1 - 2 * np.exp(-t / T1)) - y).ravel()

    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            y = signals[:, i, j]
            M0_init = float(np.max(np.abs(y)))
            T1_init = 1000.0
            res = least_squares(
                residuals,
                x0=[M0_init, T1_init],
                args=(TIs, y),
                bounds=([0, bounds[0]], [np.inf, bounds[1]]),
            )
            M0_map[i, j], T1_map[i, j] = res.x
    return M0_map, T1_map
