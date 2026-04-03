import numpy as np


def _bloch_rhs(M, B1x, B1y, df, T1, T2, M0=1.0):
    """Return dM/dt for a simple Bloch RHS (rotating-frame approx.).

    M: (...,3)
    B1x,B1y: scalars (rad/s)
    df: off-resonance (Hz)
    """
    Mx, My, Mz = M[..., 0], M[..., 1], M[..., 2]
    rot = 2 * np.pi * df
    # effective drive from B1 components
    drive_x = B1x * Mz
    drive_y = B1y * Mz

    dMx = -Mx / T2 + rot * My + drive_x
    dMy = -My / T2 - rot * Mx + drive_y
    dMz = -(Mz - M0) / T1 - (B1x * My + B1y * Mx)
    return np.stack([dMx, dMy, dMz], axis=-1)


def rk4_step(M, dt, B1x, B1y, df, T1, T2, M0=1.0):
    k1 = _bloch_rhs(M, B1x, B1y, df, T1, T2, M0)
    k2 = _bloch_rhs(M + 0.5 * dt * k1, B1x, B1y, df, T1, T2, M0)
    k3 = _bloch_rhs(M + 0.5 * dt * k2, B1x, B1y, df, T1, T2, M0)
    k4 = _bloch_rhs(M + dt * k3, B1x, B1y, df, T1, T2, M0)
    return M + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_bloch_timecourse(
    time_axis, B1x, B1y=None, df=0.0, T1=1000.0, T2=100.0, M0=1.0, M_init=None
):
    time_axis = np.asarray(time_axis)
    N = len(time_axis)
    if B1y is None:
        B1y = np.zeros_like(B1x)
    M = np.zeros((N, 3), dtype=float)
    if M_init is None:
        M_curr = np.array([0.0, 0.0, M0], dtype=float)
    else:
        M_curr = np.array(M_init, dtype=float)
    M[0] = M_curr
    for i in range(1, N):
        dt = time_axis[i] - time_axis[i - 1]
        b1x_mid = 0.5 * (B1x[i] + B1x[i - 1])
        b1y_mid = 0.5 * (B1y[i] + B1y[i - 1])
        M_curr = rk4_step(M_curr, dt, b1x_mid, b1y_mid, df, T1, T2, M0)
        M[i] = M_curr
    return M


def simulate_bloch_sequence(time_axis, events, M0=1.0, T1=1000.0, T2=100.0, df=0.0):
    """Lightweight wrapper expected by signal_model.

    For the purposes of the mapping demo we only need free-precession
    evolution sampled at the provided time_axis. The `events` argument is
    accepted for API compatibility but ignored by this simple implementation.
    """
    # construct zero B1 waveform (no RF) sampled at time_axis
    time_axis = np.asarray(time_axis)
    B1x = np.zeros_like(time_axis, dtype=float)
    B1y = np.zeros_like(time_axis, dtype=float)
    M = simulate_bloch_timecourse(time_axis, B1x, B1y, df=df, T1=T1, T2=T2, M0=M0)
    return M


def simulate_bloch_ir(TIs, M0, T1, T2=None, inversion_efficiency=1.0, steps_per_ms=1):
    """Simulate inversion recovery using the existing timecourse simulator.

    For each TI we simulate from t=0..TI starting from inverted Mz and return Mz(TI).
    This is a simple wrapper to provide a compatible API for the fitter.
    """
    if T2 is None:
        T2 = max(1e-3, T1 / 2.0)
    TIs = np.asarray(TIs, dtype=float)
    out = np.zeros((len(TIs),), dtype=float)
    for idx, TI in enumerate(TIs):
        if TI <= 0:
            out[idx] = -inversion_efficiency * M0
            continue
        # build time axis with steps proportional to TI
        n_steps = max(2, int(abs(TI) * steps_per_ms))
        time_axis = np.linspace(0.0, TI, n_steps + 1)
        # set M_init to post-inversion state
        M_init = np.array([0.0, 0.0, -inversion_efficiency * M0], dtype=float)
        M = simulate_bloch_timecourse(
            time_axis,
            np.zeros_like(time_axis),
            np.zeros_like(time_axis),
            df=0.0,
            T1=T1,
            T2=T2,
            M0=M0,
            M_init=M_init,
        )
        out[idx] = M[-1, 2]
    return out


def simulate_multi_echo(TEs, M0, T2, T1=None):
    TEs = np.asarray(TEs, dtype=float)
    return M0 * np.exp(-TEs / T2)
