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
