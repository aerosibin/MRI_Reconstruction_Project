import numpy as np
from .bloch import simulate_ir_signal, simulate_multi_echo
from .bloch_full import simulate_bloch_sequence


def t2_signal(TEs, M0, T2):
    TEs = np.asarray(TEs)
    return simulate_multi_echo(TEs, M0, T2)


def t1_ir_signal(TIs, M0, T1):
    return simulate_ir_signal(TIs, M0, T1, inversion_efficiency=1.0)


def bloch_ir(TIs, M0, T1, inversion_efficiency=1.0):
    return simulate_ir_signal(TIs, M0, T1, inversion_efficiency=inversion_efficiency)


def bloch_te(TEs, M0, T2):
    return simulate_multi_echo(TEs, M0, T2)


def bloch_te_full(TEs, M0, T2, T1=None, df=0.0):
    """Simulate multi-echo using the full Bloch integrator (simple wrapper).

    This constructs a short event list that free-precesses and samples at TE times.
    """
    # build time axis (seconds) assuming TE in ms
    time_axis = TEs / 1000.0
    # events: single free-precession period
    events = [(0.0, time_axis[-1] + 1e-6, "free", {})]
    out = simulate_bloch_sequence(
        time_axis, events, M0=M0, T1=T1 if T1 else 1000.0, T2=T2, df=df
    )
    # transverse signal magnitude
    sig = out[:, 0] + 1j * out[:, 1]
    return sig


def bloch_ir_full(TIs, M0, T1, inversion_efficiency=1.0, df=0.0):
    # time axis in seconds
    time_axis = TIs / 1000.0
    events = [(0.0, time_axis[-1] + 1e-6, "free", {})]
    out = simulate_bloch_sequence(time_axis, events, M0=M0, T1=T1, T2=100.0, df=df)
    sig = out[:, 2]
    return sig
