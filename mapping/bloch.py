import numpy as np


def simulate_ir_signal(TIs, M0, T1, T2=None, inversion_efficiency=1.0):
    TIs = np.asarray(TIs)
    inv = -inversion_efficiency
    return M0 * (1 - (1 + inv) * np.exp(-TIs / T1))


def simulate_multi_echo(TEs, M0, T2, T1=None):
    TEs = np.asarray(TEs)
    return M0 * np.exp(-TEs / T2)
