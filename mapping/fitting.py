import numpy as np
from scipy.optimize import least_squares
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from mapping.signal_model import t2_signal, t1_ir_signal
from mapping.bloch_full import simulate_bloch_ir

# Optional: use torch for batched GPU fitting if available
try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def fit_t2_curve(TEs, signals, mask=None, bounds=(0, 2000)):
    """Fit T2 per-pixel.

    Fast path: use a log-linear regression to estimate M0 and T2 in a vectorized way.
    signals: shape (N, H, W) where N=len(TEs)
    Returns M0_map, T2_map
    """
    # Vectorized log-linear fit: ln(y) = ln(M0) - TE * (1/T2)
    N, H, W = signals.shape
    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    TEs = np.asarray(TEs).astype(np.float32)
    t = TEs.reshape(-1, 1)  # (N,1)

    # Avoid non-positive values for log by adding small eps
    eps = 1e-8
    y = signals.astype(np.float32)
    y = np.maximum(y, eps)
    logy = np.log(y)  # (N,H,W)

    # compute means along time axis
    mean_t = TEs.mean()
    mean_logy = logy.mean(axis=0)  # (H,W)

    # numerator and denominator for slope for each pixel
    t_centered = (TEs - mean_t).reshape(-1, 1, 1)  # (N,1,1)
    num = (t_centered * (logy - mean_logy)).sum(axis=0)  # (H,W)
    den = (t_centered**2).sum()

    # slope = num / den  => slope = -1/T2
    # protect divide-by-zero
    den = den if den != 0 else 1.0
    slope = num / den
    with np.errstate(divide="ignore", invalid="ignore"):
        T2_map = -1.0 / slope
    M0_map = np.exp(mean_logy - slope * mean_t)

    # Mask out invalids
    invalid = ~mask
    T2_map[invalid] = 0.0
    M0_map[invalid] = 0.0

    # Clip to bounds
    T2_map = np.clip(T2_map, bounds[0], bounds[1]).astype(np.float32)
    M0_map = M0_map.astype(np.float32)

    return M0_map, T2_map


def _jac_ir(params, t, y=None):
    """Analytic Jacobian for the IR model residuals r = M0*(1 - 2*exp(-t/T1)) - y.

    Returns jac of shape (len(t), 2) for residuals wrt [M0, T1].
    """
    M0, T1 = params
    t = np.asarray(t)
    e = np.exp(-t / T1)
    # dr/dM0 = (1 - 2*e)
    jm0 = 1.0 - 2.0 * e
    # dr/dT1 = -2 * M0 * e * (t / T1**2)
    jt1 = -2.0 * M0 * e * (t / (T1**2))
    return np.vstack([jm0, jt1]).T


def _fit_t1_chunk(TIs, ys_chunk, bounds):
    """Worker: fit a chunk of pixels provided as ys_chunk shape (N, Pchunk).

    Returns two 1D numpy arrays (M0s, T1s) of length Pchunk
    """
    N, P = ys_chunk.shape
    M0s = np.zeros((P,), dtype=np.float32)
    T1s = np.zeros((P,), dtype=np.float32)

    t = np.asarray(TIs)
    for idx in range(P):
        y = ys_chunk[:, idx]
        # initial guesses
        M0_init = float(np.max(np.abs(y)))
        T1_init = 1000.0
        try:
            res = least_squares(
                lambda params, t, y: (
                    params[0] * (1 - 2 * np.exp(-t / params[1])) - y
                ).ravel(),
                x0=[M0_init, T1_init],
                jac=_jac_ir,
                args=(t, y),
                bounds=([0, bounds[0]], [np.inf, bounds[1]]),
                max_nfev=2000,
            )
            M0s[idx], T1s[idx] = res.x
        except Exception:
            # on failure, leave zeros
            M0s[idx], T1s[idx] = 0.0, 0.0

    return M0s, T1s


def _fit_t1_chunk_bloch(
    TIs, ys_chunk, bounds, inversion_efficiency=1.0, steps_per_ms=1
):
    """Worker: fit a chunk of pixels using Bloch RK4 forward model for IR signals.

    ys_chunk: (N, Pchunk)
    """
    N, P = ys_chunk.shape
    M0s = np.zeros((P,), dtype=np.float32)
    T1s = np.zeros((P,), dtype=np.float32)

    t = np.asarray(TIs)
    for idx in range(P):
        y = ys_chunk[:, idx]
        # initial guesses
        M0_init = float(np.max(np.abs(y)))
        T1_init = 1000.0
        try:
            res = least_squares(
                lambda params, t, y: (
                    simulate_bloch_ir(
                        t,
                        params[0],
                        params[1],
                        inversion_efficiency=inversion_efficiency,
                        steps_per_ms=steps_per_ms,
                    )
                    - y
                ).ravel(),
                x0=[M0_init, T1_init],
                args=(t, y),
                bounds=([0, bounds[0]], [np.inf, bounds[1]]),
                max_nfev=2000,
            )
            M0s[idx], T1s[idx] = res.x
        except Exception:
            M0s[idx], T1s[idx] = 0.0, 0.0

    return M0s, T1s


def fit_t1_ir(
    TIs,
    signals,
    mask=None,
    bounds=(1, 5000),
    n_workers=None,
    chunk_size=1024,
    use_bloch=False,
    steps_per_ms=1.0,
    inversion_eff=1.0,
    verbose=False,
):
    """Fit T1 per-pixel from inversion recovery signals using parallel least_squares with analytic jacobian.

    signals: shape (N, H, W)
    Returns M0_map, T1_map
    """
    # Prefer PyTorch batched fitter if available and user didn't request multiprocessing explicitly
    if TORCH_AVAILABLE and (n_workers is None):
        return fit_t1_ir_batched(TIs, signals, mask=mask, bounds=bounds)

    N, H, W = signals.shape
    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    T1_map = np.zeros((H, W), dtype=np.float32)
    M0_map = np.zeros((H, W), dtype=np.float32)

    # Flatten and select pixels
    signals_flat = signals.reshape(N, -1)
    mask_flat = mask.reshape(-1)
    indices = np.where(mask_flat)[0]
    P = len(indices)
    if P == 0:
        return M0_map, T1_map

    ys = signals_flat[:, indices]  # (N, P)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    # Chunk the columns to reduce IPC overhead
    chunks = []
    for i in range(0, P, chunk_size):
        chunks.append(ys[:, i : i + chunk_size])

    results = []
    if n_workers == 1:
        # Single process, run locally
        for ch in chunks:
            if use_bloch:
                results.append(
                    _fit_t1_chunk_bloch(
                        TIs,
                        ch,
                        bounds,
                        inversion_efficiency=inversion_eff,
                        steps_per_ms=steps_per_ms,
                    )
                )
            else:
                results.append(_fit_t1_chunk(TIs, ch, bounds))
            if verbose:
                print(f"Processed chunk, total_chunks={len(chunks)}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = []
            for ch in chunks:
                if use_bloch:
                    futures.append(
                        ex.submit(
                            _fit_t1_chunk_bloch,
                            TIs,
                            ch,
                            bounds,
                            inversion_eff,
                            steps_per_ms,
                        )
                    )
                else:
                    futures.append(ex.submit(_fit_t1_chunk, TIs, ch, bounds))

            # collect as they complete for progress reporting
            total = len(futures)
            done = 0
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception:
                    res = (np.zeros((0,)), np.zeros((0,)))
                results.append(res)
                done += 1
                if verbose:
                    print(f"Completed {done}/{total} chunks")

    # Fill results back
    ptr = 0
    for M0s, T1s in results:
        L = len(M0s)
        inds = indices[ptr : ptr + L]
        M0_map.reshape(-1)[inds] = M0s
        T1_map.reshape(-1)[inds] = T1s
        ptr += L

    return M0_map, T1_map


def fit_t1_ir_batched(
    TIs, signals, mask=None, bounds=(1, 5000), device=None, max_iter=400, lr=1e-2
):
    """Batched per-pixel fit for inversion-recovery T1 using PyTorch.

    TIs: array-like (N,)
    signals: numpy array shape (N, H, W)
    mask: bool array (H,W) selecting pixels to fit
    Returns M0_map, T1_map (H, W)

    This function optimizes per-pixel M0 and T1 in parallel on the chosen device.
    Parameters are constrained to be positive by optimizing in log-domain.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for batched fitting but is not available."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    TIs = np.asarray(TIs).astype(np.float32)
    N, H, W = signals.shape
    if mask is None:
        mask = np.ones((H, W), dtype=bool)

    # Prepare data: flatten pixels to batch
    mask_flat = mask.reshape(-1)
    P = int(mask_flat.sum())
    if P == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    y = signals.reshape(N, -1)[:, mask_flat]  # (N, P)
    y_t = torch.from_numpy(y).to(device=device)
    tis_t = torch.from_numpy(TIs.reshape(-1, 1)).to(device=device)  # (N,1)

    # Initialize parameters
    # M0 init: max abs across TIs
    M0_init = np.max(np.abs(y), axis=0).astype(np.float32)  # (P,)
    T1_init = np.full((P,), 1000.0, dtype=np.float32)

    # log-domain params for positivity
    log_M0 = torch.nn.Parameter(
        torch.log(torch.from_numpy(M0_init + 1e-6).to(device=device))
    )
    log_T1 = torch.nn.Parameter(torch.log(torch.from_numpy(T1_init).to(device=device)))

    optimizer = torch.optim.Adam([log_M0, log_T1], lr=lr)

    # Optimization loop
    for it in range(max_iter):
        optimizer.zero_grad()
        M0_pred = torch.exp(log_M0).unsqueeze(0)  # (1,P)
        T1_pred = torch.exp(log_T1).unsqueeze(0)  # (1,P)
        # predicted signal: M0*(1 - 2*exp(-TI/T1)) broadcast over (N,P)
        pred = M0_pred * (1.0 - 2.0 * torch.exp(-tis_t / T1_pred))
        loss = torch.mean((pred - y_t) ** 2)
        loss.backward()
        optimizer.step()

        # simple stopping criteria (optional)
        if it % 50 == 0:
            # avoid spamming; still allow external monitoring if needed
            pass

    M0_fit = torch.exp(log_M0).detach().cpu().numpy()  # (P,)
    T1_fit = torch.exp(log_T1).detach().cpu().numpy()  # (P,)

    # Place back into maps
    M0_map = np.zeros(H * W, dtype=np.float32)
    T1_map = np.zeros(H * W, dtype=np.float32)
    M0_map[mask_flat] = M0_fit
    T1_map[mask_flat] = T1_fit

    M0_map = M0_map.reshape(H, W)
    T1_map = T1_map.reshape(H, W)

    # Clip to bounds
    T1_map = np.clip(T1_map, bounds[0], bounds[1]).astype(np.float32)
    M0_map = M0_map.astype(np.float32)

    return M0_map, T1_map
