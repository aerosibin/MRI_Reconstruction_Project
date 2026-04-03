import numpy as np


def synthesize_moving_gaussian(H=64, W=64, T=32, num_blobs=1, amplitude=1.0):
    """Create a simple moving Gaussian cine stack.

    Returns array shape (T, H, W) float32 in [0,1].
    """
    yy, xx = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    stack = np.zeros((T, H, W), dtype=np.float32)

    # For each blob create a circular Gaussian whose center moves sinusoidally
    rng = np.random.RandomState(0)
    for b in range(num_blobs):
        sx = rng.uniform(0.05, 0.25)  # spatial sigma
        sy = sx
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.5, 1.5)
        # initial offset
        cx0 = rng.uniform(-0.5, 0.5)
        cy0 = rng.uniform(-0.5, 0.5)

        for t in range(T):
            # center moves in a small circular trajectory
            theta = phase + 2 * np.pi * freq * (t / float(T))
            cx = cx0 + 0.4 * np.cos(theta)
            cy = cy0 + 0.4 * np.sin(theta)
            # gaussian
            g = amplitude * np.exp(
                -(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2))
            )
            stack[t] += g

    # normalize to 0-1
    stack = stack - stack.min()
    if stack.max() > 0:
        stack = stack / stack.max()
    return stack


def synthesize_moving_rect(H=64, W=64, T=32):
    """Simple moving bright rectangle — fallback/demo helper."""
    stack = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        x = int((W - 16) * (0.5 + 0.5 * np.sin(2 * np.pi * t / T)))
        y = int((H - 16) * (0.5 + 0.5 * np.cos(2 * np.pi * t / T)))
        stack[t, y : y + 16, x : x + 16] = 1.0
    return stack
