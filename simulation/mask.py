import numpy as np


def variable_density_mask(shape, center_fraction=0.08, acceleration=4, seed=None):
    H, W = shape
    rng = np.random.RandomState(seed)

    mask = np.zeros((H, W), dtype=bool)

    center_w = int(W * center_fraction)
    start = W // 2 - center_w // 2
    end = start + center_w
    mask[:, start:end] = True

    # target number of sampled columns
    target = W // acceleration
    # probability for each outer column
    outer_indices = list(range(0, start)) + list(range(end, W))
    prob = max(0.0, (target - center_w) / max(1, len(outer_indices)))

    for i in outer_indices:
        if rng.rand() < prob:
            mask[:, i] = True

    return mask


def multi_layer_variable_density_mask(
    shape,
    layers=None,
    center_fraction=0.08,
    default_acceleration=4,
    seed=None,
):
    """Create a multi-layer variable-density Cartesian mask.

    Parameters
    - shape: (H, W)
    - layers: list of dicts describing concentric sampling layers. Each dict may contain:
        - 'r_max_frac': maximum radius (0..1) relative to kx/kx max (outer bound)
        - 'accel': effective acceleration for that layer (smaller -> denser sampling)
        - 'center_fraction': optional fully-sampled center fraction for that layer
      Layers are applied from innermost to outermost. If layers is None a default two-layer
      scheme is used: dense center + sparse outer.
    - center_fraction: fallback center fraction if not specified in layers
    - default_acceleration: fallback accel for layers that do not specify 'accel'
    - seed: RNG seed

    Returns boolean mask (H, W)
    """
    H, W = shape
    rng = np.random.RandomState(seed)

    # default two-layer if not provided
    if layers is None:
        layers = [
            {"r_max_frac": 0.2, "accel": 1},  # nearly fully sampled center
            {"r_max_frac": 1.0, "accel": default_acceleration},
        ]

    # create a radial coordinate (normalized 0..1 where 0 is center)
    ky = np.linspace(-0.5, 0.5, W, endpoint=False)
    kx = np.linspace(-0.5, 0.5, H, endpoint=False)
    yy, xx = np.meshgrid(ky, kx)
    r = np.sqrt(xx**2 + yy**2)
    r = r / r.max()

    mask = np.zeros((H, W), dtype=bool)

    for layer in layers:
        r_max = float(layer.get("r_max_frac", 1.0))
        accel = float(layer.get("accel", default_acceleration))
        cf = float(layer.get("center_fraction", 0.0))

        # decide which pixels fall into this layer's annulus (inner..r_max]
        prev_r_max = layer.get("r_min_frac", 0.0)
        in_layer = (r > prev_r_max) & (r <= r_max)

        # fully sample inner small center if center_fraction > 0
        if cf > 0:
            center_w = int(W * cf)
            start = W // 2 - center_w // 2
            end = start + center_w
            mask[:, start:end] = True

        # target number of samples in this layer
        layer_indices = np.where(in_layer)
        P = len(layer_indices[0])
        if P == 0:
            continue

        # expected sampling probability for pixels in this layer
        prob = 1.0 / max(1.0, accel)

        draw = rng.rand(P) < prob
        mask[layer_indices[0][draw], layer_indices[1][draw]] = True

    return mask
