"""Analytic SDF rendering of a vessel graph (Phase 5 prototype).

Each edge (a, b) with radii (r_a, r_b) is a round cone; the graph SDF is the min (union) or a smooth
min over edges. A graph with radii is therefore a parametric, connected-by-construction representation
of the same geometry as the SDF. Used to quantify how much of the reference geometry the imperfect,
quantized graph captures.
"""
import numpy as np


def round_cone_sdf(p, a, b, ra, rb):
    """Exact SDF of a round cone (spheres at a, b joined by tangent cone), vectorized over points p (N, 3).

    Falls back to the larger sphere when one end sphere contains the other.
    """
    ba = b - a
    l2 = float(ba @ ba)
    rr = ra - rb
    if l2 < 1e-12 or rr * rr >= l2:  # degenerate: one sphere contains the other
        return np.minimum(np.linalg.norm(p - a, axis=1) - ra, np.linalg.norm(p - b, axis=1) - rb)
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    x2 = np.sum((pa * l2 - np.outer(y, ba)) ** 2, axis=1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    out = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - ra
    cap_b = np.sign(z) * a2 * z2 > k
    cap_a = ~cap_b & (np.sign(y) * a2 * y2 < k)
    out = np.where(cap_b, np.sqrt(x2 + z2) * il2 - rb, out)
    out = np.where(cap_a, np.sqrt(x2 + y2) * il2 - ra, out)
    return out


def render(world_grid, shape, positions, radii, edges, smooth_k_mm=0.0, margin_mm=2.0, far_mm=5.0):
    """SDF (mm) of the union of round cones on a grid given as world points (prod(shape), 3).

    Each edge is evaluated only inside its bounding box grown by its radius + margin_mm; elsewhere the
    field is far_mm (renderings are compared within the truncation band). smooth_k_mm > 0 applies an
    exponential smooth min, smin = m - k log sum exp(-(d_i - m) / k), over the edges in reach.
    """
    flat = np.full(len(world_grid), np.inf)
    spacing_index = _grid_index(world_grid, shape)
    boxes = []
    for e, (i, j) in enumerate(edges):
        a, b = positions[i], positions[j]
        reach = max(radii[i], radii[j]) + margin_mm
        idx = spacing_index(np.minimum(a, b) - reach, np.maximum(a, b) + reach)
        if idx.size == 0:
            continue
        d = round_cone_sdf(world_grid[idx], a, b, radii[i], radii[j])
        flat[idx] = np.minimum(flat[idx], d)
        boxes.append((idx, d))
    if smooth_k_mm > 0:
        accum = np.zeros(len(world_grid))
        for idx, d in boxes:
            accum[idx] += np.exp(-(d - flat[idx]) / smooth_k_mm)
        touched = accum > 0
        flat[touched] = flat[touched] - smooth_k_mm * np.log(accum[touched])
    flat[~np.isfinite(flat)] = far_mm
    return np.minimum(flat, far_mm).reshape(shape)


def _grid_index(world_grid, shape):
    """Returns f(lo, hi) -> flat indices of grid points whose world coords lie in the box [lo, hi].

    Works for oblique grids by bounding in voxel space: uses the voxel->world map recovered from the grid.
    """
    shape = tuple(shape)
    origin = world_grid[0]
    steps = np.stack([world_grid[np.ravel_multi_index(tuple(np.eye(3, dtype=int)[k]), shape)] - origin for k in range(3)], 1)
    inverse = np.linalg.inv(steps)

    def query(lo, hi):
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        vox = (corners - origin) @ inverse.T
        start = np.clip(np.floor(vox.min(0)).astype(int), 0, np.array(shape) - 1)
        stop = np.clip(np.ceil(vox.max(0)).astype(int) + 1, 0, np.array(shape))
        if np.any(stop <= start):
            return np.zeros(0, dtype=np.int64)
        block = np.stack(np.meshgrid(*[np.arange(s, t) for s, t in zip(start, stop)], indexing='ij'), -1).reshape(-1, 3)
        return np.ravel_multi_index(block.T, shape)
    return query
