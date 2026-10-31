"""Alignment diagnostics shared by the Step 0 data analysis and the alignment pilots.

Large-vessel recurrence: for sampled large-vessel centreline points of subject i (mapped to a common
frame), the fraction of the other subjects whose large-vessel centreline lies within d mm. Large vessel
= graph edge with both endpoint radii >= min_radius_mm (graph radius, mm); centrelines are densified
along those edges at step_mm so the nearest-point distance approximates the centreline distance.
"""
import numpy as np
from scipy.spatial import cKDTree

from .graph_coords import edge_samples

DISTANCES_MM = (1.0, 2.0, 3.0)


def large_vessel_centerline(graph, min_radius_mm=1.2, step_mm=0.4):
    """World-mm points along edges whose two endpoints have graph radius >= min_radius_mm."""
    edges = graph['edges']
    if not len(edges):
        return np.zeros((0, 3))
    keep = (graph['radius'][edges[:, 0]] >= min_radius_mm) & (graph['radius'][edges[:, 1]] >= min_radius_mm)
    if not keep.any():
        return np.zeros((0, 3))
    return edge_samples(graph['pos'], edges[keep], step_mm)[0]


def recurrence(clouds, queries_per_subject=300, distances=DISTANCES_MM, seed=0):
    """clouds: list of (n_i, 3) arrays in a common mm frame. Returns per-query fractions {d: array} and owner ids."""
    trees = [cKDTree(c) if len(c) else None for c in clouds]
    rng = np.random.default_rng(seed)
    n = len(clouds)
    out = {d: [] for d in distances}
    owner = []
    for i, cloud in enumerate(clouds):
        if not len(cloud):
            continue
        pick = cloud[rng.choice(len(cloud), size=min(queries_per_subject, len(cloud)), replace=False)]
        hits = {d: np.zeros(len(pick)) for d in distances}
        others = 0
        for j, tree in enumerate(trees):
            if j == i or tree is None:
                continue
            others += 1
            dist = tree.query(pick, distance_upper_bound=max(distances))[0]
            for d in distances:
                hits[d] += dist <= d
        for d in distances:
            out[d].append(hits[d] / max(others, 1))
        owner.append(np.full(len(pick), i))
    return {d: np.concatenate(v) if v else np.zeros(0) for d, v in out.items()}, (np.concatenate(owner) if owner else np.zeros(0, int))


def summarize_recurrence(fractions):
    return {f'{d:g}mm': {'median': float(np.median(v)), 'mean': float(np.mean(v)), 'p25': float(np.percentile(v, 25)),
                         'p75': float(np.percentile(v, 75)), 'queries': int(len(v))} for d, v in fractions.items()}


def principal_axes(points):
    """Eigenvalues (descending) and eigenvectors (columns) of the point covariance."""
    centred = points - points.mean(0)
    values, vectors = np.linalg.eigh(centred.T @ centred / len(points))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def axis_angles_deg(vectors, reference):
    """Sign-invariant angle (deg) between each column of vectors and the matching column of reference."""
    cosine = np.abs(np.sum(vectors * reference, axis=0)).clip(0, 1)
    return np.degrees(np.arccos(cosine))


def mean_axes(all_vectors):
    """Sign-aligned mean of per-subject principal axes (list of 3x3, columns = axes), re-orthonormalized."""
    reference = all_vectors[0]
    total = np.zeros((3, 3))
    for v in all_vectors:
        total += v * np.sign(np.sum(v * reference, axis=0))[None, :]
    u, _, vt = np.linalg.svd(total)
    return u @ vt


def transform_statistics(matrices, translations_mm):
    """Rotation angle (deg), translation norm (mm), scale (det^(1/3)) and anisotropy of per-subject maps."""
    angles, scales, aniso = [], [], []
    for a in matrices:
        u, s, vt = np.linalg.svd(a)
        r = u @ vt
        angles.append(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
        scales.append(float(np.prod(s) ** (1 / 3)))
        aniso.append(float(s.max() / s.min()))
    t = np.linalg.norm(translations_mm, axis=1)

    def pct(x):
        x = np.asarray(x, dtype=float)
        return {'median': float(np.median(x)), 'p95': float(np.percentile(x, 95)), 'max': float(x.max())}
    return {'rotation_deg': pct(angles), 'translation_mm': pct(t), 'scale': pct(scales),
            'scale_spread_p5_p95': [float(np.percentile(scales, 5)), float(np.percentile(scales, 95))],
            'anisotropy': pct(aniso), 'mean_translation_mm': np.asarray(translations_mm).mean(0).tolist()}
