"""Stage G0.3: resolution / measurement phantom (in memory only; nothing is written to data directories).

Synthetic tubes (straight or circular arc with curvature radius 30 mm, length 20 mm) with a radius profile along the
centreline, random orientation and sub-voxel offset, on the real SDF grid spacing. Vessel mask = voxel centres within
the local radius of the centreline; SDF = vatl_sdf.preprocess_sdf.signed_distance (voxel-centre EDT difference, the
convention of the data files); graph = chain of nodes every 1 mm on the centreline with the true radius. Measurement
with the pipeline's rules (identity transform, the true centreline as the population edge path):
  segmatch_med / segmatch_q25  median / 25th percentile of -SDF (trilinear, unclipped) at the matched interior
                               samples (vatl_sdf.deviation.matched_interior, 0.4 mm graph samples);
  node_allr                    median -SDF over graph samples within 3 mm of the middle node (radius_allr rule);
  node_absz                    the same restricted to edges with both endpoint radii >= 1.2 mm (radius_absz rule;
                               NaN if none).
Optimistic lower bound: no bifurcations, no segmentation noise, ideal perturbation (no attachment restore).
Usage: python analysis/deviation/phantom.py OUTPUT_JSON [--reps 30] [--workers 16]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from vatl_sdf.deviation import matched_interior
from vatl_sdf.graph_coords import edge_samples, world_to_voxel
from vatl_sdf.perturbation import stenosis_profile
from vatl_sdf.preprocess_sdf import signed_distance

SPACING = (0.9375, 0.9375, 0.8)      # training SDF grid (IXI633 header; all training SDFs share the 256 x 256 x 100 grid)
LENGTH_MM, CURVATURE_MM, NODE_STEP_MM, NODE_WINDOW_MM, LARGE_MM = 20., 30., 1., 3., 1.2
RADII = np.round(np.arange(0.8, 3.0001, 0.2), 1)
CONDITIONS = (('base', None), ('thin', 0.5), ('thin', 0.3), ('stenosis', 0.5), ('stenosis', 0.3))
RULES = ('segmatch_med', 'segmatch_q25', 'node_allr', 'node_absz')


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


def centreline(kind, rng, step=0.1):
    """Centreline points (step mm) of a straight or curved tube of LENGTH_MM, randomly rotated, centred at 0."""
    s = np.arange(0., LENGTH_MM + 1e-9, step)
    if kind == 'straight':
        pts = np.c_[s, np.zeros_like(s), np.zeros_like(s)]
    else:
        a = s / CURVATURE_MM
        pts = np.c_[CURVATURE_MM * np.sin(a), CURVATURE_MM * (1 - np.cos(a)), np.zeros_like(s)]
    pts = pts - pts.mean(0)
    return pts @ random_rotation(rng).T


def profile(path, radius, kind, factor):
    """True radius at the centreline points: base, thin (factor x radius) or stenosis (perturbation profile around the
    middle point)."""
    if kind == 'base':
        return np.full(len(path), float(radius))
    if kind == 'thin':
        return np.full(len(path), factor * radius)
    d = np.linalg.norm(path - path[len(path) // 2], axis=1)
    return radius * stenosis_profile(d, factor)


def tube(path, radii, spacing, offset, margin_mm=4.):
    """(sdf, affine, graph) of a tube along path with per-point radii; grid origin shifted by offset (mm)."""
    spacing = np.asarray(spacing, float)
    lo = path.min(0) - radii.max() - margin_mm - offset
    shape = np.ceil((path.max(0) + radii.max() + margin_mm - lo) / spacing).astype(int) + 1
    affine = np.diag([*spacing, 1.]); affine[:3, 3] = lo
    dense = np.vstack([np.linspace(path[i], path[i + 1], 5, endpoint=False) for i in range(len(path) - 1)] + [path[-1:]])
    dense_r = np.interp(np.linspace(0, len(path) - 1, len(dense)), np.arange(len(path)), radii)
    world = np.indices(shape).reshape(3, -1).T * spacing + lo
    d, j = cKDTree(dense).query(world, distance_upper_bound=radii.max() + 1.)
    inside = np.zeros(len(world), bool)
    ok = np.isfinite(d)
    inside[ok] = d[ok] <= dense_r[j[ok]]
    sdf = signed_distance(inside.reshape(shape), affine)
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    idx = np.unique(np.searchsorted(length, np.arange(0., length[-1] + 1e-9, NODE_STEP_MM)).clip(0, len(path) - 1))
    graph = {'pos': path[idx], 'radius': radii[idx], 'edges': np.c_[np.arange(len(idx) - 1), np.arange(1, len(idx))]}
    return sdf, affine, graph


def measure(sdf, affine, graph, path):
    pts, eid = edge_samples(graph['pos'], graph['edges'], 0.4)
    raw = -ndi.map_coordinates(sdf, world_to_voxel(affine, pts).T, order=1, mode='nearest')
    keep, _ = matched_interior(pts, eid, path, path[0], path[-1], 3.)
    out = {'segmatch_med': float(np.median(raw[keep])) if len(keep) else np.nan,
           'segmatch_q25': float(np.quantile(raw[keep], 0.25)) if len(keep) else np.nan}
    node = graph['pos'][len(graph['pos']) // 2]
    near = np.linalg.norm(pts - node, axis=1) <= NODE_WINDOW_MM
    r = np.asarray(graph['radius'])
    large = (r[graph['edges'][eid, 0]] >= LARGE_MM) & (r[graph['edges'][eid, 1]] >= LARGE_MM)
    out['node_allr'] = float(np.median(raw[near])) if near.any() else np.nan
    out['node_absz'] = float(np.median(raw[near & large])) if (near & large).any() else np.nan
    return out


def true_summaries(path, radii, graph):
    """The same summaries of the TRUE radius profile at the measurement samples (reference for slopes)."""
    pts, eid = edge_samples(graph['pos'], graph['edges'], 0.4)
    true = radii[cKDTree(path).query(pts)[1]]
    keep, _ = matched_interior(pts, eid, path, path[0], path[-1], 3.)
    node = graph['pos'][len(graph['pos']) // 2]
    near = np.linalg.norm(pts - node, axis=1) <= NODE_WINDOW_MM
    return {'segmatch_med': float(np.median(true[keep])), 'segmatch_q25': float(np.quantile(true[keep], 0.25)),
            'node_allr': float(np.median(true[near])), 'node_absz': float(np.median(true[near]))}


def placement(args):
    radius, kind, seed = args
    rng = np.random.default_rng(seed)
    path = centreline(kind, rng, step=0.5)
    offset = rng.uniform(0., 1., 3) * np.asarray(SPACING)
    out = {}
    for cond, factor in CONDITIONS:
        radii = profile(path, radius, cond, factor)
        sdf, affine, graph = tube(path, radii, SPACING, offset)
        out[f'{cond}{factor or ""}'] = {'measured': measure(sdf, affine, graph, path), 'true': true_summaries(path, radii, graph)}
    return radius, kind, out


def summarise(results):
    report = {}
    for kind in ('straight', 'curved'):
        for radius in RADII:
            rows = [o for r, k, o in results if k == kind and abs(r - radius) < 1e-9]
            entry = {}
            for rule in RULES:
                base = np.array([o['base']['measured'][rule] for o in rows])
                e = {'bias': float(np.nanmean(base - radius)), 'sd': float(np.nanstd(base, ddof=1)) if np.isfinite(base).sum() > 1 else None,
                     'nan_fraction': float(np.mean(~np.isfinite(base)))}
                for cond in ('thin0.5', 'thin0.3', 'stenosis0.5', 'stenosis0.3'):
                    dm = np.array([o[cond]['measured'][rule] - o['base']['measured'][rule] for o in rows])
                    dt = np.array([o[cond]['true'][rule] - o['base']['true'][rule] for o in rows])
                    ok = np.isfinite(dm) & (np.abs(dt) > 1e-6)
                    e[cond] = {'measured_change_median': float(np.median(dm[ok])) if ok.any() else None,
                               'true_change_median': float(np.median(dt[ok])) if ok.any() else None,
                               'slope': float(np.median(dm[ok] / dt[ok])) if ok.any() else None,
                               'change_sd': float(np.std(dm[ok], ddof=1)) if ok.sum() > 1 else None,
                               'nan_fraction': float(np.mean(~np.isfinite(dm)))}
                entry[rule] = e
            report[f'{kind}/r{radius:.1f}'] = entry
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('output'); p.add_argument('--reps', type=int, default=30); p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    tasks = [(float(r), kind, 1000 * i + rep) for i, r in enumerate(RADII) for kind in ('straight', 'curved') for rep in range(a.reps)]
    tasks = [(r, k, s + (0 if k == 'straight' else 500)) for r, k, s in tasks]
    with Pool(a.workers) as pool:
        results = pool.map(placement, tasks)
    report = {'spacing_mm': SPACING, 'length_mm': LENGTH_MM, 'curvature_mm': CURVATURE_MM, 'reps': a.reps,
              'note': 'optimistic lower bound: no bifurcations, no segmentation noise, ideal perturbation',
              'radii': summarise(results)}
    Path(a.output).write_text(json.dumps(report, indent=1))
    print(a.output)


if __name__ == '__main__':
    main()
