"""Connectivity and radius atlases on the population vessel graph (inference only; no graph repair).

Inputs: a checkpoint (trained subject -> template maps), the population graphs of vatl_sdf.graph_atlas (nodes and
edges in template mm) and the test-time fitted transforms of the held-out subjects.

Per subject (native space, own graph and own reference SDF; vessel = SDF < 0 = label 1):
  large-vessel centreline: points densified at 0.4 mm along graph edges whose two endpoint radii are >= 1.2 mm
      (graph-radius selection, as recurrence / graph atlas); mapped to the template by the subject's transform.
  selection comparison: the same points selected by local radius -SDF >= 1.2 mm (training alignment target).
  node presence: large-vessel centreline within r of the population node (template space); the nearest such point
      is the node's anchor in the subject.
  connectivity of a population edge (a, b), when both nodes are present, two criteria side by side:
      graph  the anchors' nearer edge endpoints are connected in the subject's FULL graph (all radii) by a path of
             length <= path_factor x the straight-line distance between the anchors (native mm);
      mask   the anchors fall in the same 26-connected component of the reference mask (anchor voxel, or the
             nearest vessel voxel within snap_mm; none -> not connected).
  radius: local radius -SDF (trilinear) at centreline points (ground truth) and graph radius (linear along the edge)
      as a secondary measure, native mm and scaled by the subject's learned scale det(A)^(1/3); floor-radius
      (<= 0.801 mm graph radius) points of the whole graph are summarised separately.
Atlas: per population edge P(both present), P(connected | both), P(connected) for each criterion and the fraction
mask-connected but graph-disconnected; per node / edge radius median, IQR and number of subjects; per-voxel radius
maps. Evaluation: held-out subjects (fitted transforms; centroid-translation-only control) and the first
--train-queries training subjects with leave-one-out statistics.
"""
import argparse
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .graph_atlas import held_out_rows
from .graph_coords import edge_samples, load_graph, voxel_to_world, world_to_voxel
from .inference import world_grid
from .model import SDFModel
from .population_maps import save
from .probabilistic_atlas import within_radius

FLOOR_MM = 0.801
STRUCTURE_26 = np.ones((3, 3, 3), bool)


def centreline_samples(graph, step_mm=0.4):
    """Points along all graph edges with edge id, graph radius (linear interpolation) and the edge endpoints."""
    pos, radius, edges = graph['pos'], np.asarray(graph['radius'], float), graph['edges']
    if not len(edges):
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0)
    points, eid = edge_samples(pos, edges, step_mm)
    a, b = pos[edges[eid, 0]], pos[edges[eid, 1]]
    length = np.linalg.norm(b - a, axis=1)
    t = np.where(length > 0, np.linalg.norm(points - a, axis=1) / np.maximum(length, 1e-9), 0.)
    return points, eid, radius[edges[eid, 0]] * (1 - t) + radius[edges[eid, 1]] * t


def graph_matrix(graph):
    pos, edges = graph['pos'], graph['edges']
    n = len(pos)
    if not len(edges):
        return coo_matrix((n, n)).tocsr()
    w = np.maximum(np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1), 1e-6)
    return coo_matrix((np.r_[w, w], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()


def mask_label(labels, sdf, affine, world, snap_mm):
    """Component label at a world point: its voxel if inside the mask, else the nearest vessel voxel within snap_mm."""
    voxel = np.rint(world_to_voxel(affine, world[None])[0]).astype(int)
    spacing = nib.affines.voxel_sizes(affine)
    reach = np.ceil(snap_mm / spacing).astype(int)
    lo = np.maximum(voxel - reach, 0); hi = np.minimum(voxel + reach + 1, sdf.shape)
    if (lo >= hi).any():
        return 0
    window = labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    inside = np.argwhere(window > 0)
    if not len(inside):
        return 0
    centres = voxel_to_world(affine, inside + lo)
    d = np.linalg.norm(centres - world, axis=1)
    k = d.argmin()
    return int(window[tuple(inside[k])]) if d[k] <= snap_mm else 0


def subject_attributes(args):
    """Per-subject presence, connectivity and radius on every population graph (see module docstring)."""
    row, A, t, populations, p = args
    image = nib.load(row['SDF'])
    sdf = np.asarray(image.dataobj, dtype=np.float32)
    sdf = sdf[..., 0] if sdf.ndim == 4 else sdf
    affine = image.affine
    centre = voxel_to_world(affine, np.asarray(sdf.shape, float) / 2.)
    graph = load_graph(row['Graph'])
    points, eid, graph_r = centreline_samples(graph)
    sdf_r = -ndi.map_coordinates(sdf, world_to_voxel(affine, points).T, order=1, mode='nearest') if len(points) else np.zeros(0)
    edges = graph['edges']; radius = np.asarray(graph['radius'], float)
    by_graph = (radius[edges[eid, 0]] >= p['min_radius_mm']) & (radius[edges[eid, 1]] >= p['min_radius_mm']) if len(points) else np.zeros(0, bool)
    by_sdf = sdf_r >= p['min_radius_mm']
    floor = graph_r <= FLOOR_MM
    union = (by_graph | by_sdf).sum()
    selection = {'points': int(len(points)), 'graph_selected': int(by_graph.sum()), 'sdf_selected': int(by_sdf.sum()),
                 'jaccard': float((by_graph & by_sdf).sum() / union) if union else None,
                 'graph_selected_outside_mask': float((sdf_r[by_graph] <= 0).mean()) if by_graph.any() else None,
                 'sdf_selected_floor_graph_radius': float(floor[by_sdf].mean()) if by_sdf.any() else None,
                 'graph_selected_median_sdf_minus_graph_mm': float(np.median(sdf_r[by_graph] - graph_r[by_graph])) if by_graph.any() else None,
                 'floor_points': int(floor.sum()),
                 'floor_median_sdf_radius_mm': float(np.median(sdf_r[floor])) if floor.any() else None,
                 'floor_inside_mask': float((sdf_r[floor] > 0).mean()) if floor.any() else None}
    keep = np.flatnonzero(by_graph)
    native = points[keep]
    template = (native - centre) @ A.T + t
    scale = float(np.cbrt(abs(np.linalg.det(A))))
    out = {'subject_id': row['subject_id'], 'scale': scale, 'selection': selection, 'populations': {}}
    if not len(keep):
        for key, pop in populations.items():
            ne, nn = len(pop['edges']), len(pop['nodes_mm'])
            out['populations'][key] = {'present': np.zeros(nn, bool), 'both': np.zeros(ne, bool), 'graph': np.zeros(ne, bool),
                                       'mask': np.zeros(ne, bool), 'node_radius': np.full((nn, 2), np.nan),
                                       'edge_radius': np.full((ne, 2), np.nan)}
        out['voxels'] = (np.zeros((0, 3), int), np.zeros((0, 2)))
        return out
    tree = cKDTree(template)
    labels, _ = ndi.label(sdf < 0, STRUCTURE_26)
    matrix = graph_matrix(graph)
    for key, pop in populations.items():
        r = pop['radius_mm']
        nodes = np.asarray(pop['nodes_mm'])
        d, idx = tree.query(nodes) if len(nodes) else (np.zeros(0), np.zeros(0, int))
        present = d <= r
        anchors = native[idx] if len(nodes) else np.zeros((0, 3))
        anchor_edge = eid[keep][idx] if len(nodes) else np.zeros(0, int)
        anchor_node = np.array([edges[e][np.argmin(np.linalg.norm(graph['pos'][edges[e]] - x, axis=1))]
                                for e, x in zip(anchor_edge, anchors)], int) if len(nodes) else np.zeros(0, int)
        labels_at = np.array([mask_label(labels, sdf, affine, x, p['snap_mm']) if ok else 0 for x, ok in zip(anchors, present)], int)
        both, g_conn, m_conn = [], [], []
        for a, b in pop['edges']:
            ok = bool(present[a] and present[b])
            both.append(ok)
            if not ok:
                g_conn.append(False); m_conn.append(False); continue
            limit = p['path_factor'] * max(np.linalg.norm(anchors[a] - anchors[b]), p['min_limit_mm'])
            if anchor_node[a] == anchor_node[b]:
                g_conn.append(True)
            else:
                dist = dijkstra(matrix, indices=anchor_node[a], limit=limit)[anchor_node[b]]
                g_conn.append(bool(np.isfinite(dist) and dist <= limit))
            m_conn.append(bool(labels_at[a] > 0 and labels_at[a] == labels_at[b]))
        # radius per population node and per edge path: median over the subject's points within r
        node_r = np.full((len(nodes), 2), np.nan)
        for k, near in enumerate(tree.query_ball_point(nodes, r) if len(nodes) else []):
            if near:
                node_r[k] = np.median(sdf_r[keep][near]), np.median(graph_r[keep][near])
        edge_r = np.full((len(pop['edges']), 2), np.nan)
        for k, path in enumerate(pop['edge_points_mm']):
            near = sorted(set().union(*tree.query_ball_point(path, r)))
            if near:
                edge_r[k] = np.median(sdf_r[keep][near]), np.median(graph_r[keep][near])
        out['populations'][key] = {'present': present, 'both': np.array(both, bool), 'graph': np.array(g_conn, bool),
                                   'mask': np.array(m_conn, bool), 'node_radius': node_r, 'edge_radius': edge_r}
    grid_affine = p['grid_affine']
    voxel = np.rint(world_to_voxel(grid_affine, template)).astype(int)
    inside = (voxel >= 0).all(1) & (voxel < p['grid_shape']).all(1)
    out['voxels'] = (voxel[inside], np.c_[sdf_r[keep][inside], graph_r[keep][inside]])
    return out


def edge_atlas(results, key):
    both = np.array([r['populations'][key]['both'] for r in results])
    g = np.array([r['populations'][key]['graph'] for r in results]); m = np.array([r['populations'][key]['mask'] for r in results])
    return both, g, m


def connectivity_stats(both, g, m):
    n = len(both)
    nb = both.sum(0)
    ratio = lambda x, d: np.where(d > 0, x / np.maximum(d, 1), np.nan)
    return {'p_both': both.mean(0), 'n_both': nb,
            'graph_p_conn_given_both': ratio(g.sum(0), nb), 'graph_p_conn': g.sum(0) / n,
            'mask_p_conn_given_both': ratio(m.sum(0), nb), 'mask_p_conn': m.sum(0) / n,
            'mask_not_graph_given_both': ratio((m & ~g).sum(0), nb), 'graph_not_mask_given_both': ratio((g & ~m).sum(0), nb)}


def bernoulli(k, n, y):
    """Smoothed predictive probability (k + 1) / (n + 2) of y."""
    p = (k + 1.) / (n + 2.)
    return np.where(y, np.log(p), np.log1p(-p)), p


def score_connectivity(obs, counts, n):
    """obs: dict both/graph/mask (edges,) of one subject; counts: dict of atlas sums excluding the subject; n subjects."""
    out = {}
    for c in ('graph', 'mask'):
        both = obs['both']
        ll_cond, p_cond = bernoulli(counts[c], counts['both'], obs[c])
        ll_all, p_all = bernoulli(counts[c], n, obs[c])
        observed = both & obs[c]
        out[c] = {'edges_both_present': int(both.sum()), 'edges_connected': int(observed.sum()),
                  'loglik_given_both': float(ll_cond[both].mean()) if both.any() else None,
                  'loglik_unconditional': float(ll_all.mean()) if len(ll_all) else None,
                  'coverage_connected_p_ge_0.5': float((p_all[observed] >= 0.5).mean()) if observed.any() else None,
                  'brier_given_both': float(np.mean((p_cond[both] - obs[c][both]) ** 2)) if both.any() else None}
    return out


def radius_errors(values, distribution):
    """values: (k,) subject node radii; distribution: (subjects, k) atlas values (NaN = absent). Median abs error,
    robust z (IQR / 1.349) and percentile per node with >= 3 atlas subjects."""
    abs_err, z, pct = [], [], []
    for k, v in enumerate(values):
        column = distribution[:, k]; column = column[np.isfinite(column)]
        if not np.isfinite(v) or len(column) < 3:
            continue
        med = np.median(column); q1, q3 = np.percentile(column, [25, 75])
        abs_err.append(abs(v - med)); z.append((v - med) / max((q3 - q1) / 1.349, 1e-6)); pct.append(float((column <= v).mean()))
    if not abs_err:
        return None
    return {'nodes': len(abs_err), 'median_abs_error_mm': float(np.median(abs_err)), 'median_abs_z': float(np.median(np.abs(z))),
            'fraction_abs_z_le_2': float((np.abs(z) <= 2).mean()), 'median_percentile': float(np.median(pct)),
            'fraction_percentile_5_95': float(np.mean([(0.05 <= q <= 0.95) for q in pct]))}


def radius_summary(values):
    """values (subjects, k): per node/edge median, IQR and subject count."""
    finite = np.isfinite(values)
    out = []
    for k in range(values.shape[1]):
        column = values[finite[:, k], k]
        out.append({'n_subjects': int(len(column)), 'median': float(np.median(column)) if len(column) else None,
                    'q25': float(np.percentile(column, 25)) if len(column) else None,
                    'q75': float(np.percentile(column, 75)) if len(column) else None})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--fitted', required=True)
    p.add_argument('--graph-atlas', required=True, help='output directory of vatl_sdf.graph_atlas')
    p.add_argument('--output', required=True)
    p.add_argument('--populations', nargs='+', default=['r2mm_tau0.1', 'r3mm_tau0.1'])
    p.add_argument('--path-factor', type=float, default=3.0); p.add_argument('--min-limit-mm', type=float, default=2.0)
    p.add_argument('--snap-mm', type=float, default=1.0); p.add_argument('--min-radius-mm', type=float, default=1.2)
    p.add_argument('--train-queries', type=int, default=100); p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    start = time.time()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    model = SDFModel(config, len(frame)); model.load_state_dict(checkpoint['model'])
    train_a, train_t = (x.double().numpy() for x in model.maps_mm())
    fitted = torch.load(a.fitted, map_location='cpu', weights_only=False)
    tests = held_out_rows(config, fitted['subject_ids'])
    if set(tests.subject_id) & set(frame.subject_id):
        raise ValueError('Held-out subjects overlap the training subjects')
    populations = {}
    for key in a.populations:
        with open(Path(a.graph_atlas) / f'population_graph_{key}.pkl', 'rb') as stream:
            populations[key] = pickle.load(stream)
    _, shape, grid_affine = world_grid(config, [1.0] * 3)
    params = {'min_radius_mm': a.min_radius_mm, 'path_factor': a.path_factor, 'min_limit_mm': a.min_limit_mm,
              'snap_mm': a.snap_mm, 'grid_affine': grid_affine, 'grid_shape': np.asarray(shape)}
    from analysis.alignment.evaluate_alignment import subject_geometry
    with Pool(a.workers) as pool:
        train = pool.map(subject_attributes, [(row, train_a[i], train_t[i], populations, params) for i, (_, row) in enumerate(frame.iterrows())])
        geo = pool.map(subject_geometry, [(row, a.min_radius_mm) for _, row in frame.iterrows()])
        test_geo = pool.map(subject_geometry, [(row, a.min_radius_mm) for _, row in tests.iterrows()])
        mean_centroid = np.mean([train_a[i] @ (cen - c) + train_t[i] for i, (c, cen, _) in enumerate(geo)], 0)
        held = {'learned': pool.map(subject_attributes, [(row, fitted['A'][k], fitted['t_mm'][k], populations, params) for k, (_, row) in enumerate(tests.iterrows())]),
                'centroid_translated': pool.map(subject_attributes, [(row, np.eye(3), mean_centroid - (cen - c), populations, params)
                                                                     for (_, row), (c, cen, _) in zip(tests.iterrows(), test_geo)])}
    n = len(train)
    selection = pd.DataFrame([r['selection'] for r in train])
    summary = {'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint['step'], 'subjects': n,
               'test_subject_ids': fitted['subject_ids'], 'settings': {k: v for k, v in vars(a).items()},
               'large_vessel_selection': {
                   'used': f'graph radius >= {a.min_radius_mm} mm at both edge endpoints (as recurrence and graph atlas)',
                   'compared_with': f'local radius -SDF >= {a.min_radius_mm} mm at the centreline point (training alignment target)',
                   'per_subject_median': selection.median(numeric_only=True).to_dict()},
               'scale_det_cuberoot': {'median': float(np.median([r['scale'] for r in train])),
                                      'p5_p95': [float(x) for x in np.percentile([r['scale'] for r in train], [5, 95])]},
               'populations': {}}
    q = list(range(min(a.train_queries, n)))
    for key, pop in populations.items():
        both, g, m = edge_atlas(train, key)
        stats = connectivity_stats(both, g, m)
        node_r = np.array([r['populations'][key]['node_radius'] for r in train])  # (subjects, nodes, 2)
        edge_r = np.array([r['populations'][key]['edge_radius'] for r in train])
        scales = np.array([r['scale'] for r in train])[:, None]
        # maps: per-edge values painted on the population edge paths (voxels within 1 mm)
        for name in ('p_both', 'graph_p_conn_given_both', 'graph_p_conn', 'mask_p_conn_given_both', 'mask_p_conn', 'mask_not_graph_given_both'):
            field = np.full(shape, np.nan, np.float32)
            for value, path in zip(stats[name], pop['edge_points_mm']):
                field[within_radius(path, shape, grid_affine, 1.0)] = value
            save(np.nan_to_num(field, nan=-1.), grid_affine, out / f'{key}_{name}.nii.gz')
        counts = {'both': both.sum(0), 'graph': g.sum(0), 'mask': m.sum(0)}
        evaluation = {}
        for frame_name, results in held.items():
            records = []
            for r in results:
                obs = r['populations'][key]
                records.append({'subject_id': r['subject_id'], 'connectivity': score_connectivity(obs, counts, n),
                                'radius_node_sdf': radius_errors(obs['node_radius'][:, 0], node_r[:, :, 0]),
                                'radius_node_sdf_scaled': radius_errors(obs['node_radius'][:, 0] * r['scale'], node_r[:, :, 0] * scales),
                                'radius_node_graph': radius_errors(obs['node_radius'][:, 1], node_r[:, :, 1])})
            evaluation[f'test/{frame_name}'] = records
        records = []
        for i in q:
            obs = train[i]['populations'][key]
            loo = {c: counts[c] - obs[c] for c in counts}
            others = np.delete(np.arange(n), i)
            records.append({'subject_id': train[i]['subject_id'], 'connectivity': score_connectivity(obs, loo, n - 1),
                            'radius_node_sdf': radius_errors(obs['node_radius'][:, 0], node_r[others, :, 0]),
                            'radius_node_sdf_scaled': radius_errors(obs['node_radius'][:, 0] * train[i]['scale'], node_r[others, :, 0] * scales[others]),
                            'radius_node_graph': radius_errors(obs['node_radius'][:, 1], node_r[others, :, 1])})
        evaluation['train_loo/learned'] = records

        def med(records, path):
            values = []
            for rec in records:
                v = rec
                for k in path:
                    v = v.get(k) if isinstance(v, dict) else None
                if v is not None:
                    values.append(v)
            return float(np.median(values)) if values else None
        medians = {name: {f'{c}_{m_}': med(recs, ['connectivity', c, m_]) for c in ('graph', 'mask')
                          for m_ in ('loglik_given_both', 'loglik_unconditional', 'coverage_connected_p_ge_0.5', 'brier_given_both', 'edges_both_present')}
                   | {f'{rk}_{m_}': med(recs, [rk, m_]) for rk in ('radius_node_sdf', 'radius_node_sdf_scaled', 'radius_node_graph')
                      for m_ in ('median_abs_error_mm', 'median_abs_z', 'median_percentile', 'fraction_percentile_5_95')}
                   for name, recs in evaluation.items()}
        to_list = lambda x: [None if (isinstance(v, float) and np.isnan(v)) else (float(v) if not isinstance(v, (int, np.integer)) else int(v)) for v in x]
        summary['populations'][key] = {
            'nodes': int(len(pop['nodes_mm'])), 'edges': int(len(pop['edges'])), 'radius_mm': float(pop['radius_mm']),
            'edge_connectivity': {k: to_list(v) for k, v in stats.items()},
            'edge_connectivity_median': {k: float(np.nanmedian(v)) if np.isfinite(v).any() else None for k, v in stats.items()},
            'node_radius_sdf_mm': radius_summary(node_r[:, :, 0]), 'node_radius_sdf_scaled_mm': radius_summary(node_r[:, :, 0] * scales),
            'node_radius_graph_mm': radius_summary(node_r[:, :, 1]), 'edge_radius_sdf_mm': radius_summary(edge_r[:, :, 0]),
            'edge_radius_graph_mm': radius_summary(edge_r[:, :, 1]),
            'evaluation_median': medians, 'evaluation': evaluation}
        print(json.dumps({'population': key, 'edge_connectivity_median': summary['populations'][key]['edge_connectivity_median'],
                          'evaluation_median': medians}), flush=True)
    # per-voxel radius maps: per-subject median per voxel, then median / IQR / count over subjects
    rows = [pd.DataFrame({'v': np.ravel_multi_index(v.T, shape), 'subject': i, 'sdf': x[:, 0], 'graph': x[:, 1],
                          'sdf_scaled': x[:, 0] * train[i]['scale']}) for i, r in enumerate(train) for v, x in [r['voxels']] if len(v)]
    table = pd.concat(rows).groupby(['v', 'subject']).median().reset_index()
    grouped = table.groupby('v')
    for column in ('sdf', 'sdf_scaled', 'graph'):
        for stat, fn in (('median', 'median'), ('q25', lambda s: s.quantile(.25)), ('q75', lambda s: s.quantile(.75))):
            field = np.zeros(int(np.prod(shape)), np.float32)
            values = grouped[column].agg(fn)
            field[values.index.values] = values.values
            save(field.reshape(shape), grid_affine, out / f'Radius_{column}_{stat}.nii.gz')
    count = np.zeros(int(np.prod(shape)), np.float32); sizes = grouped.size(); count[sizes.index.values] = sizes.values
    save(count.reshape(shape), grid_affine, out / 'Radius_subject_count.nii.gz')
    summary['voxel_radius_maps'] = {'voxels': int(len(sizes)), 'voxels_ge_10_subjects': int((sizes >= 10).sum())}
    summary['runtime_s'] = time.time() - start
    (out / 'vessel_attribute_atlas.json').write_text(json.dumps(summary, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


if __name__ == '__main__':
    main()
