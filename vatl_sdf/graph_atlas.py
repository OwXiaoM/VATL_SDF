"""Probabilistic vessel-graph atlas, step (1) (inference only; training objective unchanged).

Each training subject's large-vessel centreline (graph edges with both endpoint radii >= min_radius_mm, densified;
vatl_sdf.alignment_metrics) is mapped to the template by its trained transform (template_mm = A_i (world -
centre_i) + t_i). For each radius r:
  centreline probability  c_r(x) = fraction of subjects with centreline within r of x (template grid);
  population graph        skeleton of {c_r >= tau}, converted to nodes (junction clusters / end points) and edges
                          (voxel paths); edge occurrence p_e = fraction of subjects whose centreline lies within r
                          of at least `edge_cover` of the edge's points.
Evaluation of held-out subjects (test, mapped by their test-time fitted transforms; centroid-translation-only
control) and of training subjects (leave-one-out probabilities p_e^(-i) = (N p_e - y_ie) / (N - 1), so a training
subject never scores against its own contribution):
  coverage    fraction of the subject's centreline points within r of the population skeleton (all edges, and edges
              with p_e >= 0.5), and fraction at which c_r >= 0.1 / 0.25 / 0.5;
  likelihood  per-edge presence y_e (same rule as p_e): Bernoulli log-likelihood with (N p + 1) / (N + 2)
              smoothing, Brier, both against the constant mean(p_e); probability-weighted edge recall.
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
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .graph_coords import world_to_voxel
from .inference import world_grid
from .model import SDFModel
from .population_maps import save
from .probabilistic_atlas import within_radius

OFFSETS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)])


def skeleton_graph(skeleton):
    """Nodes and edges of a 3D skeleton (26-connectivity). Node = 26-connected cluster of voxels whose degree is not
    2 (end points, junctions); edge = path of degree-2 voxels between two nodes (possibly empty). Cycles without any
    node get one of their voxels promoted to a node. Returns (node voxel positions (n, 3) float, edges as
    [(a, b, voxel path (k, 3) including the end-node positions)])."""
    voxels = np.argwhere(skeleton)
    if not len(voxels):
        return np.zeros((0, 3)), []
    index = {tuple(v): i for i, v in enumerate(voxels)}
    neighbours = [[index[t] for t in map(tuple, v + OFFSETS) if t in index] for v in voxels]
    degree = np.array([len(n) for n in neighbours])
    is_node = degree != 2
    node_of = -np.ones(len(voxels), int)
    positions = []

    def label_clusters():
        for i in np.flatnonzero(is_node & (node_of < 0)):
            if node_of[i] >= 0:  # absorbed by an earlier cluster in this pass
                continue
            stack, members = [i], []
            node_of[i] = len(positions)
            while stack:
                v = stack.pop(); members.append(v)
                for u in neighbours[v]:
                    if is_node[u] and node_of[u] < 0:
                        node_of[u] = node_of[i]; stack.append(u)
            positions.append(voxels[members].mean(0))

    label_clusters()
    visited = np.zeros(len(voxels), bool)
    edges, seen_direct = [], set()

    def trace_from(start_node_voxel, first):
        path, prev, cur = [first], start_node_voxel, first
        visited[first] = True
        while True:
            nxt = [u for u in neighbours[cur] if u != prev and (is_node[u] or not visited[u])]
            if not nxt:  # dead end inside a cycle already traced
                return None
            node_next = [u for u in nxt if is_node[u]]
            step = node_next[0] if node_next else nxt[0]
            if is_node[step]:
                return step, path
            visited[step] = True; path.append(step); prev, cur = cur, step

    def trace_all():
        for v in np.flatnonzero(is_node):
            for u in neighbours[v]:
                if is_node[u]:
                    a, b = node_of[v], node_of[u]
                    if a != b and (min(a, b), max(a, b)) not in seen_direct:
                        seen_direct.add((min(a, b), max(a, b)))
                        edges.append((a, b, np.vstack([positions[a], positions[b]])))
                elif not visited[u]:
                    result = trace_from(v, u)
                    if result is not None:
                        end, path = result
                        a, b = node_of[v], node_of[end]
                        edges.append((a, b, np.vstack([positions[a], voxels[path], positions[b]])))

    trace_all()
    while (~visited & ~is_node).any():  # node-free cycles
        i = np.flatnonzero(~visited & ~is_node)[0]
        is_node[i] = True; label_clusters(); trace_all()
    return np.asarray(positions), edges


def edge_points_mm(edges, affine):
    return [nib.affines.apply_affine(affine, path) for _, _, path in edges]


def presence(trees, edge_points, radius_mm, edge_cover):
    """(subjects, edges) bool: subject's centreline within radius of >= edge_cover of the edge points."""
    out = np.zeros((len(trees), len(edge_points)), bool)
    if not edge_points:
        return out
    flat = np.vstack(edge_points); owner = np.repeat(np.arange(len(edge_points)), [len(p) for p in edge_points])
    counts = np.bincount(owner, minlength=len(edge_points))
    for i, tree in enumerate(trees):
        if tree is None:
            continue
        near = tree.query(flat, distance_upper_bound=radius_mm)[0] <= radius_mm
        out[i] = np.bincount(owner, weights=near, minlength=len(edge_points)) >= edge_cover * counts
    return out


def edge_scores(p, y, n):
    """Bernoulli log-likelihood (smoothed), Brier and recall of presence y under edge probabilities p (n subjects)."""
    if not len(p):
        return None
    s = (n * p + 1.) / (n + 2.); c = (n * p.mean() + 1.) / (n + 2.)
    ll = y * np.log(s) + (1 - y) * np.log1p(-s); llc = y * np.log(c) + (1 - y) * np.log1p(-c)
    brier, brier_c = np.mean((p - y) ** 2), np.mean((p.mean() - y) ** 2)
    return {'edges': int(len(p)), 'present': int(y.sum()), 'loglik_per_edge': float(ll.mean()),
            'loglik_constant_per_edge': float(llc.mean()), 'brier': float(brier), 'brier_constant': float(brier_c),
            'brier_skill': float(1 - brier / brier_c) if brier_c > 0 else None,
            'weighted_recall': float((p * y).sum() / p.sum()) if p.sum() > 0 else None}


def subject_scores(points, skeleton_tree, high_tree, cmap, affine, radius_mm, p, y, n):
    record = {'points': int(len(points))}
    if len(points):
        d_all = skeleton_tree.query(points)[0] if skeleton_tree is not None else np.full(len(points), np.inf)
        d_high = high_tree.query(points)[0] if high_tree is not None else np.full(len(points), np.inf)
        values = ndi.map_coordinates(cmap, world_to_voxel(affine, points).T, order=1, mode='constant', cval=0.)
        record.update({'coverage_skeleton': float((d_all <= radius_mm).mean()), 'coverage_skeleton_p_ge_0.5': float((d_high <= radius_mm).mean()),
                       **{f'map_ge_{t:g}': float((values >= t).mean()) for t in (0.1, 0.25, 0.5)}})
    record['edges'] = edge_scores(p, y.astype(float), n)
    return record


def aggregate(records):
    keys = ['coverage_skeleton', 'coverage_skeleton_p_ge_0.5', 'map_ge_0.1', 'map_ge_0.25', 'map_ge_0.5']
    out = {k: float(np.median([r[k] for r in records if k in r])) for k in keys}
    for k in ('loglik_per_edge', 'loglik_constant_per_edge', 'brier', 'brier_skill', 'weighted_recall'):
        values = [r['edges'][k] for r in records if r['edges'] and r['edges'][k] is not None]
        out[f'edge_{k}'] = float(np.median(values)) if values else None
    out['subjects'] = len(records)
    return out


def held_out_rows(config, ids):
    rows = pd.read_csv(config['data']['tsv'], sep='\t', dtype={'subject_id': str}).set_index('subject_id', drop=False)
    parent = Path(config['data']['tsv']).resolve().parent
    rows = rows.loc[ids].reset_index(drop=True)
    for key in ('SDF', 'Graph'):
        rows[key] = rows[key].map(lambda path: str((parent / path).resolve()))
    return rows


def main():
    from analysis.alignment.evaluate_alignment import subject_geometry  # CPU helper shared with Step 0/2
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--fitted', required=True, help='fitted_test.pt of the generalization run on this checkpoint')
    p.add_argument('--radii-mm', type=float, nargs='+', default=[2.0, 3.0])
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.1, 0.25])
    p.add_argument('--edge-cover', type=float, default=0.5); p.add_argument('--min-skeleton-voxels', type=int, default=5)
    p.add_argument('--spacing', type=float, default=1.0); p.add_argument('--train-queries', type=int, default=100)
    p.add_argument('--min-radius-mm', type=float, default=1.2); p.add_argument('--workers', type=int, default=16)
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
    with Pool(a.workers) as pool:
        geo = pool.map(subject_geometry, [(row, a.min_radius_mm) for _, row in frame.iterrows()])
        test_geo = pool.map(subject_geometry, [(row, a.min_radius_mm) for _, row in tests.iterrows()])
    n = len(frame)
    learned = [(lines - c) @ train_a[i].T + train_t[i] for i, (c, _, lines) in enumerate(geo)]
    mean_centroid = np.mean([train_a[i] @ (cen - c) + train_t[i] for i, (c, cen, _) in enumerate(geo)], 0)
    centroid_frame = lambda g: [lines - cen + mean_centroid for _, cen, lines in g]
    train_trees = [cKDTree(x) if len(x) else None for x in learned]
    train_centroid_trees = [cKDTree(x) if len(x) else None for x in centroid_frame(geo)]
    test_clouds = {'learned': [(lines - c) @ fitted['A'][k].T + fitted['t_mm'][k] for k, (c, _, lines) in enumerate(test_geo)],
                   'centroid_translated': centroid_frame(test_geo)}
    _, shape, affine = world_grid(config, [a.spacing] * 3)
    summary = {'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint['step'], 'subjects': n,
               'test_subject_ids': fitted['subject_ids'], 'settings': {k: v for k, v in vars(a).items()},
               'train_query_note': f'first {a.train_queries} training subjects, leave-one-out edge probabilities', 'radii': {}}
    q = list(range(min(a.train_queries, n)))
    for r in a.radii_mm:
        cmap = np.zeros(shape)
        for cloud in learned:
            cmap += within_radius(cloud, shape, affine, r)
        cmap /= n
        save(cmap, affine, out / f'CentrelineProbability_r{r:g}mm.nii.gz')
        per_r = {'map_max': float(cmap.max()), 'thresholds': {}}
        for tau in a.thresholds:
            mask = cmap >= tau
            labels, count = ndi.label(mask, np.ones((3, 3, 3), bool))
            sizes = np.bincount(labels.ravel())
            skeleton = skeletonize(mask) > 0
            skel_labels, _ = ndi.label(skeleton, np.ones((3, 3, 3), bool))
            skel_sizes = np.bincount(skel_labels.ravel())
            skeleton &= skel_sizes[skel_labels] >= a.min_skeleton_voxels
            nodes, edges = skeleton_graph(skeleton)
            points = edge_points_mm(edges, affine)
            y_train = presence(train_trees, points, r, a.edge_cover)
            p_edge = y_train.mean(0) if len(edges) else np.zeros(0)
            nodes_mm = nib.affines.apply_affine(affine, nodes) if len(nodes) else nodes
            lengths = [float(np.linalg.norm(np.diff(pt, axis=0), axis=1).sum()) for pt in points]
            tag = f'r{r:g}mm_tau{tau:g}'
            save(skeleton.astype(np.float32), affine, out / f'PopulationSkeleton_{tag}.nii.gz')
            with open(out / f'population_graph_{tag}.pkl', 'wb') as stream:
                pickle.dump({'nodes_mm': nodes_mm, 'edges': [(int(a_), int(b_)) for a_, b_, _ in edges], 'edge_points_mm': points,
                             'edge_probability': p_edge, 'edge_length_mm': lengths, 'radius_mm': r, 'tau': tau,
                             'edge_cover': a.edge_cover, 'subjects': n}, stream)
            skeleton_pts = np.vstack(points) if points else np.zeros((0, 3))
            high_pts = np.vstack([pt for pt, pe in zip(points, p_edge) if pe >= 0.5]) if (p_edge >= 0.5).any() else np.zeros((0, 3))
            sk_tree = cKDTree(skeleton_pts) if len(skeleton_pts) else None
            hi_tree = cKDTree(high_pts) if len(high_pts) else None
            evaluation = {}
            for frame_name, clouds in test_clouds.items():
                trees = [cKDTree(x) if len(x) else None for x in clouds]
                y_test = presence(trees, points, r, a.edge_cover)
                records = [dict(subject_id=s, **subject_scores(c, sk_tree, hi_tree, cmap, affine, r, p_edge, y_test[k], n))
                           for k, (s, c) in enumerate(zip(tests.subject_id, clouds))]
                evaluation[f'test/{frame_name}'] = {'median': aggregate(records), 'subjects': records}
            for frame_name, (clouds, trees) in {'learned': (learned, train_trees),
                                                'centroid_translated': (centroid_frame(geo), train_centroid_trees)}.items():
                y_q = presence([trees[i] for i in q], points, r, a.edge_cover)
                records = []
                for k, i in enumerate(q):
                    loo = (n * p_edge - y_train[i]) / (n - 1) if len(edges) else p_edge
                    # leave-one-out map: remove the subject's own learned-frame contribution
                    own = within_radius(learned[i], shape, affine, r)
                    loo_map = (cmap * n - own) / (n - 1)
                    loo_pts = np.vstack([pt for pt, pe in zip(points, loo) if pe >= 0.5]) if (loo >= 0.5).any() else np.zeros((0, 3))
                    records.append(dict(subject_id=frame.subject_id.iloc[i], **subject_scores(
                        clouds[i], sk_tree, cKDTree(loo_pts) if len(loo_pts) else None, loo_map, affine, r, loo, y_q[k], n - 1)))
                evaluation[f'train_loo/{frame_name}'] = {'median': aggregate(records), 'subjects': records}
            per_r['thresholds'][f'{tau:g}'] = {
                'mask': {'voxels': int(mask.sum()), 'components_26': int(count), 'largest_fraction': float(sizes[1:].max() / sizes[1:].sum()) if count else 0.},
                'skeleton_voxels': int(skeleton.sum()), 'nodes': int(len(nodes)), 'edges': int(len(edges)),
                'total_length_mm': float(sum(lengths)),
                'edge_probability': {'median': float(np.median(p_edge)) if len(p_edge) else None,
                                     'fraction_ge_0.5': float((p_edge >= 0.5).mean()) if len(p_edge) else None},
                'evaluation': evaluation}
            print(json.dumps({'radius': r, 'tau': tau, 'edges': len(edges), **{k: v['median'] for k, v in evaluation.items()}}), flush=True)
        summary['radii'][f'{r:g}'] = per_r
        (out / 'graph_atlas.json').write_text(json.dumps(summary, indent=1))
    summary['runtime_s'] = time.time() - start
    (out / 'graph_atlas.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
