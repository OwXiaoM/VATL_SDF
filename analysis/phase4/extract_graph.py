"""Phase 4 prototype: extract a centreline graph from predicted implicit fields and compare it with the
subject's reference graph (reference-relative metrics). Training subjects only (fitting).

Field-guided extraction (needs centerline_distance and tangent heads; junction optional):
  candidates = grid points with predicted SDF < 0 and predicted centreline distance < d_max;
  edges = 26-neighbour candidate pairs whose step direction is aligned with the predicted tangent at
          both ends (|cos| >= cos_min) -- tracking along the tangent field, expressed as a graph;
  junctions = connected clusters of candidates with junction heatmap >= 0.5.
Comparator (any checkpoint): 3D skeleton of the predicted mask (skimage), 26-connected.
Metrics vs the mask-supported reference graph (samples every 0.5 mm): precision/recall at 1 mm,
component counts (pieces with >= 5 points) relative to the reference graph's, junction counts.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from vatl_sdf.config import condition_vector
from vatl_sdf.dataset import graph_support
from vatl_sdf.graph_coords import edge_samples, load_graph, normalize_world, voxel_to_world
from vatl_sdf.model import SDFModel

OFFSETS = np.array([o for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]) - 1


@torch.no_grad()
def predict(model, coords, index, condition, chunk=200000):
    device = next(model.parameters()).device
    out = {'sdf': [], 'fields': {k: [] for k in model.field_heads}}
    for s in range(0, len(coords), chunk):
        u = torch.as_tensor(coords[s:s + chunk], device=device)
        ids = torch.full((len(u),), index, device=device, dtype=torch.long)
        cond = torch.as_tensor(condition, device=device).reshape(1, 2).expand(len(u), -1)
        sdf, _, fields = model(u, ids, cond, return_fields=True)
        out['sdf'].append(sdf.cpu().numpy())
        for k, v in fields.items():
            out['fields'][k].append(v.cpu().numpy())
    return np.concatenate(out['sdf'])[:, 0], {k: np.concatenate(v) for k, v in out['fields'].items()}


def components_of(points_index, shape, adjacency=None):
    """Connected components of grid points; adjacency=None means plain 26-connectivity."""
    grid = np.zeros(shape, bool); grid[tuple(points_index.T)] = True
    if adjacency is None:
        labels, n = ndi.label(grid, np.ones((3, 3, 3), bool))
        return labels[tuple(points_index.T)], n
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    m = len(points_index)
    graph = sp.coo_matrix((np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])), shape=(m, m))
    n, labels = connected_components(graph, directed=False)
    return labels, n


def field_graph(sdf, fields, shape, affine, d_max, cos_min):
    distance = fields['centerline_distance'][:, 0] * 5.0
    candidate = (sdf < 0) & (distance < d_max)
    index = np.argwhere(candidate.reshape(shape))
    lookup = -np.ones(shape, np.int64); lookup[tuple(index.T)] = np.arange(len(index))
    tangent = fields['tangent'][np.ravel_multi_index(index.T, shape)]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    edges = []
    linear = affine[:3, :3]
    for o in OFFSETS[:13]:  # half the neighbourhood; pairs are undirected
        n = index + o
        ok = ((n >= 0) & (n < np.array(shape))).all(1)
        j = np.full(len(index), -1); j[ok] = lookup[tuple(n[ok].T)]
        has = j >= 0
        step = linear @ o; step = step / np.linalg.norm(step)
        aligned = (np.abs(tangent[has] @ step) >= cos_min) & (np.abs(tangent[j[has]] @ step) >= cos_min)
        edges.append(np.c_[np.flatnonzero(has)[aligned], j[has][aligned]])
    edges = np.concatenate(edges) if edges else np.zeros((0, 2), int)
    junction = None
    if 'junction' in fields:
        heat = 1 / (1 + np.exp(-fields['junction'][np.ravel_multi_index(index.T, shape), 0]))
        junction = int(components_of(index[heat >= 0.5], shape)[1]) if (heat >= 0.5).any() else 0
    return index, edges, junction


def score(points_world, labels, reference_points, reference_labels, min_points=5):
    tree_ref, tree_pred = cKDTree(reference_points), cKDTree(points_world)
    precision = float((tree_ref.query(points_world)[0] <= 1.0).mean()) if len(points_world) else 0.
    recall = float((tree_pred.query(reference_points)[0] <= 1.0).mean()) if len(points_world) else 0.
    count = lambda lab: int((np.bincount(lab) >= min_points).sum()) if len(lab) else 0
    pred_c, ref_c = count(labels), count(reference_labels)
    return {'precision_1mm': precision, 'recall_1mm': recall, 'components_ge5': pred_c,
            'reference_components_ge5': ref_c, 'component_ratio': pred_c / max(ref_c, 1), 'points': len(points_world)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--subjects', nargs='*'); p.add_argument('--device', default='cuda')
    p.add_argument('--d-max', type=float, default=0.6); p.add_argument('--cos-min', type=float, default=0.5)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = ck['config'], pd.DataFrame(ck['metadata'])
    model = SDFModel(config, len(frame)).to(a.device); model.load_state_dict(ck['model']); model.eval()
    from analysis.phase0_graph.diagnose_checkpoint import select_subjects
    results = []
    for subject in a.subjects or select_subjects(frame):
        index = int(frame.index[frame.subject_id == subject][0]); row = frame.iloc[index]
        image = nib.load(row.SDF); reference = image.get_fdata(dtype=np.float32); shape = reference.shape
        voxels = np.indices(shape).reshape(3, -1).T
        coords = normalize_world(voxel_to_world(image.affine, voxels), image.affine, shape,
                                 config['data']['world_bbox_mm']).astype(np.float32)
        sdf, fields = predict(model, coords, index, condition_vector(config, row.scan_age, row.sex))
        sdf = sdf * config['sdf']['truncation_mm']
        g = load_graph(row.Graph); g.update(graph_support(reference < 0, image.affine, g, config.get('graph') or
                                                         {'support_tolerance_mm': .4, 'support_strict': False, 'edge_step_mm': .4}))
        edges = g['edges'][g['edge_supported']]
        ref_points, owner = edge_samples(g['snapped'], edges, step_mm=0.5)
        import networkx as nx
        sub = nx.Graph(); sub.add_edges_from(edges.tolist())
        piece = {n: k for k, comp in enumerate(nx.connected_components(sub)) for n in comp}
        ref_labels = np.array([piece[edges[o, 0]] for o in owner])
        ref_junctions = int((np.bincount(edges.ravel()) >= 3).sum())
        record = {'subject_id': subject, 'reference_junctions': ref_junctions}
        skeleton = np.argwhere(skeletonize((sdf < 0).reshape(shape)))
        labels, _ = components_of(skeleton, shape)
        record['skeleton'] = score(voxel_to_world(image.affine, skeleton), labels, ref_points, ref_labels)
        if {'centerline_distance', 'tangent'} <= set(fields):
            index_f, adjacency, junctions = field_graph(sdf, fields, shape, image.affine, a.d_max, a.cos_min)
            labels, _ = components_of(index_f, shape, adjacency)
            record['field_graph'] = {**score(voxel_to_world(image.affine, index_f), labels, ref_points, ref_labels),
                                     'junction_clusters': junctions}
        results.append(record)
        print(json.dumps(record), flush=True)
        (out / 'extraction.json').write_text(json.dumps({'checkpoint': a.checkpoint, 'subjects': results}, indent=1))


if __name__ == '__main__':
    main()
