"""Step F0: why do training subjects with large-vessel presence have seg_radius ~ 0 at the target edges? (CPU)

Inputs: a deviation run with --save-elements (deviation.json targets, elements_learned.npz: population geometry,
training seg_radius and large-vessel presence under the trained transforms), the checkpoint (trained transforms,
training metadata), the subjects' SDFs and graphs.
Per target edge, training subjects with seg_radius <= --max-seg-radius and large-vessel presence: the zero-valued
interior samples of the population path (vatl_sdf.deviation.interior_samples) are split into
  no_point  no all-radius subject graph point within r;
  clipped   nearest point within r, but -SDF <= 0 there (outside the vessel mask; clipped to 0 by seg_radius);
class = the majority cause (ties: no_point); other = no zero-valued sample (0 < value <= tolerance).
Separately: neighbour = no subject graph edge matched to the population edge by the F1 rule (presence then comes from
a vessel passing by); for matched subjects the F1 value (median -SDF over the matched segment interior).
Usage: python analysis/deviation/segment_zero_diagnostics.py DEVIATION_DIR --checkpoint CKPT --output JSON
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from vatl_sdf.deviation import interior_samples, matched_interior
from vatl_sdf.graph_coords import edge_samples, load_graph, voxel_to_world, world_to_voxel
from vatl_sdf.model import SDFModel


def zero_cause(d, raw, r):
    """(class, no_point zeros, clipped zeros) of the interior samples' nearest-point distances d and -SDF raw."""
    no_point = int((d > r).sum()); clipped = int(((d <= r) & (raw <= 0)).sum())
    if not no_point and not clipped:
        return 'other', 0, 0
    return ('no_point' if no_point >= clipped else 'clipped'), no_point, clipped


def subject_worker(args):
    from vatl_sdf.sources import load_image
    row, A, t, items, r = args
    image = load_image(row['SDF'])
    sdf = np.asarray(image.dataobj, dtype=np.float32)
    sdf = sdf[..., 0] if sdf.ndim == 4 else sdf
    centre = voxel_to_world(image.affine, np.asarray(sdf.shape, float) / 2.)
    graph = load_graph(row['Graph'])
    pts, eid = edge_samples(graph['pos'], graph['edges'], 0.4) if len(graph['edges']) else (np.zeros((0, 3)), np.zeros(0, int))
    template = (pts - centre) @ np.asarray(A).T + np.asarray(t)
    raw = -ndi.map_coordinates(sdf, world_to_voxel(image.affine, pts).T, order=1, mode='nearest') if len(pts) else np.zeros(0)
    tree = cKDTree(template) if len(pts) else None
    out = []
    for k, path, a, b in items:
        idx = interior_samples(path, a, b)
        d, j = tree.query(path[idx]) if tree is not None else (np.full(len(idx), np.inf), np.zeros(len(idx), int))
        cls, n_np, n_cl = zero_cause(d, raw[np.minimum(j, max(len(raw) - 1, 0))] if len(raw) else np.zeros(len(idx)), r)
        keep, match = matched_interior(template, eid, path, a, b, r)
        out.append({'subject_id': row['subject_id'], 'edge': int(k), 'class': cls, 'no_point_samples': n_np, 'clipped_samples': n_cl,
                    'interior_samples': int(len(idx)), 'matched_edges': int(len(match)),
                    'f1_value': float(np.median(raw[keep])) if len(keep) else None})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run'); p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=16); p.add_argument('--max-seg-radius', type=float, default=0.1)
    p.add_argument('--any-presence', action='store_true', help='test fixture: do not require large-vessel presence')
    a = p.parse_args()
    run = Path(a.run)
    summary = json.loads((run / 'deviation.json').read_text())
    pop_key = f"r{summary['settings']['radii_mm'][0]:g}mm_tau{summary['settings']['tau']:g}"
    z = np.load(run / 'elements_learned.npz', allow_pickle=False)
    g = lambda name: z[f'{pop_key}/{name}']
    off, paths = g('pop_path_offsets'), g('pop_paths')
    nodes, edges, r = g('pop_nodes_mm'), g('pop_edges'), float(g('radius_mm'))
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    frame = pd.DataFrame(checkpoint['metadata'])
    model = SDFModel(checkpoint['config'], len(frame)); model.load_state_dict(checkpoint['model'])
    train_a, train_t = (x.double().numpy() for x in model.maps_mm())
    if list(frame.subject_id) != list(z['train_ids']):
        raise ValueError('training subjects of the checkpoint and the dump differ')
    targets = {t_['population_edge']: t_['p_edge'] for t_ in summary['perturbations']['targets']}
    seg, pres = g('train_seg_radius'), g('train_presence').astype(bool)
    jobs = {}
    for k in targets:
        flagged = (seg[:, k] <= a.max_seg_radius) & (pres[:, k] | a.any_presence)
        for i in np.flatnonzero(flagged):
            jobs.setdefault(int(i), []).append((k, paths[off[k]:off[k + 1]], nodes[edges[k][0]], nodes[edges[k][1]]))
    tasks = [(frame.iloc[i].to_dict(), train_a[i], train_t[i], items, r) for i, items in sorted(jobs.items())]
    with Pool(a.workers) as pool:
        rows = [x for res in pool.map(subject_worker, tasks) for x in res]
    report = {'run': str(run), 'population': pop_key, 'radius_mm': r, 'max_seg_radius_mm': a.max_seg_radius,
              'require_presence': not a.any_presence, 'training_subjects': int(len(frame)), 'targets': {}}
    for k, p_edge in targets.items():
        mine = [x for x in rows if x['edge'] == k]
        f1 = [x['f1_value'] for x in mine if x['f1_value'] is not None]
        report['targets'][int(k)] = {
            'p_edge': p_edge, 'present_training_subjects': int(pres[:, k].sum()), 'flagged': len(mine),
            'classes': {c: sum(x['class'] == c for x in mine) for c in ('no_point', 'clipped', 'other')},
            'neighbour_no_matched_edge': sum(x['matched_edges'] == 0 for x in mine),
            'class_by_matched': {c: {'matched': sum(x['class'] == c and x['matched_edges'] > 0 for x in mine),
                                     'unmatched': sum(x['class'] == c and x['matched_edges'] == 0 for x in mine)}
                                 for c in ('no_point', 'clipped', 'other')},
            'f1_value_matched': {'n': len(f1), 'median': float(np.median(f1)) if f1 else None,
                                 'q10_q90': [float(np.quantile(f1, 0.1)), float(np.quantile(f1, 0.9))] if f1 else None}}
    report['subjects'] = rows
    Path(a.output).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'subjects'}, indent=1))


if __name__ == '__main__':
    main()
