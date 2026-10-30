"""Read-only inspection of per-subject vessel graphs against the reference SDF.

Graph pickles are dicts with keys graph (networkx.Graph; node attrs pos [world mm],
radius [mm]; edge attr weight), node_radius_map, voxel_sizes, input_file.
Nothing is written except the output TSV/JSON.
"""
import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

GRAPH_TSV = '/medias/medical/data/Atlas_training/v1/ixi_metadata_label_t1_fast_graph.tsv'
SDF_TSV = '/medias/medical/data/Atlas_training/v1/ixi_metadata_label_t1_fast_sdf.tsv'
RADIUS_BINS = [0, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, np.inf]


def world_to_voxel(affine, points):
    return (np.linalg.inv(affine) @ np.c_[points, np.ones(len(points))].T)[:3]


def inspect(row):
    payload = pickle.load(open(row.Graph, 'rb'))
    graph = payload['graph']
    nodes = list(graph.nodes)
    pos = np.array([graph.nodes[n]['pos'] for n in nodes], dtype=np.float64)
    radius = np.array([graph.nodes[n]['radius'] for n in nodes], dtype=np.float64)
    radius_map = payload['node_radius_map']
    image = nib.load(row.SDF)
    sdf = image.get_fdata(dtype=np.float32)
    mask = nib.load(row.Seg).get_fdata() == 1
    vox = world_to_voxel(image.affine, pos)
    shape = np.array(sdf.shape)[:, None]
    in_fov = ((vox >= 0) & (vox <= shape - 1)).all(0)
    sdf_at = map_coordinates(sdf, vox, order=1, mode='nearest')
    in_mask = map_coordinates(mask.astype(np.float32), vox, order=0, mode='constant') > 0
    residual = sdf_at + radius  # zero if SDF(c) = -r(c)
    lengths = np.array([np.linalg.norm(graph.nodes[a]['pos'] - graph.nodes[b]['pos']) for a, b in graph.edges])
    weights = np.array([d.get('weight', np.nan) for *_, d in graph.edges(data=True)], dtype=np.float64)
    components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    degree = Counter(dict(graph.degree()).values())
    # Mask coverage: mask voxel centres farther than (local radius + 1 mm) from every node.
    mask_world = (image.affine @ np.c_[np.argwhere(mask), np.ones(mask.sum())].T)[:3].T
    dist, idx = cKDTree(pos).query(mask_world)
    uncovered = dist > radius[idx] + 1.0
    hist = np.histogram(radius, RADIUS_BINS)[0]
    return {
        'subject_id': row.subject_id,
        'input_file_matches': Path(payload['input_file']).name.startswith(row.subject_id),
        'voxel_sizes': ','.join(f'{v:.4f}' for v in payload['voxel_sizes']),
        'radius_map_consistent': all(np.isclose(radius_map[n], graph.nodes[n]['radius']) for n in nodes),
        'nodes': len(nodes), 'edges': graph.number_of_edges(), 'self_loops': nx.number_of_selfloops(graph),
        'cycles_rank': graph.number_of_edges() - len(nodes) + len(components),
        'components': len(components), 'largest_component_frac': components[0] / len(nodes),
        'deg1': degree.get(1, 0), 'deg2': degree.get(2, 0), 'deg3plus': sum(v for k, v in degree.items() if k >= 3),
        'contraction_nodes': sum('contraction' in graph.nodes[n] for n in nodes),
        'edge_len_median': float(np.median(lengths)), 'edge_len_max': float(lengths.max()),
        'edge_weight_vs_len_maxdiff': float(np.nanmax(np.abs(weights - lengths))),
        'radius_min': radius.min(), 'radius_median': float(np.median(radius)), 'radius_max': radius.max(),
        'radius_frac_lt_0p5': float((radius < 0.5).mean()),
        **{f'radius_hist_{lo}_{hi}': int(h) for lo, hi, h in zip(RADIUS_BINS[:-1], RADIUS_BINS[1:], hist)},
        'radius_unique_values': len(np.unique(np.round(radius, 4))),
        'frac_in_fov': float(in_fov.mean()), 'frac_in_mask': float(in_mask.mean()),
        'sdf_at_node_median': float(np.median(sdf_at)), 'frac_sdf_negative': float((sdf_at < 0).mean()),
        'residual_median': float(np.median(residual)), 'residual_mae': float(np.abs(residual).mean()),
        'corr_negsdf_radius': float(np.corrcoef(-sdf_at, radius)[0, 1]),
        'frac_nodes_sdf_gt_1mm': float((sdf_at > 1.0).mean()),
        'mask_voxels': int(mask.sum()), 'mask_frac_uncovered': float(uncovered.mean()),
        'sdf_neg_equals_mask': bool(((sdf < 0) == mask).all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    graphs = pd.read_csv(GRAPH_TSV, sep='\t', dtype={'subject_id': str})
    sdf = pd.read_csv(SDF_TSV, sep='\t', dtype={'subject_id': str})[['subject_id', 'SDF']]
    frame = graphs.merge(sdf, on='subject_id', how='outer', indicator=True)
    unmatched = frame[frame._merge != 'both'].subject_id.tolist()
    frame = frame[frame._merge == 'both']
    if args.limit:
        frame = frame.head(args.limit)
    rows = []
    for row in frame.itertuples():
        rows.append(inspect(row))
        print(json.dumps({k: rows[-1][k] for k in ('subject_id', 'nodes', 'frac_in_mask', 'residual_median')}), flush=True)
    table = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'graph_inspection.tsv', sep='\t', index=False)
    numeric = table.select_dtypes('number')
    summary = {'subjects': len(table), 'unmatched_ids': unmatched,
               'all_input_file_match': bool(table.input_file_matches.all()),
               'all_radius_map_consistent': bool(table.radius_map_consistent.all()),
               'all_sdf_neg_equals_mask': bool(table.sdf_neg_equals_mask.all()),
               'voxel_sizes': table.voxel_sizes.value_counts().to_dict(),
               'quantiles': numeric.quantile([0, 0.05, 0.5, 0.95, 1]).to_dict()}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
