"""Alignment Step 0 (data only, CPU): how far apart are subjects before any learned alignment?

Per training subject: vessel-mask (SDF < 0, identical to label 1) centroid in world mm and relative to the
image bbox centre (the frame the model normalizes by), principal axes and their angles to the population
mean axes, principal extents (sqrt eigenvalues, mm) and vessel volume. Large-vessel recurrence
(graph radius >= 1.2 mm; vatl_sdf.alignment_metrics) in two frames: bbox-centred (what the model sees with
identity transforms) and centroid-translated (each subject shifted so its vessel centroid is at 0).
Reported for all training subjects and for the first 100 (the pilot subset).
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from vatl_sdf.alignment_metrics import (axis_angles_deg, large_vessel_centerline, mean_axes, principal_axes,
                                        recurrence, summarize_recurrence)
from vatl_sdf.config import load_config
from vatl_sdf.dataset import read_metadata
from vatl_sdf.graph_coords import load_graph, voxel_to_world


def measure(args):
    row, min_radius = args
    image = nib.load(row['SDF'])
    sdf = np.asarray(image.dataobj, dtype=np.float32)
    if sdf.ndim == 4:
        sdf = sdf[..., 0]
    voxels = np.argwhere(sdf < 0)
    world = voxel_to_world(image.affine, voxels)
    centre = voxel_to_world(image.affine, np.asarray(sdf.shape) / 2.)
    values, vectors = principal_axes(world)
    spacing = nib.affines.voxel_sizes(image.affine)
    lines = large_vessel_centerline(load_graph(row['Graph']), min_radius)
    return {'subject_id': row['subject_id'], 'centroid': world.mean(0), 'centre': centre, 'values': values,
            'vectors': vectors, 'volume_ml': len(voxels) * float(np.prod(spacing)) / 1000., 'lines': lines}


def recurrence_block(results, subset):
    out = {}
    for frame in ('bbox_centred', 'centroid_translated'):
        clouds = [r['lines'] - (r['centre'] if frame == 'bbox_centred' else r['centroid']) for r in subset]
        fractions, _ = recurrence(clouds)
        out[frame] = summarize_recurrence(fractions)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/sdf/00_sdf_baseline.yaml'); p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=16); p.add_argument('--min-radius-mm', type=float, default=1.2)
    p.add_argument('--max-subjects', type=int)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    frame = read_metadata(load_config(a.config))
    if a.max_subjects:
        frame = frame.head(a.max_subjects)
    with Pool(a.workers) as pool:
        results = pool.map(measure, [(row, a.min_radius_mm) for _, row in frame.iterrows()])
    reference = mean_axes([r['vectors'] for r in results])
    rows = []
    for r in results:
        angles = axis_angles_deg(r['vectors'], reference)
        offset = r['centroid'] - r['centre']
        rows.append({'subject_id': r['subject_id'], **{f'centroid_{k}': r['centroid'][i] for i, k in enumerate('xyz')},
                     **{f'offset_{k}': offset[i] for i, k in enumerate('xyz')},
                     **{f'axis{i + 1}_angle_deg': angles[i] for i in range(3)},
                     **{f'extent{i + 1}_mm': float(np.sqrt(r['values'][i])) for i in range(3)},
                     'volume_ml': r['volume_ml'], 'large_vessel_points': len(r['lines'])})
    table = pd.DataFrame(rows)
    table.to_csv(out / 'mask_geometry.tsv', sep='\t', index=False)

    def spread(columns):
        return {c: {'mean': float(table[c].mean()), 'std': float(table[c].std()), 'p5': float(table[c].quantile(.05)),
                    'p95': float(table[c].quantile(.95))} for c in columns}
    offsets = table[[f'offset_{k}' for k in 'xyz']].to_numpy()
    deviation = np.linalg.norm(offsets - offsets.mean(0), axis=1)
    summary = {
        'subjects': len(table), 'min_radius_mm': a.min_radius_mm,
        'centroid_offset_from_bbox_centre_mm': spread([f'offset_{k}' for k in 'xyz']),
        'centroid_offset_deviation_from_mean_mm': {'median': float(np.median(deviation)),
                                                   'p95': float(np.percentile(deviation, 95)), 'max': float(deviation.max())},
        'centroid_world_mm': spread([f'centroid_{k}' for k in 'xyz']),
        'principal_axis_angle_to_mean_deg': spread([f'axis{i}_angle_deg' for i in (1, 2, 3)]),
        'principal_extent_mm': spread([f'extent{i}_mm' for i in (1, 2, 3)]),
        'extent_cv': {f'extent{i}': float(table[f'extent{i}_mm'].std() / table[f'extent{i}_mm'].mean()) for i in (1, 2, 3)},
        'volume_ml': spread(['volume_ml']),
        'recurrence_all': recurrence_block(results, results),
        'recurrence_first100': recurrence_block(results, results[:100]),
        'note': 'Centroids depend on the vessel content of each field of view; axes are sign-invariant.'}
    (out / 'step0_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
