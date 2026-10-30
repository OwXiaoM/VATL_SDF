"""Phase 5: how much reference geometry does the (imperfect, quantized) graph capture?

For each subject, render the graph as a union of round cones on the native SDF grid and compare with
the reference SDF: zero-level Dice, band MAE (|reference| <= 5 mm, both clamped at 5 mm), and
connectivity relative to the reference mask. Variants: radius = stored graph radius (quantized, floor
0.8 mm) or radius = -reference SDF at the node (floored at 0.3 mm); hard min or smooth min (k = 0.3 mm).
All edges are rendered (the graph as given), so missing connections and spurious loops show up.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from vatl_sdf.evaluate import dice, reference_relative_connectivity
from vatl_sdf.graph_coords import load_graph, voxel_to_world, world_to_voxel
from vatl_sdf.render_graph import render

TSV = '/medias/medical/data/Atlas_training/v1/ixi_metadata_label_t1_fast_sdf.tsv'
TAU = 5.0


def compare(subject_id, row, out):
    image = nib.load(row.SDF)
    reference = image.get_fdata(dtype=np.float32)
    shape = reference.shape
    world = voxel_to_world(image.affine, np.indices(shape).reshape(3, -1).T)
    g = load_graph(row.Graph)
    sdf_radius = np.maximum(-map_coordinates(reference, world_to_voxel(image.affine, g['pos']).T, order=1, mode='nearest'), 0.3)
    ref_mask = reference < 0
    band = np.abs(reference) <= TAU
    rows = []
    for radius_name, radius in (('graph_radius', g['radius']), ('sdf_radius', sdf_radius)):
        for k in (0.0, 0.3):
            field = render(world, shape, g['pos'], radius, g['edges'], smooth_k_mm=k, far_mm=TAU)
            mask = field < 0
            rows.append({'subject_id': subject_id, 'radius': radius_name, 'smooth_k_mm': k,
                         'dice': dice(mask, ref_mask), 'volume_ratio': float(mask.sum() / ref_mask.sum()),
                         'band_mae_mm': float(np.abs(np.clip(field[band], -TAU, TAU) - np.clip(reference[band], -TAU, TAU)).mean()),
                         'recall': float((mask & ref_mask).sum() / ref_mask.sum()),
                         'precision': float((mask & ref_mask).sum() / max(mask.sum(), 1)),
                         **reference_relative_connectivity(mask, ref_mask)})
            if radius_name == 'graph_radius' and k == 0.0:
                image_out = nib.Nifti1Image(field.astype(np.float32), image.affine)
                nib.save(image_out, out / f'{subject_id}_rendered_graph_radius_sdf_mm.nii.gz')
    return rows


def _run(task):
    return compare(*task)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True); p.add_argument('--subjects', type=int, default=20)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(TSV, sep='\t', dtype={'subject_id': str}).set_index('subject_id')
    ids = frame.index[np.linspace(0, len(frame) - 1, a.subjects).astype(int)]
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.map(_run, [(s, frame.loc[s], out) for s in ids]) for r in rs]
    table = pd.DataFrame(rows)
    table.to_csv(out / 'render_compare.tsv', sep='\t', index=False)
    summary = table.groupby(['radius', 'smooth_k_mm']).median(numeric_only=True)
    (out / 'summary.json').write_text(summary.to_json(orient='index', indent=1))
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
