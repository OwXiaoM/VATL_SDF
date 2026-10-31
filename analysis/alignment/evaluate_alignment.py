"""Alignment of a trained checkpoint: learned transform statistics and large-vessel recurrence (graph radius
>= 1.2 mm; vatl_sdf.alignment_metrics) in three frames for the checkpoint's training subjects:
bbox-centred (identity transforms), centroid-translated (Step 0 reference) and learned
(template_mm = A_i (world - image centre_i) + t_i from SDFModel.maps_mm).
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from vatl_sdf.alignment_metrics import large_vessel_centerline, recurrence, summarize_recurrence, transform_statistics
from vatl_sdf.graph_coords import load_graph, voxel_to_world
from vatl_sdf.model import SDFModel


def subject_geometry(args):
    row, min_radius = args
    image = nib.load(row['SDF'])
    sdf = np.asarray(image.dataobj, dtype=np.float32)
    if sdf.ndim == 4:
        sdf = sdf[..., 0]
    centroid = voxel_to_world(image.affine, np.argwhere(sdf < 0)).mean(0)
    centre = voxel_to_world(image.affine, np.asarray(sdf.shape) / 2.)
    return centre, centroid, large_vessel_centerline(load_graph(row['Graph']), min_radius)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--max-subjects', type=int, default=100); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--min-radius-mm', type=float, default=1.2)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    model = SDFModel(config, len(frame)); model.load_state_dict(checkpoint['model'])
    matrices, translations = (x.double().numpy() for x in model.maps_mm())
    stats_all = transform_statistics(matrices, translations)
    frame = frame.head(a.max_subjects)
    with Pool(a.workers) as pool:
        geometry = pool.map(subject_geometry, [(row, a.min_radius_mm) for _, row in frame.iterrows()])
    clouds = {'bbox_centred': [lines - centre for centre, _, lines in geometry],
              'centroid_translated': [lines - centroid for _, centroid, lines in geometry],
              'learned': [(lines - centre) @ matrices[i].T + translations[i] for i, (centre, _, lines) in enumerate(geometry)]}
    result = {'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint.get('step'),
              'transform_mode': model.transform_mode, 'transform_type': model.transform_type,
              'transform_statistics_all_subjects': stats_all, 'subjects_for_recurrence': len(frame),
              'min_radius_mm': a.min_radius_mm,
              'recurrence': {name: summarize_recurrence(recurrence(c)[0]) for name, c in clouds.items()}}
    (out / 'alignment.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
