"""Phase 5: population centreline density in template space and recurrence vs calibre.

Each training subject's graph (mask-supported edges, snapped nodes) is mapped into template space by
the subject's LEARNED rigid transform: the model decodes subject i at subject-normalized coordinates u
as the template decoder at rigid_transform(u, transform_i) (verified in tests/test_template_mapping.py).
Outputs:
- CentrelineDensity.nii.gz: expected centreline length (mm) per template voxel per subject, from
  samples every 0.5 mm splatted with a Gaussian (sigma 1 mm); also per calibre bin (graph radius
  floor / 0.9-1.6 mm / >= 1.6 mm).
- recurrence.tsv: for sampled nodes of each subject, the fraction of OTHER subjects with a supported
  centreline within d mm in template space, for d in 1, 2, 3 mm, by radius bin.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from vatl_sdf.dataset import graph_support
from vatl_sdf.graph_coords import edge_samples, load_graph, normalize_world
from vatl_sdf.inference import world_grid
from vatl_sdf.model import rigid_transform

BINS = [('floor', 0, 0.801), ('0.9-1.6', 0.801, 1.6), ('>=1.6', 1.6, np.inf)]


def template_points(row, transform, config):
    image = nib.load(row.SDF)
    sdf = np.asarray(image.dataobj, dtype=np.float32)
    g = load_graph(row.Graph)
    g.update(graph_support(sdf < 0, image.affine, g, config['graph']))
    edges = g['edges'][g['edge_supported']]
    points, owner = edge_samples(g['snapped'], edges, step_mm=0.5)
    radius = np.maximum(g['radius'][edges[owner, 0]], g['radius'][edges[owner, 1]])
    u = normalize_world(points, image.affine, sdf.shape, config['data']['world_bbox_mm'])
    half = torch.tensor(config['data']['world_bbox_mm'], dtype=torch.float32) / 2
    t = rigid_transform(torch.tensor(u, dtype=torch.float32), torch.as_tensor(transform)[None].expand(len(u), -1), half)
    return (t.numpy() * half.numpy()), radius  # template coordinates in mm (centred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--max-subjects', type=int); p.add_argument('--recurrence-nodes', type=int, default=300)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    if 'graph' not in config:
        config['graph'] = {'support_tolerance_mm': 0.4, 'support_strict': False, 'edge_step_mm': 0.4}
    transforms = checkpoint['model']['transforms'].numpy()
    if a.max_subjects:
        frame = frame.head(a.max_subjects)
    config['atlas']['spacing_mm'] = [1, 1, 1]
    _, shape, affine = world_grid(config)
    densities = {name: np.zeros(shape) for name in ['all'] + [b[0] for b in BINS]}
    clouds = []
    for index, row in frame.iterrows():
        points, radius = template_points(row, transforms[index], config)
        clouds.append((cKDTree(points), points, radius))
        voxel = np.floor((points - affine[:3, 3]) / 1.0 + 0.5).astype(int)
        inside = ((voxel >= 0) & (voxel < np.array(shape))).all(1)
        for name, lo, hi in [('all', 0, np.inf)] + BINS:
            sel = inside & (radius > lo) & (radius <= hi)
            np.add.at(densities[name], tuple(voxel[sel].T), 0.5)  # 0.5 mm of centreline per sample
        print(row.subject_id, len(points), flush=True)
    n = len(frame)
    for name, field in densities.items():
        field = gaussian_filter(field / n, sigma=1.0)
        image = nib.Nifti1Image(field.astype(np.float32), affine); image.header.set_xyzt_units('mm')
        nib.save(image, out / f'CentrelineDensity_{name.replace(">=", "ge").replace(".", "p")}.nii.gz')
    rng = np.random.default_rng(0)
    rows = []
    for i, (_, points, radius) in enumerate(clouds):
        pick = rng.choice(len(points), size=min(a.recurrence_nodes, len(points)), replace=False)
        hits = {d: np.zeros(len(pick)) for d in (1, 2, 3)}
        for j, (tree, _, _) in enumerate(clouds):
            if j == i:
                continue
            dist = tree.query(points[pick], distance_upper_bound=3.0)[0]
            for d in hits:
                hits[d] += dist <= d
        for k, node in enumerate(pick):
            rows.append({'subject': frame.subject_id.iloc[i], 'radius': radius[node],
                         **{f'recurrence_{d}mm': hits[d][k] / (n - 1) for d in hits}})
    table = pd.DataFrame(rows)
    table['radius_bin'] = pd.cut(table.radius, [b[1] for b in BINS] + [np.inf], labels=[b[0] for b in BINS])
    table.to_csv(out / 'recurrence.tsv', sep='\t', index=False)
    summary = table.groupby('radius_bin', observed=True)[[c for c in table if c.startswith('recurrence')]].describe(percentiles=[.25, .5, .75])
    (out / 'recurrence_summary.json').write_text(summary.to_json(indent=1))
    print(table.groupby('radius_bin', observed=True)[[c for c in table if c.startswith('recurrence')]].median().round(3).to_string())
    json.dump({'subjects': n, 'checkpoint': a.checkpoint, 'step': checkpoint.get('step')}, open(out / 'provenance.json', 'w'))


if __name__ == '__main__':
    main()
