"""Manifest of synthetic circle-of-Willis perturbation copies (vatl_sdf/perturbation.py); CPU, reads data only.

Uses the learned primary population graph and its centreline map (vatl_sdf.graph_atlas output on the same
checkpoint), the validation subjects and their refitted (unperturbed) transforms. Copies are ordered by priority:
removal, thinning, displacement (3 mm, then 5 mm). Output: manifest JSON (no image or graph data).
"""
import argparse
import json
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from vatl_sdf.graph_atlas import held_out_rows
from vatl_sdf.graph_coords import load_graph, voxel_to_world
from vatl_sdf.perturbation import perpendicular_direction, segment_geometry, select_targets, subject_segments, willis_centre
from vatl_sdf.deviation import load_fitted

KINDS = (('remove', 0.), ('thin', 0.), ('displace', 3.), ('displace', 5.))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--graph-atlas', required=True)
    p.add_argument('--population', default='r3mm_tau0.1'); p.add_argument('--sets', required=True)
    p.add_argument('--refit', required=True, nargs='+'); p.add_argument('--output', required=True)
    p.add_argument('--radius-mm', type=float, default=25.); p.add_argument('--min-p', type=float, default=0.5)
    p.add_argument('--min-c', type=float, default=0.5); p.add_argument('--min-radius-mm', type=float, default=1.5)
    p.add_argument('--min-large-radius-mm', type=float, default=1.2); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    with open(Path(a.graph_atlas) / f'population_graph_{a.population}.pkl', 'rb') as stream:
        pop = pickle.load(stream)
    radius_tag = a.population.split('mm')[0][1:]
    image = nib.load(Path(a.graph_atlas) / f'CentrelineProbability_r{radius_tag}mm.nii.gz')
    centre, chosen, cmean, z_median = willis_centre(pop, np.asarray(image.dataobj, float), image.affine, a.min_c)
    targets = select_targets(pop, centre, a.radius_mm, a.min_p)
    sets = json.loads(Path(a.sets).read_text())
    fitted = load_fitted(a.refit, 'validation')
    rows = held_out_rows(config, sets['validation'])
    subjects, copies = [], []
    for _, row in rows.iterrows():
        A, t = fitted[row.subject_id]
        sdf_image = nib.load(row.SDF)
        shape = sdf_image.shape[:3]
        native_centre = voxel_to_world(sdf_image.affine, np.asarray(shape, float) / 2.)
        graph = load_graph(row.Graph)
        segs = subject_segments(graph, native_centre, A, t, pop, targets, pop['radius_mm'], pop['edge_cover'],
                                a.min_large_radius_mm, a.min_radius_mm)
        subjects.append({'subject_id': row.subject_id, 'targets': segs, 'eligible_edges': [s['population_edge'] for s in segs if s['eligible']]})
        for s in segs:
            if not s['eligible']:
                continue
            points = segment_geometry(graph, np.asarray(s['matched_edges']))[0]
            for kind, magnitude in KINDS:
                copy_id = f'{row.subject_id}__e{s["population_edge"]}__{kind}{magnitude:g}'
                seed = int.from_bytes(copy_id.encode(), 'little') % (2 ** 32) ^ a.seed
                copies.append({'copy_id': copy_id, 'subject_id': row.subject_id, 'scan_age': float(row.scan_age), 'sex': row.sex,
                               'source_sdf': row.SDF, 'source_graph': row.Graph, 'population_edge': s['population_edge'],
                               'p_edge': float(pop['edge_probability'][s['population_edge']]), 'matched_edges': s['matched_edges'],
                               'segment_radius_mm': s['segment_radius_mm'], 'kind': kind, 'magnitude': magnitude,
                               'direction': perpendicular_direction(points, seed).tolist() if kind == 'displace' else None})
    priority = {k: i for i, k in enumerate(KINDS)}
    copies.sort(key=lambda c: (priority[(c['kind'], c['magnitude'])], c['subject_id'], c['population_edge']))
    eligible_subjects = sum(bool(s['eligible_edges']) for s in subjects)
    manifest = {'checkpoint': str(Path(a.checkpoint).resolve()), 'graph_atlas': str(Path(a.graph_atlas).resolve()),
                'population': a.population, 'settings': vars(a),
                'willis_centre_mm': centre.tolist(), 'centre_edges': chosen, 'centre_rule':
                    f'length-weighted centroid of population edges with mean c_{radius_tag} >= {a.min_c} and mean z <= median edge mean z '
                    f'({z_median:.1f} mm)',
                'edge_mean_centreline_probability': cmean.tolist(),
                'targets': [{'population_edge': k, 'p_edge': float(pop['edge_probability'][k]),
                             'nodes': list(pop['edges'][k]), 'mean_mm': pop['edge_points_mm'][k].mean(0).tolist(),
                             'length_mm': float(pop['edge_length_mm'][k])} for k in targets],
                'subjects': subjects, 'eligible_subjects': eligible_subjects, 'copies': copies}
    Path(a.output).write_text(json.dumps(manifest, indent=1))
    print(json.dumps({'centre': centre.round(1).tolist(), 'centre_edges': chosen, 'targets': targets,
                      'eligible_subjects': eligible_subjects, 'copies': len(copies)}))


if __name__ == '__main__':
    main()
