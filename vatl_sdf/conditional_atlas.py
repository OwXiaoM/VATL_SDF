"""Age- and sex-conditional vessel atlases and graph atlases (inference only; the trained model is unchanged).

Each cohort (age a, sex s) weights the training subjects of sex s by a Gaussian kernel on age, w_i ~ exp(-(age_i -
a)^2 / 2 sigma^2), normalized to 1. Per cohort, on the 1 mm template grid:
  decoded/VesselSDF_atlas, VesselMask_atlas   decoder output for the w-weighted mean latent with condition (a, s).
                                              Mean-latent decodings can be fragmented (CONTEXT.md).
  VesselOccupancy                             sum_i w_i 1[SDF_i < 0], from the cached individual decodings (each
                                              subject decoded with its OWN age / sex).
  CentrelineProbability_r<r>mm                sum_i w_i 1[large-vessel centreline of i within r], trained transforms.
  SharedGraphEdgeProbability_<tag>            the fixed population graph, skeleton voxels = cohort edge probability.
                                              Same edge ids in every cohort, so cohorts are comparable.
  CohortGraphEdgeProbability_<tag>            graph rebuilt from this cohort's centreline map (own topology; edge
                                              ids NOT comparable between cohorts).
  graph_<tag>.json                            nodes, edges, paths and probabilities of both graphs.
The effective age (weighted mean age) differs from the nominal age near the ends of the training range, and the
effective sample size (ESS) falls there: both are reported per cohort. Differences between cohorts are descriptive.
condition_effect isolates the decoder's direct use of age / sex by changing the condition at a fixed latent.
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
from scipy.spatial import cKDTree

from . import CHANNELS
from .config import condition_vector
from .deviation import ess, kernel_weights
from .graph_atlas import build_population, presence
from .inference import export_fields, predict_field, world_grid
from .population_maps import components, save
from .probabilistic_atlas import load_model, within_radius

LOW_ESS = 30.


def cohort_tag(age, sex):
    return f'age{age:g}_{sex}'


def cohort_weights(ages, sexes, age, sex, sigma):
    """Normalized Gaussian age weights over the subjects of one sex."""
    w = kernel_weights(ages, age, sigma) * (np.asarray(sexes) == sex)
    if w.sum() <= 0:
        raise ValueError(f'No subjects for cohort {cohort_tag(age, sex)}')
    return w / w.sum()


def load_decodings(directory, frame, shape, affine, step):
    """Packed individual decodings (subjects, bytes) in frame order, checked against the checkpoint and grid."""
    packed = None; seen = 0
    for path in sorted(Path(directory).glob('subjects_*_of_*.npz')):
        part = np.load(path)
        if int(part['step']) != step or tuple(part['shape']) != tuple(shape) or not np.allclose(part['affine'], affine):
            raise ValueError(f'{path}: decodings do not match the checkpoint step or template grid')
        if list(part['subject_ids']) != [str(s) for s in frame.subject_id.values[part['indices']]]:
            raise ValueError(f'{path}: subject order does not match the checkpoint')
        if packed is None:
            packed = np.zeros((len(frame), part['packed'].shape[1]), np.uint8)
        packed[part['indices']] = part['packed']; seen += len(part['indices'])
    if seen != len(frame):
        raise ValueError(f'Decodings cover {seen} of {len(frame)} subjects')
    return packed


def weighted_occupancy(packed, weights, voxels, block=16):
    """(cohorts, voxels) weighted means of bit-packed subject masks."""
    out = np.zeros((len(weights), voxels))
    for start in range(0, len(packed), block):
        bits = np.unpackbits(packed[start:start + block], axis=1, count=voxels).astype(np.float32)
        out += weights[:, start:start + block].astype(np.float32) @ bits
    return out


def weighted_map(flat_indices, weights, shape):
    """Weighted mean of subject masks given as flat voxel indices."""
    out = np.zeros(int(np.prod(shape)))
    for index, w in zip(flat_indices, weights):
        out[index] += w
    return out.reshape(shape)


def paint_edges(edge_paths_mm, values, shape, affine):
    """Volume with each edge's voxels set to its value (largest value where edges share a voxel)."""
    out = np.zeros(shape, np.float32); inverse = np.linalg.inv(affine)
    for path, value in zip(edge_paths_mm, values):
        voxels = np.rint(nib.affines.apply_affine(inverse, path)).astype(int)
        voxels = voxels[((voxels >= 0) & (voxels < shape)).all(1)]
        out[tuple(voxels.T)] = np.maximum(out[tuple(voxels.T)], value)
    return out


def edge_standard_error(p, effective_n):
    return np.sqrt(np.clip(p * (1 - p), 0, None) / effective_n)


def graph_record(nodes_mm, edges, paths, p_edge, lengths, effective_n):
    return {'nodes_mm': np.asarray(nodes_mm).tolist(), 'edges': [list(map(int, e)) for e in edges],
            'edge_points_mm': [np.asarray(x).tolist() for x in paths], 'edge_probability': np.asarray(p_edge).tolist(),
            'edge_probability_se': edge_standard_error(np.asarray(p_edge), effective_n).tolist(),
            'edge_length_mm': list(map(float, lengths))}


def mean_latent(model, weights):
    w = torch.as_tensor(weights, dtype=model.latents.dtype, device=model.latents.device)
    return torch.tensordot(w, model.latents.detach(), dims=1)[None]


def decode(model, coords, latent, config, age, sex):
    return predict_field(model, coords, condition_vector(config, age, sex), config, latent_override=latent)


def dice(a, b):
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def condition_effect(model, coords, config, weights, age, sex, others):
    """Change the decoder condition at one fixed latent: the direct effect of age / sex, without any cohort effect."""
    tau = config['sdf']['truncation_mm']; channel = CHANNELS['vessel_sdf_normalized']
    latent = mean_latent(model, weights)
    reference = decode(model, coords, latent, config, age, sex)[:, channel] * tau
    rows = []
    for other_age, other_sex in others:
        sdf = decode(model, coords, latent, config, other_age, other_sex)[:, channel] * tau
        near = (reference < 0) | (sdf < 0)
        rows.append({'condition': cohort_tag(other_age, other_sex), 'dice_vs_reference': dice(reference < 0, sdf < 0),
                     'volume_ratio': float((sdf < 0).sum() / max((reference < 0).sum(), 1)),
                     'mean_abs_sdf_change_mm_in_vessels': float(np.abs(sdf - reference)[near].mean())})
    return {'latent_and_reference_condition': cohort_tag(age, sex), 'rows': rows}


def centreline_clouds(frame, model, min_radius_mm, workers):
    from analysis.alignment.evaluate_alignment import subject_geometry
    matrices, translations = (x.detach().double().cpu().numpy() for x in model.maps_mm())
    with Pool(workers) as pool:
        geometry = pool.map(subject_geometry, [(row, min_radius_mm) for _, row in frame.iterrows()])
    return [(lines - c) @ matrices[i].T + translations[i] for i, (c, _, lines) in enumerate(geometry)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--decodings', required=True, help='directory with subjects_*_of_*.npz of this checkpoint')
    p.add_argument('--population-graph', required=True, help='population_graph_<tag>.pkl of this checkpoint')
    p.add_argument('--ages', type=float, nargs='+', default=[20, 30, 40, 50, 60, 70, 80])
    p.add_argument('--sexes', nargs='+', default=['M', 'F']); p.add_argument('--sigma-years', type=float, default=5.)
    p.add_argument('--min-radius-mm', type=float, default=1.2); p.add_argument('--min-skeleton-voxels', type=int, default=5)
    p.add_argument('--device', default='cuda'); p.add_argument('--chunk-size', type=int, default=400000)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    start = time.time()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    checkpoint, config, frame, model = load_model(a.checkpoint, a.device)
    config['atlas']['chunk_size'] = a.chunk_size
    lo, hi = config['data']['age_range']
    if min(a.ages) < lo or max(a.ages) > hi:
        raise ValueError(f'Ages must lie within the training condition range {lo}-{hi}')
    coords, shape, affine = world_grid(config, [1., 1., 1.])
    ages, sexes = frame.scan_age.values.astype(float), frame.sex.values
    cohorts = [(age, sex) for age in a.ages for sex in a.sexes]
    weights = np.stack([cohort_weights(ages, sexes, age, sex, a.sigma_years) for age, sex in cohorts])

    with open(a.population_graph, 'rb') as stream:
        shared = pickle.load(stream)
    r, tau, cover = shared['radius_mm'], shared['tau'], shared['edge_cover']
    tag = f'r{r:g}mm_tau{tau:g}'
    clouds = centreline_clouds(frame, model, a.min_radius_mm, a.workers)
    y_shared = presence([cKDTree(x) if len(x) else None for x in clouds], shared['edge_points_mm'], r, cover)
    drift = float(np.abs(y_shared.mean(0) - shared['edge_probability']).max())
    if drift > 1e-9:  # the rebuilt centrelines must reproduce the saved uniform graph atlas
        raise ValueError(f'Uniform edge probabilities differ from {a.population_graph} by {drift}')
    flat = [np.flatnonzero(within_radius(x, shape, affine, r)) for x in clouds]
    occupancy = weighted_occupancy(load_decodings(a.decodings, frame, shape, affine, checkpoint['step']), weights,
                                   int(np.prod(shape)))

    rows = []
    for k, (age, sex) in enumerate(cohorts):
        w, name = weights[k], cohort_tag(age, sex)
        directory = out / name; directory.mkdir()
        effective_n = ess(w)
        decoded = decode(model, coords, mean_latent(model, w), config, age, sex)
        export_fields(decoded, shape, affine, directory / 'decoded', config['sdf']['truncation_mm'])
        decoded_mask = decoded[:, CHANNELS['vessel_sdf_normalized']].reshape(shape) < 0
        occupancy_map = occupancy[k].reshape(shape)
        save(occupancy_map, affine, directory / 'VesselOccupancy.nii.gz')
        cmap = weighted_map(flat, w, shape)
        save(cmap, affine, directory / f'CentrelineProbability_r{r:g}mm.nii.gz')
        p_shared = w @ y_shared
        pop = build_population(clouds, shape, affine, r, tau, cover, a.min_skeleton_voxels, cmap=cmap, weights=w)
        save(paint_edges(shared['edge_points_mm'], p_shared, shape, affine), affine,
             directory / f'SharedGraphEdgeProbability_{tag}.nii.gz')
        save(paint_edges(pop['edge_points_mm'], pop['edge_probability'], shape, affine), affine,
             directory / f'CohortGraphEdgeProbability_{tag}.nii.gz')
        (directory / f'graph_{tag}.json').write_text(json.dumps({
            'shared': graph_record(shared['nodes_mm'], shared['edges'], shared['edge_points_mm'], p_shared,
                                   shared['edge_length_mm'], effective_n),
            'cohort': graph_record(pop['nodes_mm'], pop['edges'], pop['edge_points_mm'], pop['edge_probability'],
                                   pop['edge_length_mm'], effective_n)}))
        occupied = components(occupancy_map >= 0.25)
        decoded_parts = components(decoded_mask)
        rows.append({'cohort': name, 'age': age, 'sex': sex, 'subjects': int((sexes == sex).sum()), 'ess': effective_n,
                     'effective_age': float(w @ ages), 'age_sd': float(np.sqrt(w @ (ages - w @ ages) ** 2)),
                     'low_ess': effective_n < LOW_ESS, 'decoded_voxels': decoded_parts['voxels'],
                     'decoded_components': decoded_parts['components_26'],
                     'decoded_largest_fraction': decoded_parts['largest_fraction'],
                     'occupancy_ge_0.25_voxels': occupied['voxels'], 'occupancy_ge_0.25_components': occupied['components_26'],
                     'occupancy_ge_0.25_largest_fraction': occupied['largest_fraction'],
                     'cohort_graph_nodes': len(pop['nodes_mm']), 'cohort_graph_edges': len(pop['edges']),
                     'cohort_graph_length_mm': float(sum(pop['edge_length_mm'])),
                     'shared_edges_p_ge_0.5': int((p_shared >= 0.5).sum())})
        print(json.dumps(rows[-1]), flush=True)
    table = pd.DataFrame(rows); table.to_csv(out / 'cohorts.tsv', sep='\t', index=False)

    middle, last = a.ages[len(a.ages) // 2], a.sexes[-1]
    others = [(middle, s) for s in a.sexes if s != last] + [(x, last) for x in (a.ages[0], a.ages[-1])]
    effect = condition_effect(model, coords, config, weights[cohorts.index((middle, last))], middle, last, others)
    print(json.dumps(effect), flush=True)
    (out / 'conditional_atlas.json').write_text(json.dumps({
        'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint['step'], 'settings': vars(a),
        'graph_tag': tag, 'uniform_edge_probability_max_diff': drift, 'cohorts': rows, 'condition_effect': effect,
        'runtime_s': time.time() - start}, indent=1))


if __name__ == '__main__':
    main()
