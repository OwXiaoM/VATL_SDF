"""Probabilistic population atlas in template space (built at inference only; training objective unchanged).

From the training subjects' template-space decodings (latent_override: no transform, own age/sex condition, as in
vatl_sdf/population_maps) with UNIFORM subject weights:
  occupancy        p(x) = (1/N) sum_i 1[SDF_i(x) < 0]
  soft occupancy   q(x) = (1/N) sum_i sigmoid(-SDF_i(x) / tau)          (tau in mm)
and from the training graphs mapped by the trained transforms (template_mm = A_i (world - centre_i) + t_i):
  centreline       c(x) = (1/N) sum_i 1[large-vessel centreline of i within r mm of x]
Evaluation: connectivity at thresholds; calibration/likelihood of held-out subjects' vessel masks mapped into the
template (Brier, Brier skill vs a constant, Bernoulli log-likelihood with (N p + 1)/(N + 2) smoothing,
reliability curve); coverage of held-out large-vessel centrelines by high-probability regions.

Subcommands: `shard` (GPU: decode a subset of subjects, save partial sums) and `merge` (CPU: sum shards, maps,
centreline probability, evaluation).
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from . import CHANNELS
from .config import condition_vector
from .graph_coords import world_to_voxel
from .inference import predict_field, world_grid
from .model import SDFModel
from .population_maps import components, save

THRESHOLDS = (0.1, 0.25, 0.5)
RELIABILITY_EDGES = (0., 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9)


def accumulate(model, frame, config, coords, indices, tau_soft_mm):
    """Partial sums over `indices` of 1[SDF < 0] and sigmoid(-SDF / tau) on the template grid."""
    tau = config['sdf']['truncation_mm']
    hard = np.zeros(len(coords)); soft = np.zeros(len(coords))
    for index in indices:
        row = frame.iloc[int(index)]
        sdf = predict_field(model, coords, condition_vector(config, row.scan_age, row.sex), config,
                            latent_override=model.latents[int(index):int(index) + 1])[:, CHANNELS['vessel_sdf_normalized']] * tau
        hard += sdf < 0
        soft += 1. / (1. + np.exp(np.clip(sdf / tau_soft_mm, -60, 60)))
    return hard, soft


def within_radius(points_mm, shape, affine, radius_mm):
    """Grid voxels (template grid, diagonal affine) whose centre lies within radius_mm of any point."""
    mask = np.zeros(shape, bool)
    if not len(points_mm):
        return mask
    spacing, origin = np.diag(affine)[:3], affine[:3, 3]
    reach = int(np.ceil(radius_mm / spacing.min())) + 1
    base = np.floor((points_mm - origin) / spacing).astype(np.int64)
    offsets = np.stack(np.meshgrid(*[np.arange(-reach, reach + 1)] * 3, indexing='ij'), -1).reshape(-1, 3)
    for offset in offsets:
        voxel = base + offset
        centre = voxel * spacing + origin
        keep = (np.linalg.norm(centre - points_mm, axis=1) <= radius_mm) & (voxel >= 0).all(1) & (voxel < shape).all(1)
        mask[tuple(voxel[keep].T)] = True
    return mask


def template_world(shape, affine):
    """Template-mm coordinates of all grid voxels (C order)."""
    index = np.indices(shape).reshape(3, -1).T
    return nib.affines.apply_affine(affine, index)


def subject_labels_in_template(sdf_path, A, t_mm, template_mm, world_bbox_mm):
    """Vessel label (SDF < 0) of a subject at template points, through world = A^-1 (x - t) + centre.
    valid = inside the subject image and its physical bounding box (the region used in training)."""
    image = nib.load(sdf_path)
    sdf = np.asarray(image.dataobj, dtype=np.float32)
    sdf = sdf[..., 0] if sdf.ndim == 4 else sdf
    centre = nib.affines.apply_affine(image.affine, np.asarray(sdf.shape, float) / 2.)
    world = (template_mm - t_mm) @ np.linalg.inv(A).T + centre
    voxel = world_to_voxel(image.affine, world)
    inside = ((voxel >= -0.5) & (voxel <= np.asarray(sdf.shape) - 0.5)).all(1)
    inside &= (np.abs(world - centre) <= np.asarray(world_bbox_mm) / 2.).all(1)
    values = map_coordinates(sdf, voxel.T, order=1, mode='nearest')
    return values < 0, inside


def calibration(p, y, valid, n_subjects):
    """Brier, Brier skill vs the constant mean(p), smoothed Bernoulli log-likelihood and reliability bins."""
    p, y = p[valid].astype(np.float64), y[valid].astype(np.float64)
    if not len(p):
        return None
    brier = float(np.mean((p - y) ** 2))
    constant = float(p.mean())
    reference = float(np.mean((constant - y) ** 2))
    smoothed = (n_subjects * p + 1.) / (n_subjects + 2.)
    loglik = y * np.log(smoothed) + (1 - y) * np.log1p(-smoothed)
    const_s = (n_subjects * constant + 1.) / (n_subjects + 2.)
    loglik_const = y * np.log(const_s) + (1 - y) * np.log1p(-const_s)
    bins = np.digitize(p, RELIABILITY_EDGES) - 1
    reliability = []
    for b in range(len(RELIABILITY_EDGES) - 1):
        sel = bins == b
        reliability.append({'bin': [RELIABILITY_EDGES[b], min(RELIABILITY_EDGES[b + 1], 1.)], 'count': int(sel.sum()),
                            'mean_p': float(p[sel].mean()) if sel.any() else None,
                            'observed': float(y[sel].mean()) if sel.any() else None})
    counts = np.array([r['count'] for r in reliability])
    gaps = np.array([abs(r['mean_p'] - r['observed']) if r['count'] else 0. for r in reliability])
    return {'voxels': int(len(p)), 'prevalence': float(y.mean()), 'brier': brier, 'brier_constant': reference,
            'brier_skill': 1 - brier / reference if reference > 0 else None,
            'loglik_per_voxel': float(loglik.mean()), 'loglik_constant_per_voxel': float(loglik_const.mean()),
            'loglik_vessel_voxels': float(loglik[y > 0].mean()) if (y > 0).any() else None,
            'expected_calibration_error': float((counts * gaps).sum() / counts.sum()), 'reliability': reliability}


def coverage(maps, points_mm, affine):
    """Fraction of points at which each probability map (trilinear) reaches each threshold."""
    if not len(points_mm):
        return None
    voxel = world_to_voxel(affine, points_mm).T
    out = {'points': int(len(points_mm))}
    for name, field in maps.items():
        values = map_coordinates(field, voxel, order=1, mode='constant', cval=0.)
        out[name] = {f'ge_{t:g}': float((values >= t).mean()) for t in THRESHOLDS}
        out[name]['median_p'] = float(np.median(values))
    return out


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    model = SDFModel(config, len(frame)).to(device); model.load_state_dict(checkpoint['model']); model.eval()
    return checkpoint, config, frame, model


def shard_main(a):
    checkpoint, config, frame, model = load_model(a.checkpoint, a.device)
    config['atlas']['chunk_size'] = a.chunk_size
    coords, shape, affine = world_grid(config, [a.spacing] * 3)
    indices = np.array_split(np.arange(len(frame)), a.num_shards)[a.shard]
    start = time.time()
    hard, soft = accumulate(model, frame, config, coords, indices, a.tau_mm)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'shard_{a.shard:02d}_of_{a.num_shards:02d}.npz', hard=hard.astype(np.float32),
                        soft=soft.astype(np.float32), indices=indices, shape=shape, affine=affine,
                        step=checkpoint['step'], tau_mm=a.tau_mm, seconds=time.time() - start)
    print(json.dumps({'shard': a.shard, 'subjects': len(indices), 'seconds': time.time() - start}), flush=True)


def merge_main(a):
    from analysis.alignment.evaluate_alignment import subject_geometry  # CPU-only helper, shared with Step 0/2
    start = time.time()
    checkpoint, config, frame, model = load_model(a.checkpoint, 'cpu')
    shards = sorted(Path(a.shards).glob('shard_*.npz'))
    parts = [np.load(s) for s in shards]
    indices = np.concatenate([p['indices'] for p in parts])
    if sorted(indices.tolist()) != list(range(len(frame))) or any(int(p['step']) != checkpoint['step'] for p in parts):
        raise ValueError('Shards do not cover every training subject exactly once at this checkpoint')
    shape, affine, tau_soft = tuple(int(x) for x in parts[0]['shape']), parts[0]['affine'], float(parts[0]['tau_mm'])
    n = len(frame)
    occupancy = (sum(p['hard'].astype(np.float64) for p in parts) / n).reshape(shape)
    soft = (sum(p['soft'].astype(np.float64) for p in parts) / n).reshape(shape)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    # Centreline probability from the training graphs mapped by the trained transforms.
    train_a, train_t = (x.double().numpy() for x in model.maps_mm())
    with Pool(a.workers) as pool:
        geometry = pool.map(subject_geometry, [(row, a.min_radius_mm) for _, row in frame.iterrows()])
    centreline = np.zeros(shape)
    for i, (centre, _, lines) in enumerate(geometry):
        centreline += within_radius((lines - centre) @ train_a[i].T + train_t[i], shape, affine, a.centreline_radius_mm)
    centreline /= n
    maps = {'occupancy': occupancy, f'soft_tau{tau_soft:g}': soft, f'centreline_r{a.centreline_radius_mm:g}': centreline}
    names = {'occupancy': 'OccupancyProbability', f'soft_tau{tau_soft:g}': f'OccupancySoft_tau{tau_soft:g}mm',
             f'centreline_r{a.centreline_radius_mm:g}': f'CentrelineProbability_r{a.centreline_radius_mm:g}mm'}
    for key, field in maps.items():
        save(field, affine, out / f'{names[key]}.nii.gz')
    summary = {'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint['step'],
               'epoch': checkpoint.get('position', {}).get('epoch'), 'subjects': n, 'spacing_mm': float(affine[0, 0]),
               'shape': list(shape), 'tau_soft_mm': tau_soft, 'centreline_radius_mm': a.centreline_radius_mm,
               'large_vessel_min_radius_mm': a.min_radius_mm,
               'connectivity': {k: {f'ge_{t:g}': components(f >= t) for t in THRESHOLDS} for k, f in maps.items()}}
    # Held-out subjects: learned (test-time fitted) transforms and centroid-translation-only control.
    if a.fitted:
        fitted = torch.load(a.fitted, map_location='cpu', weights_only=False)
        rows = pd.read_csv(config['data']['tsv'], sep='\t', dtype={'subject_id': str}).set_index('subject_id', drop=False)
        parent = Path(config['data']['tsv']).resolve().parent
        test_rows = rows.loc[fitted['subject_ids']].reset_index(drop=True)
        for key in ('SDF', 'Graph'):
            test_rows[key] = test_rows[key].map(lambda path: str((parent / path).resolve()))
        if set(test_rows.subject_id) & set(frame.subject_id):
            raise ValueError('Held-out subjects overlap the training subjects')
        mean_centroid = np.mean([train_a[i] @ (c - centre) + train_t[i] for i, (centre, c, _) in enumerate(geometry)], 0)
        with Pool(min(a.workers, len(test_rows))) as pool:
            test_geo = pool.map(subject_geometry, [(row, a.min_radius_mm) for _, row in test_rows.iterrows()])
        template_mm = template_world(shape, affine)
        frames = {'learned': [(fitted['A'][k], fitted['t_mm'][k]) for k in range(len(test_rows))],
                  'centroid_translated': [(np.eye(3), mean_centroid - (c - centre)) for centre, c, _ in test_geo]}
        evaluation = {}
        for frame_name, transforms in frames.items():
            per_subject, pooled = [], {k: ([], [], []) for k in maps}
            for k, (row, (A, t)) in enumerate(zip(test_rows.itertuples(), transforms)):
                y, valid = subject_labels_in_template(row.SDF, A, t, template_mm, config['data']['world_bbox_mm'])
                y, valid = y.reshape(shape), valid.reshape(shape)
                roi = ndi.binary_dilation((occupancy > 0) | y, iterations=a.roi_dilation) & valid
                centre, _, lines = test_geo[k]
                record = {'subject_id': row.subject_id, 'vessel_voxels_in_template': int((y & valid).sum()),
                          'calibration_fov': {m: calibration(f, y, valid, n) for m, f in maps.items() if not m.startswith('centreline')},
                          'calibration_roi': {m: calibration(f, y, roi, n) for m, f in maps.items() if not m.startswith('centreline')},
                          'coverage_large_vessel_centreline': coverage(maps, (lines - centre) @ A.T + t, affine)}
                for m, f in maps.items():
                    pooled[m][0].append(f[valid]); pooled[m][1].append(y[valid]); pooled[m][2].append(roi[valid])
                per_subject.append(record)
            pooled_cal = {}
            for m, (ps, ys, rs) in pooled.items():
                if m.startswith('centreline'):
                    continue
                p_all, y_all, r_all = np.concatenate(ps), np.concatenate(ys), np.concatenate(rs)
                pooled_cal[m] = {'fov': calibration(p_all, y_all, np.ones(len(p_all), bool), n),
                                 'roi': calibration(p_all, y_all, r_all, n)}
            evaluation[frame_name] = {'pooled_calibration': pooled_cal, 'subjects': per_subject}
        summary['heldout'] = {'subject_ids': fitted['subject_ids'], 'fitted_file': str(Path(a.fitted).resolve()),
                              'roi': f'dilation ({a.roi_dilation} voxels) of occupancy > 0 or subject vessel, within the subject FOV',
                              'control': 'centroid_translated: A = I, vessel centroid moved onto the mean training centroid in template space',
                              **evaluation}
    summary['runtime_s'] = {'merge': time.time() - start, 'shards': [float(p['seconds']) for p in parts]}
    (out / 'probabilistic_atlas.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({'connectivity': summary['connectivity'], 'runtime_s': summary['runtime_s']}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('shard'); s.add_argument('--checkpoint', required=True); s.add_argument('--output', required=True)
    s.add_argument('--shard', type=int, required=True); s.add_argument('--num-shards', type=int, required=True)
    s.add_argument('--spacing', type=float, default=1.0); s.add_argument('--tau-mm', type=float, default=0.5)
    s.add_argument('--device', default='cuda'); s.add_argument('--chunk-size', type=int, default=400000)
    m = sub.add_parser('merge'); m.add_argument('--checkpoint', required=True); m.add_argument('--shards', required=True)
    m.add_argument('--output', required=True); m.add_argument('--fitted', help='fitted_test.pt of the generalization run')
    m.add_argument('--centreline-radius-mm', type=float, default=1.0); m.add_argument('--min-radius-mm', type=float, default=1.2)
    m.add_argument('--roi-dilation', type=int, default=3); m.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    shard_main(a) if a.command == 'shard' else merge_main(a)


if __name__ == '__main__':
    main()
