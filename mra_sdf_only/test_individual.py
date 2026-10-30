"""Held-out SDF evaluation: conditioned atlas and explicitly supervised test-time fitting.

No training parameters are updated. Missing/contaminated splits fail closed.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import scipy
import skimage

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import torch
import yaml

from .config import condition_vector, validate
from .inference import native_grid, weighted_latent
from .individual_metrics import component_metrics, evaluate_sdf, summarize
from .model import MRASDFModel, rigid_transform
from .splits import read_splits, ordered_rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def select_subjects(metadata, training, n, split_yaml=None, split_key=None, split_column=None):
    """Select an explicitly disjoint evaluation split in authoritative YAML order."""
    if n not in (3, 5):
        raise ValueError('--num-test-subjects must be 3 or 5')
    frame = pd.read_csv(metadata, sep='\t', dtype={'subject_id': str})
    required = {'subject_id', 'scan_age', 'sex', 'SDF'}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError('Metadata needs nonmissing subject_id, scan_age, sex, SDF')
    if any(Path(x).name != x or x in ('.', '..') for x in frame.subject_id):
        raise ValueError('Subject IDs must be safe directory names')
    if frame.subject_id.duplicated().any():
        raise ValueError('Duplicate subject IDs in metadata')
    if split_column:
        if split_column not in frame:
            raise ValueError(f'Missing split column {split_column}')
        ids = frame.loc[frame[split_column] == 'test', 'subject_id'].tolist()
    elif split_yaml and split_key:
        root_key, _, partition = split_key.rpartition('.')
        if partition in ('train', 'val', 'test'):
            if partition == 'train':
                raise ValueError('Individual evaluation cannot select the population train split')
            splits = read_splits(split_yaml, root_key)
            ids = splits[partition]
            if set(training.subject_id) != set(splits['train']):
                raise ValueError('Checkpoint training IDs must match declared population train split exactly')
        else:
            raise ValueError('Evaluation split key must end in .val or .test')
    else:
        raise ValueError('No explicit test split supplied; training subjects will not be substituted')
    if not ids:
        raise ValueError('Declared evaluation split is empty; no held-out subjects are available')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs in declared test split')
    overlap = sorted(set(ids) & set(training.subject_id))
    if overlap:
        raise ValueError(f'Evaluation/training contamination: {len(overlap)} IDs overlap, including {overlap[:5]}')
    missing = sorted(set(ids)-set(frame.subject_id))
    if missing:
        raise ValueError(f'Test IDs missing from metadata: {missing[:5]}')
    selected = ordered_rows(frame, ids).head(n)
    if len(selected) != n:
        raise ValueError(f'Only {len(selected)} declared test subjects available; requested {n}')
    parent = Path(metadata).resolve().parent
    for col in ('SDF', 'VesselMask'):
        if col in selected:
            selected[col] = selected[col].map(lambda p: str((parent / p).resolve()))
            for path in selected[col]:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
    # Do not filter age/sex or reorder silently; report invalid conditioning instead.
    selected['scan_age'] = pd.to_numeric(selected.scan_age, errors='raise')
    return selected.reset_index(drop=True)


def subject_seed(seed, subject_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{subject_id}'.encode()).digest()[:4], 'little')


def observation_partition(shape, valid, settings, seed):
    """Coordinate-only spatial fold assignment: never inspect GT to choose observations."""
    indices = np.indices(shape, dtype=np.int64)
    block = settings['block_size_voxels']
    hashed = ((indices[0]//block)*73856093 ^ (indices[1]//block)*19349663 ^
              (indices[2]//block)*83492791 ^ seed)
    adaptation = valid & ((hashed % settings['adaptation_folds']) == settings['adaptation_fold'])
    evaluation = valid & ~adaptation
    guard = settings['evaluation_guard_voxels']
    if guard:
        evaluation = ndi.binary_erosion(evaluation, structure=np.ones((3, 3, 3)), iterations=guard)
    pool = np.flatnonzero(adaptation)
    if not len(pool) or not evaluation.any():
        raise ValueError('Empty observation/evaluation partition; check FOV and block settings')
    rng = np.random.default_rng(seed)
    observations = np.sort(rng.choice(pool, size=min(settings['num_observations'], len(pool)), replace=False))
    assert not evaluation.ravel()[observations].any()
    return observations, adaptation, evaluation


def load_reference(row, config):
    image = nib.load(row.SDF)
    if len(image.shape) != 3 or image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Reference SDF must be 3D with explicit mm spatial units')
    affine = image.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-9:
        raise ValueError('Invalid reference affine')
    sdf = image.get_fdata(dtype=np.float32)
    if not np.isfinite(sdf).all():
        raise ValueError('Nonfinite reference SDF')
    coords, shape, _ = native_grid(image, config)
    valid = np.all(np.abs(coords) <= 1., axis=1).reshape(shape)
    if not valid.any():
        raise ValueError('No reference voxels in model FOV')
    if 'VesselMask' in row.index:
        mask_image = nib.load(row.VesselMask)
        if mask_image.shape != shape or not np.allclose(mask_image.affine, affine, rtol=0, atol=1e-5):
            raise ValueError('VesselMask must already share reference SDF grid/affine')
        mask = mask_image.get_fdata()
        if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError('VesselMask must be binary (0/1), not a multi-tissue segmentation')
        if not np.array_equal(mask[valid] > 0, sdf[valid] < 0):
            raise ValueError('Reference vessel mask disagrees with negative-inside SDF within FOV')
    return image, sdf, coords, valid


def forward_sdf(model, coords, condition, latent, transform):
    # latent_override bypasses the model's stored rigid table. Apply the SAME
    # physical XYZ Euler mapping explicitly, before grid interpolation AND decoding.
    if model.model_config['rigid']:
        coords = rigid_transform(coords, transform.expand(len(coords), -1), model.half_bbox)
    ids = torch.zeros(len(coords), device=coords.device, dtype=torch.long)  # never used for stored parameters
    cond = torch.as_tensor(condition, device=coords.device).reshape(1, 2).expand(len(coords), -1)
    return model(coords, ids, cond, latent_override=latent, apply_transform=False)[0][:, 1]


def fit_subject(model, initial, condition, observation_coords, observation_sdf_mm, config, settings, seed):
    """ONLY observed SDF values enter this function. No evaluation values/stopping criteria."""
    device = next(model.parameters()).device
    latent = torch.nn.Parameter(initial.detach().clone())
    transform = torch.nn.Parameter(torch.zeros(1, 6, device=device),
                                   requires_grad=settings['optimize_rigid'] and config['model']['rigid'])
    groups = [{'params': [latent], 'lr': settings['lr_latent']}]
    if transform.requires_grad:
        groups.append({'params': [transform], 'lr': settings['lr_rigid']})
    optimizer = torch.optim.Adam(groups)
    coords = torch.as_tensor(observation_coords, device=device)
    tau = config['sdf']['truncation_mm']
    targets = torch.as_tensor(np.clip(observation_sdf_mm, -tau, tau)/tau, device=device)
    rng = np.random.default_rng(seed)
    trace = []
    for step in range(settings['steps']):
        index = torch.as_tensor(rng.integers(len(coords), size=settings['batch_size']), device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_sdf(model, coords[index], condition, latent, transform)
        residual = (prediction-targets[index]).abs().mean()
        prior = (latent-initial).square().mean()*settings['latent_prior_weight']
        scaled = torch.cat([transform[:, :3], transform[:, 3:]/model.half_bbox], -1)
        loss = residual+prior+scaled.square().mean()*settings['rigid_prior_weight']
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite test-time objective')
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            transform[:, :3].clamp_(-settings['max_rotation_radians'], settings['max_rotation_radians'])
            transform[:, 3:].clamp_(-settings['max_translation_mm'], settings['max_translation_mm'])
        if step == 0 or (step+1) % 50 == 0 or step+1 == settings['steps']:
            trace.append({'step': step+1, 'adaptation_normalized_sdf_l1': float(residual.detach()),
                          'adaptation_objective': float(loss.detach())})
    assert all(p.grad is None for p in model.parameters())
    return latent.detach(), transform.detach(), trace


@torch.no_grad()
def infer_native(model, coords, valid, condition, latent, transform, config, settings):
    # Outside the model FOV is explicitly undefined, not clamped into supported anatomy.
    result = np.full(len(coords), np.nan, dtype=np.float32)
    indices = np.flatnonzero(valid)
    device = next(model.parameters()).device
    for start in range(0, len(indices), settings['chunk_size']):
        ids = indices[start:start+settings['chunk_size']]
        points = torch.as_tensor(coords[ids], device=device)
        result[ids] = forward_sdf(model, points, condition, latent, transform).cpu().numpy()*config['sdf']['truncation_mm']
    if not np.isfinite(result[indices]).all():
        raise FloatingPointError('Nonfinite prediction in valid FOV')
    return result.reshape(valid.shape)


def save_volume(path, data, reference):
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    image = nib.Nifti1Image(data, reference.affine, header)
    image.header.set_xyzt_units('mm')
    image.set_qform(reference.get_qform(), int(reference.header['qform_code']))
    image.set_sform(reference.get_sform(), int(reference.header['sform_code']))
    nib.save(image, path)
    check = nib.load(path)
    if check.shape != reference.shape or not np.allclose(check.affine, reference.affine, rtol=0, atol=1e-5):
        raise ValueError('Export changed physical grid')


def comparison_figure(path, reference, prediction, valid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gt = (reference < 0) & valid
    pred = (prediction < 0) & valid
    # Fixed physical voxel-plane rule: median reference vessel coordinate, else FOV center.
    positions = np.median(np.argwhere(gt), axis=0).astype(int) if gt.any() else np.array(gt.shape)//2
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for a in range(3):
        sl = [slice(None)]*3
        sl[a] = int(positions[a])
        g, p = gt[tuple(sl)].T, pred[tuple(sl)].T
        rgb = np.zeros(g.shape+(3,))
        rgb[g & p] = 1
        rgb[g & ~p] = (0.2, 1., 0.3)
        rgb[p & ~g] = (1., 0.2, 0.3)
        axes[0, a].imshow(rgb, origin='lower', interpolation='nearest')
        axes[0, a].set_title(f'Axis {a}, index {positions[a]}')
        gm, pm = gt.max(a).T, pred.max(a).T
        overlay = np.zeros(gm.shape+(3,)); overlay[gm & pm] = 1
        overlay[gm & ~pm] = (.2, 1., .3); overlay[pm & ~gm] = (1., .2, .3)
        axes[1, a].imshow(overlay, origin='lower', interpolation='nearest')
        for ax in axes[:, a]:
            ax.axis('off')
    fig.suptitle(f'{path.parent.parent.name} / {path.parent.name}\nGT green / prediction red / overlap white; slices and projections; full-FOV descriptive display', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .9]); fig.savefig(path, dpi=140); plt.close(fig)


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_settings(s):
    for k in ('block_size_voxels', 'num_observations', 'steps', 'batch_size', 'chunk_size'):
        if not isinstance(s[k], int) or isinstance(s[k], bool) or s[k] <= 0:
            raise ValueError(f'{k} must be a positive integer')
    if not isinstance(s['optimize_rigid'], bool):
        raise ValueError('optimize_rigid must be boolean')
    for key in ('seed', 'adaptation_folds', 'adaptation_fold', 'evaluation_guard_voxels'):
        if not isinstance(s[key], int) or isinstance(s[key], bool) or s[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if s['adaptation_folds'] < 2 or not 0 <= s['adaptation_fold'] < s['adaptation_folds'] or s['evaluation_guard_voxels'] < 1:
        raise ValueError('Need >=2 folds, valid adaptation fold, and >=1 evaluation guard voxel')
    for k in ('lr_latent', 'lr_rigid', 'latent_prior_weight', 'rigid_prior_weight',
              'max_rotation_radians', 'max_translation_mm', 'metric_truncation_mm', 'surface_tolerance_mm'):
        if not np.isfinite(s[k]) or s[k] <= 0:
            raise ValueError(f'Invalid {k}')


def run(args):
    settings = yaml.safe_load(Path(args.settings).read_text())
    for key in ('checkpoint', 'metadata', 'split_yaml', 'split_key'):
        if getattr(args, key, None) is not None:
            settings[key] = getattr(args, key)
    validate_settings(settings)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    if not settings.get('checkpoint'):
        raise ValueError('Supply --checkpoint from the new 497-subject run; contaminated 552-subject checkpoints are forbidden')
    checkpoint_path = Path(settings['checkpoint']).resolve()
    provenance = {'status': 'preflight', 'settings': settings, 'num_test_subjects': args.num_test_subjects,
                  'protocols': args.protocols, 'device': args.device, 'determinism': args.determinism,
                  'checkpoint': str(checkpoint_path), 'checkpoint_sha256': None,
                  'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                  'scipy': scipy.__version__, 'skimage': skimage.__version__, 'nibabel': nib.__version__,
                  'split_column': args.split_column, 'cohort_manifest': args.cohort_manifest,
                  'cuda_grid_sample_backward_nondeterminism': args.device.startswith('cuda') and 'test_time_sdf_fit' in args.protocols,
                  'settings_sha256': sha256(args.settings),
                  'source_sha256': {name: sha256(Path(__file__).parent/name) for name in
                      ('test_individual.py', 'individual_metrics.py', 'model.py', 'layers.py', 'config.py', 'inference.py', 'splits.py')},
                  'command': __import__('sys').argv}
    try:
        provenance['checkpoint_sha256'] = sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = copy.deepcopy(checkpoint['config']); validate(config)
        if config['loss']['mra_weight'] != 0:
            raise ValueError('This pipeline accepts SDF-only checkpoints (mra_weight=0)')
        training = pd.DataFrame(checkpoint['metadata'])
        if training.subject_id.duplicated().any() or len(training) != len(checkpoint['model']['latents']):
            raise ValueError('Checkpoint metadata/latent table mismatch')
        selected = select_subjects(settings['metadata'], training, args.num_test_subjects,
                                   settings.get('split_yaml'), settings.get('split_key'), args.split_column)
        for _, row in selected.iterrows():
            if not config['data']['age_range'][0] <= row.scan_age <= config['data']['age_range'][1]:
                raise ValueError(f'{row.subject_id}: age outside checkpoint conditioning range')
            condition_vector(config, row.scan_age, row.sex)
            if Path(row.SDF).resolve() in {Path(x).resolve() for x in training.SDF}:
                raise ValueError('Test SDF path aliases a training SDF')
        frozen_settings = {k: v for k, v in settings.items() if k not in
                           ('checkpoint', 'metadata', 'split_yaml', 'split_key')}
        if args.cohort_manifest:
            fixed = json.loads(Path(args.cohort_manifest).read_text())
            if selected.subject_id.tolist() != fixed['subject_ids'][:args.num_test_subjects]:
                raise ValueError('Selected subjects/order differ from frozen cohort manifest')
            if fixed['evaluation_settings'] != frozen_settings or fixed['world_bbox_mm'] != config['data']['world_bbox_mm']:
                raise ValueError('Evaluation settings or FOV changed relative to frozen cohort manifest')
            if sha256(settings['metadata']) != fixed['metadata_sha256']:
                raise ValueError('Metadata changed relative to frozen cohort manifest')
        selection = {'subject_ids': selected.subject_id.tolist(), 'subjects': selected.to_dict(orient='records'),
                     'training_subject_ids': training.subject_id.tolist(), 'train_test_overlap': [],
                     'evaluation_settings': frozen_settings, 'world_bbox_mm': config['data']['world_bbox_mm'],
                     'metadata': str(Path(settings['metadata']).resolve()), 'metadata_sha256': sha256(settings['metadata']),
                     'split_yaml': settings.get('split_yaml'), 'split_key': settings.get('split_key'),
                     'split_column': args.split_column, 'selection_order': 'YAML list order (TSV order only for split-column mode)',
                     'split_sha256': sha256(settings['split_yaml']) if not args.split_column else None}
        dump(out/'selected_subjects.json', selection); selected.to_csv(out/'selected_metadata.tsv', sep='\t', index=False)
        dump(out/'checkpoint_config.json', config)
        # Preflight validates all selected physical grids before any adaptation or GPU allocation use.
        for _, row in selected.iterrows():
            load_reference(row, config)
        if args.prepare_only:
            provenance['status'] = 'prepared'; dump(out/'run.json', provenance)
            return
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        random.seed(settings['seed']); np.random.seed(settings['seed']); torch.manual_seed(settings['seed'])
        torch.cuda.manual_seed_all(settings['seed'])
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=args.determinism == 'seeded')
        model = MRASDFModel(config, len(training)); model.load_state_dict(checkpoint['model'])
        del checkpoint
        model.to(args.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        before = parameter_digest(model)
        records = []
        for _, row in selected.iterrows():
            print('Subject:', row.subject_id, flush=True)
            seed = subject_seed(settings['seed'], row.subject_id)
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); torch.cuda.manual_seed_all(seed)
            image, sdf, coords, valid = load_reference(row, config)
            observations, adaptation, evaluation = observation_partition(image.shape, valid, settings, seed)
            directory = out/row.subject_id; directory.mkdir()
            np.save(directory/'observation_flat_indices.npy', observations)
            save_volume(directory/'valid_fov.nii.gz', valid.astype(np.uint8), image)
            save_volume(directory/'adaptation_region.nii.gz', adaptation.astype(np.uint8), image)
            save_volume(directory/'evaluation_support.nii.gz', evaluation.astype(np.uint8), image)
            dump(directory/'subject.json', {'subject_id': row.subject_id, 'scan_age': float(row.scan_age), 'sex': row.sex,
                 'seed': seed, 'reference_sdf': row.SDF, 'reference_sdf_sha256': sha256(row.SDF),
                 'mask_reference': row.get('VesselMask', 'reference_SDF < 0'), 'shape': list(image.shape),
                 'affine': image.affine.tolist(), 'spacing_mm': list(map(float, image.header.get_zooms())),
                 'fov_voxels': int(valid.sum()), 'observation_count': len(observations),
                 'evaluation_voxels': int(evaluation.sum()), 'condition': condition_vector(config, row.scan_age, row.sex).tolist(),
                 'coordinates': '(affine @ voxel - affine @ (shape/2)) / (world_bbox_mm/2)',
                 'outside_fov': 'SDF=NaN, mask=0; never scored', 'no_train_test_subject_overlap': True})
            with torch.no_grad():
                initial, weights = weighted_latent(model, training, config, float(row.scan_age))
            dump(directory/'initialization.json', {'method': 'training_age_weighted_mean_latent; sex enters decoder condition',
                 'training_subject_weights': weights.cpu().tolist(), 'initial_rigid': [0.]*6})
            condition = condition_vector(config, row.scan_age, row.sex)
            for protocol in args.protocols:
                dest = directory/protocol; dest.mkdir()
                if protocol == 'test_time_sdf_fit':
                    latent, transform, trace = fit_subject(model, initial, condition, coords[observations],
                        sdf.ravel()[observations], config, settings, seed)
                    dump(dest/'adaptation_trace.json', trace)
                    torch.save({'latent': latent.cpu(), 'rigid_xyz_radians_translation_mm': transform.cpu(),
                                'protocol': protocol, 'subject_id': row.subject_id}, dest/'subject_parameters.pt')
                else:
                    latent, transform = initial, torch.zeros(1, 6, device=args.device)
                prediction = infer_native(model, coords, valid, condition, latent, transform, config, settings)
                save_volume(dest/'predicted_sdf_mm.nii.gz', prediction.astype(np.float32), image)
                save_volume(dest/'predicted_vessel_mask.nii.gz', ((prediction < 0) & valid).astype(np.uint8), image)
                # Primary scores: unobserved spatial fold only, identical in both protocols.
                metrics = evaluate_sdf(prediction, sdf, image.affine, valid, evaluation,
                                       settings['metric_truncation_mm'], settings['surface_tolerance_mm'])
                primary = {'subject_id': row.subject_id, 'protocol': protocol, 'scope': 'heldout_spatial_folds', **metrics}
                # Global topology is descriptive; masking observation holes would create artificial fragments.
                whole = evaluate_sdf(prediction, sdf, image.affine, valid, valid,
                                     settings['metric_truncation_mm'], settings['surface_tolerance_mm'])
                dense = {'subject_id': row.subject_id, 'protocol': protocol, 'scope': 'full_fov_descriptive', **whole,
                         **{'pred_'+k: v for k, v in component_metrics((prediction < 0) & valid).items()},
                         **{'reference_'+k: v for k, v in component_metrics((sdf < 0) & valid).items()}}
                dump(dest/'metrics.json', {'heldout_spatial_folds': primary, 'full_fov_descriptive': dense})
                comparison_figure(dest/'comparison.png', sdf, prediction, valid)
                records.extend([primary, dense])
                print(protocol, 'held-out Dice:', primary['dice'], flush=True)
        after = parameter_digest(model)
        if before != after:
            raise RuntimeError('Pretrained state changed during inference')
        summaries = {}
        for protocol in args.protocols:
            summaries[protocol] = {scope: summarize([r for r in records if r['protocol'] == protocol and r['scope'] == scope])
                                   for scope in ('heldout_spatial_folds', 'full_fov_descriptive')}
        dump(out/'metrics.json', {'per_subject': records, 'summary': summaries})
        for protocol in args.protocols:
            subset = [r for r in records if r['protocol'] == protocol]
            dump(out/(protocol+'_metrics.json'), {'per_subject': subset, 'summary': summaries[protocol]})
            protocol_keys = list(dict.fromkeys(k for r in subset for k in r))
            with (out/(protocol+'_metrics.csv')).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=protocol_keys)
                writer.writeheader(); writer.writerows(subset)
        keys = list(dict.fromkeys(k for r in records for k in r))
        with (out/'metrics.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(records)
        with (out/'summary.csv').open('w') as stream:
            writer = csv.writer(stream); writer.writerow(['protocol', 'scope', 'metric', 'mean', 'std', 'n_defined', 'n_total'])
            for protocol, scopes in summaries.items():
                for scope, metrics in scopes.items():
                    for key, values in metrics.items():
                        writer.writerow([protocol, scope, key, *[values[k] for k in ('mean', 'std', 'n_defined', 'n_total')]])
        provenance.update(status='complete', frozen_state_sha256_before=before, frozen_state_sha256_after=after,
                          cuda_version=torch.version.cuda, device_name=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'cpu',
                          checkpoint_unchanged=sha256(checkpoint_path) == provenance['checkpoint_sha256'])
        if not provenance['checkpoint_unchanged']:
            raise RuntimeError('Original checkpoint file changed during evaluation')
        dump(out/'run.json', provenance)
    except Exception as exc:
        provenance.update(status='blocked_or_failed', error=str(exc))
        dump(out/'run.json', provenance)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default='configs/evaluation/sdf_only.yaml')
    parser.add_argument('--checkpoint'); parser.add_argument('--metadata')
    parser.add_argument('--split-yaml'); parser.add_argument('--split-key'); parser.add_argument('--split-column')
    parser.add_argument('--num-test-subjects', type=int, choices=(3, 5), required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--device', default='cpu')
    parser.add_argument('--determinism', choices=('strict', 'seeded'), default='strict')
    parser.add_argument('--protocols', nargs='+', choices=('zero_shot_atlas', 'test_time_sdf_fit'),
                        default=['zero_shot_atlas', 'test_time_sdf_fit'])
    parser.add_argument('--cohort-manifest', help='Freeze subject prefix and metadata hash against an earlier selection')
    parser.add_argument('--prepare-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
