"""Fitting diagnostics (Phase 0c, Phase 2 screening): individual fits vs atlas construction.

Read-only on the checkpoint. For fixed training subjects spanning age and sex:
- native-grid reconstruction vs reference SDF: zero-level Dice, band MAE, volume, and
  connectivity RELATIVE TO THE SUBJECT'S REFERENCE MASK (26-connectivity);
- optionally the same on a supersampled native grid (reference SDF trilinearly interpolated),
  to separate voxel undersampling from missing geometry;
- grid-free centreline check: at mask-supported graph nodes (0.4 mm tolerance, snapped,
  vatl_sdf.graph_coords), fraction with predicted SDF < 0, by radius, next to the reference.
Mean-latent atlas exports with component statistics.
Training subjects measure fitting, not generalisation.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from vatl_sdf import CHANNELS
from vatl_sdf.config import condition_vector
from vatl_sdf.evaluate import reference_relative_connectivity
from vatl_sdf.graph_coords import load_graph, mask_support, normalize_world, world_to_voxel
from vatl_sdf.inference import generate_atlas, predict_field
from vatl_sdf.model import SDFModel

RADIUS_BINS = [('floor', 0, 0.801), ('0.80-0.90', 0.801, 0.9), ('0.90-1.20', 0.9, 1.2), ('1.20-1.60', 1.2, 1.6),
               ('>=1.60', 1.6, np.inf)]
FULL = np.ones((3, 3, 3), bool)


def component_stats(mask):
    out = {'voxels': int(mask.sum())}
    for name, structure in (('c6', None), ('c26', FULL)):
        labels, count = ndi.label(mask, structure=structure)
        sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1]
        out[name] = {'count': int(count), 'largest_frac': float(sizes[0] / sizes.sum()) if count else None,
                     'largest': int(sizes[0]) if count else 0, 'top5': sizes[:5].tolist(),
                     'count_ge_10vox': int((sizes >= 10).sum())}
    return out


def dice(a, b):
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else None


def select_subjects(frame, per_sex=3):
    chosen = []
    for sex in sorted(frame.sex.unique()):
        part = frame[frame.sex == sex]
        for q in np.linspace(0.1, 0.9, per_sex):
            target = part.scan_age.quantile(q)
            chosen.append(part.iloc[(part.scan_age - target).abs().argsort()].subject_id.iloc[0])
    return list(dict.fromkeys(chosen))


def subject_grid(affine, shape, factor):
    """Voxel-centre world points of a grid `factor` times finer, sharing the FOV."""
    axes = [np.arange(n * factor) / factor - (factor - 1) / (2 * factor) for n in shape]
    vox = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    fine = affine.copy()
    fine[:3, :3] = affine[:3, :3] / factor
    fine[:3, 3] = nib.affines.apply_affine(affine, [-(factor - 1) / (2 * factor)] * 3)
    return vox, nib.affines.apply_affine(affine, vox), tuple(n * factor for n in shape), fine


def normalized(world, affine, shape, config):
    return normalize_world(world, affine, shape, config['data']['world_bbox_mm']).astype(np.float32)


def averaged_field(model, coords, conditions, config, index):
    """Subject field averaged over condition vectors (one vector: the plain forward pass)."""
    total = 0.
    for condition in conditions:
        total = total + predict_field(model, coords, condition, config, subject_index=index).astype(np.float64)
    return (total / len(conditions)).astype(np.float32)


def evaluate_subject(model, config, index, row, out, factors, save, conditions=None):
    """conditions: average the field over these condition vectors (default: the subject's own condition)."""
    tau = config['sdf']['truncation_mm']
    image = nib.load(row.SDF)
    reference = image.get_fdata(dtype=np.float32)
    conditions = [condition_vector(config, row.scan_age, row.sex)] if conditions is None else conditions
    result = {'subject_id': row.subject_id, 'scan_age': float(row.scan_age), 'sex': row.sex}
    for factor in factors:
        vox, world, shape, affine = subject_grid(image.affine, image.shape, factor)
        coords = normalized(world, image.affine, image.shape, config)
        valid = (np.abs(coords) <= 1).all(1).reshape(shape)
        pred = averaged_field(model, coords, conditions, config, index)
        pred = pred[:, CHANNELS['vessel_sdf_normalized']].reshape(shape) * tau
        ref = reference if factor == 1 else map_coordinates(reference, vox.T, order=1, mode='nearest').reshape(shape)
        p, r = (pred < 0) & valid, (ref < 0) & valid
        band = (np.abs(ref) <= tau) & valid
        result[f'grid_x{factor}'] = {
            'zero_level_dice': dice(p, r), 'sign_agreement': float((p == r)[valid].mean()),
            'band_mae_mm': float(np.abs(pred[band] - ref[band]).mean()),
            'volume_ratio_pred_over_ref': float(p.sum() / max(r.sum(), 1)),
            'connectivity': reference_relative_connectivity(p, r),
            'pred': component_stats(p), 'ref': component_stats(r)}
        if save and factor == 1:
            directory = out / 'subjects' / row.subject_id
            directory.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(pred.astype(np.float32), affine), directory / 'Subject_VesselSDF.nii.gz')
            nib.save(nib.Nifti1Image(p.astype(np.uint8), affine), directory / 'Subject_VesselMask.nii.gz')
    graph = load_graph(row.Graph)
    mask = reference < 0
    supported, snapped, _ = mask_support(mask, image.affine, graph['pos'])
    radius = graph['radius']
    pred_nodes = averaged_field(model, normalized(snapped, image.affine, image.shape, config), conditions, config,
                                index)[:, CHANNELS['vessel_sdf_normalized']] * tau
    ref_nodes = map_coordinates(reference, world_to_voxel(image.affine, snapped).T, order=1, mode='nearest')
    centreline = {'nodes': len(radius), 'supported': int(supported.sum())}
    for label, lo, hi in RADIUS_BINS:
        keep = supported & (radius > lo) & (radius <= hi)
        centreline[label] = {'nodes': int(keep.sum()),
                             'pred_inside': float((pred_nodes[keep] < 0).mean()) if keep.any() else None,
                             'ref_inside': float((ref_nodes[keep] < 0).mean()) if keep.any() else None,
                             'pred_median_sdf_mm': float(np.median(pred_nodes[keep])) if keep.any() else None}
    inside_ref = supported & (ref_nodes < 0)
    centreline['pred_inside_given_ref_inside'] = float((pred_nodes[inside_ref] < 0).mean())
    result['centerline'] = centreline
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--subjects', nargs='*')
    p.add_argument('--device', default='cuda')
    p.add_argument('--factors', nargs='+', type=int, default=[1, 2])
    p.add_argument('--atlas-spacings', nargs='*', type=float, default=[1.0, 0.5])
    p.add_argument('--chunk-size', type=int, default=200000)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Output not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    config['atlas']['chunk_size'] = a.chunk_size
    model = SDFModel(config, len(frame)).to(a.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    subjects = a.subjects or select_subjects(frame)
    results = {'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint.get('step'),
               'epoch': checkpoint.get('position', {}).get('epoch'),
               'note': 'training subjects: fitting quality, not held-out generalisation', 'subjects': []}
    for subject in subjects:
        index = int(frame.index[frame.subject_id == subject][0])
        results['subjects'].append(evaluate_subject(model, config, index, frame.iloc[index], out, a.factors, save=True))
        print(json.dumps({'subject': subject, **results['subjects'][-1]['grid_x1']['connectivity']}), flush=True)
        (out / 'diagnostics.json').write_text(json.dumps(results, indent=1))
    results['atlas'] = {}
    for spacing in a.atlas_spacings:
        config['atlas']['spacing_mm'] = [spacing] * 3
        directory = generate_atlas(model, frame, config, out / f'atlas_{spacing:g}mm')
        mask = nib.load(directory / 'VesselMask_atlas.nii.gz').get_fdata() > 0
        results['atlas'][f'{spacing:g}mm'] = component_stats(mask)
        (out / 'diagnostics.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
