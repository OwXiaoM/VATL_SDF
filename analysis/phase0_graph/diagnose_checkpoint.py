"""Phase 0 diagnostics: individual fitting (Case 1) versus atlas construction (Case 2).

Read-only on the checkpoint. For fixed training subjects spanning age and sex:
- native-grid reconstruction vs reference SDF (zero-level Dice, volume, 6/26 components);
- the same on a 2x supersampled native grid (reference SDF trilinearly interpolated),
  to separate voxel undersampling from missing geometry;
- centerline recall: fraction of graph skeleton nodes with predicted SDF < 0, by radius,
  which is grid-free. The reference SDF is evaluated at the same nodes.
Atlas (mean-latent) exports at 1 mm and 0.5 mm with component statistics.
Training subjects measure fitting, not generalisation.
"""
import argparse
import json
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from mra_sdf_only import CHANNELS
from mra_sdf_only.config import condition_vector
from mra_sdf_only.inference import generate_atlas, predict_field
from mra_sdf_only.model import MRASDFModel

GRAPH_DIR = Path('/medias/medical/data/Atlas_training/v1/graph')
SDF_DIR = Path('/medias/medical/data/Atlas_training/v1/sdf')
RADIUS_BINS = [(0, 0.9), (0.9, 1.5), (1.5, 2.0), (2.0, np.inf)]
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
    center = nib.affines.apply_affine(affine, np.asarray(shape) / 2.)
    return ((world - center) / (np.asarray(config['data']['world_bbox_mm']) / 2.)).astype(np.float32)


def evaluate_subject(model, config, index, row, out, save):
    tau = config['sdf']['truncation_mm']
    image = nib.load(SDF_DIR / f'{row.subject_id}-Mask.nii.gz')
    reference = image.get_fdata(dtype=np.float32)
    condition = condition_vector(config, row.scan_age, row.sex)
    result = {'subject_id': row.subject_id, 'scan_age': float(row.scan_age), 'sex': row.sex}
    for factor in (1, 2):
        vox, world, shape, affine = subject_grid(image.affine, image.shape, factor)
        coords = normalized(world, image.affine, image.shape, config)
        pred = predict_field(model, coords, condition, config, subject_index=index)
        pred = pred[:, CHANNELS['vessel_sdf_normalized']].reshape(shape) * tau
        ref = reference if factor == 1 else map_coordinates(reference, vox.T, order=1, mode='nearest').reshape(shape)
        p, r = pred < 0, ref < 0
        band = np.abs(ref) <= tau
        result[f'grid_x{factor}'] = {
            'zero_level_dice': dice(p, r), 'sign_agreement': float((p == r).mean()),
            'band_mae_mm': float(np.abs(pred[band] - ref[band]).mean()),
            'volume_ratio_pred_over_ref': float(p.sum() / max(r.sum(), 1)),
            'pred': component_stats(p), 'ref': component_stats(r)}
        if save and factor == 1:
            directory = out / 'subjects' / row.subject_id
            directory.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(pred.astype(np.float32), affine), directory / 'Subject_VesselSDF.nii.gz')
            nib.save(nib.Nifti1Image(p.astype(np.uint8), affine), directory / 'Subject_VesselMask.nii.gz')
    graph = pickle.load(open(GRAPH_DIR / f'{row.subject_id}-Graph.pkl', 'rb'))['graph']
    pos = np.array([d['pos'] for _, d in graph.nodes(data=True)])
    radius = np.array([d['radius'] for _, d in graph.nodes(data=True)])
    pred_nodes = predict_field(model, normalized(pos, image.affine, image.shape, config), condition, config,
                               subject_index=index)[:, CHANNELS['vessel_sdf_normalized']] * tau
    ref_nodes = map_coordinates(reference, (np.linalg.inv(image.affine) @ np.c_[pos, np.ones(len(pos))].T)[:3],
                                order=1, mode='nearest')
    recall = {}
    for lo, hi in RADIUS_BINS:
        keep = (radius >= lo) & (radius < hi)
        recall[f'r_{lo}_{hi}'] = {'nodes': int(keep.sum()),
                                  'pred_inside': float((pred_nodes[keep] < 0).mean()) if keep.any() else None,
                                  'ref_inside': float((ref_nodes[keep] < 0).mean()) if keep.any() else None,
                                  'pred_median_sdf': float(np.median(pred_nodes[keep])) if keep.any() else None}
    # Nodes inside the reference but outside the prediction: centreline breaks attributable to fitting.
    both = ref_nodes < 0
    recall['pred_inside_given_ref_inside'] = float((pred_nodes[both] < 0).mean())
    result['centerline'] = recall
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--subjects', nargs='*')
    p.add_argument('--device', default='cuda')
    p.add_argument('--skip-atlas', action='store_true')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Output not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    model = MRASDFModel(config, len(frame)).to(a.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    subjects = a.subjects or select_subjects(frame)
    results = {'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint.get('step'),
               'note': 'training subjects: fitting quality, not held-out generalisation', 'subjects': []}
    for subject in subjects:
        index = int(frame.index[frame.subject_id == subject][0])
        results['subjects'].append(evaluate_subject(model, config, index, frame.iloc[index], out, save=True))
        print(json.dumps(results['subjects'][-1]['grid_x1']['pred']), flush=True)
        (out / 'diagnostics.json').write_text(json.dumps(results, indent=1))
    if not a.skip_atlas:
        results['atlas'] = {}
        for spacing in (1.0, 0.5):
            config['atlas']['spacing_mm'] = [spacing] * 3
            directory = generate_atlas(model, frame, config, out / f'atlas_{spacing:g}mm')
            mask = nib.load(directory / 'VesselMask_atlas.nii.gz').get_fdata() > 0
            results['atlas'][f'{spacing:g}mm'] = component_stats(mask)
            (out / 'diagnostics.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
