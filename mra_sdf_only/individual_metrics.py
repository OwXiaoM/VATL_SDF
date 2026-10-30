"""SDF-only native-grid metrics. Surface samples are boundary voxel centers."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .evaluate import dice


def component_metrics(mask):
    out = {}
    for neighbors, connectivity in ((6, 1), (26, 3)):
        labels, count = ndi.label(mask, structure=ndi.generate_binary_structure(3, connectivity))
        sizes = np.bincount(labels.ravel())[1:]
        out[f'components_{neighbors}'] = int(count)
        out[f'largest_component_voxels_{neighbors}'] = int(sizes.max()) if count else 0
        out[f'largest_component_fraction_{neighbors}'] = float(sizes.max()/mask.sum()) if count else 0.
    return out


def surface_metrics(pred, reference, affine, valid, evaluation, tolerance_mm):
    # Extract surfaces BEFORE restricting support: never create surfaces at observation holes.
    structure = ndi.generate_binary_structure(3, 1)
    interior_fov = ndi.binary_erosion(valid, structure=structure, border_value=0)
    support = evaluation & interior_fov
    boundaries = [m & ~ndi.binary_erosion(m, structure=structure, border_value=0) & support
                  for m in (pred, reference)]
    counts = [int(b.sum()) for b in boundaries]
    out = {'surface_pred_voxels': counts[0], 'surface_reference_voxels': counts[1],
           'surface_status': 'defined', 'assd_mm': None, 'hd95_mm': None,
           'surface_dice_tolerance': None}
    if not all(counts):
        out['surface_status'] = ('both_surfaces_empty' if not any(counts) else
                                 'prediction_surface_empty' if not counts[0] else 'reference_surface_empty')
        return out
    points = [np.argwhere(b) @ affine[:3, :3].T + affine[:3, 3] for b in boundaries]
    distances = [cKDTree(points[1]).query(points[0])[0], cKDTree(points[0]).query(points[1])[0]]
    out.update(assd_mm=float((distances[0].mean()+distances[1].mean())/2),
               hd95_mm=float(np.percentile(np.concatenate(distances), 95)),
               surface_dice_tolerance=float(sum((d <= tolerance_mm).sum() for d in distances)/sum(counts)))
    return out


def evaluate_sdf(pred_sdf, reference_sdf, affine, valid, evaluation, tau=5., tolerance_mm=1.):
    arrays = (pred_sdf, reference_sdf, valid, evaluation)
    if any(a.shape != reference_sdf.shape for a in arrays) or not evaluation.any():
        raise ValueError('Identical grids and nonempty evaluation support required')
    if np.any(evaluation & ~valid) or not np.isfinite(reference_sdf[valid]).all() or not np.isfinite(pred_sdf[valid]).all():
        raise ValueError('Invalid FOV or nonfinite SDF within FOV')
    pred, reference = pred_sdf < 0, reference_sdf < 0
    p, r = pred[evaluation], reference[evaluation]
    error = np.abs(pred_sdf[evaluation]-reference_sdf[evaluation])
    clipped = np.abs(np.clip(pred_sdf[evaluation], -tau, tau)-np.clip(reference_sdf[evaluation], -tau, tau))
    band = np.abs(reference_sdf[evaluation]) <= tau
    intersection = int((p & r).sum())
    metrics = {'evaluation_voxels': int(evaluation.sum()), 'dice': dice(p, r),
               'precision': float(intersection/p.sum()) if p.any() else None,
               'recall': float(intersection/r.sum()) if r.any() else None,
               'empty_case': 'both_empty' if not p.any() and not r.any() else
                             'prediction_empty' if not p.any() else 'reference_empty' if not r.any() else 'neither_empty',
               'sdf_mae_mm': float(error.mean()), 'sdf_clipped_mae_mm': float(clipped.mean()),
               'sdf_reference_band_mae_mm': float(error[band].mean()) if band.any() else None,
               'pred_voxels': int(p.sum()), 'reference_voxels': int(r.sum()),
               'pred_volume_mm3': float(p.sum()*abs(np.linalg.det(affine[:3, :3]))),
               'reference_volume_mm3': float(r.sum()*abs(np.linalg.det(affine[:3, :3])))}
    metrics.update(surface_metrics(pred, reference, affine, valid, evaluation, tolerance_mm))
    return metrics


def summarize(records):
    """Equal subject weights, sample SD (ddof=1), finite values only, explicit denominators."""
    keys = sorted({k for record in records for k, v in record.items()
                   if (isinstance(v, (int, float)) and not isinstance(v, bool)) or v is None})
    out = {}
    for key in keys:
        values = [r[key] for r in records if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        out[key] = {'mean': float(np.mean(values)) if values else None,
                    'std': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    'n_defined': len(values), 'n_total': len(records)}
    return out
