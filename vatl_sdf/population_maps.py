"""Population variability maps in template space (Phase 3).

POPULATION VARIABILITY (this module): how vessel geometry differs ACROSS SUBJECTS at a template
location. Each age-weighted training subject's latent is decoded on the template grid without its
rigid transform (the network works in template space, so decodings are rigidly aligned), with that
subject's own age/sex condition. Outputs, with the same Gaussian age weights w_i as the atlas:
  p(x)      = sum_i w_i 1[SDF_i(x) < 0]          vessel frequency (VesselFrequency.nii.gz)
  mean(x)   = sum_i w_i SDF_i(x)                  (SDF_mean_mm.nii.gz)
  std(x)    = sqrt(sum_i w_i (SDF_i(x) - mean)^2)  (SDF_std_mm.nii.gz)
plus the existing mean-latent atlas for comparison. Fields are accumulated subject by subject;
all individual fields are never held at once.

This is NOT individual fitting uncertainty (e.g. a Laplace scale b(x): how hard one subject is to
fit), which is meaningless when decoded from a mean latent and is not computed here.

These are MARGINAL statistics. They carry no connectivity: a thin vessel present in every subject
but shifted by a few mm has low p(x) everywhere along its course, so thresholding p or the mean SDF
can be disconnected even when every individual decoding is connected.
"""
import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from . import CHANNELS
from .config import condition_vector
from .inference import generate_atlas, predict_field, weighted_latent, world_grid
from .model import SDFModel

STRUCTURE_26 = np.ones((3, 3, 3), bool)


def components(mask):
    labels, count = ndi.label(mask, STRUCTURE_26)
    sizes = np.bincount(labels.ravel())[1:]
    return {'voxels': int(mask.sum()), 'components_26': int(count),
            'largest_fraction': float(sizes.max() / sizes.sum()) if count else 0.}


def accumulate(model, frame, config, age, coords, min_weight=1e-4):
    """Weighted frequency, mean and std of individual template-space decodings."""
    _, weights = weighted_latent(model, frame, config, age)
    weights = weights.detach().cpu().numpy().astype(np.float64)
    chosen = np.flatnonzero(weights >= min_weight)
    weights = weights[chosen] / weights[chosen].sum()  # renormalize after dropping negligible weights
    tau = config['sdf']['truncation_mm']
    frequency = np.zeros(len(coords)); total = np.zeros(len(coords)); total_sq = np.zeros(len(coords))
    for index, w in zip(chosen, weights):
        row = frame.iloc[int(index)]
        sdf = predict_field(model, coords, condition_vector(config, row.scan_age, row.sex), config,
                            latent_override=model.latents[int(index):int(index) + 1])[:, CHANNELS['vessel_sdf_normalized']] * tau
        frequency += w * (sdf < 0); total += w * sdf; total_sq += w * sdf.astype(np.float64) ** 2
    mean = total
    std = np.sqrt(np.clip(total_sq - mean ** 2, 0, None))
    return frequency, mean, std, {'subject_ids': frame.subject_id.iloc[chosen].tolist(), 'weights': weights.tolist(),
                                  'subjects_used': int(len(chosen)), 'min_weight': min_weight}


def save(data, affine, path):
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    image.header.set_xyzt_units('mm'); image.set_sform(affine, code=1)
    nib.save(image, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True); p.add_argument('--output', required=True)
    p.add_argument('--age', type=float); p.add_argument('--sex', choices=['M', 'F'])
    p.add_argument('--spacing', type=float, default=1.0); p.add_argument('--device', default='cuda')
    p.add_argument('--chunk-size', type=int, default=200000)
    a = p.parse_args()
    start = time.time()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    config, frame = checkpoint['config'], pd.DataFrame(checkpoint['metadata'])
    config['atlas'].update(spacing_mm=[a.spacing] * 3, chunk_size=a.chunk_size)
    age = config['atlas']['age'] if a.age is None else a.age
    sex = config['atlas']['sex'] if a.sex is None else a.sex
    model = SDFModel(config, len(frame)).to(a.device); model.load_state_dict(checkpoint['model']); model.eval()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    coords, shape, affine = world_grid(config)
    frequency, mean, std, provenance = accumulate(model, frame, config, age, coords)
    for name, field in (('VesselFrequency', frequency), ('SDF_mean_mm', mean), ('SDF_std_mm', std)):
        save(field.reshape(shape), affine, out / f'{name}.nii.gz')
    accumulated = time.time() - start
    atlas = generate_atlas(model, frame, config, out / 'mean_latent_atlas', age, sex)
    atlas_mask = nib.load(atlas / 'VesselMask_atlas.nii.gz').get_fdata() > 0
    summary = {
        'checkpoint': str(Path(a.checkpoint).resolve()), 'step': checkpoint.get('step'), 'age': age,
        'atlas_sex': sex, 'individual_condition': "each subject's own age/sex", 'spacing_mm': a.spacing,
        'shape': list(shape), **provenance,
        'runtime_s': {'individual_accumulation': accumulated, 'total': time.time() - start},
        'descriptive_masks': {
            'mean_latent_atlas': components(atlas_mask),
            **{f'frequency_ge_{t:g}': components(frequency.reshape(shape) >= t) for t in (0.1, 0.25, 0.5)},
            'mean_sdf_lt_0': components(mean.reshape(shape) < 0)},
        'note': 'Marginal statistics: thresholded maps may be disconnected even if every decoding is connected.'}
    (out / 'population_maps.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('weights', 'subject_ids')}, indent=1))


if __name__ == '__main__':
    main()
