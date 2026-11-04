"""Stabilized and viewable versions of a conditional atlas directory (adds files; existing ones are not changed).

The decoder is very sensitive to the age input at a fixed latent (one year changes the mean-latent mask by Dice ~0.2;
controls/condition_stability.log), so one forward pass reflects the exact condition value. Per cohort:
  mean_latent_field/VesselSDF_atlas      the cohort's mean latent decoded at the cohort's weighted age quantiles
                                         (fixed sex), SDF fields averaged. Mean of SDFs, not an exact distance field.
  mean_latent_field/VesselMask_atlas     SDF < 0 of that field.
  mean_latent_field/VesselSoft           sigmoid(-SDF / 1 mm).
  VesselOccupancy_smooth<s>mm            Gaussian-smoothed occupancy (sigma s mm).
  *GraphEdgeProbability_tube<r>mm_<tag>  graph edges drawn as tubes of radius r for viewers; the 1-voxel files remain.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .conditional_atlas import cohort_weights, decode, dice, mean_latent
from .inference import world_grid
from .population_maps import components, save
from .probabilistic_atlas import load_model, within_radius
import nibabel as nib


def weighted_age_quantiles(ages, weights, count):
    """`count` equally spaced quantiles of the weighted age distribution (actual subject ages)."""
    order = np.argsort(ages)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    positions = np.searchsorted(cumulative, (np.arange(count) + 0.5) / count)
    return np.asarray(ages)[order][np.minimum(positions, len(ages) - 1)]


def tube_volume(paths_mm, values, shape, affine, radius_mm):
    """Edges as tubes; the largest edge value wins where tubes overlap."""
    out = np.zeros(shape, np.float32)
    for path, value in zip(paths_mm, values):
        tube = within_radius(np.asarray(path), shape, affine, radius_mm)
        out[tube] = np.maximum(out[tube], value)
    return out


def soft_vessel(sdf_mm, scale_mm=1.):
    return 1. / (1. + np.exp(np.clip(sdf_mm / scale_mm, -60, 60)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--atlas', required=True); p.add_argument('--age-samples', type=int, default=24)
    p.add_argument('--smooth-mm', type=float, default=1.); p.add_argument('--tube-mm', type=float, default=1.5)
    p.add_argument('--device', default='cuda'); p.add_argument('--chunk-size', type=int, default=400000)
    a = p.parse_args()
    root = Path(a.atlas)
    summary = json.loads((root / 'conditional_atlas.json').read_text())
    table = pd.read_csv(root / 'cohorts.tsv', sep='\t'); tag = summary['graph_tag']
    sigma = summary['settings']['sigma_years']
    _, config, frame, model = load_model(summary['checkpoint'], a.device)
    config['atlas']['chunk_size'] = a.chunk_size; tau = config['sdf']['truncation_mm']
    coords, shape, affine = world_grid(config, [1., 1., 1.])
    ages, sexes = frame.scan_age.values.astype(float), frame.sex.values
    rows, masks = [], {}
    for row in table.itertuples():
        directory = root / row.cohort; target = directory / 'mean_latent_field'; target.mkdir(exist_ok=False)
        w = cohort_weights(ages, sexes, row.age, row.sex, sigma); latent = mean_latent(model, w)
        samples = weighted_age_quantiles(ages, w, a.age_samples)
        sdf = np.zeros(len(coords))
        for age in samples:
            sdf += decode(model, coords, latent, config, age, row.sex)[:, 0] * tau
        sdf = (sdf / len(samples)).reshape(shape)
        save(sdf, affine, target / 'VesselSDF_atlas.nii.gz'); save(sdf < 0, affine, target / 'VesselMask_atlas.nii.gz')
        save(soft_vessel(sdf), affine, target / 'VesselSoft.nii.gz')
        occupancy = np.asarray(nib.load(directory / 'VesselOccupancy.nii.gz').dataobj, dtype=np.float32)
        save(ndi.gaussian_filter(occupancy, a.smooth_mm), affine, directory / f'VesselOccupancy_smooth{a.smooth_mm:g}mm.nii.gz')
        graph = json.loads((directory / f'graph_{tag}.json').read_text())
        for kind, name in (('shared', 'SharedGraph'), ('cohort', 'CohortGraph')):
            save(tube_volume(graph[kind]['edge_points_mm'], graph[kind]['edge_probability'], shape, affine, a.tube_mm), affine,
                 directory / f'{name}EdgeProbability_tube{a.tube_mm:g}mm_{tag}.nii.gz')
        parts = components(sdf < 0); masks[row.cohort] = sdf < 0
        rows.append({'cohort': row.cohort, 'age_samples_min': float(samples.min()), 'age_samples_max': float(samples.max()),
                     'voxels': parts['voxels'], 'components': parts['components_26'], 'largest_fraction': parts['largest_fraction'],
                     'dice_occupancy_0.25': dice(sdf < 0, occupancy >= 0.25)})
        print(json.dumps(rows[-1]), flush=True)
    pd.DataFrame(rows).to_csv(root / 'mean_latent_field.tsv', sep='\t', index=False)
    ages_list = sorted(table.age.unique())
    pairs = {'male_vs_female': {f'{x:g}': dice(masks[f'age{x:g}_M'], masks[f'age{x:g}_F']) for x in ages_list},
             'adjacent_female': {f'{x:g}-{y:g}': dice(masks[f'age{x:g}_F'], masks[f'age{y:g}_F']) for x, y in zip(ages_list, ages_list[1:])}}
    (root / 'mean_latent_field.json').write_text(json.dumps({'settings': vars(a), 'pairs': pairs, 'cohorts': rows}, indent=1))
    print(json.dumps(pairs), flush=True)


if __name__ == '__main__':
    main()
