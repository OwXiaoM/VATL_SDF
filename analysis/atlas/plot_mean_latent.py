"""Figures of mean_latent_field (vatl_sdf.conditional_atlas_smooth): overviews and one panel per cohort."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_conditional_atlas import overview, projection, read, section, show, title


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--atlas', required=True); p.add_argument('--manifest', required=True)
    a = p.parse_args()
    root = Path(a.atlas); figures = root / 'figures'
    table = pd.read_csv(root / 'cohorts.tsv', sep='\t')
    centre = json.loads(Path(a.manifest).read_text())['willis_centre_mm']
    data = {}
    for name in table.cohort:
        soft, affine = read(root / name / 'mean_latent_field' / 'VesselSoft.nii.gz')
        mask, _ = read(root / name / 'mean_latent_field' / 'VesselMask_atlas.nii.gz')
        data[name] = {'soft': soft, 'mask': mask > 0}
    shape = soft.shape
    found = np.argwhere(np.max([d['soft'] for d in data.values()], axis=0) >= 0.5)
    crop = [(affine[i, 3] + found[:, i].min() - 8, affine[i, 3] + found[:, i].max() + 8) for i in range(3)]
    slices = {'axial': [centre[2], centre[2] + 15, centre[2] + 35], 'sagittal': [centre[0], centre[0] - 15, centre[0] + 15]}
    for row in table.itertuples():
        d = data[row.cohort]
        figure, axes = plt.subplots(2, 4, figsize=(12.4, 6.2), constrained_layout=True)
        for i, plane in enumerate(('axial', 'sagittal')):
            for j, position in enumerate(slices[plane]):
                handle = show(axes[i, j], section(d['soft'], plane, position, affine), plane, shape, affine, crop, vmax=1.)
                axes[i, j].set_title(f'{plane} {position:.0f} mm', fontsize=8)
            show(axes[i, 3], projection(d['soft'], plane), plane, shape, affine, crop, vmax=1.)
            axes[i, 3].set_title(f'{plane} maximum projection', fontsize=8)
        figure.colorbar(handle, ax=axes, shrink=0.6, label='sigmoid(-SDF / 1 mm)')
        figure.suptitle('Mean latent, age-averaged field: ' + title(row), fontsize=11)
        figure.savefig(figures / f'mean_latent_{row.cohort}.png', dpi=170); plt.close(figure)
    for plane, position in (('axial', centre[2]), ('sagittal', centre[0])):
        overview(data, table, lambda ax, c: show(ax, projection(c['soft'], plane), plane, shape, affine, crop, vmax=1.),
                 figures / f'overview_mean_latent_{plane}_projection.png', f'Mean latent, age-averaged field, {plane} maximum projection')
        overview(data, table, lambda ax, c: show(ax, section(c['soft'], plane, position, affine), plane, shape, affine, crop, vmax=1.),
                 figures / f'overview_mean_latent_{plane}_CoW.png', f'Mean latent, age-averaged field, {plane} slice through the CoW centre')
    print(figures)


if __name__ == '__main__':
    main()
