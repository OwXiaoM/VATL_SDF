"""Figures of a conditional atlas directory (vatl_sdf.conditional_atlas): one panel per cohort and age x sex overviews.

Slices are 1 mm template slices, positioned relative to the data-defined Circle-of-Willis centre (perturbation
manifest, willis_centre_mm). Colour = vessel occupancy; cyan line = decoded mean-latent mask. There is no anatomical
background: the model has no intensity output. Axes are template mm, RAS (+x right, +y anterior, +z superior).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

VMAX = 0.7
PLANES = {'axial': (2, (0, 1), 'x (mm, +R)', 'y (mm, +A)'), 'sagittal': (0, (1, 2), 'y (mm, +A)', 'z (mm, +S)')}


def read(path):
    image = nib.load(path)
    return np.asarray(image.dataobj, dtype=np.float32), image.affine


def voxel(position_mm, axis, affine):
    return int(round((position_mm - affine[axis, 3]) / affine[axis, axis]))


def section(volume, plane, position_mm, affine):
    axis = PLANES[plane][0]
    return np.take(volume, voxel(position_mm, axis, affine), axis=axis).T


def projection(volume, plane):
    return volume.max(axis=PLANES[plane][0]).T


def full_extent(plane, shape, affine):
    a, b = PLANES[plane][1]
    return [affine[a, 3], affine[a, 3] + shape[a], affine[b, 3], affine[b, 3] + shape[b]]


def show(axis, image, plane, shape, affine, crop, cmap='inferno', vmax=VMAX, contour=None):
    box = full_extent(plane, shape, affine)
    handle = axis.imshow(image, origin='lower', extent=box, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    if contour is not None and contour.any():
        axis.contour(contour, levels=[0.5], colors='cyan', linewidths=0.5, extent=box, origin='lower')
    a, b = PLANES[plane][1]
    axis.set_xlim(*crop[a]); axis.set_ylim(*crop[b]); axis.set_facecolor('black')
    axis.tick_params(labelsize=6)
    return handle


def draw_graph(axis, graph, plane, background, shape, affine, crop):
    a, b = PLANES[plane][1]
    show(axis, background, plane, shape, affine, crop, cmap='gray', vmax=1.)
    segments = [np.asarray(x)[:, [a, b]] for x in graph['edge_points_mm']]
    p = np.asarray(graph['edge_probability'])
    lines = LineCollection(segments, array=p, cmap='viridis', norm=plt.Normalize(0, 1), linewidths=0.8 + 2.2 * p)
    axis.add_collection(lines)
    nodes = np.asarray(graph['nodes_mm'])
    if len(nodes):
        axis.scatter(nodes[:, a], nodes[:, b], s=5, c='white', linewidths=0, zorder=3)
    return lines


def load_cohort(directory, tag, radius):
    occupancy, affine = read(directory / 'VesselOccupancy.nii.gz')
    mask, _ = read(directory / 'decoded' / 'VesselMask_atlas.nii.gz')
    cmap, _ = read(directory / f'CentrelineProbability_r{radius}mm.nii.gz')
    graph = json.loads((directory / f'graph_{tag}.json').read_text())
    return {'occupancy': occupancy, 'mask': mask > 0, 'cmap': cmap, 'graph': graph, 'affine': affine}


def title(row):
    flag = ' LOW ESS' if row.low_ess else ''
    return f'{row.sex} {row.age:g} y (effective {row.effective_age:.1f}, ESS {row.ess:.0f}{flag})'


def cohort_figure(data, row, slices, crop, path):
    shape, affine = data['occupancy'].shape, data['affine']
    columns = max(len(v) for v in slices.values()) + 1
    figure, axes = plt.subplots(3, columns, figsize=(3.1 * columns, 8.6), constrained_layout=True)
    for line, plane in enumerate(('axial', 'sagittal')):
        label = 'z' if plane == 'axial' else 'x'
        for column, (name, position) in enumerate(slices[plane]):
            handle = show(axes[line, column], section(data['occupancy'], plane, position, affine), plane, shape, affine,
                          crop, contour=section(data['mask'], plane, position, affine))
            axes[line, column].set_title(f'{plane} {name}, {label} = {position:.0f} mm', fontsize=8)
        show(axes[line, -1], projection(data['occupancy'], plane), plane, shape, affine, crop)
        axes[line, -1].set_title(f'{plane} maximum projection', fontsize=8)
    bottom = [('shared', 'axial'), ('shared', 'sagittal'), ('cohort', 'axial'), ('cohort', 'sagittal')]
    for column, (kind, plane) in enumerate(bottom[:columns]):
        lines = draw_graph(axes[2, column], data['graph'][kind], plane, projection(data['cmap'], plane), shape, affine, crop)
        axes[2, column].set_title(f'{kind} graph, {plane}', fontsize=8)
    figure.colorbar(handle, ax=axes[:2], shrink=0.5, label='vessel occupancy')
    figure.colorbar(lines, ax=axes[2], shrink=0.7, label='edge probability')
    figure.suptitle(title(row), fontsize=11)
    figure.savefig(path, dpi=170); plt.close(figure)


def overview(cohorts, table, draw, path, heading):
    ages, sexes = sorted(table.age.unique()), list(table.sex.unique())
    figure, axes = plt.subplots(len(sexes), len(ages), figsize=(2.5 * len(ages), 2.9 * len(sexes)), squeeze=False,
                                constrained_layout=True)
    for i, sex in enumerate(sexes):
        for j, age in enumerate(ages):
            row = table[(table.age == age) & (table.sex == sex)].iloc[0]
            handle = draw(axes[i, j], cohorts[row.cohort])
            axes[i, j].set_title(f'{sex} {age:g} (eff. {row.effective_age:.0f}, ESS {row.ess:.0f})', fontsize=8)
    figure.colorbar(handle, ax=axes, shrink=0.7)
    figure.suptitle(heading, fontsize=11)
    figure.savefig(path, dpi=170); plt.close(figure)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--atlas', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--axial-offsets-mm', type=float, nargs='+', default=[0., 15., 35.])
    p.add_argument('--sagittal-offsets-mm', type=float, nargs='+', default=[0., -15., 15.])
    a = p.parse_args()
    root = Path(a.atlas); figures = root / 'figures'; figures.mkdir(exist_ok=True)
    summary = json.loads((root / 'conditional_atlas.json').read_text())
    tag = summary['graph_tag']; radius = tag.split('mm')[0][1:]
    table = pd.read_csv(root / 'cohorts.tsv', sep='\t')
    centre = json.loads(Path(a.manifest).read_text())['willis_centre_mm']
    cohorts = {name: load_cohort(root / name, tag, radius) for name in table.cohort}
    first = next(iter(cohorts.values())); affine, shape = first['affine'], first['occupancy'].shape
    pooled = np.max([c['occupancy'] for c in cohorts.values()], axis=0) >= 0.05
    found = np.argwhere(pooled)
    crop = [(affine[i, 3] + found[:, i].min() - 8, affine[i, 3] + found[:, i].max() + 8) for i in range(3)]
    slices = {'axial': [(f'CoW{o:+g}', centre[2] + o) for o in a.axial_offsets_mm],
              'sagittal': [(f'CoW{o:+g}', centre[0] + o) for o in a.sagittal_offsets_mm]}
    for row in table.itertuples():
        cohort_figure(cohorts[row.cohort], row, slices, crop, figures / f'{row.cohort}.png')
    for plane, position in (('axial', centre[2]), ('sagittal', centre[0])):
        overview(cohorts, table, lambda ax, c: show(ax, section(c['occupancy'], plane, position, affine), plane, shape, affine, crop,
                                                    contour=section(c['mask'], plane, position, affine)),
                 figures / f'overview_{plane}_CoW.png', f'Vessel occupancy, {plane} slice through the CoW centre (cyan: decoded mask)')
        overview(cohorts, table, lambda ax, c: show(ax, projection(c['occupancy'], plane), plane, shape, affine, crop),
                 figures / f'overview_{plane}_projection.png', f'Vessel occupancy, {plane} maximum projection')
        for kind in ('shared', 'cohort'):
            overview(cohorts, table, lambda ax, c: draw_graph(ax, c['graph'][kind], plane, projection(c['cmap'], plane), shape, affine, crop),
                     figures / f'overview_graph_{kind}_{plane}.png', f'{kind.capitalize()} graph atlas, {plane} projection, colour = edge probability')
    print(figures)


if __name__ == '__main__':
    main()
