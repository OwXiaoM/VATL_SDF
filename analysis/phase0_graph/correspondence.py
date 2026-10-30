"""Phase 0a/0b: graph-mask coordinate correspondence and graph imperfection statistics.

Read-only with respect to data. Modes:
  headers      shape/sform/qform/codes of mask, SDF, MRA; sdf < 0 vs label 1
  conventions  score candidate graph->voxel conventions, resampled-grid hypotheses, radius units
  stats        per-subject inside fractions, SDF residuals, mask vs graph components, loops
  qc           PNG overlays (current convention vs best translation)
"""
import argparse
import heapq
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import map_coordinates

from vatl_sdf.graph_coords import (MaskDistance, inside_mask, load_graph, vessel_mask, voxel_to_world,
                                       world_to_voxel)

ROOT = '/medias/medical/data/Atlas_training/v1'
GRAPH_TSV = f'{ROOT}/ixi_metadata_label_t1_fast_graph.tsv'
SDF_TSV = f'{ROOT}/ixi_metadata_label_t1_fast_sdf.tsv'
DEFAULT_SUBJECTS = ['IXI002-Guys-0828', 'IXI012-HH-1211', 'IXI120-Guys-0766', 'IXI363-HH-2050', 'IXI586-HH-2451']
RADIUS_BINS = [('floor', 0, 0.801), ('0.80-0.90', 0.801, 0.9), ('0.90-1.20', 0.9, 1.2), ('1.20-1.60', 1.2, 1.6),
               ('>=1.60', 1.6, np.inf)]
STRUCT26 = np.ones((3, 3, 3), bool)
SHORT_LOOP_MM = 5.0
SNAP_MM = 1.0


def table():
    graphs = pd.read_csv(GRAPH_TSV, sep='\t', dtype={'subject_id': str})
    sdf = pd.read_csv(SDF_TSV, sep='\t', dtype={'subject_id': str})[['subject_id', 'SDF']]
    return graphs.merge(sdf, on='subject_id').set_index('subject_id')


def load_subject(row):
    label = nib.load(row.Seg)
    return {'graph': load_graph(row.Graph), 'label': label, 'mask': vessel_mask(label), 'affine': label.affine,
            'zooms': np.array(label.header.get_zooms()[:3], dtype=np.float64)}


def floor_of(radius):
    return radius <= RADIUS_BINS[0][2]


# ---------------------------------------------------------------- headers

def header_record(subject_id, row):
    out = {'subject_id': subject_id}
    images = {k: nib.load(row[k]) for k in ('Seg', 'SDF', 'MRA')}
    for name, image in images.items():
        h = image.header
        out[f'{name}_shape'] = 'x'.join(map(str, image.shape))
        out[f'{name}_zooms'] = ','.join(f'{z:.5f}' for z in h.get_zooms()[:3])
        out[f'{name}_sform_code'] = int(h['sform_code'])
        out[f'{name}_qform_code'] = int(h['qform_code'])
        out[f'{name}_sform_vs_qform_maxdiff'] = float(np.abs(image.get_sform() - image.get_qform()).max())
    ref = images['Seg']
    for name in ('SDF', 'MRA'):
        out[f'{name}_same_shape'] = images[name].shape == ref.shape
        out[f'{name}_sform_maxdiff_vs_seg'] = float(np.abs(images[name].get_sform() - ref.get_sform()).max())
        out[f'{name}_qform_maxdiff_vs_seg'] = float(np.abs(images[name].get_qform() - ref.get_qform()).max())
    labels = np.asarray(ref.dataobj)
    sdf = np.asarray(images['SDF'].dataobj)
    out['labels_present'] = ','.join(map(str, np.unique(labels)))
    out['sdf_neg_equals_label1'] = bool(((sdf < 0) == (labels == 1)).all())
    return out


# ---------------------------------------------------------------- conventions

def candidate_voxels(subject, label_image):
    """Map node pos to continuous voxel coordinates of the mask grid under each candidate."""
    pos = subject['graph']['pos']
    sform, qform = label_image.get_sform(), label_image.get_qform()
    zooms = subject['zooms']
    base = world_to_voxel(sform, pos)
    out = {'current: world mm, inv(sform)': base, 'world mm, inv(qform)': world_to_voxel(qform, pos)}
    for shift in itertools.product((-0.5, 0, 0.5), repeat=3):
        if any(shift):
            out[f'current + voxel shift {shift}'] = base + np.array(shift)
    out['pos = voxel index (i,j,k)'] = pos.copy()
    out['pos = voxel index (k,j,i)'] = pos[:, ::-1].copy()
    out['pos = index * spacing (no origin/rotation)'] = pos / zooms
    out['pos = index * 0.8 iso (no origin/rotation)'] = pos / 0.8
    for signs in itertools.product((1, -1), repeat=3):
        if signs != (1, 1, 1):
            tag = 'LPS->RAS' if signs == (-1, -1, 1) else f'sign flip {signs}'
            out[f'world {tag}'] = world_to_voxel(sform, pos * np.array(signs))
    for perm in itertools.permutations(range(3)):
        if perm != (0, 1, 2):
            out[f'world axis permutation {perm}'] = world_to_voxel(sform, pos[:, perm])
    return out


def score(subject, voxels, distance):
    mask, affine, radius = subject['mask'], subject['affine'], subject['graph']['radius']
    world = voxel_to_world(affine, voxels)
    inside = inside_mask(mask, affine, world)
    d = distance.distance(world)
    floor = floor_of(radius)
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else np.nan
    return {'inside_all': inside.mean(), 'inside_floor': inside[floor].mean(), 'inside_nonfloor': inside[~floor].mean(),
            'dist_median_all': q(d, .5), 'dist_p95_all': q(d, .95),
            'dist_median_floor': q(d[floor], .5), 'dist_p95_floor': q(d[floor], .95)}


def best_translation(subject, distance, step=0.1, extent=1.0):
    """Global world-mm translation maximizing the floor-node inside fraction (grid search)."""
    mask, affine = subject['mask'], subject['affine']
    pos = subject['graph']['pos'][floor_of(subject['graph']['radius'])]
    grid = np.arange(-extent, extent + 1e-9, step)
    best = (-1, None)
    for t in itertools.product(grid, repeat=3):
        frac = inside_mask(mask, affine, pos + np.array(t)).mean()
        if frac > best[0]:
            best = (frac, np.array(t))
    return best[1]


def resampled_masks(subject):
    """0.8 mm isotropic regrids of the vessel mask the graph pipeline may have used."""
    mask, affine, zooms = subject['mask'], subject['affine'], subject['zooms']
    shape = np.asarray(mask.shape)
    iso = 0.7999918
    scale = zooms / iso
    out = {}
    # Same rotation; voxel 0 centre preserved (origin-aligned) or FOV corner preserved (corner-aligned).
    for name, offset in (('origin-aligned', np.zeros(3)), ('corner-aligned', 0.5 / scale - 0.5)):
        new_shape = np.ceil(shape * scale).astype(int)
        new_affine = affine.copy()
        new_affine[:3, :3] = affine[:3, :3] / scale
        new_affine[:3, 3] = voxel_to_world(affine, offset)
        coords = (np.indices(new_shape).reshape(3, -1).T * (1 / scale) + offset).T
        for order in (0, 1):
            values = map_coordinates(mask.astype(np.float32), coords, order=order, mode='constant')
            out[f'{name}, order {order}'] = ((values.reshape(new_shape) >= 0.5), new_affine)
    return out


def conventions_subject(subject_id, row):
    subject = load_subject(row)
    distance = MaskDistance(subject['mask'], subject['affine'])
    rows = []
    for name, voxels in candidate_voxels(subject, subject['label']).items():
        rows.append({'subject_id': subject_id, 'convention': name, **score(subject, voxels, distance)})
    t = best_translation(subject, distance)
    base = world_to_voxel(subject['affine'], subject['graph']['pos'] + t)
    rows.append({'subject_id': subject_id, 'convention': f'best global translation {np.round(t, 2).tolist()} mm',
                 **score(subject, base, distance)})

    # Resampled-grid hypotheses: are nodes inside the regridded mask, and does its EDT reproduce radius?
    pos, radius = subject['graph']['pos'], subject['graph']['radius']
    floor = floor_of(radius)
    resample = []
    for name, (iso_mask, iso_affine) in resampled_masks(subject).items():
        inside = inside_mask(iso_mask, iso_affine, pos)
        index = np.clip(np.rint(world_to_voxel(iso_affine, pos)).astype(int), 0, np.asarray(iso_mask.shape) - 1)
        edt = ndimage.distance_transform_edt(iso_mask, sampling=nib.affines.voxel_sizes(iso_affine))
        edt_at = edt[tuple(index.T)]
        frac = np.abs(world_to_voxel(iso_affine, pos) - np.rint(world_to_voxel(iso_affine, pos)))
        resample.append({'subject_id': subject_id, 'grid': name, 'iso_voxels': int(iso_mask.sum()),
                         'inside_all': inside.mean(), 'inside_floor': inside[floor].mean(),
                         'radius_equals_iso_edt': float(np.isclose(edt_at, radius, atol=1e-3).mean()),
                         'nodes_on_iso_lattice': float((frac < 0.02).all(1).mean()),
                         'median_offset_from_iso_centre_vox': float(np.median(np.linalg.norm(frac, axis=1)))})

    # Radius units: interior distance of the native mask at the node.
    mask, affine, zooms = subject['mask'], subject['affine'], subject['zooms']
    edt_mm = ndimage.distance_transform_edt(mask, sampling=zooms)
    edt_vox = ndimage.distance_transform_edt(mask)
    index = np.clip(np.rint(world_to_voxel(affine, pos)).astype(int), 0, np.asarray(mask.shape) - 1)
    background = MaskDistance(~mask, affine)
    inside = inside_mask(mask, affine, pos)
    units = []
    for label, lo, hi in RADIUS_BINS:
        sel = (radius > lo) & (radius <= hi) & inside
        if sel.sum() == 0:
            continue
        units.append({'subject_id': subject_id, 'radius_bin': label, 'n_inside': int(sel.sum()),
                      'radius_median_mm': float(np.median(radius[sel])),
                      'edt_mm_median': float(np.median(edt_mm[tuple(index[sel].T)])),
                      'edt_vox_median': float(np.median(edt_vox[tuple(index[sel].T)])),
                      'dist_to_background_centre_mm_median': float(np.median(background.centre_distance(pos[sel]))),
                      'radius_over_edt_mm_median': float(np.median(radius[sel] / edt_mm[tuple(index[sel].T)])),
                      'radius_vox_0p8_median': float(np.median(radius[sel] / 0.7999918))})
    return rows, resample, units


# ---------------------------------------------------------------- stats (0b)

def chains(core):
    """Decompose a 2-core into chains between junctions (degree != 2): (u, v, length_mm, nodes)."""
    out, seen = [], set()
    junctions = {n for n in core if core.degree(n) != 2}
    for start in junctions:
        for nxt in core.neighbors(start):
            if (start, nxt) in seen:
                continue
            path, prev, cur, length = [start], start, nxt, core[start][nxt]['weight']
            seen.add((start, nxt))
            while cur not in junctions:
                path.append(cur)
                a, b = list(core.neighbors(cur))
                step = b if a == prev else a
                seen.add((cur, step))
                length += core[cur][step]['weight']
                prev, cur = cur, step
            seen.add((cur, prev))
            path.append(cur)
            out.append((start, cur, length, path))
    for component in nx.connected_components(core):
        if not component & junctions:  # isolated ring: every node has degree 2
            sub = core.subgraph(component)
            out.append((None, None, sum(d['weight'] for *_, d in sub.edges(data=True)), list(component)))
    return out


def loops(graph):
    """Approximate minimum cycle basis: shortest cycle through each chain, greedy GF(2) independent set."""
    core = nx.k_core(graph, 2)
    if core.number_of_edges() == 0:
        return []
    parts = chains(core)
    rank = core.number_of_edges() - core.number_of_nodes() + nx.number_connected_components(core)
    adjacency = {}
    for c, (u, v, length, _) in enumerate(parts):
        if u is None:
            continue
        adjacency.setdefault(u, []).append((v, length, c))
        adjacency.setdefault(v, []).append((u, length, c))
    candidates = []
    for c, (u, v, length, nodes) in enumerate(parts):
        if u is None:
            candidates.append((length, 1 << c, nodes))
            continue
        dist, back, heap = {u: 0.}, {}, [(0., u)]
        while heap:
            d, x = heapq.heappop(heap)
            if x == v or d > dist.get(x, np.inf):
                if x == v:
                    break
                continue
            for y, w, e in adjacency[x]:
                if e != c and d + w < dist.get(y, np.inf):
                    dist[y], back[y] = d + w, (x, e)
                    heapq.heappush(heap, (d + w, y))
        if v not in dist:
            continue
        bits, x, members = 1 << c, v, list(nodes)
        while x != u:
            x, e = back[x]
            bits |= 1 << e
            members += parts[e][3]
        candidates.append((length + dist[v], bits, members))
    candidates.sort(key=lambda t: t[0])
    basis, chosen = {}, []
    for length, bits, members in candidates:
        reduced = bits
        while reduced:
            top = reduced.bit_length() - 1
            if top not in basis:
                basis[top] = reduced
                chosen.append((length, set(members)))
                break
            reduced ^= basis[top]
        if len(chosen) == rank:
            break
    return chosen


def stats_subject(subject_id, row):
    subject = load_subject(row)
    g, mask, affine = subject['graph'], subject['mask'], subject['affine']
    pos, radius, graph = g['pos'], g['radius'], g['graph']
    index_of = {n: i for i, n in enumerate(g['nodes'])}
    distance = MaskDistance(mask, affine)
    inside = inside_mask(mask, affine, pos)
    d = distance.distance(pos)
    sdf_image = nib.load(row.SDF)
    sdf_at = map_coordinates(np.asarray(sdf_image.dataobj, dtype=np.float32), world_to_voxel(sdf_image.affine, pos).T,
                             order=1, mode='nearest')
    out = {'subject_id': subject_id, 'nodes': len(pos), 'inside_all': inside.mean()}
    for label, lo, hi in RADIUS_BINS:
        sel = (radius > lo) & (radius <= hi)
        out[f'n_{label}'] = int(sel.sum())
        out[f'inside_{label}'] = float(inside[sel].mean()) if sel.any() else np.nan
        out[f'sdf_plus_r_median_{label}'] = float(np.median(sdf_at[sel] + radius[sel])) if sel.any() else np.nan
    floor_out = floor_of(radius) & ~inside
    out['dist_floor_outside_median'] = float(np.median(d[floor_out])) if floor_out.any() else np.nan
    out['dist_floor_outside_p95'] = float(np.quantile(d[floor_out], .95)) if floor_out.any() else np.nan
    out['frac_nodes_dist_gt_0p5mm'] = float((d > 0.5).mean())
    out['frac_nodes_dist_gt_1mm'] = float((d > 1.0).mean())

    # Mask components (26-connectivity).
    labels, n_mask = ndimage.label(mask, STRUCT26)
    sizes = np.bincount(labels.ravel())[1:]
    out.update({'mask_components': n_mask, 'mask_largest_frac': sizes.max() / sizes.sum(),
                'mask_comp_1vox': int((sizes == 1).sum()), 'mask_comp_2_9vox': int(((sizes >= 2) & (sizes < 10)).sum()),
                'mask_comp_10_99vox': int(((sizes >= 10) & (sizes < 100)).sum()), 'mask_comp_ge100vox': int((sizes >= 100).sum())})

    # Assign each node to a mask component: its own voxel, else the nearest mask voxel within SNAP_MM.
    vox = np.rint(world_to_voxel(affine, pos)).astype(int)
    vox_c = np.clip(vox, 0, np.asarray(mask.shape) - 1)
    comp = np.where(inside, labels[tuple(vox_c.T)], 0)
    near_idx = distance.tree.query(pos)[1]
    near_vox = np.rint(world_to_voxel(affine, distance.centres[near_idx])).astype(int)
    snapped = (~inside) & (d <= SNAP_MM)
    comp[snapped] = labels[tuple(near_vox[snapped].T)]
    out['nodes_unassigned_gt_snap'] = int((comp == 0).sum())

    # Graph components and their relation to mask components.
    graph_comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    sizes_g = np.array([len(c) for c in graph_comps])
    comp_len = np.array([sum(graph[a][b]['weight'] for a, b in graph.subgraph(c).edges) for c in graph_comps])
    out.update({'graph_components': len(graph_comps), 'graph_largest_frac_nodes': sizes_g[0] / sizes_g.sum(),
                'graph_largest_frac_length': comp_len.max() / max(comp_len.sum(), 1e-9),
                'graph_comp_1node': int((sizes_g == 1).sum()), 'graph_comp_2_4': int(((sizes_g >= 2) & (sizes_g < 5)).sum()),
                'graph_comp_5_19': int(((sizes_g >= 5) & (sizes_g < 20)).sum()), 'graph_comp_ge20': int((sizes_g >= 20).sum()),
                'graph_comp_length_median_mm': float(np.median(comp_len))})
    within_one, spanning, unassigned = 0, 0, 0
    graph_pieces_per_mask = {}
    for c in graph_comps:
        ids = set(comp[[index_of[n] for n in c]]) - {0}
        if not ids:
            unassigned += 1
        elif len(ids) == 1:
            within_one += 1
        else:
            spanning += 1
        for m in ids:
            graph_pieces_per_mask[m] = graph_pieces_per_mask.get(m, 0) + 1
    covered = np.zeros(n_mask + 1, bool)
    covered[list(graph_pieces_per_mask)] = True
    pieces = np.array(list(graph_pieces_per_mask.values()))
    out.update({'graph_comp_within_one_mask_comp': within_one, 'graph_comp_spanning_mask_comps': spanning,
                'graph_comp_unassigned': unassigned,
                'mask_comp_with_ge2_graph_comps': int((pieces >= 2).sum()),
                'graph_comps_in_largest_mask_comp': int(graph_pieces_per_mask.get(int(np.argmax(sizes)) + 1, 0)),
                'mask_comp_without_nodes': int((~covered[1:]).sum()),
                'mask_voxel_frac_in_comps_without_nodes': float(sizes[~covered[1:]].sum() / sizes.sum())})
    ea, eb = g['edges'][:, 0], g['edges'][:, 1]
    cross = (comp[ea] > 0) & (comp[eb] > 0) & (comp[ea] != comp[eb])
    out['edges_crossing_mask_comps'] = int(cross.sum())
    out['edges_with_endpoint_outside_mask'] = int((~inside[ea] | ~inside[eb]).sum())

    # Loops.
    loop_rows = []
    for length, members in loops(graph):
        idx = [index_of[n] for n in members]
        loop_rows.append({'subject_id': subject_id, 'length_mm': length, 'nodes': len(idx),
                          'max_radius_mm': float(radius[idx].max()), 'frac_nodes_inside': float(inside[idx].mean()),
                          'mask_components': len(set(comp[idx]) - {0})})
    lengths = np.array([r['length_mm'] for r in loop_rows])
    out.update({'cycle_rank': graph.number_of_edges() - graph.number_of_nodes() + len(graph_comps),
                'loops_found': len(loop_rows), 'loops_short': int((lengths < SHORT_LOOP_MM).sum()),
                'loops_long': int((lengths >= SHORT_LOOP_MM).sum()),
                'loop_length_median_mm': float(np.median(lengths)) if len(lengths) else np.nan})
    return out, loop_rows


# ---------------------------------------------------------------- qc

def qc_subject(subject_id, row, output, translation=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    subject = load_subject(row)
    mask, affine, pos, radius = subject['mask'], subject['affine'], subject['graph']['pos'], subject['graph']['radius']
    variants = [('current (world mm, inv sform)', pos)]
    if translation is not None and np.any(translation):
        variants.append((f'translated {np.round(translation, 2).tolist()} mm', pos + translation))
    floor = floor_of(radius)
    vox0 = world_to_voxel(affine, pos)
    out_floor = floor & ~inside_mask(mask, affine, pos)
    # Zoom centre: the 20x20-voxel window holding most outside floor nodes.
    xy = np.rint(vox0[out_floor, :2] / 10).astype(int)
    keys, counts = np.unique(xy, axis=0, return_counts=True)
    centres = [keys[i] * 10 for i in np.argsort(-counts)[:3]]
    fig, axes = plt.subplots(len(variants), 2, figsize=(12, 6 * len(variants)), squeeze=False)
    for r, (name, p) in enumerate(variants):
        vox = world_to_voxel(affine, p)
        inside = inside_mask(mask, affine, p)
        mip = mask.max(2).T
        for c, window in enumerate([None, centres[0]]):
            ax = axes[r, c]
            ax.imshow(mip, cmap='gray', origin='lower', interpolation='nearest',
                      extent=(-0.5, mask.shape[0] - 0.5, -0.5, mask.shape[1] - 0.5))
            ax.scatter(vox[inside, 0], vox[inside, 1], s=2 if window is None else 10, c='tab:green', label='inside')
            ax.scatter(vox[~inside, 0], vox[~inside, 1], s=2 if window is None else 10, c='tab:red', label='outside')
            if window is not None:
                ax.set_xlim(window[0] - 15, window[0] + 15)
                ax.set_ylim(window[1] - 15, window[1] + 15)
            ax.set_title(f'{subject_id} axial MIP, {name}\ninside {inside.mean():.3f}, floor {inside[floor].mean():.3f}')
            ax.legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    fig.savefig(output / f'{subject_id}_mip.png', dpi=110)
    plt.close(fig)

    # Single axial slices around outside floor nodes: nodes within half a slice of the plane.
    for n, centre in enumerate(centres):
        sel_nodes = out_floor & (np.abs(vox0[:, 0] - centre[0]) < 15) & (np.abs(vox0[:, 1] - centre[1]) < 15)
        if not sel_nodes.any():
            continue
        k = int(np.median(np.rint(vox0[sel_nodes, 2])))
        fig, axes = plt.subplots(1, len(variants) * 3, figsize=(6 * len(variants) * 3, 6), squeeze=False)
        for v, (name, p) in enumerate(variants):
            vox = world_to_voxel(affine, p)
            inside = inside_mask(mask, affine, p)
            for s, kk in enumerate((k - 1, k, k + 1)):
                ax = axes[0, v * 3 + s]
                ax.imshow(mask[:, :, kk].T, cmap='gray', origin='lower', interpolation='nearest',
                          extent=(-0.5, mask.shape[0] - 0.5, -0.5, mask.shape[1] - 0.5))
                near = np.abs(vox[:, 2] - kk) <= 0.5
                ax.scatter(vox[near & inside, 0], vox[near & inside, 1], s=25, c='tab:green')
                ax.scatter(vox[near & ~inside, 0], vox[near & ~inside, 1], s=25, c='tab:red')
                ax.set_xlim(centre[0] - 12, centre[0] + 12)
                ax.set_ylim(centre[1] - 12, centre[1] + 12)
                ax.set_xticks(np.arange(centre[0] - 12, centre[0] + 13) - 0.5, minor=True)
                ax.set_yticks(np.arange(centre[1] - 12, centre[1] + 13) - 0.5, minor=True)
                ax.grid(which='minor', color='tab:blue', linewidth=0.2)
                ax.set_title(f'{name}\nslice k={kk} (nodes |k-kk|<=0.5)', fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f'{subject_id}_slices_{n}.png', dpi=100)
        plt.close(fig)


# ---------------------------------------------------------------- main

def _run(task):
    mode, subject_id, row = task
    try:
        if mode == 'stats':
            return stats_subject(subject_id, row)
        return conventions_subject(subject_id, row)
    except Exception as exc:  # report and continue; failures are listed in the summary
        return {'subject_id': subject_id, 'error': repr(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['headers', 'conventions', 'stats', 'qc'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--subjects', nargs='*', default=DEFAULT_SUBJECTS)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame = table()
    ids = list(frame.index) if args.all else args.subjects
    if args.mode == 'headers':
        pd.DataFrame([header_record(s, frame.loc[s]) for s in ids]).to_csv(out / 'headers.tsv', sep='\t', index=False)
        return
    if args.mode == 'qc':
        translations = {}
        conv = out / 'conventions.tsv'
        if conv.exists():
            c = pd.read_csv(conv, sep='\t')
            for s, t in c[c.convention.str.startswith('best global translation')][['subject_id', 'convention']].values:
                translations[s] = np.array(json.loads(t.split('translation ')[1].split(' mm')[0]))
        qc_dir = out / 'qc'
        qc_dir.mkdir(exist_ok=True)
        for s in ids:
            qc_subject(s, frame.loc[s], qc_dir, translations.get(s))
        return
    with Pool(args.workers) as pool:
        results = pool.map(_run, [(args.mode, s, frame.loc[s]) for s in ids], chunksize=1)
    errors = [r for r in results if isinstance(r, dict)]
    results = [r for r in results if not isinstance(r, dict)]
    if args.mode == 'conventions':
        pd.DataFrame([x for r in results for x in r[0]]).to_csv(out / 'conventions.tsv', sep='\t', index=False)
        pd.DataFrame([x for r in results for x in r[1]]).to_csv(out / 'resampled_grids.tsv', sep='\t', index=False)
        pd.DataFrame([x for r in results for x in r[2]]).to_csv(out / 'radius_units.tsv', sep='\t', index=False)
    else:
        stats = pd.DataFrame([r[0] for r in results])
        stats.to_csv(out / 'graph_mask_stats.tsv', sep='\t', index=False)
        pd.DataFrame([x for r in results for x in r[1]]).to_csv(out / 'loops.tsv', sep='\t', index=False)
        summary = {'subjects': len(stats), 'errors': errors,
                   'quantiles': stats.select_dtypes('number').quantile([0, .05, .5, .95, 1]).to_dict()}
        (out / 'stats_summary.json').write_text(json.dumps(summary, indent=1, default=float))
    if errors:
        (out / f'{args.mode}_errors.json').write_text(json.dumps(errors, indent=1))


if __name__ == '__main__':
    main()
