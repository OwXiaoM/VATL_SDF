"""Synthetic perturbations of large circle-of-Willis vessels for the sensitivity test (in memory only).

Targets come from the learned primary population graph (r3mm_tau0.1), without labelling vessels by hand:
  CoW centre  length-weighted centroid of the population edges whose mean centreline probability c_3 >= min_c and
              whose mean z is at or below the median of the edges' mean z (skull base half of the graph);
  targets     edges with p_edge >= min_p whose whole path lies within radius_mm of the centre.
Per subject (validation, mapped by its refitted transform), a target is perturbable when it is present (large-vessel
centreline within r of >= edge_cover of the edge path) and the matched subject segment (large-vessel graph edges
with >= 50% of their samples within r of the path) has median graph radius >= min_radius_mm.
Perturbations of the matched segment (one edge per copy), on the reference SDF with CSG operations, and on the graph:
  remove     vessel minus a tube of radius (r_local + 1 mm): sdf' = max(sdf, -tube); matched graph edges dropped;
  thin       remove, then add a tube of radius 0.5 r_local; interior node radii x 0.5;
             (thin and displace keep the original vessel within r + 1 mm of the attachment nodes so that the end caps
             of the removal tube do not disconnect the segment)
  displace   remove, then add the tube shifted by magnitude mm along a seeded direction perpendicular to the segment,
             tapered linearly to 0 within TAPER_MM of the segment's attachment nodes; interior nodes moved likewise.
  thin       optional factor (default 0.5, the legacy copies) in place of 0.5.
  stenosis   local narrowing: centre = dense segment sample farthest from the segment's chain end nodes; radius factor
             f(d) of the distance d to the centre (stenosis_profile: factor within STENOSIS_PLATEAU_MM, cosine taper
             to 1 over STENOSIS_TAPER_MM); remove a tube r + 1 mm around samples with f < 1, add a tube of radius f r
             over the segment samples within the removal reach (f = 1 refills the end caps); original vessel kept
             within r + 1 of attachments except inside the lumen (graph radius) of the narrowed part (nearest
             sample f < 1), so a zone reaching an attachment still narrows the segment; interior node radii x f(d).
Copies are addressed as 'perturb:<copy id>' and regenerated from the manifest in every process (vatl_sdf.sources);
nothing is written next to the data.
"""
import functools
import json
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from . import sources
from .graph_coords import edge_samples, voxel_to_world, world_to_voxel

PREFIX = 'perturb'
TAPER_MM = 3.0
STENOSIS_PLATEAU_MM = 2.0   # half-length of the plateau (4 mm at the full factor)
STENOSIS_TAPER_MM = 2.0     # cosine taper back to the original radius (8 mm affected length in total)
KIND_NAMES = ('remove', 'thin', 'displace', 'stenosis')
SDF_MARGIN_MM = 6.0
_COPIES = {}


# ---------------------------------------------------------------- target selection
def edge_centreline_probability(pop, cmap, affine):
    return np.array([ndi.map_coordinates(cmap, world_to_voxel(affine, pts).T, order=1, mode='constant', cval=0.).mean()
                     for pts in pop['edge_points_mm']])


def willis_centre(pop, cmap, affine, min_c=0.5):
    c = edge_centreline_probability(pop, cmap, affine)
    z_median = float(np.median([pts[:, 2].mean() for pts in pop['edge_points_mm']]))
    chosen = [k for k, pts in enumerate(pop['edge_points_mm']) if c[k] >= min_c and pts[:, 2].mean() <= z_median]
    if not chosen:
        raise ValueError('No population edge qualifies for the circle-of-Willis centre')
    centre = np.vstack([pop['edge_points_mm'][k] for k in chosen]).mean(0)
    return centre, chosen, c, float(z_median)


def select_targets(pop, centre, radius_mm, min_p):
    return [k for k, pts in enumerate(pop['edge_points_mm'])
            if pop['edge_probability'][k] >= min_p and np.linalg.norm(pts - centre, axis=1).max() <= radius_mm]


def edge_point_samples(graph, step_mm=0.4):
    """Points along all graph edges with edge id and linearly interpolated graph radius."""
    pos, radius, edges = graph['pos'], np.asarray(graph['radius'], float), graph['edges']
    if not len(edges):
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0)
    points, eid = edge_samples(pos, edges, step_mm)
    a, b = pos[edges[eid, 0]], pos[edges[eid, 1]]
    length = np.linalg.norm(b - a, axis=1)
    t = np.where(length > 0, np.linalg.norm(points - a, axis=1) / np.maximum(length, 1e-9), 0.)
    return points, eid, radius[edges[eid, 0]] * (1 - t) + radius[edges[eid, 1]] * t


def subject_segments(graph, centre, A, t, pop, targets, r, cover, min_large=1.2, min_radius=1.5):
    """Per target: presence, matched large-vessel subject edges, their median graph radius and eligibility."""
    points, eid, rad = edge_point_samples(graph)
    edges, radius = graph['edges'], np.asarray(graph['radius'], float)
    large = (radius[edges[eid, 0]] >= min_large) & (radius[edges[eid, 1]] >= min_large) if len(points) else np.zeros(0, bool)
    template = (points - centre) @ np.asarray(A).T + np.asarray(t)
    tree = cKDTree(template[large]) if large.any() else None
    out = []
    for k in targets:
        path = pop['edge_points_mm'][k]
        frac = float((tree.query(path, distance_upper_bound=r)[0] <= r).mean()) if tree is not None else 0.
        near_path = cKDTree(path).query(template)[0] <= r if len(points) else np.zeros(0, bool)
        matched = []
        for e in np.unique(eid[large]):
            sel = eid == e
            if near_path[sel].mean() >= 0.5:
                matched.append(int(e))
        seg_r = float(np.median(rad[np.isin(eid, matched)])) if matched else 0.
        out.append({'population_edge': int(k), 'presence_fraction': frac, 'present': frac >= cover,
                    'matched_edges': matched, 'segment_radius_mm': seg_r,
                    'eligible': bool(frac >= cover and matched and seg_r >= min_radius)})
    return out


# ---------------------------------------------------------------- perturbations
def segment_geometry(graph, edge_ids, step_mm=0.2, min_large=1.2):
    """Dense samples of the segment, its nodes and its attachment nodes: segment nodes shared with unmatched LARGE
    edges (both endpoint radii >= min_large). Small side branches are not attachments (cut at their roots in the
    mask by the removal tube; their graph edges stay attached to the moved / thinned nodes)."""
    sub = {'pos': graph['pos'], 'radius': graph['radius'], 'edges': graph['edges'][edge_ids]}
    points, _, rad = edge_point_samples(sub, step_mm)
    seg_nodes = np.unique(graph['edges'][edge_ids])
    radius = np.asarray(graph['radius'], float)
    others = np.setdiff1d(np.arange(len(graph['edges'])), edge_ids)
    others = others[(radius[graph['edges'][others, 0]] >= min_large) & (radius[graph['edges'][others, 1]] >= min_large)] if len(others) else others
    boundary = np.intersect1d(seg_nodes, np.unique(graph['edges'][others])) if len(others) else np.zeros(0, int)
    return points, rad, seg_nodes, boundary


def perpendicular_direction(points, seed):
    centred = points - points.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0] if len(points) > 1 else np.array([1., 0, 0])
    v = np.random.default_rng(seed).normal(size=3)
    v -= (v @ axis) * axis
    return v / np.linalg.norm(v)


def taper(points, boundary_positions):
    if not len(boundary_positions):
        return np.ones(len(points))
    return np.clip(cKDTree(boundary_positions).query(points)[0] / TAPER_MM, 0., 1.)


def stenosis_profile(d, factor):
    """Radius factor at distance d (mm) from the stenosis centre."""
    u = np.clip((np.asarray(d, float) - STENOSIS_PLATEAU_MM) / STENOSIS_TAPER_MM, 0., 1.)
    return factor + (1. - factor) * 0.5 * (1. - np.cos(np.pi * u))


def stenosis_centre(graph, edge_ids, points):
    seg = graph['edges'][edge_ids]
    nodes, degree = np.unique(seg, return_counts=True)
    ends = nodes[degree == 1] if (degree == 1).any() else nodes
    return points[np.argmax(cKDTree(graph['pos'][ends]).query(points)[0])]


def perturb(sdf, affine, graph, edge_ids, kind, magnitude=0., direction=None, factor=None):
    """Returns (sdf', graph') for one perturbation of the segment made of graph edges `edge_ids`."""
    edge_ids = np.asarray(edge_ids, int)
    points, rad, seg_nodes, boundary = segment_geometry(graph, edge_ids)
    shifted, centre = None, None
    if kind == 'thin':
        factor = 0.5 if factor is None else factor
    elif kind == 'stenosis':
        centre = stenosis_centre(graph, edge_ids, points)
    if kind == 'displace':
        shift = magnitude * taper(points, graph['pos'][boundary])[:, None] * np.asarray(direction)
        shifted = points + shift
    reach = rad.max() + 1. + SDF_MARGIN_MM + (magnitude if kind == 'displace' else 0.)
    cloud = points if shifted is None else np.vstack([points, shifted])
    corners = world_to_voxel(affine, np.array([[x, y, z] for x in (cloud[:, 0].min() - reach, cloud[:, 0].max() + reach)
                                                for y in (cloud[:, 1].min() - reach, cloud[:, 1].max() + reach)
                                                for z in (cloud[:, 2].min() - reach, cloud[:, 2].max() + reach)]))
    lo = np.clip(np.floor(corners.min(0)).astype(int), 0, sdf.shape); hi = np.clip(np.ceil(corners.max(0)).astype(int) + 1, 0, sdf.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    voxels = np.indices(hi - lo).reshape(3, -1).T + lo
    world = voxel_to_world(affine, voxels)
    out = sdf.copy()
    if kind == 'stenosis':
        dc = np.linalg.norm(points - centre, axis=1)
        f = stenosis_profile(dc, factor)
        zone = f < 1.
        d, i = cKDTree(points[zone]).query(world)
        region = np.maximum(sdf[box].reshape(-1), -(d - (rad[zone][i] + 1.)))
        fill = dc <= STENOSIS_PLATEAU_MM + STENOSIS_TAPER_MM + rad.max() + 1.
        d2, i2 = cKDTree(points[fill]).query(world)
        region = np.minimum(region, d2 - (rad * f)[fill][i2])
    else:
        d, i = cKDTree(points).query(world)
        region = np.maximum(sdf[box].reshape(-1), -(d - (rad[i] + 1.)))
    if kind == 'thin':
        region = np.minimum(region, d - factor * rad[i])
    elif kind == 'displace':
        d2, i2 = cKDTree(shifted).query(world)
        region = np.minimum(region, d2 - rad[i2])
    elif kind not in ('remove', 'stenosis'):
        raise ValueError(kind)
    if kind in ('thin', 'displace', 'stenosis') and len(boundary):
        # keep the original vessel within (r + 1) of the attachment nodes: the removal tube's end caps would
        # otherwise cut a ~1 mm notch that disconnects the thinned / displaced segment (removal keeps the gap)
        db, ib = cKDTree(graph['pos'][boundary]).query(world)
        inside = db <= np.asarray(graph['radius'], float)[boundary][ib] + 1.
        if kind == 'stenosis':  # inside the lumen of the narrowed part (f < 1) the stenosis applies; elsewhere restored
            ds, js = cKDTree(points).query(world)
            inside &= (ds > rad[js]) | (f[js] >= 1.)
        region = np.where(inside, np.minimum(region, sdf[box].reshape(-1)), region)
    out[box] = region.reshape(hi - lo).astype(sdf.dtype)
    return out, perturb_graph(graph, edge_ids, kind, magnitude, direction, seg_nodes, boundary, factor, centre)


def perturb_graph(graph, edge_ids, kind, magnitude, direction, seg_nodes, boundary, factor=None, centre=None):
    pos, radius, edges = graph['pos'].copy(), np.asarray(graph['radius'], float).copy(), graph['edges'].copy()
    interior = np.setdiff1d(seg_nodes, boundary)
    if kind == 'remove':
        edges = np.delete(edges, edge_ids, axis=0)
    elif kind == 'thin':
        radius[interior] *= factor
    elif kind == 'stenosis':
        radius[interior] *= stenosis_profile(np.linalg.norm(pos[interior] - centre, axis=1), factor)
    elif kind == 'displace':
        pos[interior] += magnitude * taper(pos[interior], graph['pos'][boundary])[:, None] * np.asarray(direction)
    used = np.unique(edges) if len(edges) else np.zeros(0, int)
    keep = np.union1d(used, np.setdiff1d(np.arange(len(pos)), seg_nodes))  # drop only nodes orphaned by removal
    remap = -np.ones(len(pos), int); remap[keep] = np.arange(len(keep))
    pos, radius, edges = pos[keep], radius[keep], remap[edges]
    g = nx.Graph()
    for n in range(len(pos)):
        g.add_node(n, pos=pos[n], radius=radius[n])
    g.add_edges_from(map(tuple, edges))
    return {'nodes': list(range(len(pos))), 'pos': pos, 'radius': radius, 'edges': edges.reshape(-1, 2), 'graph': g}


# ---------------------------------------------------------------- virtual source
def parse_kinds(specs):
    """'remove', 'thin[:factor]', 'stenosis:factor', 'displace:mm' -> {kind, magnitude (mm), factor}."""
    out = []
    for spec in specs:
        kind, _, value = spec.partition(':')
        if kind not in KIND_NAMES or (kind in ('displace', 'stenosis') and not value):
            raise ValueError(f'Unknown perturbation {spec!r}')
        out.append({'kind': kind, 'magnitude': float(value) if kind == 'displace' else 0.,
                    'factor': float(value) if kind in ('thin', 'stenosis') and value else None})
    return out


def copy_label(c):
    """Perturbation label of a copy / kind: legacy '<kind><magnitude>' (remove0, thin0 = factor 0.5, displace3), else
    '<kind><factor>' (thin0.3, stenosis0.5)."""
    return f"{c['kind']}{c['factor']:g}" if c.get('factor') is not None else f"{c['kind']}{c['magnitude']:g}"


def configure(manifest_path, reset=True):
    manifest = json.loads(Path(manifest_path).read_text())
    if reset:
        _COPIES.clear()
    _COPIES.update({c['copy_id']: c for c in manifest['copies']})
    provide.cache_clear()
    sources.register(PREFIX, provide)
    return manifest


@functools.lru_cache(maxsize=8)
def provide(path):
    from .graph_coords import load_graph
    spec = _COPIES[path.split(':', 1)[1]]
    image = nib.load(spec['source_sdf'])
    sdf = np.asarray(image.dataobj, dtype=np.float32)
    sdf = sdf[..., 0] if sdf.ndim == 4 else sdf
    graph = load_graph(spec['source_graph'])
    new_sdf, new_graph = perturb(sdf, image.affine, graph, spec['matched_edges'], spec['kind'], spec['magnitude'], spec['direction'],
                                 spec.get('factor'))
    header = image.header.copy(); header.set_data_shape(new_sdf.shape)
    return nib.Nifti1Image(new_sdf, image.affine, header), new_graph


def copy_rows(manifest, kinds=None):
    """Rows (subject_id = copy id, SDF/Graph = virtual paths) for dataset-style code."""
    import pandas as pd
    return pd.DataFrame([{'subject_id': c['copy_id'], 'scan_age': c['scan_age'], 'sex': c['sex'], 'MRA': 'unused_never_read',
                          'SDF': f'{PREFIX}:{c["copy_id"]}', 'Graph': f'{PREFIX}:{c["copy_id"]}'}
                         for c in manifest['copies'] if kinds is None or c['kind'] in kinds])
