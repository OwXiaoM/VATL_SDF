"""Implicit graph fields (Phase 4): targets derived from the mask-supported graph only.

Fields at a query point x (world mm):
  centerline_distance  distance to the nearest supported centreline segment (clamped at truncation)
  radius               local vessel radius at that nearest centreline point, taken from the REFERENCE
                       SDF (-SDF_ref(c), floored at 0.1 mm), not from the quantized graph radius
  tangent              unit direction of that segment (sign-invariant loss)
  junction             exp(-d_j^2 / (2 sigma^2)), d_j = distance to the nearest junction (degree >= 3
                       in the supported subgraph)
Weights: 0 near spurious structure (points whose nearest centreline sample lies on a short loop
(< short_loop_mm) or within exclusion_mm of an unsupported node); radius and tangent additionally
only within band: distance <= radius + band_mm.
"""
import networkx as nx
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from .graph_coords import world_to_voxel

FIELDS = {'centerline_distance': 1, 'radius': 1, 'tangent': 3, 'junction': 1}


def short_loop_edges(edges, positions, supported_edges, short_loop_mm):
    """Indices of supported edges lying on a cycle shorter than short_loop_mm (shortest cycle through the edge)."""
    graph = nx.Graph()
    for e in np.flatnonzero(supported_edges):
        a, b = edges[e]
        graph.add_edge(int(a), int(b), weight=float(np.linalg.norm(positions[a] - positions[b])), index=int(e))
    core = nx.k_core(graph, 2).copy()  # only edges of the 2-core can lie on a cycle
    short = set()
    for a, b, data in list(core.edges(data=True)):
        core.remove_edge(a, b)
        cutoff = short_loop_mm - data['weight']
        if cutoff > 0:
            reach = nx.single_source_dijkstra_path_length(core, a, cutoff=cutoff, weight='weight')
            if b in reach and reach[b] + data['weight'] < short_loop_mm:
                short.add(data['index'])
        core.add_edge(a, b, **data)
    return np.array(sorted(short), dtype=np.int64)


def build_centerline(subject, settings, step_mm=0.2):
    """Dense samples along supported edges (snapped endpoints) with radius, tangent and flags."""
    g = subject['graph']
    edges = g['edges'][g['edge_supported']] if len(g['edges']) else np.zeros((0, 2), np.int64)
    edge_ids = np.flatnonzero(g['edge_supported']) if len(g['edges']) else np.zeros(0, np.int64)
    if not len(edges):
        return None
    a, b = g['snapped'][edges[:, 0]], g['snapped'][edges[:, 1]]
    length = np.linalg.norm(b - a, axis=1)
    counts = np.maximum(np.ceil(length / step_mm).astype(int), 1) + 1
    owner = np.repeat(np.arange(len(edges)), counts)
    t = np.concatenate([np.linspace(0, 1, n) for n in counts])[:, None]
    points = a[owner] * (1 - t) + b[owner] * t
    tangent = (b - a) / np.maximum(length, 1e-6)[:, None]
    reference = map_coordinates(subject['sdf'], world_to_voxel(subject['affine'], points).T, order=1, mode='nearest')
    short = np.isin(edge_ids[owner], short_loop_edges(g['edges'], g['snapped'], g['edge_supported'], settings['short_loop_mm']))
    degree = np.bincount(edges.ravel(), minlength=len(g['pos']))
    junctions = g['snapped'][degree >= 3]
    unsupported = g['pos'][~g['supported']]
    return {'tree': cKDTree(points), 'radius': np.maximum(-reference, 0.1), 'tangent': tangent[owner],
            'short_loop': short, 'junction_tree': cKDTree(junctions) if len(junctions) else None,
            'unsupported_tree': cKDTree(unsupported) if len(unsupported) else None}


def field_targets(centerline, world, settings, truncation_mm):
    """Targets (N, 6): distance_mm, radius_mm, tangent xyz, junction; weights (N, 4) per field."""
    n = len(world)
    targets = np.zeros((n, 6), np.float32)
    weights = np.zeros((n, 4), np.float32)
    if centerline is None:
        return targets, weights
    # Bounded queries: distance is clamped at the truncation, exclusion matters within exclusion_mm,
    # and the junction heatmap is < 3.4e-4 beyond 4 sigma. Points beyond a bound get inf.
    distance, nearest = centerline['tree'].query(world, distance_upper_bound=truncation_mm)
    near = np.isfinite(distance)
    nearest = np.where(near, nearest, 0)
    distance = np.where(near, distance, truncation_mm)
    radius = centerline['radius'][nearest]
    usable = ~(centerline['short_loop'][nearest] & near)
    if centerline['unsupported_tree'] is not None:
        usable &= ~np.isfinite(centerline['unsupported_tree'].query(world, distance_upper_bound=settings['exclusion_mm'])[0])
    band = near & (distance <= radius + settings['band_mm'])
    targets[:, 0] = distance
    targets[:, 1] = np.where(band, radius, 0)
    targets[:, 2:5] = np.where(band[:, None], centerline['tangent'][nearest], 0)
    if centerline['junction_tree'] is not None:
        sigma = settings['junction_sigma_mm']
        dj = centerline['junction_tree'].query(world, distance_upper_bound=4 * sigma)[0]
        targets[:, 5] = np.where(np.isfinite(dj), np.exp(-np.minimum(dj, 4 * sigma) ** 2 / (2 * sigma ** 2)), 0)
    weights[:, 0] = usable
    weights[:, 1] = usable & band
    weights[:, 2] = usable & band
    weights[:, 3] = usable
    return targets, weights
