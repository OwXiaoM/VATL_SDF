"""Coordinate convention shared by vessel graphs, masks and SDF volumes.

Verified in Phase 0a (CONTEXT.md; job 46230, 552 subjects): node["pos"] is world mm
in the frame of the segmentation sform (mask, SDF and MRA share shape and sform;
qform agrees within 1.5e-5 mm). World -> voxel is inv(affine) of the segmentation;
integer voxel indices are voxel centres, so the voxel containing a point is
round(inv(affine) @ pos). No flip, permutation, half-voxel shift or index
interpretation scores better. node["radius"] is mm (EDT on the 0.8 mm isotropic
grid the graphs were extracted on: values 0.8 * sqrt(k)), so RADIUS_SCALE_TO_MM = 1.
Normalized coords follow dataset.py and the reference loader:
(world - affine @ (shape / 2)) / (world_bbox_mm / 2).
The segmentation file is multi-label; the vessel class is label 1 (sdf < 0 == label 1).

Thin vessels: graph nodes are sub-voxel points extracted on a 0.8 mm regrid, while the
mask grid is 0.9375 x 0.9375 x 0.8 mm. About 11% of floor-radius nodes fall in a
background voxel next to a one-voxel-thick vessel, yet 99.8% of all nodes lie within
0.4 mm (half a graph voxel) of the voxelized mask (MaskDistance.distance).
"""
import pickle

import networkx as nx
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

VESSEL_LABEL = 1
RADIUS_SCALE_TO_MM = 1.0


def load_graph(path):
    """Return node ids, world positions (N, 3), radii in mm (N,), edge index (E, 2), graph."""
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    graph = payload['graph'] if isinstance(payload, dict) else payload
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        raise ValueError(f'{path}: expected a non-empty networkx graph')
    nodes = list(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    pos = np.array([graph.nodes[n]['pos'] for n in nodes], dtype=np.float64)
    radius = np.array([graph.nodes[n]['radius'] for n in nodes], dtype=np.float64) * RADIUS_SCALE_TO_MM
    edges = np.array([(index[a], index[b]) for a, b in graph.edges], dtype=np.int64).reshape(-1, 2)
    return {'nodes': nodes, 'pos': pos, 'radius': radius, 'edges': edges, 'graph': graph}


def world_to_voxel(affine, world):
    return nib.affines.apply_affine(np.linalg.inv(affine), np.asarray(world, dtype=np.float64))


def voxel_to_world(affine, voxel):
    return nib.affines.apply_affine(affine, np.asarray(voxel, dtype=np.float64))


def nearest_voxel(affine, world, shape):
    """Index of the voxel containing each point and whether it lies in the grid."""
    index = np.rint(world_to_voxel(affine, world)).astype(np.int64)
    valid = ((index >= 0) & (index < np.asarray(shape))).all(1)
    return index, valid


def normalize_world(world, affine, shape, world_bbox_mm):
    centre = voxel_to_world(affine, np.asarray(shape, dtype=np.float64) / 2.)
    return (np.asarray(world, dtype=np.float64) - centre) / (np.asarray(world_bbox_mm, dtype=np.float64) / 2.)


def vessel_mask(label_image):
    return np.asarray(label_image.dataobj) == VESSEL_LABEL


def inside_mask(mask, affine, world):
    """Nearest-voxel lookup; points outside the grid count as outside."""
    index, valid = nearest_voxel(affine, world, mask.shape)
    inside = np.zeros(len(index), bool)
    i = index[valid]
    inside[valid] = mask[i[:, 0], i[:, 1], i[:, 2]]
    return inside


class MaskDistance:
    """Exact distances from arbitrary points to the voxelized mask (union of voxel boxes).

    distance(world) is 0 iff the containing voxel is in the mask, so it agrees with
    inside_mask. centre_distance is the distance to the nearest mask voxel centre.
    """

    def __init__(self, mask, affine, neighbours=16):
        self.affine = np.asarray(affine, dtype=np.float64)
        self.linear = self.affine[:3, :3]
        self.inverse = np.linalg.inv(self.linear)
        self.centres = voxel_to_world(self.affine, np.argwhere(mask))
        self.tree = cKDTree(self.centres)
        self.k = min(neighbours, len(self.centres))

    def centre_distance(self, world):
        return self.tree.query(world)[0]

    def distance(self, world):
        world = np.asarray(world, dtype=np.float64)
        _, idx = self.tree.query(world, k=self.k)
        idx = idx.reshape(len(world), -1)
        offset = (world[:, None, :] - self.centres[idx]) @ self.inverse.T
        excess = np.sign(offset) * np.clip(np.abs(offset) - 0.5, 0, None)
        return np.linalg.norm(excess @ self.linear.T, axis=-1).min(1)
