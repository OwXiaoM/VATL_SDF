"""Graph/mask coordinate convention on a synthetic oblique, anisotropic image."""
import pickle
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np

from analysis.phase0_graph.correspondence import loops
from mra_sdf_only.graph_coords import (MaskDistance, RADIUS_SCALE_TO_MM, edge_samples, inside_mask, load_graph,
                                       mask_support, nearest_voxel, normalize_world, supported_edges, vessel_mask,
                                       voxel_to_world, world_to_voxel)

ANGLE = np.deg2rad(7)
ROTATION = np.array([[1, 0, 0], [0, np.cos(ANGLE), -np.sin(ANGLE)], [0, np.sin(ANGLE), np.cos(ANGLE)]])
AFFINE = np.eye(4)
AFFINE[:3, :3] = ROTATION @ np.diag([0.9375, 0.9375, 0.8])
AFFINE[:3, 3] = [-119.4, -111.9, 4.9]
SHAPE = (12, 10, 8)


class GraphCoordsTest(unittest.TestCase):
    def setUp(self):
        labels = np.zeros(SHAPE, np.uint8)
        labels[3:9, 4, 5] = 1    # one-voxel-thick vessel along i
        labels[:, :, 0] = 3      # a tissue label that must not count as vessel
        self.image = nib.Nifti1Image(labels, AFFINE)
        self.mask = vessel_mask(self.image)

    def test_round_trip_and_voxel_centres(self):
        index = np.array([[3, 4, 5], [0, 0, 0], [11, 9, 7]])
        world = voxel_to_world(AFFINE, index)
        np.testing.assert_allclose(world_to_voxel(AFFINE, world), index, atol=1e-9)
        np.testing.assert_allclose(world[1], AFFINE[:3, 3])  # integer index = voxel centre

    def test_nearest_voxel_rounds_within_half_voxel(self):
        offsets = np.array([[0.49, -0.49, 0.49], [-0.49, 0.49, -0.49]])
        world = voxel_to_world(AFFINE, np.array([5, 4, 5]) + offsets)
        index, valid = nearest_voxel(AFFINE, world, SHAPE)
        np.testing.assert_array_equal(index, [[5, 4, 5], [5, 4, 5]])
        self.assertTrue(valid.all())
        self.assertTrue(inside_mask(self.mask, AFFINE, world).all())

    def test_inside_uses_vessel_label_and_grid_bounds(self):
        self.assertEqual(self.mask.sum(), 6)
        world = voxel_to_world(AFFINE, np.array([[6, 4, 5], [6, 5, 5], [6, 4, 0], [-1, 4, 5], [6, 4.6, 5]]))
        np.testing.assert_array_equal(inside_mask(self.mask, AFFINE, world), [True, False, False, False, False])

    def test_distance_to_voxelized_mask_in_mm(self):
        distance = MaskDistance(self.mask, AFFINE)
        # 0.25 voxel beyond the vessel's j face = 0.25 * 0.9375 mm; zero inside a mask voxel.
        world = voxel_to_world(AFFINE, np.array([[6, 4.75, 5], [6, 4.3, 5.4], [6, 4, 6.5]]))
        np.testing.assert_allclose(distance.distance(world), [0.25 * 0.9375, 0, 0.8], atol=1e-9)
        np.testing.assert_allclose(distance.centre_distance(world[:1]), [0.75 * 0.9375], atol=1e-9)
        # Corner case: offsets beyond the box along two axes combine through the oblique affine.
        corner = voxel_to_world(AFFINE, np.array([[6, 4.9, 5.9]]))
        expected = np.linalg.norm(AFFINE[:3, :3] @ np.array([0, 0.4, 0.4]))
        np.testing.assert_allclose(distance.distance(corner), [expected], atol=1e-9)

    def test_normalization_matches_dataset_convention(self):
        centre = voxel_to_world(AFFINE, np.asarray(SHAPE) / 2.)
        bbox = np.array([240., 240., 120.])
        np.testing.assert_allclose(normalize_world(centre[None] + [[120, -120, 60]], AFFINE, SHAPE, bbox), [[1, -1, 1]])

    def test_load_graph_reads_world_mm_and_mm_radius(self):
        graph = nx.Graph()
        graph.add_node(7, pos=voxel_to_world(AFFINE, [3, 4, 5]), radius=np.float32(0.7999918))
        graph.add_node(9, pos=voxel_to_world(AFFINE, [4, 4, 5]), radius=np.float32(1.13))
        graph.add_edge(7, 9, weight=0.9375)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'g.pkl'
            with open(path, 'wb') as f:
                pickle.dump({'graph': graph, 'node_radius_map': {}, 'voxel_sizes': np.full(3, 0.7999918)}, f)
            loaded = load_graph(path)
        self.assertEqual(RADIUS_SCALE_TO_MM, 1.0)
        np.testing.assert_allclose(loaded['radius'], [0.7999918, 1.13], rtol=1e-6)
        np.testing.assert_array_equal(loaded['edges'], [[0, 1]])
        self.assertTrue(inside_mask(self.mask, AFFINE, loaded['pos']).all())


class MaskSupportTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros(SHAPE, bool)
        self.mask[3:9, 4, 5] = True   # vessel A
        self.mask[3:9, 7, 5] = True   # vessel B, parallel, two background voxels away
        voxels = np.array([[6, 4, 5], [6, 4.75, 5], [6, 4.9, 5], [6, 5.0, 5], [6, 7, 5], [3, 4, 5], [8, 4, 5]])
        self.world = voxel_to_world(AFFINE, voxels)

    def test_tolerance_and_snapping(self):
        supported, positions, d = mask_support(self.mask, AFFINE, self.world)
        # 0.25 and 0.4 voxel past the face (0.23 / 0.375 mm) are tolerated; 0.5 voxel (0.47 mm) is not.
        np.testing.assert_array_equal(supported, [True, True, True, False, True, True, True])
        np.testing.assert_allclose(positions[[1, 2]], voxel_to_world(AFFINE, [[6, 4, 5], [6, 4, 5]]), atol=1e-9)
        np.testing.assert_allclose(positions[[0, 3, 4]], self.world[[0, 3, 4]])
        # Hinge targets are never in background voxels.
        self.assertTrue(inside_mask(self.mask, AFFINE, positions[supported]).all())

    def test_strict_rule_has_no_tolerance_or_snapping(self):
        supported, positions, _ = mask_support(self.mask, AFFINE, self.world, strict=True)
        np.testing.assert_array_equal(supported, [True, False, False, False, True, True, True])
        np.testing.assert_allclose(positions, self.world)

    def test_edge_filter_drops_edges_leaving_the_mask(self):
        edges = np.array([[5, 6],   # along vessel A: kept
                          [1, 6],   # from a snapped node along A: kept
                          [0, 4],   # A -> B across background: dropped
                          [3, 0]])  # unsupported endpoint: dropped
        supported, positions, _ = mask_support(self.mask, AFFINE, self.world)
        np.testing.assert_array_equal(supported_edges(self.mask, AFFINE, positions, edges, supported),
                                      [True, True, False, False])
        strict, raw, _ = mask_support(self.mask, AFFINE, self.world, strict=True)
        np.testing.assert_array_equal(supported_edges(self.mask, AFFINE, raw, edges, strict, strict=True),
                                      [True, False, False, False])

    def test_edge_samples_spacing(self):
        points, edge_id = edge_samples(np.array([[0., 0, 0], [1., 0, 0]]), np.array([[0, 1]]), step_mm=0.4)
        np.testing.assert_allclose(points[:, 0], [0, 1 / 3, 2 / 3, 1])
        np.testing.assert_array_equal(edge_id, 0)


class LoopTest(unittest.TestCase):
    def test_short_and_long_loops_with_tree_branches(self):
        graph = nx.Graph()
        square = [(0, 1), (1, 2), (2, 3), (3, 0)]          # 4 mm loop
        ring = [(10 + i, 10 + (i + 1) % 12) for i in range(12)]  # 12 mm isolated ring
        tail = [(2, 20), (20, 21), (0, 30)]                   # tree branches, no loops
        graph.add_edges_from(square + ring + tail, weight=1.0)
        graph.add_edge(1, 3, weight=1.5)                      # chord: splits the square into two triangles
        lengths = sorted(round(length, 6) for length, _ in loops(graph))
        self.assertEqual(lengths, [3.5, 3.5, 12.0])
        rank = graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)
        self.assertEqual(len(lengths), rank)

    def test_tree_has_no_loops(self):
        self.assertEqual(loops(nx.path_graph(5)), [])


if __name__ == '__main__':
    unittest.main()
