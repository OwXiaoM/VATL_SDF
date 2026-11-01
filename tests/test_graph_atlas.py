"""Probabilistic vessel-graph atlas (CPU)."""
import unittest

import numpy as np
from scipy.spatial import cKDTree

from vatl_sdf.graph_atlas import edge_scores, presence, skeleton_graph


class SkeletonGraphTests(unittest.TestCase):
    def test_line(self):
        s = np.zeros((9, 3, 3), bool); s[1:8, 1, 1] = True
        nodes, edges = skeleton_graph(s)
        self.assertEqual(len(nodes), 2); self.assertEqual(len(edges), 1)
        self.assertEqual(len(edges[0][2]), 7)  # 2 end nodes + 5 interior voxels

    def test_t_junction(self):
        s = np.zeros((9, 9, 3), bool); s[1:8, 4, 1] = True; s[4, 5:8, 1] = True
        nodes, edges = skeleton_graph(s)
        self.assertEqual(len(edges), 3)
        self.assertEqual(len(nodes), 4)  # three end points and one junction cluster
        degree = np.bincount(np.array([(a, b) for a, b, _ in edges]).ravel(), minlength=len(nodes))
        self.assertEqual(sorted(degree.tolist()), [1, 1, 1, 3])

    def test_cycle_without_nodes(self):
        # 8-voxel ring, every voxel with exactly two 26-neighbours (a square ring has degree-3 voxels at corners)
        s = np.zeros((6, 6, 3), bool)
        for i, j in [(1, 2), (1, 3), (2, 4), (3, 4), (4, 3), (4, 2), (3, 1), (2, 1)]:
            s[i, j, 1] = True
        nodes, edges = skeleton_graph(s)
        self.assertEqual(len(nodes), 1); self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][0], edges[0][1])
        covered = len(edges[0][2]) - 2 + 1
        self.assertEqual(covered, s.sum())

    def test_empty(self):
        nodes, edges = skeleton_graph(np.zeros((3, 3, 3), bool))
        self.assertEqual(len(nodes), 0); self.assertEqual(edges, [])


class PresenceTests(unittest.TestCase):
    def test_presence_and_scores(self):
        edge_a = np.stack([np.arange(10.), np.zeros(10), np.zeros(10)], 1)
        edge_b = edge_a + [0, 20, 0]
        trees = [cKDTree(edge_a + [0, 1, 0]), cKDTree(edge_b[:3]), None]
        y = presence(trees, [edge_a, edge_b], 2.0, 0.5)
        np.testing.assert_array_equal(y, [[True, False], [False, False], [False, False]])
        perfect = edge_scores(np.array([1., 0.]), np.array([1., 0.]), 10)
        constant = edge_scores(np.array([.5, .5]), np.array([1., 0.]), 10)
        self.assertEqual(perfect['brier'], 0.); self.assertGreater(perfect['loglik_per_edge'], constant['loglik_per_edge'])
        self.assertEqual(perfect['weighted_recall'], 1.)


if __name__ == '__main__':
    unittest.main()
