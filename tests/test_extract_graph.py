"""Phase 4 field-guided extraction on a synthetic straight vessel."""
import unittest

import numpy as np

from analysis.phase4.extract_graph import components_of, field_graph


class FieldGraphTest(unittest.TestCase):
    def setUp(self):
        self.shape = (10, 10, 10)
        idx = np.indices(self.shape).reshape(3, -1).T
        radial = np.linalg.norm(idx[:, 1:] - 5, axis=1)
        self.sdf = radial - 0.8                    # one-voxel-thick vessel along x at y = z = 5
        self.distance = (radial / 5.0)[:, None]    # normalized by the 5 mm truncation

    def run_with_tangent(self, tangent):
        fields = {'centerline_distance': self.distance, 'tangent': np.tile(tangent, (1000, 1)).astype(float),
                  'junction': np.full((1000, 1), -10.)}
        index, edges, junctions = field_graph(self.sdf, fields, self.shape, np.eye(4), d_max=0.6, cos_min=0.5)
        labels, n = components_of(index, self.shape, edges)
        return index, edges, n, junctions

    def test_tangent_aligned_steps_connect_the_vessel(self):
        index, edges, n, junctions = self.run_with_tangent([1, 0, 0])
        self.assertEqual(len(index), 10); self.assertEqual(len(edges), 9); self.assertEqual(n, 1)
        self.assertEqual(junctions, 0)
        _, _, flipped, _ = self.run_with_tangent([-1, 0, 0])  # sign-invariant
        self.assertEqual(flipped, 1)

    def test_misaligned_tangent_breaks_tracking(self):
        _, edges, n, _ = self.run_with_tangent([0, 1, 0])
        self.assertEqual(len(edges), 0); self.assertEqual(n, 10)


if __name__ == '__main__':
    unittest.main()
