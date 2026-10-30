"""Phase 5 round-cone rendering: exact capsule, convex hull of spheres, grid union and smooth min."""
import unittest

import numpy as np

from tests.test_pipeline import AFFINE
from vatl_sdf.graph_coords import voxel_to_world
from vatl_sdf.render_graph import render, round_cone_sdf


def segment_distance(p, a, b):
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * (b - a)), axis=1)


class RoundConeTest(unittest.TestCase):
    def setUp(self):
        self.p = np.random.default_rng(0).uniform(-4, 9, (2000, 3))
        self.a, self.b = np.array([0., 0, 0]), np.array([5., 1, 0])

    def test_equal_radii_is_capsule(self):
        np.testing.assert_allclose(round_cone_sdf(self.p, self.a, self.b, 1.2, 1.2),
                                   segment_distance(self.p, self.a, self.b) - 1.2, atol=1e-9)

    def test_tapered_cone_bounds_swept_spheres(self):
        t = np.linspace(0, 1, 2001)
        centres = self.a + t[:, None] * (self.b - self.a); radii = 1.0 + 0.3 * t
        swept = (np.linalg.norm(self.p[:, None] - centres[None], axis=2) - radii).min(1)
        cone = round_cone_sdf(self.p, self.a, self.b, 1.0, 1.3)
        self.assertTrue(np.all(cone <= swept + 1e-6))      # convex hull contains the swept spheres
        self.assertLess(np.max(swept - cone), 0.01)         # and differs little for a gentle taper
        # End spheres are exact far behind each end.
        far = np.array([[-6., 0, 0]])
        self.assertAlmostEqual(round_cone_sdf(far, self.a, self.b, 1.0, 1.3)[0], 5.0, places=9)

    def test_degenerate_sphere_inside_sphere(self):
        d = round_cone_sdf(self.p, self.a, np.array([0.5, 0, 0]), 2.0, 0.5)
        np.testing.assert_allclose(d, np.linalg.norm(self.p - self.a, axis=1) - 2.0, atol=1e-9)


class RenderTest(unittest.TestCase):
    def test_union_on_oblique_grid_and_smooth_min(self):
        shape = (24, 20, 16)
        world = voxel_to_world(AFFINE, np.indices(shape).reshape(3, -1).T)
        positions = voxel_to_world(AFFINE, np.array([[4, 10, 8], [12, 10, 8], [18, 16, 8.]]))
        radii = np.array([1.5, 1.2, 0.8]); edges = np.array([[0, 1], [1, 2]])
        field = render(world, shape, positions, radii, edges, margin_mm=20)
        exact = np.minimum(round_cone_sdf(world, positions[0], positions[1], 1.5, 1.2),
                           round_cone_sdf(world, positions[1], positions[2], 1.2, 0.8))
        np.testing.assert_allclose(field.ravel(), np.minimum(exact, 5.0), atol=1e-9)
        smooth = render(world, shape, positions, radii, edges, smooth_k_mm=0.3, margin_mm=20)
        self.assertTrue(np.all(smooth <= field + 1e-9))
        sharp = render(world, shape, positions, radii, edges, smooth_k_mm=1e-3, margin_mm=20)
        self.assertLess(np.abs(sharp - field).max(), 1e-2)
        # Restricted boxes: far from every edge the field is the far value.
        self.assertEqual(render(world, shape, positions, radii, edges, margin_mm=0.5).max(), 5.0)


if __name__ == '__main__':
    unittest.main()
