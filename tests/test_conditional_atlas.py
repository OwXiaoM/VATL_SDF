"""Conditional atlas weights, weighted maps and graph helpers (CPU)."""
import unittest

import numpy as np

from vatl_sdf.conditional_atlas import cohort_tag, cohort_weights, edge_standard_error, paint_edges, weighted_map, weighted_occupancy
from vatl_sdf.conditional_atlas_smooth import soft_vessel, tube_volume, weighted_age_quantiles
from vatl_sdf.deviation import ess
from vatl_sdf.graph_atlas import build_population


class Weights(unittest.TestCase):
    ages = np.array([20., 25., 50., 80., 21.]); sexes = np.array(['M', 'F', 'M', 'F', 'M'])

    def test_sex_matched_and_normalized(self):
        w = cohort_weights(self.ages, self.sexes, 20, 'M', 5.)
        self.assertAlmostEqual(w.sum(), 1.)
        self.assertTrue((w[self.sexes == 'F'] == 0).all())
        self.assertGreater(w[0], w[4]); self.assertGreater(w[4], w[2])

    def test_effective_age_is_pulled_inward_at_the_range_end(self):
        w = cohort_weights(self.ages, self.sexes, 15, 'M', 50.)
        self.assertGreater(w @ self.ages, 15)
        self.assertLessEqual(ess(w), 3)

    def test_empty_cohort_raises(self):
        with self.assertRaises(ValueError):
            cohort_weights(self.ages, self.sexes, 20, 'X', 5.)

    def test_tag(self):
        self.assertEqual(cohort_tag(50.0, 'F'), 'age50_F')


class Maps(unittest.TestCase):
    def test_weighted_occupancy_matches_dense_mean(self):
        rng = np.random.default_rng(0)
        masks = rng.random((7, 53)) < 0.3  # 53 is not a multiple of 8: exercises the packed tail
        w = rng.random((3, 7)); w /= w.sum(1, keepdims=True)
        got = weighted_occupancy(np.packbits(masks, axis=1), w, 53, block=2)
        np.testing.assert_allclose(got, w @ masks, atol=1e-6)

    def test_weighted_map(self):
        got = weighted_map([np.array([0, 3]), np.array([3])], [0.25, 0.75], (2, 2))
        np.testing.assert_allclose(got, [[0.25, 0], [0, 1.]])

    def test_paint_edges_keeps_largest_and_ignores_outside(self):
        path = np.array([[1., 1., 1.], [2., 1., 1.], [99., 1., 1.]])
        got = paint_edges([path, path[:1]], [0.2, 0.9], (4, 4, 4), np.eye(4))
        self.assertAlmostEqual(float(got[1, 1, 1]), 0.9, places=6); self.assertAlmostEqual(float(got[2, 1, 1]), 0.2, places=6)
        self.assertEqual(int((got > 0).sum()), 2)

    def test_standard_error(self):
        np.testing.assert_allclose(edge_standard_error(np.array([0., 0.5]), 25.), [0, 0.1])


class Population(unittest.TestCase):
    def setUp(self):
        line = np.stack([np.arange(4., 20.), np.full(16, 10.), np.full(16, 10.)], 1)
        self.clouds = [line, line, line + [0, 12., 0]]
        self.args = ((24, 24, 24), np.eye(4), 2., 0.5, 0.5, 3)

    def test_uniform_weights_equal_default(self):
        plain = build_population(self.clouds, *self.args)
        weighted = build_population(self.clouds, *self.args, cmap=plain['cmap'], weights=np.full(3, 1 / 3))
        np.testing.assert_allclose(weighted['edge_probability'], plain['edge_probability'])

    def test_weights_change_edge_probability(self):
        plain = build_population(self.clouds, *self.args)
        weighted = build_population(self.clouds, *self.args, cmap=plain['cmap'], weights=np.array([0.5, 0.5, 0.]))
        self.assertTrue((weighted['edge_probability'] >= plain['edge_probability']).all())
        self.assertGreater(weighted['edge_probability'].max(), plain['edge_probability'].max() - 1e-12)

    def test_weights_need_weighted_map(self):
        with self.assertRaises(ValueError):
            build_population(self.clouds, *self.args, weights=np.full(3, 1 / 3))


class Smooth(unittest.TestCase):
    def test_quantiles_follow_the_weights(self):
        ages = np.array([40., 20., 30.]); weights = np.array([0., 0.5, 0.5])
        got = weighted_age_quantiles(ages, weights, 4)
        self.assertEqual(sorted(set(got)), [20., 30.]); self.assertEqual(len(got), 4)
        self.assertEqual(float((got == 20.).mean()), 0.5)

    def test_tube_is_wider_than_the_path_and_keeps_largest(self):
        path = np.array([[5., 5., 5.], [6., 5., 5.]])
        got = tube_volume([path, path], [0.3, 0.8], (12, 12, 12), np.eye(4), 1.5)
        self.assertGreater(int((got > 0).sum()), 2); self.assertAlmostEqual(float(got.max()), 0.8, places=6)
        self.assertAlmostEqual(float(got[got > 0].min()), 0.8, places=6)

    def test_soft_vessel(self):
        np.testing.assert_allclose(soft_vessel(np.array([0., 100., -100.])), [0.5, 0., 1.], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
