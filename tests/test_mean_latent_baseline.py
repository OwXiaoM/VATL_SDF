"""Mean-latent baseline: latent construction, fit tables, paired statistics, averaged forward pass (CPU)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from analysis.generalization.mean_latent_baseline import VARIANTS, baseline_specs, load_fit_tables, paired_differences
from analysis.phase0_graph.diagnose_checkpoint import averaged_field


class Specs(unittest.TestCase):
    def setUp(self):
        n = 120
        self.ages = np.linspace(20, 80, n); self.sexes = np.where(np.arange(n) % 2, 'F', 'M')
        self.frame = pd.DataFrame({'scan_age': self.ages, 'sex': self.sexes})
        self.latents = torch.arange(n, dtype=torch.float32)[:, None, None, None, None].expand(n, 2, 1, 1, 1).clone()
        self.config = {'atlas': {'gaussian_span_years': 99., 'max_subjects': 1}}
        self.specs = baseline_specs(self.latents, self.frame, self.config, 50., 'F', 24)

    def test_all_variants_defined(self):
        self.assertEqual(set(self.specs), set(VARIANTS))

    def test_primary_is_cinema(self):
        _, _, w = self.specs['cinema_single']
        self.assertAlmostEqual(w.sum(), 1., places=5); self.assertLessEqual(int((w > 0).sum()), 100)
        self.assertGreater(w[self.sexes == 'M'].sum(), 0); self.assertGreater(w[self.sexes == 'F'].sum(), 0)
        expected = np.exp(-(self.ages - 50) ** 2 / (2 * 2.5 ** 2)); expected[np.argsort(-expected)[100:]] = 0
        np.testing.assert_allclose(w, expected / expected.sum(), atol=1e-6)
        self.assertEqual(self.config['atlas']['max_subjects'], 1)  # caller's config untouched

    def test_secondary_is_same_sex(self):
        _, _, w = self.specs['samesex_single']
        self.assertEqual(w[self.sexes == 'M'].sum(), 0); self.assertAlmostEqual(w.sum(), 1.)

    def test_latent_is_the_weighted_mean(self):
        for name in ('cinema_single', 'samesex_single'):
            latent, _, w = self.specs[name]
            self.assertAlmostEqual(float(latent[0, 0]), float(w @ np.arange(120.)), places=3)

    def test_age24_shares_the_latent_and_has_24_ages(self):
        for single, averaged in (('cinema_single', 'cinema_age24'), ('samesex_single', 'samesex_age24')):
            self.assertTrue(torch.equal(self.specs[single][0], self.specs[averaged][0]))
            self.assertEqual(len(self.specs[averaged][1]), 24); self.assertIsNone(self.specs[single][1])

    def test_fitted_and_zero(self):
        self.assertIsNone(self.specs['fitted'][0]); self.assertEqual(float(self.specs['zero'][0].abs().sum()), 0.)


class Tables(unittest.TestCase):
    def test_concatenates_in_order_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            for name, ids, value in (('a', ['s1', 's2'], 1.), ('b', ['s3'], 2.), ('c', ['s1'], 3.)):
                torch.save({'subject_ids': ids, 'model_subject_tables': {'latents': torch.full((len(ids), 2), value)}}, Path(d) / f'{name}.pt')
            ids, tables = load_fit_tables([Path(d) / 'a.pt', Path(d) / 'b.pt'])
            self.assertEqual(ids, ['s1', 's2', 's3']); self.assertEqual(tables['latents'][:, 0].tolist(), [1., 1., 2.])
            with self.assertRaises(ValueError):
                load_fit_tables([Path(d) / 'a.pt', Path(d) / 'c.pt'])


class Paired(unittest.TestCase):
    def test_differences_match_by_subject(self):
        table = pd.DataFrame({'subject_id': ['b', 'a', 'a', 'b', 'c'], 'variant': ['fitted', 'fitted', 'zero', 'zero', 'zero'],
                              'dice': [0.8, 0.7, 0.2, 0.5, 0.9]})
        got = paired_differences(table, 'zero', ('dice',))
        self.assertEqual(got['subjects'], 2); self.assertAlmostEqual(got['dice']['median'], 0.4); self.assertEqual(got['dice']['fitted_higher'], 2)


class AveragedField(unittest.TestCase):
    def fake(self, model, coords, condition, config, subject_index=None):
        return np.full((len(coords), 1), condition, np.float32)

    def test_single_condition_is_the_plain_forward_pass(self):
        with mock.patch('analysis.phase0_graph.diagnose_checkpoint.predict_field', self.fake):
            got = averaged_field(None, np.zeros((3, 3)), [0.1], {}, 0)
        self.assertEqual(got.dtype, np.float32); np.testing.assert_array_equal(got, np.full((3, 1), 0.1, np.float32))

    def test_mean_over_conditions(self):
        with mock.patch('analysis.phase0_graph.diagnose_checkpoint.predict_field', self.fake):
            np.testing.assert_allclose(averaged_field(None, np.zeros((2, 3)), [1., 3.], {}, 0), 2.)


if __name__ == '__main__':
    unittest.main()
