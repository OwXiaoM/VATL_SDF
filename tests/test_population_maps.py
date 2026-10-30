"""Phase 3: population frequency/mean/std accumulate individual template-space decodings correctly."""
import tempfile
import unittest

import numpy as np
import torch

from tests.test_pipeline import tiny_config
from vatl_sdf import CHANNELS
from vatl_sdf.config import condition_vector
from vatl_sdf.dataset import read_metadata
from vatl_sdf.inference import predict_field, weighted_latent, world_grid
from vatl_sdf.model import SDFModel
from vatl_sdf.population_maps import accumulate


class PopulationMapTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.c = tiny_config(self.temp.name)
        self.c['atlas'].update(spacing_mm=[1, 1, 1], gaussian_span_years=20.)
        torch.manual_seed(3)
        self.model = SDFModel(self.c, 2).eval()
        self.frame = read_metadata(self.c)
        self.coords, _, _ = world_grid(self.c)

    def tearDown(self):
        self.temp.cleanup()

    def test_matches_explicit_weighted_statistics(self):
        frequency, mean, std, info = accumulate(self.model, self.frame, self.c, 20.5, self.coords)
        _, weights = weighted_latent(self.model, self.frame, self.c, 20.5)
        weights = weights.numpy() / weights.numpy().sum()
        fields = []
        for i, row in self.frame.iterrows():
            fields.append(predict_field(self.model, self.coords, condition_vector(self.c, row.scan_age, row.sex), self.c,
                                        latent_override=self.model.latents[i:i + 1])[:, CHANNELS['vessel_sdf_normalized']] * 5)
        fields = np.array(fields, dtype=np.float64)
        np.testing.assert_allclose(frequency, (weights[:, None] * (fields < 0)).sum(0), atol=1e-6)
        np.testing.assert_allclose(mean, (weights[:, None] * fields).sum(0), atol=1e-5)
        np.testing.assert_allclose(std, np.sqrt((weights[:, None] * (fields - mean) ** 2).sum(0)), atol=1e-4)
        self.assertEqual(info['subjects_used'], 2)
        self.assertTrue(((frequency >= 0) & (frequency <= 1 + 1e-9)).all())

    def test_identical_subjects_have_zero_variability(self):
        with torch.no_grad():
            self.model.latents[1] = self.model.latents[0]
        self.frame.loc[1, ['scan_age', 'sex']] = self.frame.loc[0, ['scan_age', 'sex']].values
        frequency, mean, std, _ = accumulate(self.model, self.frame, self.c, 20., self.coords)
        self.assertLess(std.max(), 1e-3)
        self.assertTrue(np.isin(np.round(frequency, 6), [0, 1]).all())


if __name__ == '__main__':
    unittest.main()
