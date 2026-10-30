"""Rigid transform direction: subject decoding at u == template decoding at rigid_transform(u, t_i)."""
import tempfile
import unittest

import torch

from tests.test_pipeline import tiny_config
from vatl_sdf.model import SDFModel, rigid_transform


class TemplateMappingTest(unittest.TestCase):
    def test_subject_to_template_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = tiny_config(tmp)
        torch.manual_seed(0)
        model = SDFModel(c, 2).eval()
        with torch.no_grad():
            model.transforms[1] = torch.tensor([0.2, -0.1, 0.3, 0.5, -0.4, 0.2])
        u = torch.rand(50, 3) * 1.6 - 0.8
        cond = torch.zeros(50, 2)
        with torch.no_grad():
            subject, _ = model(u, torch.ones(50, dtype=torch.long), cond)
            template_points = rigid_transform(u, model.transforms[1].expand(50, -1), model.half_bbox)
            template, _ = model(template_points, torch.zeros(50, dtype=torch.long), cond,
                                latent_override=model.latents[1:2], apply_transform=False)
            wrong, _ = model(u, torch.zeros(50, dtype=torch.long), cond, latent_override=model.latents[1:2],
                             apply_transform=False)
        torch.testing.assert_close(subject, template, atol=1e-6, rtol=1e-5)
        self.assertGreater((subject - wrong).abs().max().item(), 1e-4)


if __name__ == '__main__':
    unittest.main()
