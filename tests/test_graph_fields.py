"""Phase 4 implicit graph fields: targets, exclusions, sign-invariant tangent loss, CPU training."""
import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from tests.test_pipeline import tiny_config
from vatl_sdf.config import field_names
from vatl_sdf.graph_fields import build_centerline, field_targets, short_loop_edges
from vatl_sdf.losses import field_losses

SETTINGS = {'band_mm': 1.0, 'junction_sigma_mm': 1.0, 'short_loop_mm': 5.0, 'exclusion_mm': 1.0}


def synthetic_subject():
    """Isotropic 1 mm grid; vessel along x (y=z=10) of radius ~1.5 mm, a branch along +y from x=10,
    and a 3-node triangle (short spurious loop) far away; one unsupported node."""
    shape = (30, 30, 21)
    world = np.indices(shape).reshape(3, -1).T.astype(float)
    trunk = np.linalg.norm(world[:, 1:] - [10, 10], axis=1) - 1.5
    branch = np.where(world[:, 1] >= 10, np.linalg.norm(world[:, [0, 2]] - [10, 10], axis=1) - 1.5, np.inf)
    sdf = np.minimum(trunk, branch).reshape(shape).astype(np.float32)
    pos = np.array([[2, 10, 10], [10, 10, 10], [20, 10, 10], [10, 20, 10],   # Y: junction at node 1
                    [25, 25, 5], [26, 25, 5], [25.5, 26, 5],                  # short triangle loop
                    [2, 25, 18]], float)                                      # unsupported stray node
    edges = np.array([[0, 1], [1, 2], [1, 3], [4, 5], [5, 6], [6, 4]])
    supported = np.array([True] * 7 + [False])
    graph = {'pos': pos, 'snapped': pos.copy(), 'edges': edges, 'supported': supported,
             'edge_supported': np.ones(len(edges), bool), 'radius': np.full(8, 1.5)}
    return {'sdf': sdf, 'affine': np.eye(4), 'graph': graph}


class FieldTargetTest(unittest.TestCase):
    def setUp(self):
        self.subject = synthetic_subject()
        self.centerline = build_centerline(self.subject, SETTINGS)

    def test_short_loop_detection(self):
        g = self.subject['graph']
        np.testing.assert_array_equal(short_loop_edges(g['edges'], g['pos'], g['edge_supported'], 5.0), [3, 4, 5])
        self.assertEqual(len(short_loop_edges(g['edges'], g['pos'], g['edge_supported'], 2.0)), 0)

    def test_targets_on_trunk_and_branch(self):
        world = np.array([[15, 10, 10], [15, 10.5, 10], [15, 13, 10], [10, 15, 10.8], [10, 10, 10], [4, 10, 10]], float)
        targets, weights = field_targets(self.centerline, world, SETTINGS, 5.)
        np.testing.assert_allclose(targets[:4, 0], [0, .5, 3, .8], atol=1e-6)
        # Radius = -reference SDF at the nearest centreline point (1.5 mm on the axis), not the graph radius.
        np.testing.assert_allclose(targets[[0, 1, 3], 1], 1.5, atol=1e-5)
        np.testing.assert_allclose(np.abs(targets[0, 2:5]), [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(np.abs(targets[3, 2:5]), [0, 1, 0], atol=1e-6)
        # Band: radius/tangent only within distance <= radius + 1 mm; distance/junction everywhere usable.
        np.testing.assert_array_equal(weights[:4, 1], [1, 1, 0, 1]); np.testing.assert_array_equal(weights[:4, 0], 1)
        self.assertAlmostEqual(float(targets[4, 5]), 1.0, places=6)            # at the junction
        self.assertAlmostEqual(float(targets[5, 5]), np.exp(-36 / 2), places=6)  # 6 mm away

    def test_exclusions_near_short_loop_and_unsupported_node(self):
        world = np.array([[25.3, 25.3, 5], [2, 24.5, 18], [15, 10, 10]], float)
        _, weights = field_targets(self.centerline, world, SETTINGS, 5.)
        np.testing.assert_array_equal(weights[:, 0], [0, 0, 1])

    def test_tangent_loss_is_sign_invariant(self):
        config = {'sdf': {'truncation_mm': 5.}, 'graph_fields': {'tangent': {'weight': 1.}}}
        targets = torch.zeros(2, 6); targets[:, 2] = 1
        weights = torch.ones(2, 4)
        for prediction in (torch.tensor([[2., 0, 0], [-3., 0, 0]]), torch.tensor([[0., 1, 0], [0, 0, -1.]])):
            parts = field_losses({'total': torch.tensor(0.)}, {'tangent': prediction}, targets, weights, config)
            expected = 0. if prediction[0, 0] else 1.
            self.assertAlmostEqual(parts['field_tangent_raw'].item(), expected, places=6)


class FieldTrainingTest(unittest.TestCase):
    def test_cpu_training_with_all_heads(self):
        from vatl_sdf.train import train
        with tempfile.TemporaryDirectory() as tmp:
            c = tiny_config(tmp)
            for name in ('centerline_distance', 'radius', 'tangent', 'junction'):
                c['graph_fields'][name]['enabled'] = True
            c['sampling']['sdf']['points'] = 64
            c['training'].update(max_steps=30, epochs=100, task_points_per_step=64)
            c['optimizer'].update(lr_model=1e-3, lr_latent=1e-3)
            torch.manual_seed(0)
            directory = train(c)
            rows = [json.loads(l) for l in (Path(directory) / 'losses.jsonl').read_text().splitlines()]
            state = torch.load(next(Path(directory).glob('checkpoint_step_*.pt')), map_location='cpu', weights_only=False)
        self.assertEqual(field_names(c), ['centerline_distance', 'radius', 'tangent', 'junction'])
        self.assertIn('field_heads.tangent.weight', state['model'])
        for name in field_names(c):
            self.assertTrue(all(np.isfinite(r[f'field_{name}_raw']) for r in rows))
        d0 = np.mean([r['field_centerline_distance_raw'] for r in rows[:6]])
        d1 = np.mean([r['field_centerline_distance_raw'] for r in rows[-6:]])
        self.assertLess(d1, d0)


if __name__ == '__main__':
    unittest.main()
