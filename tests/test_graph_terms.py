"""Phase 2 graph supervision: analytic cylinder values/gradients, edge filtering, CPU training."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np
import torch

from tests.test_pipeline import AFFINE, tiny_config
from vatl_sdf.config import graph_terms
from vatl_sdf.dataset import load_subject, read_metadata, sample_graph
from vatl_sdf.graph_coords import MaskDistance, normalize_world, voxel_to_world, world_to_voxel
from vatl_sdf.losses import eikonal_loss, graph_term_loss

TAU = 5.
HALF = torch.tensor([120., 120., 60.])  # anisotropic bbox half-lengths, as in the real config


def cylinder_sdf_mm(world, radius):
    """Exact SDF of an infinite cylinder along x through the origin."""
    return np.linalg.norm(world[..., 1:], axis=-1) - radius


class CylinderLossTest(unittest.TestCase):
    def test_exact_cylinder_gives_zero_losses_and_correct_gradients(self):
        radius = 1.2
        axis = np.c_[np.linspace(-5, 5, 11), np.zeros(11), np.zeros(11)]
        exact = torch.tensor(cylinder_sdf_mm(axis, radius) / TAU, dtype=torch.float32)[:, None]
        ones = torch.ones_like(exact)
        target = torch.full_like(exact, -radius)
        for term in ('centerline_sdf', 'radius'):
            self.assertAlmostEqual(graph_term_loss(term, exact, target, ones, TAU).item(), 0., places=6)
        # Hinge with ceiling -epsilon: satisfied by the exact SDF (-1.2 mm < -0.5 mm).
        self.assertEqual(graph_term_loss('centerline_hinge', exact, torch.full_like(exact, -.5), ones, TAU).item(), 0.)
        # Perturb half the nodes outward by 2 mm: L1 gradient sign/n there; hinge active only where pred > ceiling.
        pred = target.clamp(-TAU, TAU) / TAU + torch.tensor([2. / TAU if i % 2 else 0. for i in range(11)])[:, None]
        pred.requires_grad_(True)
        graph_term_loss('centerline_sdf', pred, target, ones, TAU).backward()
        expected = torch.tensor([1. / 11 if i % 2 else 0. for i in range(11)])[:, None]
        # Unperturbed nodes sit exactly at the target (|0| has subgradient 0 in torch).
        torch.testing.assert_close(pred.grad, expected)
        pred.grad = None
        loss = graph_term_loss('centerline_hinge', pred, torch.full_like(exact, -.5), ones, TAU)
        self.assertAlmostEqual(loss.item(), 5 * (0.8 - (-.5)) / TAU / 11, places=6)  # 5 active nodes at +0.8 mm
        loss.backward()
        torch.testing.assert_close(pred.grad, expected)
        # Zero-weight rows (empty pools) contribute nothing.
        self.assertEqual(graph_term_loss('radius', pred, target, torch.zeros_like(exact), TAU).item(), 0.)

    def test_eikonal_physical_units_with_anisotropic_normalization(self):
        radius = 1.2
        world = torch.tensor(np.random.default_rng(0).uniform(-8, 8, (200, 3)), dtype=torch.float32)
        world[:, 1:] += torch.sign(world[:, 1:]) * 0.5  # keep away from the axis singularity
        u = (world / HALF).requires_grad_(True)
        s = ((u[:, 1:] * HALF[1:]).norm(dim=-1, keepdim=True) - radius) / TAU
        self.assertLess(eikonal_loss(s, u, HALF, TAU).item(), 1e-8)
        # Using normalized-coordinate gradients without the bbox/truncation conversion would be wrong.
        u2 = (world / HALF).requires_grad_(True)
        s2 = ((u2[:, 1:] * HALF[1:]).norm(dim=-1, keepdim=True) - radius) / TAU
        self.assertGreater(eikonal_loss(s2, u2, torch.ones(3), 1.).item(), 1.)
        band = torch.zeros(200, dtype=torch.bool)
        self.assertEqual(eikonal_loss(s2, u2, torch.ones(3), 1., band).item(), 0.)


def two_vessel_subject(directory, config):
    """Two parallel one-voxel vessels (j=2 and j=5) on the oblique grid; graph with one bridging edge."""
    labels = np.zeros((8, 8, 8), bool)
    labels[1:7, 2, 4] = True
    labels[1:7, 5, 4] = True
    from vatl_sdf.preprocess_sdf import signed_distance
    sdf = np.zeros((8, 8, 8), np.float32)
    # Oblique affine: EDT on the voxel grid with its spacings (columns are orthogonal here).
    sdf[:] = signed_distance(labels, AFFINE)
    image = nib.Nifti1Image(sdf, AFFINE); image.header.set_xyzt_units('mm')
    nib.save(image, Path(directory) / 'SDF0.nii.gz'); nib.save(image, Path(directory) / 'SDF1.nii.gz')
    graph = nx.Graph()
    voxels = [(1, 2, 4), (3, 2.9, 4), (6, 2, 4), (1, 5, 4), (6, 5, 4)]  # node 1: 0.4 voxel past the face, snapped
    for n, v in enumerate(voxels):
        graph.add_node(n, pos=voxel_to_world(AFFINE, np.array(v, float)), radius=np.float32(0.8 if n else 1.13))
    graph.add_edges_from([(0, 1), (1, 2), (3, 4), (2, 4)], weight=1.)  # (2, 4) bridges the background
    for i in (0, 1):
        with open(Path(directory) / f'Graph{i}.pkl', 'wb') as f:
            pickle.dump({'graph': graph}, f)
    return labels


class GraphSamplingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.c = tiny_config(self.temp.name)
        self.labels = two_vessel_subject(self.temp.name, self.c)
        for term in ('centerline_sdf', 'centerline_hinge', 'edge_hinge', 'radius'):
            self.c['graph'][term]['enabled'] = True
        self.c['sampling']['sdf']['points'] = 400

    def tearDown(self):
        self.temp.cleanup()

    def test_support_snapping_and_edge_filter(self):
        subject = load_subject(read_metadata(self.c).iloc[0], self.c)
        g = subject['graph']
        self.assertTrue(g['supported'].all())
        np.testing.assert_allclose(g['snapped'][1], voxel_to_world(AFFINE, np.array([3, 2, 4.])), atol=1e-6)
        # Edge order follows networkx insertion: (0,1), (1,2), (2,4), (3,4).
        edges = {tuple(sorted(e)): ok for e, ok in zip(g['edges'].tolist(), g['edge_supported'])}
        self.assertEqual(edges, {(0, 1): True, (1, 2): True, (3, 4): True, (2, 4): False})

    def test_sampled_targets_respect_the_reference(self):
        subject = load_subject(read_metadata(self.c).iloc[0], self.c)
        samples, report = sample_graph(subject, self.c, np.random.default_rng(0))
        self.assertEqual(set(report), set(graph_terms(self.c)))
        distance = MaskDistance(self.labels, AFFINE)
        shape, bbox = (8, 8, 8), self.c['data']['world_bbox_mm']
        centre = voxel_to_world(AFFINE, np.asarray(shape) / 2.)
        for term in graph_terms(self.c):
            world = samples[f'{term}_coords'] * (np.asarray(bbox) / 2.) + centre
            np.testing.assert_allclose(normalize_world(world, AFFINE, shape, bbox), samples[f'{term}_coords'], atol=1e-5)
            self.assertEqual(samples[f'{term}_coords'].shape, (60, 3))
            self.assertTrue((samples[f'{term}_weight'] == 1).all())
            if term == 'edge_hinge':
                # No sample on the bridging edge: every sample lies within 0.4 mm of the mask.
                self.assertLessEqual(distance.distance(world).max(), 0.4 + 1e-6)
            if term in ('centerline_hinge', 'edge_hinge'):
                vox = world_to_voxel(AFFINE, world)
                from scipy.ndimage import map_coordinates
                ref = map_coordinates(subject['sdf'], vox.T, order=1, mode='nearest')
                target = samples[f'{term}_target_mm'][:, 0]
                # Hinge ceiling never asks for SDF < 0 where the reference is positive.
                self.assertTrue(np.all(target[ref > 0] >= ref[ref > 0] - 1e-5))
                self.assertTrue(np.all(target[ref <= 0] <= 0))
            if term == 'centerline_hinge':
                self.assertTrue(self.labels[tuple(np.rint(world_to_voxel(AFFINE, world)).astype(int).T)].all())
            if term == 'radius':
                np.testing.assert_allclose(samples[f'{term}_target_mm'], -1.13, rtol=1e-5)  # only r >= 0.9 node

    def test_strict_rule_drops_off_mask_node(self):
        self.c['graph']['support_strict'] = True
        subject = load_subject(read_metadata(self.c).iloc[0], self.c)
        self.assertFalse(subject['graph']['supported'][1])
        np.testing.assert_allclose(subject['graph']['snapped'][1], subject['graph']['pos'][1])

    def test_cpu_training_all_terms_logged_and_decreasing(self):
        from vatl_sdf.train import train
        self.c['graph']['eikonal']['enabled'] = True
        self.c['sampling']['sdf']['points'] = 64
        self.c['training'].update(max_steps=40, epochs=100, task_points_per_step=64)
        self.c['optimizer'].update(lr_model=1e-3, lr_latent=1e-3)
        torch.manual_seed(0)
        directory = train(self.c)
        rows = [json.loads(l) for l in (Path(directory) / 'losses.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 40)
        for term in graph_terms(self.c) + ['eikonal']:
            self.assertIn(f'{term}_raw', rows[0]); self.assertIn(f'{term}_weighted', rows[0])
            self.assertTrue(all(np.isfinite(r[f'{term}_raw']) for r in rows))
        self.assertEqual(rows[0]['centerline_sdf_points_per_rank'], 10)
        first = np.mean([r['total'] for r in rows[:8]]); last = np.mean([r['total'] for r in rows[-8:]])
        self.assertLess(last, first)
        for term in ('sdf', 'centerline_sdf'):
            self.assertLess(np.mean([r[f'{term}_raw'] for r in rows[-8:]]), np.mean([r[f'{term}_raw'] for r in rows[:8]]))


if __name__ == '__main__':
    unittest.main()
