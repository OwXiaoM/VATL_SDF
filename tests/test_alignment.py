"""Alignment Step 1: transform parameterizations, mean-identity gauge, centroid init, staged training,
run-length scheduler horizon, alignment targets and metrics (CPU)."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from tests.test_pipeline import AFFINE, tiny_config
from vatl_sdf.alignment_metrics import recurrence, transform_statistics
from vatl_sdf.config import validate
from vatl_sdf.dataset import SDFGraphDataset, alignment_targets, load_subject, read_metadata, vessel_centroids
from vatl_sdf.graph_coords import voxel_to_world
from vatl_sdf.model import SDFModel, rigid_transform
from vatl_sdf.train import parameter_groups, scheduler_horizon, train


def normalized(c, kind='rigid', mean_identity=True, centroid_init=False):
    c['model']['transform'] = {'mode': 'normalized', 'type': kind, 'mean_identity': mean_identity,
                               'centroid_init': centroid_init, 'affine_reg_weight': 0.1}
    c['optimizer'].update(lr_rotation=1e-3, lr_translation=1e-3, lr_scale=1e-3)
    validate(c)
    return c


def staged(c, epochs=1):
    c['alignment'] = {'enabled': True, 'epochs': epochs, 'sdf_truncation_mm': 4.0, 'sdf_weight': 1.0,
                      'large_vessel_weight': 1.0, 'large_vessel_min_radius_mm': 1.2, 'latent_prior_weight': 1e-3}
    validate(c)
    return c


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.c = tiny_config(self.temp.name)
        self.u = torch.rand(20, 3) * 2 - 1

    def tearDown(self):
        self.temp.cleanup()

    def test_legacy_default_is_unchanged(self):
        model = SDFModel(self.c, 2)
        self.assertEqual(list(model.subject_tensors()), ['latents', 'transforms'])
        with torch.no_grad():
            model.transforms[1] = torch.tensor([.1, -.2, .3, .5, -.4, .2])
        ids = torch.ones(20, dtype=torch.long)
        expected = rigid_transform(self.u, model.transforms[ids], model.half_bbox)
        torch.testing.assert_close(model.transform_coords(self.u, ids), expected)
        groups = parameter_groups(model, self.c)
        self.assertEqual([g['lr'] for g in groups], [self.c['optimizer'][k] for k in ('lr_model', 'lr_latent', 'lr_transform')])
        self.assertIs(groups[2]['params'][0], model.transforms)

    def check_maps_agree(self, model):
        a, t = model.maps_mm()
        for i in range(2):
            ids = torch.full((20,), i, dtype=torch.long)
            mapped = model.transform_coords(self.u, ids) * model.half_bbox
            torch.testing.assert_close(mapped, (self.u * model.half_bbox) @ a[i].T + t[i], atol=1e-5, rtol=1e-5)

    def test_normalized_translation_units_and_maps(self):
        model = SDFModel(normalized(self.c, mean_identity=False), 2)
        with torch.no_grad():
            model.transform_translation[0] = torch.tensor([.25, 0, 0])
        ids = torch.zeros(1, dtype=torch.long)
        out = model.transform_coords(torch.zeros(1, 3), ids)
        torch.testing.assert_close(out, torch.tensor([[.25, 0, 0]]))  # a quarter of the half bbox
        self.assertAlmostEqual(model.maps_mm()[1][0, 0].item(), .25 * 4)  # half bbox x = 4 mm
        with torch.no_grad():
            model.transform_rotation[1] = torch.tensor([.1, .2, -.3])
        self.check_maps_agree(model)
        names = [g.get('name') for g in parameter_groups(model, self.c)[2:]]
        self.assertEqual(names, ['transform_rotation', 'transform_translation'])

    def test_mean_identity_removes_common_offset(self):
        model = SDFModel(normalized(self.c), 2)
        with torch.no_grad():
            model.transform_translation[:] = torch.tensor([[.1, .2, .3], [.3, .2, .1]])
        before = model.transform_coords(self.u, torch.ones(20, dtype=torch.long))
        with torch.no_grad():
            model.transform_translation += torch.tensor([.5, -.5, .5])
            model.transform_rotation += torch.tensor([.2, .1, .3])
        torch.testing.assert_close(model.transform_coords(self.u, torch.ones(20, dtype=torch.long)), before)
        torch.testing.assert_close(model.maps_mm()[1].mean(0), torch.zeros(3), atol=1e-6, rtol=0)

    def test_affine_maps_and_scale(self):
        model = SDFModel(normalized(self.c, 'affine', mean_identity=False), 2)
        with torch.no_grad():
            model.transform_log_scale[0] = torch.tensor([.1, -.05, .2])
            model.transform_shear[0] = torch.tensor([.05, 0, -.02])
            model.transform_rotation[0] = torch.tensor([.1, 0, .2])
        a, _ = model.maps_mm()
        self.assertAlmostEqual(torch.linalg.det(a[0]).item(), math.exp(.25), places=5)
        self.check_maps_agree(model)
        stats = transform_statistics(a.numpy(), np.zeros((2, 3)))
        self.assertAlmostEqual(stats['scale']['max'], math.exp(.25 / 3), places=4)

    def test_validation_rejects_bad_combinations(self):
        c = normalized(self.c)
        c['loss']['tf_weight'] = 0.1
        with self.assertRaisesRegex(ValueError, 'tf_weight'):
            validate(c)
        c['loss']['tf_weight'] = 0.
        del c['optimizer']['lr_translation']
        with self.assertRaisesRegex(ValueError, 'lr_translation'):
            validate(c)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.c = tiny_config(self.temp.name)
        # Subject 1: same sphere shifted by one voxel along x (centroid differs by the voxel step).
        image = nib.load(Path(self.temp.name) / 'SDF0.nii.gz')
        world = voxel_to_world(AFFINE, np.indices((8, 8, 8)).reshape(3, -1).T)
        sdf = (np.linalg.norm(world - voxel_to_world(AFFINE, np.array([[5, 4, 4]])), axis=1) - 1.5).reshape(8, 8, 8)
        shifted = nib.Nifti1Image(sdf.astype(np.float32), image.affine); shifted.header.set_xyzt_units('mm')
        nib.save(shifted, Path(self.temp.name) / 'SDF1.nii.gz')
        self.step_mm = voxel_to_world(AFFINE, np.array([[5, 4, 4]]))[0] - voxel_to_world(AFFINE, np.array([[4, 4, 4]]))[0]

    def tearDown(self):
        self.temp.cleanup()

    def test_centroid_init_maps_centroids_together(self):
        c = normalized(self.c, centroid_init=True)
        frame = read_metadata(c)
        centroids = vessel_centroids(frame, c)
        np.testing.assert_allclose((centroids[1] - centroids[0]) * 4, self.step_mm, atol=1e-4)  # half bbox 4 mm
        from vatl_sdf.train import initialize_subject_parameters
        model = SDFModel(c, 2); initialize_subject_parameters(model, frame, c)
        mapped = [model.transform_coords(torch.as_tensor(centroids[i:i + 1]), torch.tensor([i])) for i in range(2)]
        torch.testing.assert_close(mapped[0], mapped[1], atol=1e-6, rtol=0)

    def test_alignment_targets_large_vessels_only(self):
        c = staged(self.c)
        subject = load_subject(read_metadata(c).iloc[0], c)
        self.assertIsNotNone(subject['align_tree'])
        node = subject['graph']['pos'][0]
        target, weight = alignment_targets(subject, np.array([node, node + [0, 0, 2.0], node + [0, 0, 50.0]]), c)
        self.assertLess(target[0, 0], 0.3); self.assertAlmostEqual(target[2, 0], 4.0)  # clamped at truncation
        np.testing.assert_array_equal(weight[:, 0], 1)
        c['alignment']['large_vessel_min_radius_mm'] = 5.0  # no centreline that thick -> zero weight
        subject = load_subject(read_metadata(c).iloc[0], c)
        self.assertEqual(alignment_targets(subject, node[None], c)[1][0, 0], 0)

    def test_targets_only_during_alignment_epochs(self):
        data = SDFGraphDataset(staged(self.c, epochs=1))
        self.assertIn('align_target', data[0])
        data.epoch = 1
        self.assertNotIn('align_target', data[0])


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.c = tiny_config(self.temp.name)
        self.c['training'].update(max_steps=None, epochs=3, stop_after_epoch=None)
        self.c['optimizer']['scheduler_epochs'] = 'run'

    def tearDown(self):
        self.temp.cleanup()

    def run_dir(self, c):
        torch.manual_seed(0)
        d = Path(train(c))
        rows = [json.loads(l) for l in (d / 'losses.jsonl').read_text().splitlines()]
        return d, rows

    def test_run_horizon(self):
        steps = math.ceil(1 * self.c['sampling']['sdf']['points'] / self.c['training']['task_points_per_step'])
        self.assertEqual(scheduler_horizon(self.c, 2, 1), 3)
        self.assertEqual(scheduler_horizon(dict(self.c, training=dict(self.c['training'], stop_after_epoch=0)), 2, 1), 1)
        limited = dict(self.c, training=dict(self.c['training'], max_steps=2 * steps * 2 + 1))
        self.assertEqual(scheduler_horizon(limited, 2, 1), 3)
        d, rows = self.run_dir(self.c)
        self.assertEqual(len([r for r in rows if r['epoch'] == 0]), 2 * steps)

    def test_staged_training_freezes_latents_then_refines(self):
        c = normalized(staged(self.c, epochs=1), 'affine')
        d, rows = self.run_dir(c)
        align, main = [r for r in rows if r['stage'] == 'align'], [r for r in rows if r['stage'] == 'main']
        self.assertTrue(align and main)
        self.assertIn('align_vessel_raw', align[0]); self.assertNotIn('sdf_raw', align[0])
        self.assertIn('latent_prior_raw', main[0]); self.assertIn('affine_reg_raw', main[0])
        first = torch.load(d / 'checkpoint_step_{:08d}.pt'.format(len(align)), map_location='cpu', weights_only=False)['model']
        self.assertEqual(first['latents'].abs().max().item(), 0.)          # frozen at zero during alignment
        self.assertGreater(first['transform_translation'].abs().max().item(), 0.)  # transforms moved
        last = torch.load(sorted(d.glob('checkpoint_step_*.pt'))[-1], map_location='cpu', weights_only=False)['model']
        self.assertGreater(last['latents'].abs().max().item(), 0.)          # unfrozen afterwards
        log = [json.loads(l) for l in (d / 'transforms.jsonl').read_text().splitlines()]
        self.assertEqual([r['stage'] for r in log], ['align', 'main', 'main'])
        self.assertIn('scale', log[0])

    def test_staged_resume_across_stage_boundary_is_exact(self):
        c = normalized(staged(self.c, epochs=1))
        # Fixed horizon: with 'run' the stopped partial run would get a 1-epoch horizon, which the resumed
        # scheduler state keeps (documented in train.scheduler_horizon).
        c['optimizer']['scheduler_epochs'] = 3
        straight, _ = self.run_dir(c)
        partial, rows = self.run_dir(dict(c, training=dict(c['training'], stop_after_epoch=0)))
        checkpoint = sorted(partial.glob('checkpoint_step_*.pt'))[-1]
        resumed = Path(train(dict(c, resume=str(checkpoint), require_fresh_start=False)))
        a = torch.load(sorted(resumed.glob('checkpoint_step_*.pt'))[-1], map_location='cpu', weights_only=False)['model']
        b = torch.load(sorted(straight.glob('checkpoint_step_*.pt'))[-1], map_location='cpu', weights_only=False)['model']
        self.assertLess(max((a[k] - b[k]).abs().max().item() for k in a), 1e-6)


class MetricTests(unittest.TestCase):
    def test_recurrence(self):
        line = np.stack([np.linspace(0, 10, 26), np.zeros(26), np.zeros(26)], 1)
        clouds = [line, line + [0, 1.5, 0], line + [0, 10, 0]]
        fractions, owner = recurrence(clouds, queries_per_subject=26)
        self.assertEqual(len(owner), 78)
        first = fractions[2.0][owner == 0]
        np.testing.assert_allclose(first, 0.5)        # subject 1 within 2 mm, subject 2 not
        np.testing.assert_allclose(fractions[1.0][owner == 0], 0.)


if __name__ == '__main__':
    unittest.main()
