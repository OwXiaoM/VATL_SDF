"""Test-time fitting, generalization evaluation and probabilistic atlas (CPU)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml

from tests.test_alignment import normalized, staged
from tests.test_pipeline import tiny_config
from vatl_sdf.alignment_metrics import recurrence, recurrence_against
from vatl_sdf.dataset import read_metadata
from vatl_sdf.model import SDFModel
from vatl_sdf.probabilistic_atlas import calibration, coverage, subject_labels_in_template, within_radius
from vatl_sdf.test_time_fit import SubjectSampler, build_fitting_model, fit, fitting_config, training_steps_per_epoch
from vatl_sdf.train import train


def held_out_split(c, directory):
    """s0 = training subject, s1 = validation subject."""
    path = Path(directory) / 'splits_heldout.yaml'
    path.write_text(yaml.safe_dump({'train': ['s0'], 'val': ['s1'], 'test': []}))
    c['data'].update(subject_ids=str(path), expected_split_counts={'train': 1, 'val': 1, 'test': 0})
    return c


class FittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        c = normalized(staged(tiny_config(cls.temp.name), epochs=1), 'affine')
        c['graph']['centerline_sdf']['enabled'] = True
        c['graph_fields']['centerline_distance']['enabled'] = True
        c['training'].update(max_steps=None, epochs=2, stop_after_epoch=None)
        c['optimizer']['scheduler_epochs'] = 'run'
        held_out_split(c, cls.temp.name)
        torch.manual_seed(0)
        cls.run_directory = Path(train(c))
        cls.checkpoint_path = sorted(cls.run_directory.glob('checkpoint_step_*.pt'))[-1]
        cls.checkpoint = torch.load(cls.checkpoint_path, map_location='cpu', weights_only=False)
        cls.config = c
        frame = pd.read_csv(c['data']['tsv'], sep='\t', dtype={'subject_id': str})
        cls.test_rows = frame[frame.subject_id == 's1'].loc[:, ['subject_id', 'SDF', 'Graph', 'scan_age', 'sex']].reset_index(drop=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fitting_config_drops_gauge_only_for_new_subjects(self):
        fitted = fitting_config(self.checkpoint['config'])
        self.assertFalse(fitted['model']['transform']['mean_identity'])
        self.assertTrue(self.checkpoint['config']['model']['transform']['mean_identity'])

    def test_build_loads_network_and_resets_subject_tables(self):
        model, _ = build_fitting_model(self.checkpoint, 3)
        state = self.checkpoint['model']
        for name, value in model.state_dict().items():
            if name in model.subject_tensors():
                self.assertEqual(value.shape[0], 3); self.assertEqual(value.abs().max().item(), 0.)
            else:
                torch.testing.assert_close(value, state[name])
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, set(model.subject_tensors()))
        # Effective transform = own parameter (no mean over new subjects removed).
        with torch.no_grad():
            model.transform_translation[0] = torch.tensor([.1, 0., 0.])
        self.assertAlmostEqual(model.maps_mm()[1][0, 0].item(), .1 * 4., places=5)

    def test_staged_fit_freezes_network_and_latents_in_alignment(self):
        model, config = build_fitting_model(self.checkpoint, 1)
        sampler = SubjectSampler(self.test_rows, config, seed_offset=1)
        seen, rows = {}, []

        def on_epoch_end(epoch, stage):
            seen[epoch] = (stage, model.latents.detach().clone(), model.transform_translation.detach().clone())

        steps = fit(model, config, sampler, align_epochs=1, total_epochs=3, steps_per_epoch=2, on_epoch_end=on_epoch_end,
                    log=rows.append)
        self.assertEqual(steps, 6)
        self.assertTrue(seen[1][0]); self.assertFalse(seen[2][0])
        self.assertEqual(seen[1][1].abs().max().item(), 0.)        # latent fixed at zero during alignment
        self.assertGreater(seen[1][2].abs().max().item(), 0.)      # transform moved during alignment
        self.assertGreater(seen[3][1].abs().max().item(), 0.)      # latent unfrozen afterwards
        self.assertIn('align_vessel_raw', rows[0]); self.assertIn('latent_prior_raw', rows[-1])
        self.assertIn('centerline_sdf_raw', rows[-1]); self.assertIn('field_centerline_distance_raw', rows[-1])
        self.assertIn('affine_reg_raw', rows[-1])
        for name, value in model.state_dict().items():  # fit() also checks this internally
            if name not in model.subject_tensors():
                torch.testing.assert_close(value, self.checkpoint['model'][name])

    def test_fit_is_reproducible(self):
        tables = []
        for _ in range(2):
            torch.manual_seed(0)
            model, config = build_fitting_model(self.checkpoint, 1)
            fit(model, config, SubjectSampler(self.test_rows, config, seed_offset=1), 1, 2, steps_per_epoch=2)
            tables.append({k: v.detach().clone() for k, v in model.subject_tensors().items()})
        for k in tables[0]:
            torch.testing.assert_close(tables[0][k], tables[1][k])

    def test_steps_per_epoch_matches_training(self):
        c = {'training': {'subjects_per_rank': 5, 'task_points_per_step': 25000}, 'sampling': {'sdf': {'points': 250000}}}
        self.assertEqual(training_steps_per_epoch(c), 50)

    def test_test_frame_checks_split(self):
        from analysis.generalization.evaluate_generalization import test_frame
        frame = test_frame(self.config, ['s0'], None, 1)
        self.assertEqual(frame.subject_id.tolist(), ['s1'])
        with self.assertRaises(ValueError):
            test_frame(self.config, ['s0', 's1'], None, 1)       # overlap with training
        manifest = Path(self.temp.name) / 'manifest.json'
        manifest.write_text(json.dumps({'subject_ids': ['s0']}))
        with self.assertRaises(ValueError):
            test_frame(self.config, ['s0'], str(manifest), 1)    # manifest disagrees with YAML order

    def test_end_to_end_generalization_and_probabilistic_atlas(self):
        from analysis.generalization import evaluate_generalization
        from vatl_sdf import probabilistic_atlas
        out = Path(self.temp.name) / 'generalization'
        argv = ['x', '--checkpoint', str(self.checkpoint_path), '--output', str(out), '--manifest', '', '--num-test', '1',
                '--align-epochs', '1', '--total-epochs', '2', '--eval-epochs', '1', '2', '--final-factors', '1',
                '--steps-per-epoch', '2', '--device', 'cpu', '--workers', '1']
        with mock.patch.object(sys, 'argv', argv):
            evaluate_generalization.main()
        summary = json.loads((out / 'summary.json').read_text())
        self.assertEqual(summary['test_subject_ids'], ['s1'])
        self.assertEqual(sorted(summary['test']['epochs']), ['1', '2'])
        self.assertIn('zero_level_dice', summary['test']['epochs']['2']['median'])
        self.assertIn('train_refit', summary); self.assertIn('train_trained', summary)
        self.assertIn('test_vs_train/learned', summary['recurrence'])
        self.assertTrue((out / 'fitted_test.pt').is_file())
        shards = Path(self.temp.name) / 'shards'
        for k in range(2):
            with mock.patch.object(sys, 'argv', ['x', 'shard', '--checkpoint', str(self.checkpoint_path), '--output', str(shards),
                                                 '--shard', str(k), '--num-shards', '2', '--device', 'cpu', '--chunk-size', '64']):
                probabilistic_atlas.main()
        atlas = Path(self.temp.name) / 'prob_atlas'
        with mock.patch.object(sys, 'argv', ['x', 'merge', '--checkpoint', str(self.checkpoint_path), '--shards', str(shards),
                                             '--output', str(atlas), '--fitted', str(out / 'fitted_test.pt'), '--workers', '1']):
            probabilistic_atlas.main()
        result = json.loads((atlas / 'probabilistic_atlas.json').read_text())
        self.assertEqual(result['subjects'], 1)
        occupancy = nib.load(atlas / 'OccupancyProbability.nii.gz').get_fdata()
        self.assertTrue(set(np.unique(occupancy)) <= {0., 1.})      # one training subject: binary frequency
        self.assertIn('ge_0.5', result['connectivity']['occupancy'])
        self.assertEqual(set(result['heldout']), {'subject_ids', 'fitted_file', 'roi', 'control', 'learned', 'centroid_translated'})
        from vatl_sdf import graph_atlas
        graph_out = Path(self.temp.name) / 'graph_atlas'
        with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--output', str(graph_out),
                                             '--fitted', str(out / 'fitted_test.pt'), '--radii-mm', '2', '--thresholds', '0.5',
                                             '--min-skeleton-voxels', '1', '--workers', '1']):
            graph_atlas.main()
        graph = json.loads((graph_out / 'graph_atlas.json').read_text())
        level = graph['radii']['2']['thresholds']['0.5']
        self.assertGreater(level['skeleton_voxels'], 0)
        self.assertEqual(set(level['evaluation']), {'test/learned', 'test/centroid_translated', 'train_loo/learned',
                                                    'train_loo/centroid_translated'})
        from vatl_sdf import vessel_attribute_atlas
        attr_out = Path(self.temp.name) / 'attribute_atlas'
        with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--fitted', str(out / 'fitted_test.pt'),
                                             '--graph-atlas', str(graph_out), '--output', str(attr_out),
                                             '--populations', 'r2mm_tau0.5', '--workers', '1']):
            vessel_attribute_atlas.main()
        attr = json.loads((attr_out / 'vessel_attribute_atlas.json').read_text())
        self.assertIn('r2mm_tau0.5', attr['populations'])
        self.assertEqual(set(attr['populations']['r2mm_tau0.5']['evaluation']),
                         {'test/learned', 'test/centroid_translated', 'train_loo/learned'})
        self.assertTrue((attr_out / 'Radius_sdf_median.nii.gz').is_file())
        with self.assertRaises(ValueError):  # shards must cover every subject exactly once
            (shards / 'shard_00_of_02.npz').unlink()  # the shard holding the only training subject
            with mock.patch.object(sys, 'argv', ['x', 'merge', '--checkpoint', str(self.checkpoint_path), '--shards', str(shards),
                                                 '--output', str(atlas) + '_b', '--workers', '1']):
                probabilistic_atlas.main()


class MetricTests(unittest.TestCase):
    def test_recurrence_against_matches_recurrence_and_excludes_self(self):
        rng = np.random.default_rng(0)
        clouds = [rng.normal(size=(50, 3)) * 5 for _ in range(4)]
        own, _ = recurrence(clouds, queries_per_subject=20)
        cross = recurrence_against(clouds, clouds, queries_per_subject=20, exclude=list(range(4)))
        for d in own:
            np.testing.assert_allclose(own[d], cross[d])
        far = recurrence_against([clouds[0]], [clouds[0] + 100.], queries_per_subject=20)
        self.assertEqual(far[3.0].max(), 0.)

    def test_within_radius_matches_brute_force(self):
        affine = np.diag([1., 1., 1., 1.]); affine[:3, 3] = -4.
        shape = (8, 8, 8)
        points = np.random.default_rng(1).uniform(-3, 3, size=(30, 3))
        mask = within_radius(points, shape, affine, 1.0)
        centres = nib.affines.apply_affine(affine, np.indices(shape).reshape(3, -1).T)
        brute = (np.linalg.norm(centres[:, None] - points[None], axis=2) <= 1.0).any(1).reshape(shape)
        np.testing.assert_array_equal(mask, brute)

    def test_calibration_extremes(self):
        y = np.array([0, 0, 1, 1, 0, 1], bool); valid = np.ones(6, bool)
        perfect = calibration(y.astype(float), y, valid, 10)
        self.assertEqual(perfect['brier'], 0.); self.assertEqual(perfect['brier_skill'], 1.)
        constant = calibration(np.full(6, .5), y, valid, 10)
        self.assertAlmostEqual(constant['brier_skill'], 0.)
        self.assertEqual(sum(r['count'] for r in constant['reliability']), 6)
        self.assertGreater(perfect['loglik_per_voxel'], constant['loglik_per_voxel'])

    def test_subject_labels_in_template_identity_and_translation(self):
        with tempfile.TemporaryDirectory() as d:
            affine = np.diag([1., 1., 1., 1.]); affine[:3, 3] = -4.
            sdf = np.ones((8, 8, 8), np.float32); sdf[2, 3, 4] = -1
            nib.save(nib.Nifti1Image(sdf, affine), Path(d) / 'sdf.nii.gz')
            centre = nib.affines.apply_affine(affine, [4., 4., 4.])
            vessel = nib.affines.apply_affine(affine, [2., 3., 4.])
            points = np.array([vessel - centre, vessel - centre + [1., 0, 0]])
            y, valid = subject_labels_in_template(Path(d) / 'sdf.nii.gz', np.eye(3), np.zeros(3), points, [8, 8, 8])
            np.testing.assert_array_equal(y, [True, False]); self.assertTrue(valid.all())
            y, _ = subject_labels_in_template(Path(d) / 'sdf.nii.gz', np.eye(3), np.array([1., 0, 0]), points, [8, 8, 8])
            np.testing.assert_array_equal(y, [False, True])

    def test_coverage(self):
        affine = np.eye(4)
        field = np.zeros((4, 4, 4)); field[1, 1, 1] = 1.
        result = coverage({'p': field}, np.array([[1., 1., 1.], [3., 3., 3.]]), affine)
        self.assertEqual(result['p']['ge_0.5'], .5)


if __name__ == '__main__':
    unittest.main()
