"""Deviation-statistics stage (CPU): subject sets, refitting, per-subject decodings."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

from analysis.deviation.subject_sets import make_sets
from tests import test_generalization  # module import: keeps FittingTests out of this module's test discovery


class SubjectSetTests(unittest.TestCase):
    def test_disjoint_seeded_and_excluding(self):
        train = [f't{i}' for i in range(30)]
        a = make_sets(train, ['v0', 'v1'], {'null': 10, 'calibration': 5, 'ants_template': 8}, 0, ['t0', 't1'])
        b = make_sets(train, ['v0', 'v1'], {'null': 10, 'calibration': 5, 'ants_template': 8}, 0, ['t0', 't1'])
        self.assertEqual(a, b)
        groups = [set(a[k]) for k in ('null', 'calibration', 'ants_template')]
        self.assertEqual(sum(map(len, groups)), len(set().union(*groups)))
        self.assertFalse({'t0', 't1'} & set().union(*groups))
        with self.assertRaises(ValueError):
            make_sets(train, ['t3'], {'null': 2}, 0, [])
        with self.assertRaises(ValueError):
            make_sets(train, [], {'null': 40}, 0, [])


class RefitAndDecodeTests(unittest.TestCase):
    setUpClass = classmethod(test_generalization.FittingTests.setUpClass.__func__)   # tiny trained checkpoint (s0 train, s1 validation)
    tearDownClass = classmethod(test_generalization.FittingTests.tearDownClass.__func__)

    def test_refit_subjects_and_per_subject_decodings(self):
        from analysis.deviation import refit_subjects
        from vatl_sdf import probabilistic_atlas
        sets = Path(self.temp.name) / 'sets.json'
        sets.write_text(json.dumps({'null': ['s0'], 'calibration': [], 'validation': ['s1']}))
        out = Path(self.temp.name) / 'refit'
        for name in ('null', 'validation'):
            with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--set', name,
                                                 '--output', str(out), '--align-epochs', '1', '--total-epochs', '2',
                                                 '--steps-per-epoch', '2', '--device', 'cpu']):
                refit_subjects.main()
        fitted = torch.load(out / 'fitted_validation_00.pt', weights_only=False)
        self.assertEqual(fitted['subject_ids'], ['s1'])
        self.assertEqual(fitted['A'].shape, (1, 3, 3))
        self.assertEqual(torch.load(out / 'fitted_null_00.pt', weights_only=False)['subject_ids'], ['s0'])
        with self.assertRaises(FileExistsError):
            with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--set', 'null',
                                                 '--output', str(out), '--total-epochs', '2', '--device', 'cpu']):
                refit_subjects.main()
        shards = Path(self.temp.name) / 'decode'
        for k in range(2):
            with mock.patch.object(sys, 'argv', ['x', 'shard', '--checkpoint', str(self.checkpoint_path), '--output', str(shards),
                                                 '--shard', str(k), '--num-shards', '2', '--device', 'cpu', '--chunk-size', '64',
                                                 '--save-subjects']):
                probabilistic_atlas.main()
        total = sum(np.load(f)['hard'] for f in sorted(shards.glob('shard_*.npz')))
        packed = [np.load(f) for f in sorted(shards.glob('subjects_*.npz'))]
        n = int(np.prod(packed[0]['shape']))
        decoded = sum(np.unpackbits(p['packed'], axis=1, count=n).sum(0) for p in packed if len(p['packed']))
        np.testing.assert_array_equal(decoded, total)
        self.assertEqual([s for p in packed for s in p['subject_ids']], ['s0'])


if __name__ == '__main__':
    unittest.main()


def cohort_config(directory, n_train=6, n_val=2):
    """tiny_config with n_train + n_val subjects whose vessel (sphere + 3-node graph) is shifted per subject."""
    import pickle
    import networkx as nx
    import nibabel as nib
    import pandas as pd
    import yaml
    from tests.test_pipeline import AFFINE
    from vatl_sdf.graph_coords import voxel_to_world
    c = test_generalization.tiny_config(directory)
    rows = []
    world = voxel_to_world(AFFINE, np.indices((8, 8, 8)).reshape(3, -1).T)
    for i in range(n_train + n_val):
        shift = np.array([0.3 * (i % 3), 0.2 * (i % 2), 0.])
        centre = voxel_to_world(AFFINE, np.array([[4., 4., 4.]]))[0] + shift
        sdf = (np.linalg.norm(world - centre, axis=1) - 1.5 - 0.1 * (i % 2)).reshape(8, 8, 8).astype(np.float32)
        nib.save(nib.Nifti1Image(sdf, AFFINE), Path(directory) / f'SDF{i}.nii.gz')
        graph = nx.Graph()
        for n, v in enumerate(([4, 4, 4], [5, 4, 4], [4, 5, 4], [4, 4, 5])):
            graph.add_node(n, pos=voxel_to_world(AFFINE, np.array(v, float)) + shift, radius=np.float32(1.5 if n < 3 or i % 2 else 0.8))
        graph.add_edges_from([(0, 1), (0, 2), (0, 3)], weight=1.)
        with open(Path(directory) / f'Graph{i}.pkl', 'wb') as f:
            pickle.dump({'graph': graph}, f)
        rows.append({'subject_id': f's{i}', 'scan_age': 20 + 5 * i, 'sex': 'MF'[i % 2], 'MRA': 'unused_never_read.nii.gz',
                     'SDF': str(Path(directory) / f'SDF{i}.nii.gz'), 'Graph': str(Path(directory) / f'Graph{i}.pkl')})
    pd.DataFrame(rows).to_csv(c['data']['tsv'], sep='\t', index=False)
    split = Path(directory) / 'splits_cohort.yaml'
    split.write_text(yaml.safe_dump({'train': [f's{i}' for i in range(n_train)], 'val': [f's{i}' for i in range(n_train, n_train + n_val)], 'test': []}))
    c['data'].update(subject_ids=str(split), split_key=None, expected_split_counts={'train': n_train, 'val': n_val, 'test': 0})
    return c


class DeviationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_alignment import normalized, staged
        from vatl_sdf.train import train
        cls.temp = tempfile.TemporaryDirectory()
        c = normalized(staged(cohort_config(cls.temp.name), epochs=1), 'affine')
        c['graph']['centerline_sdf']['enabled'] = True
        c['graph_fields']['centerline_distance']['enabled'] = True
        c['training'].update(max_steps=None, epochs=2, stop_after_epoch=None)
        c['optimizer']['scheduler_epochs'] = 'run'
        torch.manual_seed(0)
        cls.checkpoint_path = sorted(Path(train(c)).glob('checkpoint_step_*.pt'))[-1]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_end_to_end(self):
        from analysis.deviation import refit_subjects
        from vatl_sdf import deviation, probabilistic_atlas
        d = Path(self.temp.name)
        sets = d / 'sets.json'
        sets.write_text(json.dumps({'null': ['s0', 's1', 's2'], 'calibration': ['s3', 's4'], 'ants_template': ['s5'], 'validation': ['s6', 's7']}))
        for name in ('null', 'calibration', 'validation'):
            with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--set', name,
                                                 '--output', str(d / 'refit'), '--align-epochs', '1', '--total-epochs', '2',
                                                 '--steps-per-epoch', '1', '--device', 'cpu']):
                refit_subjects.main()
        with mock.patch.object(sys, 'argv', ['x', 'shard', '--checkpoint', str(self.checkpoint_path), '--output', str(d / 'dec'),
                                             '--shard', '0', '--num-shards', '1', '--device', 'cpu', '--chunk-size', '64', '--save-subjects']):
            probabilistic_atlas.main()
        out = d / 'deviation'
        with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--decodings', str(d / 'dec'),
                                             '--refit', str(d / 'refit'), '--output', str(out), '--min-skeleton-voxels', '1',
                                             '--age-sigmas', '7', '--smooth-mm', '0', '1', '--recalibration', 'none', 'isotonic',
                                             '--workers', '1']):
            deviation.main()
        summary = json.loads((out / 'deviation.json').read_text())
        self.assertEqual(set(summary['fpr']), {f'{m}/{w}' for m in ('decoder', 'learned', 'centroid') for w in ('uniform', 'age_sigma7')})
        fpr = summary['fpr']['learned/uniform']
        self.assertEqual((fpr['null_loo']['subjects'], fpr['validation']['subjects']), (3, 2))
        self.assertIn('affine_d2', fpr['validation']['per_score'])
        self.assertNotIn('affine_d2', summary['fpr']['centroid/uniform']['validation']['per_score'])
        scores = pd.read_csv(out / 'scores.tsv', sep='\t', keep_default_na=False, na_values=[''])  # 'null' is a set name
        p = scores.p.dropna()
        self.assertTrue(((p > 0) & (p <= 1)).all())
        self.assertEqual(set(scores.set), {'null', 'validation'})       # calibration subjects never scored
        self.assertTrue((out / 'subjects' / 's6' / 'VoxelSurprise_learned.nii.gz').is_file())
        self.assertFalse((out / 'subjects' / 's0' / 'VoxelSurprise_learned.nii.gz').exists())  # maps: validation only
        report = json.loads((out / 'subjects' / 's6' / 'report_learned_uniform.json').read_text())
        self.assertIn('r3mm_tau0.1', report['populations'])


class DeviationStatisticsTests(unittest.TestCase):
    def test_weights_and_smoothing(self):
        from vatl_sdf.deviation import ess, kernel_weights, smoothed
        w = kernel_weights([20, 30, 40], 30, None, exclude=1)
        np.testing.assert_array_equal(w, [1, 0, 1])
        self.assertAlmostEqual(ess(w), 2.)
        self.assertAlmostEqual(float(smoothed(1., w.sum(), w)), (1 + 1) / (2 + 2))   # uniform: (k + 1) / (n + 2)
        w7 = kernel_weights([20, 30, 40], 30, 7.)
        self.assertLess(ess(w7), 3.); self.assertAlmostEqual(w7[1], 1.)

    def test_stack_loo_matches_weighted(self):
        from vatl_sdf.deviation import Stack, kernel_weights
        rng = np.random.default_rng(0)
        y = rng.uniform(size=(5, 40)) < 0.3; v = rng.uniform(size=(5, 40)) < 0.9
        support = np.flatnonzero(y.any(0))
        stack = Stack([np.packbits(r) for r in y], [np.packbits(r) for r in v], support)
        w = kernel_weights(np.zeros(5), 0., None, exclude=2)
        np.testing.assert_allclose(stack.values(w, uniform_exclude=2), stack.values(w), atol=1e-6)
        num, den = (y & v)[[0, 1, 3, 4]][:, support].sum(0), v[[0, 1, 3, 4]][:, support].sum(0)
        np.testing.assert_allclose(stack.values(w), np.where(den > 0, num / np.maximum(den, 1), 0), atol=1e-6)

    def test_pvalues_bh_holm(self):
        from vatl_sdf.deviation import benjamini_hochberg, empirical_p, holm, radius_deviation
        self.assertAlmostEqual(empirical_p(5., [1, 2, 3, 6]), 2 / 5)
        self.assertAlmostEqual(empirical_p(10., [1, 2, 3, 6]), 1 / 5)
        self.assertTrue(np.isnan(empirical_p(np.nan, [1.])))
        np.testing.assert_array_equal(benjamini_hochberg([0.001, 0.02, 0.04, 0.5, np.nan], 0.05), [True, True, False, False, False])
        np.testing.assert_array_equal(holm([0.01, 0.03, 0.2], 0.05), [True, False, False])
        dist = np.array([[1.], [2.], [3.], [4.], [np.nan]])
        out = radius_deviation(np.array([2.5]), dist, np.ones(5))
        self.assertAlmostEqual(out[0, 0], 0.); self.assertAlmostEqual(out[0, 2], 0.5)
