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
        # replicate refits of the validation subjects: own seed, own file name, never picked up as 'validation'
        from vatl_sdf.deviation import load_fitted
        with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--sets', str(sets), '--set', 'valrep1',
                                             '--output', str(out), '--align-epochs', '1', '--total-epochs', '2',
                                             '--steps-per-epoch', '2', '--device', 'cpu']):
            refit_subjects.main()
        rep = torch.load(out / 'fitted_valrep1_00.pt', weights_only=False)
        self.assertEqual(rep['subject_ids'], ['s1'])
        self.assertFalse(np.allclose(rep['t_mm'], fitted['t_mm']) and np.allclose(rep['A'], fitted['A']))  # other sampler seed
        np.testing.assert_array_equal(load_fitted(out, 'validation')['s1'][1], np.asarray(fitted['t_mm'][0], float))
        self.assertEqual(set(load_fitted(out, 'valrep1')), {'s1'})
        self.assertEqual(len(list(out.glob('fitted_validation_*.pt'))), 1)
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


def tube_cohort_config(directory, n_train=6, n_val=2, size=24):
    """Cohort of tubes along x (radius 1.5 mm) with a vertical side branch, shifted per subject; bbox 20 mm."""
    import pickle
    import networkx as nx
    import nibabel as nib
    import pandas as pd
    import yaml
    from vatl_sdf.graph_coords import voxel_to_world
    c = test_generalization.tiny_config(directory)
    c['data']['world_bbox_mm'] = [20, 20, 20]
    affine = np.diag([1., 1., 1., 1.]); affine[:3, 3] = -size / 2
    world = voxel_to_world(affine, np.indices((size,) * 3).reshape(3, -1).T)
    rows = []
    for i in range(n_train + n_val):
        dy, dz = 0.4 * (i % 3) - 0.4, 0.3 * (i % 2)
        line = np.linalg.norm(world[:, 1:] - [dy, dz], axis=1) - 1.5
        branch = np.where(world[:, 2] >= dz, np.linalg.norm(world[:, :2] - [0., dy], axis=1) - 1.5, 9.)
        sdf = np.minimum(line, branch).reshape((size,) * 3).astype(np.float32)
        nib.save(nib.Nifti1Image(sdf, affine), Path(directory) / f'SDF{i}.nii.gz')
        g = nx.Graph()
        xs = np.arange(-8., 9.)
        for n, x in enumerate(xs):
            g.add_node(n, pos=np.array([x, dy, dz]), radius=np.float32(1.5))
        for m, z in enumerate(np.arange(1., 9.)):
            g.add_node(len(xs) + m, pos=np.array([0., dy, dz + z]), radius=np.float32(1.5))
        g.add_edges_from([(n, n + 1) for n in range(len(xs) - 1)] + [(8, len(xs))] + [(len(xs) + m, len(xs) + m + 1) for m in range(7)])
        with open(Path(directory) / f'Graph{i}.pkl', 'wb') as f:
            pickle.dump({'graph': g}, f)
        rows.append({'subject_id': f's{i}', 'scan_age': 20 + 5 * i, 'sex': 'MF'[i % 2], 'MRA': 'unused_never_read.nii.gz',
                     'SDF': str(Path(directory) / f'SDF{i}.nii.gz'), 'Graph': str(Path(directory) / f'Graph{i}.pkl')})
    pd.DataFrame(rows).to_csv(c['data']['tsv'], sep='\t', index=False)
    split = Path(directory) / 'splits_cohort.yaml'
    split.write_text(yaml.safe_dump({'train': [f's{i}' for i in range(n_train)], 'val': [f's{i}' for i in range(n_train, n_train + n_val)], 'test': []}))
    c['data'].update(subject_ids=str(split), split_key=None, expected_split_counts={'train': n_train, 'val': n_val, 'test': 0})
    return c


class DeviationPipelineTests(unittest.TestCase):
    cohort = staticmethod(cohort_config)

    @classmethod
    def setUpClass(cls):
        from tests.test_alignment import normalized, staged
        from vatl_sdf.train import train
        cls.temp = tempfile.TemporaryDirectory()
        c = normalized(staged(cls.cohort(cls.temp.name), epochs=1), 'affine')
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
        self.assertIn('primary_family_minp', fpr['validation']); self.assertIn('elements_bh_population', fpr['validation'])
        self.assertNotIn('affine_d2', summary['fpr']['centroid/uniform']['validation']['per_score'])
        scores = pd.read_csv(out / 'scores.tsv', sep='\t', keep_default_na=False, na_values=[''])  # 'null' is a set name
        p = scores.p.dropna()
        self.assertTrue(((p > 0) & (p <= 1)).all())
        self.assertEqual(set(scores.set), {'null', 'validation'})       # calibration subjects never scored
        self.assertTrue((out / 'subjects' / 's6' / 'VoxelSurprise_learned.nii.gz').is_file())
        self.assertFalse((out / 'subjects' / 's0' / 'VoxelSurprise_learned.nii.gz').exists())  # maps: validation only
        report = json.loads((out / 'subjects' / 's6' / 'report_learned_uniform.json').read_text())
        self.assertIn('r3mm_tau0.1', report['populations'])


class PerturbationPipelineTests(DeviationPipelineTests):
    cohort = staticmethod(tube_cohort_config)

    def test_end_to_end(self):  # covered by DeviationPipelineTests
        pass

    def test_perturbation_pipeline(self):
        from analysis.deviation import perturbation_manifest, refit_subjects
        from vatl_sdf import deviation, graph_atlas, probabilistic_atlas
        d = Path(self.temp.name) / 'pert'
        d.mkdir()
        sets = d / 'sets.json'
        sets.write_text(json.dumps({'null': ['s0', 's1', 's2'], 'calibration': ['s3', 's4'], 'ants_template': ['s5'], 'validation': ['s6', 's7']}))
        common = ['--checkpoint', str(self.checkpoint_path), '--sets', str(sets)]
        for name in ('null', 'calibration', 'validation'):
            with mock.patch.object(sys, 'argv', ['x', *common, '--set', name, '--output', str(d / 'refit'), '--align-epochs', '1',
                                                 '--total-epochs', '2', '--steps-per-epoch', '1', '--device', 'cpu']):
                refit_subjects.main()
        with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--output', str(d / 'graph'),
                                             '--fitted', str(d / 'refit' / 'fitted_validation_00.pt'), '--radii-mm', '3', '2',
                                             '--thresholds', '0.1', '--min-skeleton-voxels', '1', '--workers', '1']):
            graph_atlas.main()
        manifest = d / 'manifest.json'
        with mock.patch.object(sys, 'argv', ['x', '--checkpoint', str(self.checkpoint_path), '--graph-atlas', str(d / 'graph'),
                                             '--sets', str(sets), '--refit', str(d / 'refit'), '--output', str(manifest),
                                             '--radius-mm', '100', '--min-p', '0', '--min-c', '0', '--min-radius-mm', '0']):
            perturbation_manifest.main()
        m = json.loads(manifest.read_text())
        self.assertGreater(len(m['copies']), 0)
        self.assertEqual(m['copies'][0]['kind'], 'remove')                   # priority order
        with mock.patch.object(sys, 'argv', ['x', *common, '--set', 'perturbed', '--perturbations', str(manifest), '--chunk-size', '3',
                                             '--output', str(d / 'refit'), '--align-epochs', '1', '--total-epochs', '2',
                                             '--steps-per-epoch', '1', '--device', 'cpu']):
            refit_subjects.main()                                             # first chunk only: partial copies are allowed
        with mock.patch.object(sys, 'argv', ['x', 'shard', '--checkpoint', str(self.checkpoint_path), '--output', str(d / 'dec'),
                                             '--shard', '0', '--num-shards', '1', '--device', 'cpu', '--chunk-size', '64', '--save-subjects']):
            probabilistic_atlas.main()
        out = d / 'deviation'
        with mock.patch.object(sys, 'argv', ['x', *common, '--decodings', str(d / 'dec'), '--refit', str(d / 'refit'), '--output', str(out),
                                             '--min-skeleton-voxels', '1', '--radii-mm', '3', '2', '--age-sigmas', '7',
                                             '--smooth-mm', '0', '--recalibration', 'none', '--perturbations', str(manifest), '--workers', '1']):
            deviation.main()
        summary = json.loads((out / 'deviation.json').read_text())
        self.assertEqual(summary['perturbations']['copies_scored'], min(3, len(m['copies'])))
        sens = summary['sensitivity']['learned/uniform']
        self.assertIn('remove0', sens['by_perturbation'])
        self.assertIn('population_any', sens['by_perturbation']['remove0'])
        self.assertIn('subject_detection', summary['sensitivity']['decoder/uniform']['by_perturbation']['remove0'])
        self.assertNotIn('learned/age_sigma7', summary['sensitivity'])       # perturbed copies: uniform weighting only
        self.assertFalse(any(p.name.startswith('s6__') for p in (out / 'subjects').iterdir()))  # no reports for copies


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
        self.assertAlmostEqual(out[0, 3], 1.)                               # median value: two-sided p = 1
        far = radius_deviation(np.array([9.]), dist, np.ones(5))
        self.assertAlmostEqual(far[0, 3], 2 * 1 / 5)                        # (0 + c) / (4 + c), c = 1

    def test_outcome_p(self):
        from vatl_sdf.deviation import outcome_p
        np.testing.assert_allclose(outcome_p(np.array([0.9, 0.9, 0.2, 0.2]), np.array([False, True, True, False])), [0.1, 1., 0.2, 1.])


def tube_case():
    """Straight tube along x (radius 2 mm) in a 1 mm grid, graph = chain of nodes every mm plus a side branch."""
    import networkx as nx
    shape = (40, 24, 24); affine = np.eye(4)
    world = np.indices(shape).reshape(3, -1).T.astype(float)
    sdf = (np.linalg.norm(world[:, 1:] - 12., axis=1) - 2.).reshape(shape).astype(np.float32)
    g = nx.Graph()
    for i in range(5, 36):
        g.add_node(i - 5, pos=np.array([i, 12., 12.]), radius=2.)
    g.add_node(31, pos=np.array([20., 18., 12.]), radius=1.)
    g.add_edges_from([(k, k + 1) for k in range(30)] + [(15, 31)])
    from vatl_sdf.graph_coords import load_graph
    import pickle
    return sdf, affine, g


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        import pickle
        from vatl_sdf.graph_coords import load_graph
        self.temp = tempfile.TemporaryDirectory()
        self.sdf, self.affine, g = tube_case()
        path = Path(self.temp.name) / 'g.pkl'
        with open(path, 'wb') as f:
            pickle.dump({'graph': g}, f)
        self.graph = load_graph(str(path)); self.graph_path = str(path)
        # segment = edges between nodes 10..20 (x = 15..25)
        self.edges = [int(k) for k, (a, b) in enumerate(self.graph['edges']) if 10 <= a <= 20 and 10 <= b <= 20]

    def tearDown(self):
        self.temp.cleanup()

    def test_remove(self):
        from vatl_sdf.perturbation import perturb
        sdf, g = perturb(self.sdf, self.affine, self.graph, self.edges, 'remove')
        self.assertGreater(sdf[20, 12, 12], 0.)                 # segment cleared
        self.assertLess(sdf[8, 12, 12], 0.); self.assertLess(sdf[32, 12, 12], 0.)   # rest of the tube intact
        np.testing.assert_array_equal(sdf[:, :, :2], self.sdf[:, :, :2])            # far away unchanged
        self.assertEqual(len(g['edges']), len(self.graph['edges']) - len(self.edges))
        self.assertTrue(np.all(np.bincount(g['edges'].ravel(), minlength=len(g['pos'])) > 0))  # no orphan nodes

    def test_thin(self):
        from vatl_sdf.perturbation import perturb
        tail = [int(k) for k, (a, b) in enumerate(self.graph['edges']) if a >= 21 and b >= 21 and max(a, b) <= 30]  # x = 26..35
        sdf, g = perturb(self.sdf, self.affine, self.graph, tail, 'thin')
        self.assertAlmostEqual(float(sdf[32, 12, 12]), -1., places=1)  # radius 1 mm away from the attachment
        self.assertGreater(sdf[32, 13, 13], 0.)                       # 1.41 mm off axis: outside
        self.assertTrue((sdf[22:36, 12, 12] < 0).all())               # still connected through the attachment (x = 26)
        np.testing.assert_allclose(g['radius'][22:31], 1.); self.assertEqual(g['radius'][21], 2.)  # boundary node kept
        sdf, g = perturb(self.sdf, self.affine, self.graph, self.edges, 'thin')
        np.testing.assert_allclose(g['radius'][range(11, 20)], 1.)    # node 15 joins only a small branch: not an attachment
        self.assertEqual(g['radius'][10], 2.); self.assertEqual(g['radius'][20], 2.)  # attachments to the large tube kept
        self.assertLess(sdf[20, 12, 12], -0.5); self.assertGreater(sdf[20, 13, 13], 0.)  # thinned through the small-branch junction

    def test_thin_factor(self):
        from vatl_sdf.perturbation import perturb
        tail = [int(k) for k, (a, b) in enumerate(self.graph['edges']) if a >= 21 and b >= 21 and max(a, b) <= 30]
        default, _ = perturb(self.sdf, self.affine, self.graph, tail, 'thin')
        half, _ = perturb(self.sdf, self.affine, self.graph, tail, 'thin', factor=0.5)
        np.testing.assert_array_equal(default, half)                  # legacy copies (no factor) regenerate identically
        sdf, g = perturb(self.sdf, self.affine, self.graph, tail, 'thin', factor=0.3)
        self.assertAlmostEqual(float(sdf[32, 12, 12]), -0.6, places=1)
        self.assertGreater(sdf[32, 13, 12], 0.)
        np.testing.assert_allclose(g['radius'][22:31], 0.6)

    def test_stenosis(self):
        from vatl_sdf.perturbation import perturb, stenosis_profile
        np.testing.assert_allclose(stenosis_profile(np.array([0., 2., 3., 4., 9.]), 0.5), [0.5, 0.5, 0.75, 1., 1.])
        sdf, g = perturb(self.sdf, self.affine, self.graph, self.edges, 'stenosis', factor=0.5)   # x = 15..25, centre x = 20
        self.assertAlmostEqual(float(sdf[20, 12, 12]), -1., places=1)   # plateau: radius 1 mm
        self.assertGreater(sdf[20, 13, 13], 0.)
        self.assertAlmostEqual(float(sdf[17, 12, 12]), -1.5, places=1)  # taper (d = 3 mm): radius 1.5 mm
        self.assertAlmostEqual(float(sdf[14, 12, 12]), -2., places=1)   # outside the 8 mm zone: original
        self.assertTrue((sdf[5:36, 12, 12] < 0).all())                  # lumen continuous
        np.testing.assert_array_equal(sdf[:9], self.sdf[:9])            # far away unchanged
        self.assertAlmostEqual(float(g['radius'][15]), 1.); self.assertAlmostEqual(float(g['radius'][12]), 1.5)
        self.assertAlmostEqual(float(g['radius'][10]), 2.); self.assertEqual(len(g['edges']), len(self.graph['edges']))
        np.testing.assert_array_equal(g['pos'], self.graph['pos'])

    def test_kinds_and_labels(self):
        from vatl_sdf.perturbation import copy_label, parse_kinds
        legacy = parse_kinds(['remove', 'thin', 'displace:3', 'displace:5'])
        self.assertEqual([copy_label(k) for k in legacy], ['remove0', 'thin0', 'displace3', 'displace5'])  # unchanged ids
        new = parse_kinds(['thin:0.3', 'stenosis:0.5', 'stenosis:0.3', 'displace:2'])
        self.assertEqual([copy_label(k) for k in new], ['thin0.3', 'stenosis0.5', 'stenosis0.3', 'displace2'])
        self.assertEqual(new[0], {'kind': 'thin', 'magnitude': 0., 'factor': 0.3})
        self.assertEqual(new[3], {'kind': 'displace', 'magnitude': 2., 'factor': None})
        with self.assertRaises(ValueError):
            parse_kinds(['bend:2'])

    def test_displace(self):
        from vatl_sdf.perturbation import perturb
        tail = [int(k) for k, (a, b) in enumerate(self.graph['edges']) if a >= 21 and b >= 21 and max(a, b) <= 30]  # x = 26..35
        sdf, g = perturb(self.sdf, self.affine, self.graph, tail, 'displace', 3., [0., 0., 1.])
        self.assertGreater(sdf[33, 12, 12], 0.)                  # original axis vacated (shift 3 > radius 2)
        self.assertLess(sdf[33, 12, 15], 0.)                     # shifted tube present
        self.assertLess(sdf[24, 12, 12], 0.)                     # tapered to 0 at the attachment (node 21, x = 26)
        moved = np.linalg.norm(g['pos'] - self.graph['pos'], axis=1)
        self.assertLessEqual(moved.max(), 3. + 1e-9); self.assertEqual(moved[21], 0.); self.assertAlmostEqual(moved[29], 3.)

    def test_virtual_source_roundtrip(self):
        import nibabel as nib
        from vatl_sdf import perturbation, sources
        from vatl_sdf.graph_coords import load_graph
        nib.save(nib.Nifti1Image(self.sdf, self.affine), Path(self.temp.name) / 's.nii.gz')
        manifest = Path(self.temp.name) / 'm.json'
        manifest.write_text(json.dumps({'copies': [{'copy_id': 'c0', 'scan_age': 30., 'sex': 'F', 'source_sdf': str(Path(self.temp.name) / 's.nii.gz'),
                                                     'source_graph': self.graph_path, 'matched_edges': self.edges, 'kind': 'remove',
                                                     'magnitude': 0., 'direction': None}]}))
        perturbation.configure(manifest)
        image = sources.load_image('perturb:c0')
        expected, g = perturbation.perturb(self.sdf, self.affine, self.graph, self.edges, 'remove')
        np.testing.assert_array_equal(np.asarray(image.dataobj), expected)
        self.assertEqual(len(load_graph('perturb:c0')['edges']), len(g['edges']))
        second = Path(self.temp.name) / 'm2.json'   # a second manifest (Step C) is added to the first
        second.write_text(json.dumps({'copies': [{'copy_id': 'c1', 'scan_age': 30., 'sex': 'F', 'source_sdf': str(Path(self.temp.name) / 's.nii.gz'),
                                                   'source_graph': self.graph_path, 'matched_edges': self.edges, 'kind': 'stenosis',
                                                   'magnitude': 0., 'factor': 0.5, 'direction': None}]}))
        perturbation.configure(second, reset=False)
        expected, _ = perturbation.perturb(self.sdf, self.affine, self.graph, self.edges, 'stenosis', factor=0.5)
        np.testing.assert_array_equal(np.asarray(sources.load_image('perturb:c1').dataobj), expected)
        self.assertIsNotNone(sources.load_image('perturb:c0'))
        self.assertTrue(np.allclose(nib.load(Path(self.temp.name) / 's.nii.gz').get_fdata(), self.sdf))  # source untouched

    def test_selection(self):
        from vatl_sdf.perturbation import select_targets, subject_segments, willis_centre
        pop = {'edge_points_mm': [np.c_[np.arange(15., 26.), np.full(11, 12.), np.full(11, 12.)],
                                  np.c_[np.full(15, 20.), np.arange(3., 18.), np.full(15, 30.)]],
               'edge_probability': np.array([0.9, 0.3]), 'radius_mm': 3., 'edge_cover': 0.5}
        cmap = np.zeros((40, 24, 40)); cmap[15:26, 12, 12] = 0.8
        centre, chosen, c, _ = willis_centre(pop, cmap, np.eye(4), 0.5)
        self.assertEqual(chosen, [0]); np.testing.assert_allclose(centre, [20., 12., 12.])
        self.assertEqual(select_targets(pop, centre, 25., 0.5), [0])
        segs = subject_segments(self.graph, np.zeros(3), np.eye(3), np.zeros(3), pop, [0], 3., 0.5)
        self.assertTrue(segs[0]['present'] and segs[0]['eligible'])
        self.assertAlmostEqual(segs[0]['segment_radius_mm'], 2.)
        thin_graph = dict(self.graph, radius=np.full(len(self.graph['radius']), 1.3))
        self.assertFalse(subject_segments(thin_graph, np.zeros(3), np.eye(3), np.zeros(3), pop, [0], 3., 0.5)[0]['eligible'])
