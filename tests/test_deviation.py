"""Deviation-statistics stage (CPU): subject sets, refitting, per-subject decodings."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
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
