"""Scale-gap probe (CPU)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

from tests.test_alignment import normalized, staged
from tests.test_pipeline import tiny_config
from vatl_sdf.config import condition_vector
from vatl_sdf.dataset import read_metadata
from vatl_sdf.losses import normalized_sdf
from vatl_sdf.model import SDFModel
from vatl_sdf.train import train


class ProbeTests(unittest.TestCase):
    def test_own_matches_training_forward(self):
        from analysis.scale_gap import probe_subjects
        with tempfile.TemporaryDirectory() as d:
            c = normalized(staged(tiny_config(d), epochs=1), 'affine')
            c['training'].update(max_steps=None, epochs=2, stop_after_epoch=None)
            torch.manual_seed(0)
            run = Path(train(c))
            out = Path(d) / 'probe'
            with mock.patch.object(sys, 'argv', ['x', '--runs', f'a={run}', f'b={run}', '--output', str(out),
                                                 '--subjects', '2', '--points', '8', '--device', 'cpu']):
                probe_subjects.main()
            table = pd.read_csv(out / 'per_subject.tsv', sep='\t')
            self.assertEqual(len(table), 2 * 2 * 2)  # runs x checkpoints x subjects
            # 'own' equals the training forward path (transform + own latent + own condition).
            last = sorted(run.glob('checkpoint_step_*.pt'))[-1]
            state = torch.load(last, map_location='cpu', weights_only=False)
            frame = pd.DataFrame(state['metadata'])
            model = SDFModel(state['config'], len(frame)); model.load_state_dict(state['model']); model.eval()
            coords, target = probe_subjects.probe_samples(frame, state['config'], [1], 8)[1]
            x = torch.as_tensor(coords); ids = torch.ones(len(x), dtype=torch.long)
            cond = torch.as_tensor(condition_vector(state['config'], frame.scan_age[1], frame.sex[1])).expand(len(x), -1)
            with torch.no_grad():
                pred, _ = model(x, ids, cond)
            expected = (pred[:, 0] - normalized_sdf(torch.as_tensor(target), state['config']['sdf']['truncation_mm']).reshape(-1)).abs().mean().item()
            row = table[(table.run == 'a') & (table.step == state['step']) & (table.subject_index == 1)].iloc[0]
            self.assertAlmostEqual(row.own, expected, places=5)


class SubsetTests(unittest.TestCase):
    def test_subset_include_keeps_listed_ids_and_yaml_order(self):
        with tempfile.TemporaryDirectory() as d:
            c = tiny_config(d)
            c['data']['max_subjects'] = 1
            self.assertEqual(read_metadata(c).subject_id.tolist(), ['s0'])            # default: head(N)
            c['data']['subset_include'] = ['s1']
            self.assertEqual(read_metadata(c).subject_id.tolist(), ['s1'])
            c['data']['max_subjects'] = 2
            self.assertEqual(read_metadata(c).subject_id.tolist(), ['s0', 's1'])      # YAML order kept
            for bad in (['nope'], ['s1', 's1'], ['s0', 's1', 's1']):
                c['data']['subset_include'] = bad
                with self.assertRaises(ValueError):
                    read_metadata(c)


if __name__ == '__main__':
    unittest.main()


class AblationTableTests(unittest.TestCase):
    def test_reads_full_scale_and_pilot_layouts(self):
        import json
        from analysis.scale_gap.ablation_table import row_for
        conn = {'component_ratio': 2.0, 'component_ratio_ge_min': 3.0}
        diag = {'subjects': [{'grid_x1': {'zero_level_dice': v, 'band_mae_mm': 1.0, 'volume_ratio_pred_over_ref': 1.1,
                                          'connectivity': conn}} for v in (0.2, 0.4, 0.9)]}
        median = {'zero_level_dice': 0.5, 'band_mae_mm': 1.2, 'volume_ratio_pred_over_ref': 1.3, 'component_ratio_ge_min': 4.0}
        gen = {'test': {'epochs': {'10': {'median': {**median, 'zero_level_dice': 0.1}}, '20': {'median': median}}},
               'recurrence': {'train_vs_train/learned': {'3mm': {'median': 0.41}}, 'test_vs_train/learned': {'3mm': {'median': 0.44}}}}
        with tempfile.TemporaryDirectory() as d:
            for layout in ('', 'post'):
                root = Path(d) / (layout or 'full')
                (root / 'diagnostics').mkdir(parents=True)
                (root / 'diagnostics' / 'diagnostics.json').write_text(json.dumps(diag))
                (root / layout / 'generalization').mkdir(parents=True)
                (root / layout / 'generalization' / 'summary.json').write_text(json.dumps(gen))
                (root / layout / 'inspect').mkdir(parents=True)
                pd.DataFrame({'moe1_entropy': [0.9, 0.6], 'latent_norm_median': [0.0, 1.2]}).to_csv(
                    root / layout / 'inspect' / 'checkpoints.tsv', sep='\t', index=False)
                row = row_for('x', root)
                self.assertAlmostEqual(row['0c_dice'], 0.4)
                self.assertEqual(row['test_dice'], 0.5)  # final epoch, not first
                self.assertEqual((row['rec3_train'], row['rec3_test']), (0.41, 0.44))
                self.assertEqual(row['moe1_entropy'], 0.6)
                self.assertNotIn('train_refit_dice', row)
