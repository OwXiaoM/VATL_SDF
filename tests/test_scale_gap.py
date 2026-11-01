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


if __name__ == '__main__':
    unittest.main()
