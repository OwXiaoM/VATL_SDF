"""Split integrity tests: actual authoritative cohort plus synthetic failure cases."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from mra_sdf_only.config import load_config
from mra_sdf_only.dataset import read_metadata, MRASDFDataset
from mra_sdf_only.splits import read_splits, ordered_rows, check_resume_subjects
from mra_sdf_only.test_individual import select_subjects
from mra_sdf_only.train import train
from tests.test_pipeline import tiny_config

SPLIT = Path('/leonardo/home/userexternal/xzhang01/data/v1/subject_ids_label_t1_fast.yaml')
KEY = 'mra_atlas_synthseg_label_t1_fast.subject_ids'
BASE = Path('configs/ablations/02_sdf_only.yaml')


class SplitTests(unittest.TestCase):
    def test_actual_497_55_membership_and_yaml_order(self):
        self.assertTrue(SPLIT.is_file(), 'Authoritative split must be accessible for this project integration check')
        config = load_config(BASE)
        splits = read_splits(SPLIT, KEY, {'train':497,'val':55,'test':0})
        frame = read_metadata(config)
        self.assertEqual(len(frame),497)
        self.assertEqual(frame.subject_id.tolist(),splits['train'])
        self.assertFalse(set(frame.subject_id) & set(splits['val']))
        self.assertFalse(set(splits['train']) & set(splits['val']))
        self.assertEqual(config['loss']['mra_weight'],0)
        self.assertIsNone(config['resume']);self.assertTrue(config['require_fresh_start'])
        selected3 = select_subjects(config['data']['tsv'],frame,3,SPLIT,KEY+'.val')
        selected5 = select_subjects(config['data']['tsv'],frame,5,SPLIT,KEY+'.val')
        self.assertEqual(selected3.subject_id.tolist(),splits['val'][:3])
        self.assertEqual(selected5.subject_id.tolist(),splits['val'][:5])
        self.assertEqual(selected5.subject_id.tolist()[:3],selected3.subject_id.tolist())
        contaminated = pd.read_csv(config['data']['tsv'],sep='\t')
        with self.assertRaisesRegex(ValueError,'match declared population train'):
            select_subjects(config['data']['tsv'],contaminated,3,SPLIT,KEY+'.val')
        with self.assertRaisesRegex(ValueError,'contaminated or incompatible'):
            check_resume_subjects({'metadata':contaminated.to_dict(orient='records')},frame)
        with self.assertRaisesRegex(ValueError,'outside the declared'):
            MRASDFDataset(config,frame=selected3)

    def test_overlap_duplicate_wrong_count_and_missing_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'splits.yaml'
            path.write_text(yaml.safe_dump({'train':['a'],'val':['a'],'test':[]}))
            with self.assertRaisesRegex(ValueError,'overlap'):read_splits(path,None)
            path.write_text(yaml.safe_dump({'train':['a','a'],'val':[],'test':[]}))
            with self.assertRaisesRegex(ValueError,'Duplicate'):read_splits(path,None)
            path.write_text(yaml.safe_dump({'train':['a'],'val':['b'],'test':[]}))
            with self.assertRaisesRegex(ValueError,'expected counts'):read_splits(path,None,{'train':497,'val':55,'test':0})
            with self.assertRaisesRegex(ValueError,'missing from metadata'):
                ordered_rows(pd.DataFrame({'subject_id':['b']}),['a'])

    def test_null_and_wrong_training_split_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=tiny_config(tmp)
            config['data']['subject_ids']=None
            with self.assertRaisesRegex(ValueError,'Explicit subject split'):read_metadata(config)
            config=tiny_config(tmp);config['data']['split']='val'
            with self.assertRaisesRegex(ValueError,'split=train'):read_metadata(config)
            config=tiny_config(tmp);config['data']['expected_split_counts']={'train':552,'val':0,'test':0}
            with self.assertRaisesRegex(ValueError,'expected counts'):read_metadata(config)

    def test_fresh_start_blocks_resume_before_loading_weights(self):
        config=load_config(BASE);config['resume']='/any/contaminated/checkpoint.pt'
        with patch('mra_sdf_only.train.torch.load') as load:
            with self.assertRaisesRegex(ValueError,'fresh start'):train(config)
            load.assert_not_called()

    def test_only_split_and_run_identity_changed_from_original_sdf_only(self):
        original = Path('/leonardo_scratch/large/userexternal/xzhang01/Cinema_output/mra_sdf_only/20261027_144056_557977_ablation_02_sdf_only_62280701/config_final.yaml')
        self.assertTrue(original.is_file())
        old=yaml.safe_load(original.read_text());new=load_config(BASE)
        for section in ('model','loss','sampling','sdf','optimizer','training','atlas','seed','device','output_root'):
            self.assertEqual(old[section],new[section],section)
        for key,value in old['data'].items():
            if key!='subject_ids':self.assertEqual(value,new['data'][key],key)


if __name__=='__main__':unittest.main()
