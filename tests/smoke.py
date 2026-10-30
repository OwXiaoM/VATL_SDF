"""Bounded smoke runner: synthetic/resume or real data, CPU/CUDA/torchrun."""
import argparse
import json
import os
from pathlib import Path
import torch
import yaml
from mra_sdf_only.config import load_config
from mra_sdf_only.train import train
from tests.test_pipeline import tiny_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--real',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args()
    root=Path(a.output)
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    if rank==0:root.mkdir(parents=True,exist_ok=False)
    if world>1:
        # Shared files must exist before the trainer initializes its process group.
        # torchrun provides a filesystem-independent rendezvous; create data on rank 0
        # then use an atomic config-ready flag with a bounded timeout on other ranks.
        import time
        if rank!=0:
            deadline=time.monotonic()+60
            while not (root/'smoke.yaml').exists():
                if time.monotonic()>deadline:raise TimeoutError('Rank-zero smoke config missing')
                time.sleep(.05)
    if rank==0:
        if a.real:
            c=load_config('config.yaml')
            c['data'].update(max_subjects=2,num_workers=0)
            c['sampling']['mra']['points']=256;c['sampling']['sdf']['points']=256
            c['model'].update(latent_shape=[8,2,2,2],hidden_size=32,hidden_layers=3,modulated_layers=[1,3])
            c['training'].update(epochs=2,subjects_per_rank=1,task_points_per_step=256,max_steps=2,atlas_every=0,checkpoint_every=1)
            c['atlas'].update(spacing_mm=[8,8,8],chunk_size=20000)
        else:
            c=tiny_config(root)
        c['device']=a.device;c['output_root']=str(root/'runs')
        temporary=root/'smoke.tmp';temporary.write_text(yaml.safe_dump(c));temporary.replace(root/'smoke.yaml')
    c=load_config(root/'smoke.yaml')
    directory=train(c)
    if rank==0:
        (root/'run_directory.txt').write_text(directory)
        if world==1 and not a.real:
            # Resume from step 1 and compare to an uninterrupted two-step run.
            original=torch.load(next(Path(directory).glob('checkpoint_step_*.pt')),map_location='cpu',weights_only=False)
            resumed=c.copy();resumed['resume']=str(next(Path(directory).glob('checkpoint_step_*.pt')))
            resumed['training']=dict(c['training'],max_steps=2)
            resumed_dir=train(resumed)
            result=torch.load(Path(resumed_dir)/'checkpoint_step_00000002.pt',map_location='cpu',weights_only=False)
            uninterrupted=c.copy();uninterrupted['training']=dict(c['training'],max_steps=2)
            reference_dir=train(uninterrupted)
            reference=torch.load(Path(reference_dir)/'checkpoint_step_00000002.pt',map_location='cpu',weights_only=False)
            error=max((result['model'][key]-value).abs().max().item() for key,value in reference['model'].items())
            (root/'resume_comparison.json').write_text(json.dumps({'max_parameter_difference':error,'resumed_step':result['step']}))
            if error>1e-6:raise AssertionError(f'Resume diverged by {error}')
            print('RESUME_MAX_PARAMETER_DIFFERENCE',error,flush=True)

if __name__=='__main__':main()
