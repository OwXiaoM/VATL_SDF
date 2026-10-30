"""Single-device and torchrun DDP entry point; no dependency on VATL_MM."""
import argparse
import json
import math
import os
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import yaml
from .config import load_config
from .dataset import MRASDFDataset, collate_subjects
from .model import MRASDFModel
from .losses import task_losses, gradient_diagnostics
from .inference import generate_atlas
from .splits import check_resume_subjects


def setup(config):
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world>1:
        backend='nccl' if config['device']=='cuda' else 'gloo'
        if backend=='nccl':torch.cuda.set_device(local)
        dist.init_process_group(backend=backend)
    device=torch.device('cuda',local) if config['device']=='cuda' else torch.device('cpu')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use device=cpu for CPU tests')
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    return rank,world,device


def local_rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state, device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and device.type=='cuda':torch.cuda.set_rng_state(state['cuda'].cpu())


def save_checkpoint(directory, model, optimizer, scheduler, config, frame, position, step, rank, world):
    rng=local_rng_state()
    states=[None]*world if world>1 else [rng]
    if world>1:dist.all_gather_object(states,rng)
    if rank==0:
        path=Path(directory)/f'checkpoint_step_{step:08d}.pt'
        if not path.exists():
            content={'format_version':1,'config':config,'metadata':frame.to_dict(orient='records'),
                     'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                     'position':position,'step':step,'world_size':world,'rng_states':states}
            temp=path.with_suffix('.tmp');torch.save(content,temp);temp.replace(path)
        print('Checkpoint:',path,flush=True)
    if world>1:dist.barrier()


def train(config):
    if config.get('require_fresh_start') and config.get('resume'):
        raise ValueError('Corrected baseline requires a fresh start; no checkpoint initialization/resume allowed')
    rank,world,device=setup(config)
    if world>1 and config['training']['gradient_diagnostics_every']:
        raise ValueError('Optional per-task autograd diagnostics are single-device only; keep disabled for DDP')
    checkpoint=torch.load(config['resume'],map_location='cpu',weights_only=False) if config['resume'] else None
    data=MRASDFDataset(config);frame=data.frame
    if checkpoint:
        check_resume_subjects(checkpoint, frame)
        if checkpoint['world_size']!=world:raise ValueError('Exact resume requires the same world size')
        for section in ('data','sampling','model','loss','optimizer','sdf','seed'):
            if checkpoint['config'][section]!=config[section]:raise ValueError(f'Resume config changed immutable section {section}')
        if checkpoint['metadata']!=frame.to_dict(orient='records'):raise ValueError('Resume metadata or subject order changed')
    directory=None
    if rank==0:
        if checkpoint:
            directory=str(Path(config['resume']).resolve().parent)
            (Path(directory)/f'resume_config_step_{checkpoint["step"]}.yaml').write_text(yaml.safe_dump(config))
        else:
            name=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+f'_{config["run_name"]}_{os.environ.get("SLURM_JOB_ID","local")}'
            directory=str(Path(config['output_root'])/name)
            Path(directory).mkdir(parents=True,exist_ok=False)
            (Path(directory)/'config_final.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
            frame.to_csv(Path(directory)/'training_metadata.tsv',sep='\t',index=False)
    if world>1:
        objects=[directory];dist.broadcast_object_list(objects,src=0);directory=objects[0]
    model=MRASDFModel(config,len(data)).to(device)
    if checkpoint:model.load_state_dict(checkpoint['model'])
    base_params=[p for name,p in model.named_parameters() if name not in ('latents','transforms')]
    groups=[{'params':base_params,'lr':config['optimizer']['lr_model']},
            {'params':[model.latents],'lr':config['optimizer']['lr_latent']}]
    if model.transforms.requires_grad:groups.append({'params':[model.transforms],'lr':config['optimizer']['lr_transform']})
    optimizer=torch.optim.Adam(groups,weight_decay=config['optimizer']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['optimizer']['scheduler_epochs'],eta_min=config['optimizer']['eta_min'])
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng_states'][rank],device)
    # Latents/transforms are model Parameters, so DDP gradient averaging keeps the
    # ENTIRE subject table synchronized at every step, including uneven ownership.
    network=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,find_unused_parameters=True) if world>1 else model
    position=checkpoint['position'] if checkpoint else {'epoch':0,'batch':0,'inner':0}
    step=checkpoint['step'] if checkpoint else 0
    limit=config['training']['max_steps']
    if limit is not None and step>=limit:
        if rank==0:print('Already at max_steps; no updates made',flush=True)
        if world>1:dist.destroy_process_group()
        return directory
    if rank==0:print(f'Output: {directory}; subjects={len(data)}; world_size={world}; device={device}',flush=True)
    for epoch in range(position['epoch'],config['training']['epochs']):
        data.epoch=epoch
        generator=torch.Generator().manual_seed(config['seed']+epoch)
        sampler=DistributedSampler(data,num_replicas=world,rank=rank,shuffle=True,seed=config['seed'],drop_last=False) if world>1 else RandomSampler(data,generator=generator)
        if world>1:sampler.set_epoch(epoch)
        loader=DataLoader(data,batch_size=config['training']['subjects_per_rank'],sampler=sampler,
            num_workers=config['data']['num_workers'],collate_fn=collate_subjects,generator=generator,
            pin_memory=device.type=='cuda',persistent_workers=False)
        for batch_index,batch in enumerate(loader):
            if epoch==position['epoch'] and batch_index<position['batch']:continue
            if rank==0:
                with (Path(directory)/'sampling.jsonl').open('a') as stream:
                    for report in batch['sampling_report']:stream.write(json.dumps(report)+'\n')
            if world>1:
                # Each rank has its own explicit sampling log; no loss of nonzero-rank diagnostics.
                with (Path(directory)/f'sampling_rank{rank}.jsonl').open('a') as stream:
                    for report in batch['sampling_report']:stream.write(json.dumps(report)+'\n')
            indices=batch['subject_index'].to(device)
            conditions=batch['condition'].to(device)
            tasks={}
            for name,target in (('mra','mra_target'),('sdf','sdf_target_mm')):
                b,p,_=batch[f'{name}_coords'].shape
                coords=batch[f'{name}_coords'].reshape(-1,3).to(device)
                targets=batch[target].reshape(-1,1).to(device)
                ids=indices.repeat_interleave(p);cond=conditions.repeat_interleave(p,dim=0)
                perm=torch.randperm(len(coords),generator=torch.Generator().manual_seed(config['seed']+epoch*100000+batch_index*100+rank)).to(device)
                tasks[name]=(coords[perm],targets[perm],ids[perm],cond[perm])
            nm,ns=len(tasks['mra'][0]),len(tasks['sdf'][0])
            steps=max(math.ceil(nm/config['training']['task_points_per_step']),math.ceil(ns/config['training']['task_points_per_step']))
            if steps>min(nm,ns):raise ValueError('Task counts too imbalanced to form nonempty independent task minibatches')
            for inner in range(steps):
                if epoch==position['epoch'] and batch_index==position['batch'] and inner<position['inner']:continue
                slices={name:tuple(x[inner*len(x)//steps:(inner+1)*len(x)//steps] for x in arrays) for name,arrays in tasks.items()}
                mc,mt,mi,mv=slices['mra'];sc,st,si,sv=slices['sdf']
                optimizer.zero_grad(set_to_none=True)
                output,moe=network(torch.cat([mc,sc]),torch.cat([mi,si]),torch.cat([mv,sv]))
                parts=task_losses(output,len(mc),mt,st,moe,model.transforms[indices] if model.transforms.requires_grad else None,config)
                diagnostics={}
                every=config['training']['gradient_diagnostics_every']
                if every and step%every==0:diagnostics=gradient_diagnostics(parts,model.backbone[0].linear.weight)
                if not torch.isfinite(parts['total']):raise FloatingPointError('Nonfinite loss')
                parts['total'].backward();optimizer.step();step+=1
                values=torch.stack([v.detach() for v in parts.values()])
                if world>1:dist.all_reduce(values);values/=world
                if rank==0:
                    report={'epoch':epoch,'batch':batch_index,'inner':inner,'step':step,
                        'mra_points_per_rank':len(mc),'sdf_points_per_rank':len(sc),
                        **dict(zip(parts,values.cpu().tolist())),**diagnostics}
                    with (Path(directory)/'losses.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
                    if step%10==0 or step==1:print(json.dumps(report),flush=True)
                if limit is not None and step>=limit:
                    save_checkpoint(directory,model,optimizer,scheduler,config,frame,{'epoch':epoch,'batch':batch_index,'inner':inner+1},step,rank,world)
                    if world>1:verify_synchronization(model,rank,world);dist.destroy_process_group()
                    return directory
        scheduler.step()
        position={'epoch':epoch+1,'batch':0,'inner':0}
        final=epoch==config['training']['epochs']-1 or (config['training']['stop_after_epoch'] is not None and epoch>=config['training']['stop_after_epoch'])
        if (epoch+1)%config['training']['checkpoint_every']==0 or final:
            save_checkpoint(directory,model,optimizer,scheduler,config,frame,position,step,rank,world)
        if config['training']['atlas_every'] and ((epoch+1)%config['training']['atlas_every']==0 or final):
            if rank==0:
                generate_atlas(model,frame,config,Path(directory)/'atlases'/f'epoch_{epoch:04d}')
            if world>1:dist.barrier()
            model.train()
        if final:break
    if world>1:verify_synchronization(model,rank,world);dist.destroy_process_group()
    return directory


def verify_synchronization(model,rank,world):
    # Full tables (not scalar checksums) compared once at shutdown.
    difference=[]
    for value in (model.latents.detach(),model.transforms.detach()):
        reference=value.clone();dist.broadcast(reference,src=0)
        difference.append((reference-value).abs().max())
    error=torch.stack(difference).max();dist.all_reduce(error,op=dist.ReduceOp.MAX)
    if rank==0:print('DDP_LATENT_TRANSFORM_MAX_DIFFERENCE',error.item(),flush=True)
    if error.item()>1e-6:raise RuntimeError('DDP subject parameters drifted')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');p.add_argument('--set',nargs='*',default=[])
    args=p.parse_args();train(load_config(args.config,args.set))

if __name__=='__main__':main()
