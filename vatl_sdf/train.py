"""Single-device and torchrun DDP entry point; no dependency on VATL_MM."""
import argparse
import json
import math
import os
import random
import time
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
from .dataset import SDFGraphDataset, collate_subjects
from .model import SDFModel
from .losses import sdf_losses, add_graph_losses, eikonal_loss, field_losses, alignment_losses, affine_regularization
from .dataset import vessel_centroids, alignment_enabled
from .alignment_metrics import transform_statistics
from .config import field_names, graph_terms
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
    checkpoint=torch.load(config['resume'],map_location='cpu',weights_only=False) if config['resume'] else None
    data=SDFGraphDataset(config);frame=data.frame
    if checkpoint:
        check_resume_subjects(checkpoint, frame)
        if checkpoint['world_size']!=world:raise ValueError('Exact resume requires the same world size')
        for section in ('data','sampling','model','loss','optimizer','sdf','seed','alignment'):
            if checkpoint['config'].get(section)!=config.get(section):raise ValueError(f'Resume config changed immutable section {section}')
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
    model=SDFModel(config,len(data)).to(device)
    if checkpoint:model.load_state_dict(checkpoint['model'])
    else:initialize_subject_parameters(model,frame,config)
    optimizer=torch.optim.Adam(parameter_groups(model,config),weight_decay=config['optimizer']['weight_decay'])
    horizon=scheduler_horizon(config,len(data),world)
    if rank==0:print(f'Scheduler horizon (epochs): {horizon}',flush=True)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=horizon,eta_min=config['optimizer']['eta_min'])
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng_states'][rank],device)
    # Latents/transforms are model Parameters, so DDP gradient averaging keeps the
    # ENTIRE subject table synchronized at every step, including uneven ownership.
    network=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,find_unused_parameters=True) if world>1 else model
    position=checkpoint['position'] if checkpoint else {'epoch':0,'batch':0,'inner':0}
    step=checkpoint['step'] if checkpoint else 0
    limit=config['training']['max_steps']
    stop=config['training']['stop_after_epoch']
    if stop is not None and position['epoch']>stop:
        if rank==0:print(f'Already past stop_after_epoch={stop}; no updates made',flush=True)
        if world>1:dist.destroy_process_group()
        return directory
    if limit is not None and step>=limit:
        if rank==0:print('Already at max_steps; no updates made',flush=True)
        if world>1:dist.destroy_process_group()
        return directory
    terms=graph_terms(config);eikonal=bool(config.get('graph')) and config['graph']['eikonal']['enabled'];fields_on=bool(field_names(config))
    if rank==0:print(f'Output: {directory}; subjects={len(data)}; world_size={world}; device={device}; graph_terms={terms}; eikonal={eikonal}',flush=True)
    aligning=alignment_enabled(config)
    for epoch in range(position['epoch'],config['training']['epochs']):
        data.epoch=epoch
        # Alignment stage: latents gated to zero (one shared template), alignment loss only.
        stage=aligning and epoch<config['alignment']['epochs']
        model.latent_gate=0.0 if stage else 1.0
        active=[] if stage else terms
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
            b,p,_=batch['sdf_coords'].shape
            coords=batch['sdf_coords'].reshape(-1,3).to(device)
            targets=batch['sdf_target_mm'].reshape(-1,1).to(device)
            ids=indices.repeat_interleave(p);cond=conditions.repeat_interleave(p,dim=0)
            perm=torch.randperm(len(coords),generator=torch.Generator().manual_seed(config['seed']+epoch*100000+batch_index*100+rank)).to(device)
            arrays=(coords[perm],targets[perm],ids[perm],cond[perm])
            extra=[k for k in ('field_target','field_weight','align_target','align_weight') if k in batch]
            arrays+=tuple(batch[k].reshape(-1,batch[k].shape[-1]).to(device)[perm] for k in extra)
            steps=math.ceil(len(coords)/config['training']['task_points_per_step'])
            # Extra graph points per enabled term, shuffled independently and split over the same steps.
            graph_arrays={}
            for t,term in enumerate(active):
                gp=batch[f'{term}_coords'].shape[1]
                gperm=torch.randperm(b*gp,generator=torch.Generator().manual_seed(config['seed']+epoch*100000+batch_index*100+rank+(t+1)*7919)).to(device)
                graph_arrays[term]=tuple(x.to(device)[gperm] for x in (batch[f'{term}_coords'].reshape(-1,3),batch[f'{term}_target_mm'].reshape(-1,1),
                                         batch[f'{term}_weight'].reshape(-1,1),indices.repeat_interleave(gp),conditions.repeat_interleave(gp,dim=0)))
            for inner in range(steps):
                if epoch==position['epoch'] and batch_index==position['batch'] and inner<position['inner']:continue
                sc,st,si,sv,*extra_slices=(x[inner*len(x)//steps:(inner+1)*len(x)//steps] for x in arrays)
                ex=dict(zip(extra,extra_slices))
                pieces={term:tuple(x[inner*len(x)//steps:(inner+1)*len(x)//steps] for x in graph_arrays[term]) for term in active}
                if eikonal:sc=sc.detach().requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                all_coords=torch.cat([sc]+[pieces[t][0] for t in active]);all_ids=torch.cat([si]+[pieces[t][3] for t in active])
                all_cond=torch.cat([sv]+[pieces[t][4] for t in active])
                if fields_on or aligning:output,moe,fields=network(all_coords,all_ids,all_cond,balance_count=len(sc) if active else None,return_fields=True)
                else:output,moe=network(all_coords,all_ids,all_cond,balance_count=len(sc) if active else None)
                outputs=output.split([len(sc)]+[len(pieces[t][0]) for t in active])
                legacy_tf=model.transforms is not None and model.transforms.requires_grad
                if stage:
                    parts=alignment_losses(outputs[0],st,fields['align_distance'][:len(sc)],ex['align_target'],ex['align_weight'],moe,config)
                else:
                    parts=sdf_losses(outputs[0],st,moe,model.transforms[indices] if legacy_tf else None,config)
                    if active:parts=add_graph_losses(parts,[(o,pieces[t][1],pieces[t][2]) for o,t in zip(outputs[1:],active)],active,config)
                    if fields_on:parts=field_losses(parts,{k:v[:len(sc)] for k,v in fields.items() if k!='align_distance'},ex['field_target'],ex['field_weight'],config)
                    if aligning:
                        # Refinement stage: L2 prior on the squared latent norm of the batch subjects.
                        prior=model.latents[indices].flatten(1).square().sum(1).mean()
                        parts['latent_prior_raw']=prior;parts['latent_prior_weighted']=config['alignment']['latent_prior_weight']*prior
                        parts['total']=parts['total']+parts['latent_prior_weighted']
                regularization=affine_regularization(model)
                if regularization is not None:
                    parts['affine_reg_raw']=regularization
                    parts['affine_reg_weighted']=config['model']['transform'].get('affine_reg_weight',0.)*regularization
                    parts['total']=parts['total']+parts['affine_reg_weighted']
                if eikonal and not stage:
                    band=st.abs().reshape(-1)<=config['graph']['eikonal']['band_mm']
                    eik=eikonal_loss(outputs[0],sc,model.half_bbox,config['sdf']['truncation_mm'],band)
                    parts['eikonal_raw']=eik;parts['eikonal_weighted']=config['graph']['eikonal']['weight']*eik
                    parts['total']=parts['total']+parts['eikonal_weighted']
                if not torch.isfinite(parts['total']):raise FloatingPointError('Nonfinite loss')
                parts['total'].backward();optimizer.step();step+=1
                values=torch.stack([v.detach() for v in parts.values()])
                if world>1:dist.all_reduce(values);values/=world
                if rank==0:
                    report={'epoch':epoch,'batch':batch_index,'inner':inner,'step':step,'time':time.time(),
                        'stage':'align' if stage else 'main','sdf_points_per_rank':len(sc),**{f'{t}_points_per_rank':len(pieces[t][0]) for t in active},
                        **dict(zip(parts,values.cpu().tolist()))}
                    with (Path(directory)/'losses.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
                    if step%10==0 or step==1:print(json.dumps(report),flush=True)
                if limit is not None and step>=limit:
                    save_checkpoint(directory,model,optimizer,scheduler,config,frame,{'epoch':epoch,'batch':batch_index,'inner':inner+1},step,rank,world)
                    if world>1:verify_synchronization(model,rank,world);dist.destroy_process_group()
                    return directory
        scheduler.step()
        position={'epoch':epoch+1,'batch':0,'inner':0}
        if rank==0:log_subject_parameters(directory,model,epoch,step,stage)
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


def initialize_subject_parameters(model,frame,config):
    """Fresh start only. Alignment stage: latents start at zero. centroid_init: translation_i = mean(c) - c_i
    (normalized units), so every subject's vessel centroid maps to the population mean centroid."""
    t=config['model'].get('transform',{})
    with torch.no_grad():
        if alignment_enabled(config):model.latents.zero_()
        if t.get('centroid_init'):
            centroids=torch.as_tensor(vessel_centroids(frame,config),device=model.latents.device)
            model.transform_translation.copy_(centroids.mean(0,keepdim=True)-centroids)


def parameter_groups(model,config):
    """Backbone/heads, latents and transform tables in separate Adam groups (own learning rates)."""
    o=config['optimizer'];tables=model.subject_tensors()
    groups=[{'params':[p for name,p in model.named_parameters() if name not in tables],'lr':o['lr_model']},
            {'params':[model.latents],'lr':o['lr_latent']}]
    lr={'transforms':o['lr_transform'],'transform_rotation':o.get('lr_rotation'),'transform_translation':o.get('lr_translation'),
        'transform_log_scale':o.get('lr_scale'),'transform_shear':o.get('lr_scale')}
    for name,value in tables.items():
        if name!='latents' and value.requires_grad:groups.append({'params':[value],'lr':lr[name],'name':name})
    return groups


def scheduler_horizon(config,subjects,world):
    """Cosine horizon in epochs: the configured integer, or 'run' = the number of epochs this run will train
    (epochs, stop_after_epoch + 1 and max_steps, whichever ends first). A resumed run keeps the horizon stored in
    its scheduler state, so extending stop_after_epoch in a continuation does not stretch the schedule."""
    horizon=config['optimizer']['scheduler_epochs']
    if horizon!='run':return horizon
    t=config['training'];per_rank=math.ceil(subjects/world);spr=t['subjects_per_rank']
    batches=math.ceil(per_rank/spr)
    sizes=[spr]*(batches-1)+[per_rank-spr*(batches-1)]
    steps=sum(math.ceil(b*config['sampling']['sdf']['points']/t['task_points_per_step']) for b in sizes)
    epochs=[t['epochs']]
    if t['stop_after_epoch'] is not None:epochs.append(t['stop_after_epoch']+1)
    if t['max_steps'] is not None:epochs.append(math.ceil(t['max_steps']/steps))
    return max(1,min(epochs))


def log_subject_parameters(directory,model,epoch,step,stage):
    """Per-epoch transform and latent statistics (transforms.jsonl)."""
    a,t=model.maps_mm()
    norms=model.latents.detach().flatten(1).norm(dim=1)
    record={'epoch':epoch,'step':step,'stage':'align' if stage else 'main',
            **transform_statistics(a.cpu().numpy(),t.cpu().numpy()),
            'latent_norm':{'median':norms.median().item(),'p95':norms.quantile(.95).item(),'max':norms.max().item()}}
    with (Path(directory)/'transforms.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')


def verify_synchronization(model,rank,world):
    # Full tables (not scalar checksums) compared once at shutdown.
    difference=[]
    for value in (v.detach() for v in model.subject_tensors().values()):
        reference=value.clone();dist.broadcast(reference,src=0)
        difference.append((reference-value).abs().max())
    error=torch.stack(difference).max();dist.all_reduce(error,op=dist.ReduceOp.MAX)
    if rank==0:print('DDP_LATENT_TRANSFORM_MAX_DIFFERENCE',error.item(),flush=True)
    if error.item()>1e-6:raise RuntimeError('DDP subject parameters drifted')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');p.add_argument('--set',nargs='*',default=[])
    args=p.parse_args();train(load_config(args.config,args.set))

if __name__=='__main__':main()
