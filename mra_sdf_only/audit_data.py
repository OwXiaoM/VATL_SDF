"""MRA/SDF-only QC; cannot prove sign semantics without an external vessel mask."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import load_config
from .dataset import read_metadata,load_pair,sample_tasks


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');p.add_argument('--limit',type=int,default=3)
    p.add_argument('--output',required=True);a=p.parse_args();c=load_config(a.config)
    with Path(a.output).open('x') as stream:
        for _,row in read_metadata(c).head(a.limit).iterrows():
            pair=load_pair(row,c);report=pair['report']
            mra=pair['mra'];edge=np.concatenate([mra[0].ravel(),mra[-1].ravel(),mra[:,0].ravel(),mra[:,-1].ravel(),mra[:,:,0].ravel(),mra[:,:,-1].ravel()])
            report['edge_exact_zero_fraction']=float((edge==0).mean())
            report['pre_masking_assessment']='Zeros/edges alone do not establish provenance; upstream masking is possible and cannot be undone by this pipeline.'
            report['sampling']=sample_tasks(pair,c,np.random.default_rng(c['seed']))['sampling_report']
            stream.write(json.dumps(report)+'\n');stream.flush();print(json.dumps(report),flush=True)

if __name__=='__main__':main()
