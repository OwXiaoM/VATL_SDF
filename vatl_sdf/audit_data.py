"""SDF/graph data QC: geometry, sampling pools and graph coordinates in the training frame."""
import argparse
import json
from pathlib import Path
import numpy as np
from .config import load_config
from .dataset import read_metadata, load_subject, sample_sdf
from .graph_coords import inside_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/sdf/00_sdf_baseline.yaml');p.add_argument('--limit',type=int,default=3)
    p.add_argument('--output',required=True);a=p.parse_args();c=load_config(a.config)
    with Path(a.output).open('x') as stream:
        for _,row in read_metadata(c).head(a.limit).iterrows():
            subject=load_subject(row,c);report=subject['report']
            report['graph_nodes_inside_mask']=float(inside_mask(subject['sdf']<0,subject['affine'],subject['graph']['pos']).mean())
            report['sampling']=sample_sdf(subject,c,np.random.default_rng(c['seed']))['sampling_report']
            stream.write(json.dumps(report)+'\n');stream.flush();print(json.dumps(report),flush=True)

if __name__=='__main__':main()
