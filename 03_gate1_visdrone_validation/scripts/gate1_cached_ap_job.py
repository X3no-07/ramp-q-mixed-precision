"""One immutable cached-matrix AP job. NumPy only; no image/model access."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import gate1_official_ap as AP


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def identity(path):
    path=Path(path).resolve(strict=True)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest}


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def curve_from_entries(score,flag):
    if not len(score):return {'threshold':np.empty(0),'TP':np.empty(0,np.int64),'FP':np.empty(0,np.int64)}
    order=np.argsort(-score,kind='stable');s=score[order];f=flag[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    return {'threshold':s[ends],'TP':np.cumsum(f==1)[ends],'FP':np.cumsum(f==0)[ends]}


def rank_entries(gt_images,dt_images):
    scores=[];flags=[]
    for category in range(1,11):
        for gt,dt in zip(gt_images,dt_images):
            d=dt[:500];_,m=AP.eval_res(gt[gt[:,5]==category,:5],d[d[:,5]==category,:5],.5)
            scores.extend(m[:,4]);flags.extend(m[:,5])
    return np.array(scores,np.float64),np.array(flags,np.int8)


def run(job_path):
    job=read(job_path);out=Path(job['output']).resolve(strict=True);started=time.monotonic()
    if identity(job['matrices']['path'])!=job['matrices']:raise ValueError('Frozen matrix binding differs')
    with np.load(job['matrices']['path'],allow_pickle=False) as data:
        gt=[data[f'gt_{i}'].copy() for i in range(job['images'])]
        dt=[data[f'dt_{i}'].copy() for i in range(job['images'])]
    ap=AP.calc_accuracy(gt,dt,capture_intermediates=False)
    np.savez_compressed(out/'AP_AR_arrays.npz',AP=ap['AP'],AR=ap['AR'],evalClass=ap['evalClass'])
    score,flag=rank_entries(gt,dt);curve=curve_from_entries(score,flag)
    np.savez_compressed(out/'official_IoU05_rank_curve.npz',score=score,flag=flag,**curve)
    result={'cohort':job['cohort'],'branch':job['branch'],'images':job['images'],'metrics_percent':ap['metrics'],
            'matrices':job['matrices'],'job':identity(job_path),'pid':os.getpid(),'parent_pid':os.getppid(),
            'finished_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,
            'torch_imported':'torch' in sys.modules,'scientific_acceptance':False,'new_detector_forwards':0}
    write(out/'metrics.json',result)
    print(json.dumps({'status':'COMPLETE_AP_JOB','cohort':job['cohort'],'branch':job['branch'],'images':job['images'],'AP':ap['metrics']['AP_all']}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job');args=parser.parse_args()
    try:run(Path(args.job))
    except BaseException as error:
        job=read(args.job)
        write(Path(job['output'])/'failure_receipt.json',{'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
