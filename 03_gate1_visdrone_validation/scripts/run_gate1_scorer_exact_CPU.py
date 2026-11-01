"""Single fixed exact128row feature optimization; no learning or detector calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import os
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/gate1/20261026_scorer_exact_CPU_optimization_v1'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'settings')
import numpy as np
import gate1_scorer_exact_features as O
import gate1_scorer_robustness_cost as D
P=D.P;S=O.S;E=D.E;T=D.T
read,write,identity,require=P.read,P.write,P.identity,P.require


def sources():
    result=D.source_bindings()
    for p in (Path(__file__),ROOT/'scripts/gate1_scorer_exact_features.py',ROOT/'tests/test_gate1_scorer_exact_features.py'):
        result[p.name]=identity(p)
    return result


def prepare():
    require(not (OUT/'contract.json').exists(),'Optimization contract already frozen')
    ext=read(E.OUT/'contract.json');scorer=read(S.OUT/'contract.json');threshold=read(T.OUT/'contract.json')
    members=[]
    for i,row in enumerate(ext['cohorts']['new286']):
        folder=E.OUT/'attempt_01/new_inference'/f'{i:04d}'
        members.append({'scope':'new286','index':i,'image_id':row['image_id'],'raw':identity(folder/'raw.npz'),'raw_key':'mse_r',
            'prediction_source':identity(folder/'predictions.json'),'prediction_branch':'mse_r','features':identity(folder/'scorer_features.npz')})
    for i,row in enumerate(scorer['evaluation_members']):
        cached=read(S.OUT/'attempt_01/development_cache'/f'{i:04d}.json')
        members.append({'scope':'old128','index':i,'image_id':row['image_id'],'raw':identity(cached['original_raw']['path']),'raw_key':'raw',
            'prediction_source':identity(S.OUT/'attempt_01/development_cache'/f'{i:04d}.json'),'prediction_branch':None,
            'features':identity(S.OUT/'attempt_01/development_cache'/f'{i:04d}.npz')})
    for i,row in enumerate(threshold['calibration_metadata']):
        folder=T.OUT/'attempt_01/calibration'/f'{i:04d}'
        members.append({'scope':'calibration178','index':i,'image_id':row['image_id'],'raw':identity(folder/'raw_and_features.npz'),'raw_key':'raw',
            'prediction_source':identity(folder/'predictions.json'),'prediction_branch':'mse_r','features':identity(folder/'raw_and_features.npz')})
    require(len(members)==592 and len({r['image_id'] for r in members})==592,'Expected unique saved592members')
    c={'schema':'rampq.exact_scorer_feature_optimization/v1','at_utc':P.now(),'source_bindings':sources(),
       'authorization':'User continues; reduce the measured CPU neighbor feature cost, enforce a strictly equivalent engineering implementation with no changes to the model or threshold',
       'previous_cost_contract':identity(D.OUT/'contract.json'),'previous_cost_delivery':identity(D.OUT/'delivery_validation.json'),
       'previous_cost_decision':identity(D.OUT/'machine_decision.json'),'scorer_model':threshold['scorer_model'],
       'frozen_threshold_record':identity(T.OUT/'attempt_01/frozen_threshold.json'),'members':members,
       'optimized_block_rows':128,'optimization_candidates':1,'tuning_sweep':False,
       'equivalence_rule':'ALL592members22features bitwise equal to baseline and savedfeatures; frozenmodel scores and stable sort exactlyequal. No tolerances or GT inputs.',
       'timing_scope':'Onlynew286members:1warmup each old/newimplementation;3pairedrepeats,alternate order by(imageindex+repeat)%2. Readraw/JSON outside timers. Time features+frozenCPUscoring+stablePythonindexsort.',
       'timing_repeats':3,'timing_warmups_each':1,'benchmark_samples_per_implementation':858,
       'excludes':['fileI/O','imageDecode','NMS','detector','CUDAoutputtransfer','GTlabels','submissionSerialization'],
       'failure_policy':'Any bitwise mismatch stops and preserves attempt; no fallback tolerance,parameter or threshold changes',
       'preflight_initial_failure':identity(OUT/'preflight_initial_failed.xml'),
       'preflight_initial_failure_reason':'Test assertion for firsttiedrow was applied to257randomcandidates including higher-score sameclass boxes; correctedassertion uses4isolatedtiedrows. Originalfailedtestreport retained. Feature equality already passed.',
       'new_model_fits':0,'new_detector_forwards':0,'new_threshold_candidates':0,'new_AP_scores':0,
       'scientific_method_novelty_claim':False,'end_to_end_speedup_claim':False,'Jetson_acceptance':False,
       'scientific_acceptance':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,
       'timeout_seconds':1800,'automatic_retry':False}
    write(OUT/'contract.json',c)
    print(json.dumps({'status':'FROZEN_SINGLE_EXACT_OPTIMIZATION','contract':identity(OUT/'contract.json'),'verification_members':592,'paired_timing_images':286,'block_rows':128},ensure_ascii=True))


class Worker:
    def __init__(self):
        self.contract=read(OUT/'contract.json');self.attempt=OUT/'attempt_01';self.attempt.mkdir(exist_ok=False);self.started=time.monotonic()
        self.live={'pid':os.getpid(),'parent_pid':os.getppid(),'started_at_utc':P.now(),'status':'STARTING','contract':identity(OUT/'contract.json'),
                   'new_detector_forwards':0,'new_model_fits':0,'new_threshold_candidates':0}
        self.progress('STARTING',0,1)

    def progress(self,stage,done,total):
        require(time.monotonic()-self.started<1800,'Optimization time budget exhausted')
        self.live.update(status=stage,done=done,total=total,updated_at_utc=P.now())
        write(self.attempt/'live_status.json',self.live,once=False)
        with (self.attempt/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(self.live)+'\n')
        print(json.dumps({'status':stage,'done':done,'total':total}),flush=True)

    def verify(self):
        c=self.contract;require(c['source_bindings']==sources(),'Frozen sources changed')
        for key in ('previous_cost_contract','previous_cost_delivery','previous_cost_decision','scorer_model','frozen_threshold_record','preflight_initial_failure'):
            require(identity(c[key]['path'])==c[key],'Bound source changed: '+key)
        seen={}
        for row in c['members']:
            for key in ('raw','prediction_source','features'):
                b=row[key];seen[b['path']]=b
        for b in seen.values():require(identity(b['path'])==b,'Saved member changed')

    def run(self):
        P.configure();self.verify();model=read(self.contract['scorer_model']['path']);equivalence=[];timings=[]
        for position,row in enumerate(self.contract['members']):
            record=read(row['prediction_source']['path'])
            pred=record['predictions'] if row['prediction_branch'] is None else record[row['prediction_branch']]
            if isinstance(pred,dict):pred=pred['predictions']
            with np.load(row['raw']['path'],allow_pickle=False) as d:raw=d[row['raw_key']].copy()
            with np.load(row['features']['path'],allow_pickle=False) as d:expected=d['features'].copy();p=d['original_score'].copy()
            old=S.visible_features(raw,pred);new=O.visible_features(raw,pred)
            require(np.array_equal(old,expected) and np.array_equal(new,old),'Bitwise feature mismatch; stop registered implementation')
            old_score=S.score_rows(old,p,model);new_score=S.score_rows(new,p,model)
            require(np.array_equal(new_score,old_score),'Bitwise frozen scorer mismatch')
            old_order=sorted(range(len(pred)),key=lambda i:(-float(old_score[i]),pred[i]['nms_input_order']))
            new_order=sorted(range(len(pred)),key=lambda i:(-float(new_score[i]),pred[i]['nms_input_order']))
            require(old_order==new_order,'Stable candidate sort mismatch')
            equivalence.append({'scope':row['scope'],'index':row['index'],'image_id':row['image_id'],'candidates':len(pred),
                'features_bitwise_equal':True,'scores_bitwise_equal':True,'stable_sort_equal':True,'saved_feature_binding':row['features']})
            if row['scope']=='new286':
                def full(fn):
                    x=fn(raw,pred);scores=S.score_rows(x,p,model)
                    order=sorted(range(len(pred)),key=lambda i:(-float(scores[i]),pred[i]['nms_input_order']))
                    return x,scores,order
                full(S.visible_features);full(O.visible_features)
                for repeat in range(3):
                    order=('original','optimized') if (row['index']+repeat)%2==0 else ('optimized','original')
                    measured={}
                    for implementation in order:
                        fn=S.visible_features if implementation=='original' else O.visible_features
                        start=time.perf_counter_ns();x,scores,indices=full(fn);elapsed=time.perf_counter_ns()-start
                        require(np.array_equal(x,expected) and np.array_equal(scores,old_score) and indices==old_order,'Paired timing output mismatch')
                        measured[implementation+'_ns']=elapsed
                    timings.append({'image_id':row['image_id'],'index':row['index'],'repeat':repeat,'candidate_count':len(pred),'measurement_order':list(order),**measured})
            if (position+1)%32==0 or position==591:self.progress('EXACT_FEATURE_SCORE_ORDER_REPLAY_AND_PAIRED_CPU_TIMING',position+1,592)
        write(self.attempt/'equivalence_checks.json',equivalence);write(self.attempt/'paired_CPU_timings.json',timings)
        original=np.array([r['original_ns']/1e6 for r in timings]);optimized=np.array([r['optimized_ns']/1e6 for r in timings])
        ratio=original/optimized;old=D.summarize_timing(original);new=D.summarize_timing(optimized)
        self.verify()
        result={'status':'COMPLETE_SINGLE_EXACT_CPU_FEATURE_OPTIMIZATION','bitwise_verified_members':592,
            'scope_member_counts':dict(D.Counter(r['scope'] for r in equivalence)),'paired_timing_samples':858,
            'original_CPU':old,'optimized_CPU':new,'ratio_of_median_times':old['median_ms']/new['median_ms'],
            'median_paired_time_ratio':float(np.median(ratio)),'optimized_faster_sample_count':int((optimized<original).sum()),
            'block_rows':128,'optimization_candidates':1,'all_features_scores_and_orders_exact':True,
            'frozen_threshold_record':self.contract['frozen_threshold_record'],'frozen_model':self.contract['scorer_model'],
            'existing_AP_TP_FP_results_unchanged_by_exact_implementation':True,'not_an_accuracy_improvement':True,
            'runtime':{'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'torch_num_threads':P.torch.get_num_threads(),
                'numpy':np.__version__,'python':P.sys.version},'excludes':self.contract['excludes'],
            'end_to_end_speedup_measured':False,'Jetson_latency_measured':False,'model_fits':0,'new_detector_forwards':0,'threshold_changes':0,
            'scientific_acceptance':False,'actual_INT8':False,'Selection_accessed':False,'Holdout_accessed':False}
        write(self.attempt/'optimization_result.json',result)
        files=[identity(p) for p in sorted(self.attempt.rglob('*')) if p.is_file() and p.name not in {'events.jsonl','live_status.json','manifest.json','completion_receipt.json'}]
        write(self.attempt/'manifest.json',{'files':files});self.progress('COMPLETE_SINGLE_EXACT_CPU_FEATURE_OPTIMIZATION',1,1)
        write(self.attempt/'completion_receipt.json',{**self.live,'finished_at_utc':P.now(),'elapsed_seconds':time.monotonic()-self.started,
            'manifest':identity(self.attempt/'manifest.json'),'result':identity(self.attempt/'optimization_result.json')})
        print(json.dumps({'status':result['status'],'original_median_ms':old['median_ms'],'optimized_median_ms':new['median_ms'],
            'ratio_of_median_times':result['ratio_of_median_times'],'bitwise_members':592}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));args=parser.parse_args()
    if args.mode=='prepare':prepare();return
    worker=Worker()
    try:worker.run()
    except BaseException as error:
        worker.live.update(status='FAILED_ATTEMPT_PRESERVED',error_type=type(error).__name__,error=str(error),updated_at_utc=P.now())
        write(worker.attempt/'live_status.json',worker.live,once=False);write(worker.attempt/'failure_receipt.json',{**worker.live,'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
