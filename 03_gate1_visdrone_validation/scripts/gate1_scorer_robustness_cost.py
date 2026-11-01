"""Post hoc fixed-scorer robustness and CPU overhead diagnostics.

Only saved authorized developer outputs are read. No detector forward, AP
rerun, parameter/threshold search, data split change or sealed-data access.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/gate1/20261026_scorer_robustness_cost_v1'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'settings')
import numpy as np
import torch
import gate1_separate_threshold as T
P=T.P;S=T.S;E=T.E
read,write,identity,require,now=P.read,P.write,P.identity,P.require,P.now
KINDS=('same_class_overlap','other_class_overlap','near_object_low_IoU','low_overlap_remainder')


def partition(iou,categories,category):
    same=np.asarray(iou)[np.asarray(categories)==category]
    maximum=float(np.max(iou)) if len(iou) else 0.
    if len(same) and float(np.max(same))>=.5:return KINDS[0]
    if maximum>=.5:return KINDS[1]
    if maximum>=.1:return KINDS[2]
    return KINDS[3]


def group_sensitivity(rows):
    groups=defaultdict(lambda:Counter())
    for row in rows:
        groups[row['prefix_proxy']].update(TP_delta=row['TP_delta'],FP_delta=row['FP_delta'],images=1)
    total_tp=sum(x['TP_delta'] for x in rows);total_fp=sum(x['FP_delta'] for x in rows)
    leaves=[]
    for name,g in sorted(groups.items()):
        tp=total_tp-g['TP_delta'];fp=total_fp-g['FP_delta']
        leaves.append({'omitted_prefix_proxy':name,'omitted_images':g['images'],'remaining_TP_delta':tp,'remaining_FP_delta':fp,
                       'remaining_joint_observation':tp>=0 and fp<=0})
    require(sum(g['TP_delta'] for g in groups.values())==total_tp and sum(g['FP_delta'] for g in groups.values())==total_fp,'Group conservation failure')
    return {'images':len(rows),'prefix_proxies':len(groups),'pooled_TP_delta':total_tp,'pooled_FP_delta':total_fp,
        'pooled_joint_observation':total_tp>=0 and total_fp<=0,
        'prefix_groups':[{**g,'prefix_proxy':name} for name,g in sorted(groups.items())],
        'leave_one_prefix_out':leaves,'leave_one_prefix_joint_count':sum(int(r['remaining_joint_observation']) for r in leaves),
        'all_leave_one_prefix_joint_observation':all(r['remaining_joint_observation'] for r in leaves),
        'image_joint_count':sum(int(r['TP_delta']>=0 and r['FP_delta']<=0) for r in rows),
        'prefix_proxy_is_verified_scene':False,'not_independent_replication':True,'not_a_statistical_confidence_interval':True}


def summarize_timing(values):
    a=np.asarray(values,dtype=np.float64)
    require(a.ndim==1 and len(a)>0 and np.isfinite(a).all() and (a>=0).all(),'Invalid timing vector')
    return {'samples':len(a),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),'maximum_ms':float(np.max(a)),
            'mean_ms':float(np.mean(a)),'minimum_ms':float(np.min(a))}


def source_bindings():
    result=deepcopy(T.sources())
    for path in (Path(__file__),ROOT/'scripts/launch_gate1_scorer_robustness_cost.py',ROOT/'tests/test_gate1_scorer_robustness_cost.py'):
        result[path.name]=identity(path)
    return result


def prepare():
    require(not (OUT/'contract.json').exists(),'Robustness contract already frozen')
    threshold=read(T.OUT/'contract.json');extension=read(E.OUT/'contract.json')
    reused=[]
    for branch in E.BRANCHES:
        folder=E.OUT/'attempt_01/evaluation/full414'/branch
        for name in ('saved_predictions_and_targets.json','project_metrics.json','metrics.json'):
            reused.append(identity(folder/name))
    reused += [identity(T.OUT/'attempt_01/developer_project_threshold_matches.json'),identity(T.OUT/'attempt_01/comparison_result.json')]
    for i in range(286):
        folder=E.OUT/'attempt_01/new_inference'/f'{i:04d}'
        reused += [identity(folder/'raw.npz'),identity(folder/'scorer_features.npz')]
    fp32=ROOT/'artifacts/gate1/20261024_formal_candidate_evaluation_v1/scores/fp32/metrics.json'
    c={'schema':'rampq.gate1.frozen_scorer_robustness_cost/v1','created_at_utc':now(),
       'authorization':'User: Proceed; diagnose new prefix false positives, group stability, and actual additional CPU overhead based on the current frozen results, without adjusting thresholds or models',
       'source_bindings':source_bindings(),'previous_threshold_contract':identity(T.OUT/'contract.json'),
       'previous_threshold_delivery':identity(T.OUT/'delivery_validation.json'),'previous_threshold_decision':identity(T.OUT/'machine_decision.json'),
       'extension_contract':identity(E.OUT/'contract.json'),'extension_delivery':identity(E.OUT/'delivery_validation.json'),
       'frozen_threshold_record':identity(T.OUT/'attempt_01/frozen_threshold.json'),
       'scorer_model':threshold['scorer_model'],'evaluation_protocol':threshold['evaluation_protocol'],'historical_FP32_AP':identity(fp32),
       'cohorts':threshold['evaluation_cohorts'],'reused_files':reused,'extension_root':str(E.OUT),'threshold_root':str(T.OUT),
       'fixed_threshold':read(T.OUT/'attempt_01/frozen_threshold.json')['threshold'],'baseline_threshold':.25,
       'posthoc_diagnostic_only':True,'new_acceptance_gate':False,
       'sensitivity':'For new286,new_prefix47,full414 sum per-image PROJECTTP/FP deltas; group by existing filename prefix; exhaustively remove each one group and report every result. Do not alter members,threshold or selected candidate.',
       'FP_geometry':'All414 existingGT/projectFPflags; priority sameclassIoU>=0.5,otherclassIoU>=0.5,maximumIoU>=0.1,lowoverlapremainder. Partition is diagnostic only,not causal,not proof lowoverlap is background.',
       'FP_identity':'Use immutable raw_anchor_index within sameimage/candidatepool for FPidentity gains/losses. Report added and removedFP sets separately from geometric category netcounts.',
       'CPU_benchmark':'Frozen current implementation,286new developer savedRraw images. One untimed warmup perimage,then3repeats. Timers exclude fileI/O,NMS,image decode,detector,CUDAdownload,annotation reads. Include current22feature computation and CPU residualscoring+stablePython sorting; score-only timing separately. No implementation optimization or speedup claim.',
       'benchmark_members':[{'index':i,'image_id':r['image_id']} for i,r in enumerate(extension['cohorts']['new286'])],
       'benchmark_repeats':3,'benchmark_warmups_per_image':1,'candidate_scorer_refits':0,'new_detector_image_forwards':0,
       'full_AP_scores':0,'threshold_candidates':0,'new_methods':0,'timeout_seconds':1800,'automatic_retry':False,
       'prior_art':[{'name':'IoU-Net','url':'https://arxiv.org/abs/1807.11590','point':'Localization confidence and better candidate selection are prior art'},
                    {'name':'GFLV2','url':'https://arxiv.org/abs/2011.12885','point':'Lightweight localization-quality ranking using box-distribution statistics is prior art'},
                    {'name':'DetPTQ','url':'https://arxiv.org/abs/2304.09785','point':'Detection-task-specific quantization parameter selection is prior art'}],
       'scientific_acceptance':False,'independent_test_set':False,'actual_INT8':False,'Selection_payload_access':False,'Holdout_payload_access':False,
       'no_dataset_or_threshold_changes_based_on_diagnostic':True}
    write(OUT/'contract.json',c)
    print(json.dumps({'status':'FROZEN_POSTHOC_DIAGNOSTIC','contract':identity(OUT/'contract.json'),'saved_development_images':414,'benchmark_images':286,'new_detector_forwards':0},ensure_ascii=True))


class Worker:
    def __init__(self):
        self.contract=read(OUT/'contract.json');self.attempt=OUT/'attempt_01';self.attempt.mkdir(exist_ok=False)
        self.started=time.monotonic();self.live={'pid':os.getpid(),'parent_pid':os.getppid(),'started_at_utc':now(),
            'status':'STARTING','contract':identity(OUT/'contract.json'),'new_detector_forwards':0,'new_model_fits':0}
        self.progress('STARTING',0,1)

    def progress(self,status,done,total):
        require(time.monotonic()-self.started<1800,'Finite diagnostic budget exhausted')
        self.live.update(status=status,done=done,total=total,updated_at_utc=now())
        write(self.attempt/'live_status.json',self.live,once=False)
        with (self.attempt/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(self.live)+'\n')
        print(json.dumps({'status':status,'done':done,'total':total}),flush=True)

    def verify(self):
        c=self.contract;require(c['source_bindings']==source_bindings(),'Frozen source changed')
        for key in ('previous_threshold_contract','previous_threshold_delivery','previous_threshold_decision','extension_contract','extension_delivery',
                    'frozen_threshold_record','scorer_model','evaluation_protocol','historical_FP32_AP'):
            require(identity(c[key]['path'])==c[key],'Bound source changed: '+key)
        for b in c['reused_files']:require(identity(b['path'])==b,'Saved result changed')

    def diagnose(self):
        c=self.contract;ext=Path(c['extension_root']);threshold=Path(c['threshold_root'])
        base=read(ext/'attempt_01/evaluation/full414/mse_r/saved_predictions_and_targets.json')
        scored=read(ext/'attempt_01/evaluation/full414/frozen_score_base/saved_predictions_and_targets.json')
        newer=read(threshold/'attempt_01/developer_project_threshold_matches.json')
        require([r['image_id'] for r in base]==[r['image_id'] for r in scored]==[r['image_id'] for r in newer]==[r['image_id'] for r in c['cohorts']['full414']],'Input image order differs')
        root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
        per_image=[];fp_details=[]
        for i,(meta,br,sr,nr) in enumerate(zip(c['cohorts']['full414'],base,scored,newer)):
            bmatch=br['project_conf025'];smatch=nr['matching'];require(nr['threshold']==c['fixed_threshold'],'Threshold changed')
            require({r['target_id'] for r in bmatch['targets']}=={r['target_id'] for r in smatch['targets']},'Eligible target identities differ')
            oldpred={r['detection_id']:r for r in br['predictions']};newpred={r['detection_id']:r for r in sr['predictions']}
            oldfps=[oldpred[r['detection_id']] for r in bmatch['detection_states'] if r['reason']=='false_positive']
            newfps=[newpred[r['detection_id']] for r in smatch['detection_states'] if r['reason']=='false_positive']
            oldfpids={r['raw_anchor_index'] for r in oldfps};newfpids={r['raw_anchor_index'] for r in newfps}
            oldhits={r['target_id'] for r in bmatch['matches']};newhits={r['target_id'] for r in smatch['matches']}
            ann=(root/meta['annotation_path']).resolve(strict=True)
            require(ann.is_relative_to(root) and P.sha(ann)==meta['annotation_sha256'],'Developer annotation changed')
            gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id'])
            eligible={t['target_id'] for t in bmatch['targets']};positives=[g for g in gt if g['target_id'] in eligible]
            boxes=np.asarray([P.MATCH._xyxy(g['bbox_xywh'],meta['width'],meta['height']) for g in positives]).reshape(-1,4)
            cats=np.array([g['category'] for g in positives]);oldtypes=Counter({k:0 for k in KINDS});newtypes=Counter({k:0 for k in KINDS})
            for branch,preds,counter in (('mse_r',oldfps,oldtypes),('scored_fixed_threshold',newfps,newtypes)):
                for pred in preds:
                    box=np.array([P.MATCH._xyxy(pred['bbox_xywh'],meta['width'],meta['height'])])
                    overlaps=P.box_iou_matrix(box,boxes)[0];kind=partition(overlaps,cats,pred['category']);counter[kind]+=1
                    other=newfpids if branch=='mse_r' else oldfpids
                    fp_details.append({'image_id':meta['image_id'],'prefix_proxy':meta['prefix_proxy'],'branch':branch,'raw_anchor_index':pred['raw_anchor_index'],
                        'category':pred['category'],'score':pred['score'],'kind':kind,'max_GT_IoU':float(overlaps.max()) if len(overlaps) else 0.,
                        'FP_anchor_in_other_branch':pred['raw_anchor_index'] in other})
            row={'image_id':meta['image_id'],'prefix_proxy':meta['prefix_proxy'],'base_TP':bmatch['counts']['hits'],
                 'base_FP':bmatch['counts']['false_positive_detections'],'new_TP':smatch['counts']['hits'],'new_FP':smatch['counts']['false_positive_detections'],
                 'TP_delta':len(newhits)-len(oldhits),'FP_delta':len(newfps)-len(oldfps),
                 'newly_hit_target_ids':sorted(newhits-oldhits),'lost_hit_target_ids':sorted(oldhits-newhits),
                 'added_FP_anchor_ids':sorted(newfpids-oldfpids),'removed_FP_anchor_ids':sorted(oldfpids-newfpids),
                 'base_FP_types':dict(oldtypes),'new_FP_types':dict(newtypes),'new_candidate_pool_size':len(sr['predictions'])}
            require(row['TP_delta']==row['new_TP']-row['base_TP'] and row['FP_delta']==row['new_FP']-row['base_FP'],'Target/FP conservation differs')
            per_image.append(row)
            if (i+1)%32==0 or i==413:self.progress('FIXED_THRESHOLD_TARGET_AND_FP_GEOMETRY_DIAGNOSIS',i+1,414)
        write(self.attempt/'per_image_diagnostics.json',per_image);write(self.attempt/'FP_geometry_details.json',fp_details)
        by_id={r['image_id']:r for r in per_image};cohorts={}
        old_result=read(threshold/'attempt_01/comparison_result.json')
        for name in ('new286','new_prefix47','full414'):
            rows=[by_id[r['image_id']] for r in c['cohorts'][name]];sensitivity=group_sensitivity(rows)
            original=old_result['cohorts'][name]
            require(sensitivity['pooled_TP_delta']==original['project_TP_delta_vs_R'] and sensitivity['pooled_FP_delta']==original['project_FP_delta_vs_R'],'New diagnostic does not reproduce frozen outcomes')
            counts={branch:Counter({k:0 for k in KINDS}) for branch in ('base','new')}
            for row in rows:
                counts['base'].update(row['base_FP_types']);counts['new'].update(row['new_FP_types'])
            sensitivity.update(FP_geometry_counts={k:dict(v) for k,v in counts.items()},
                FP_geometry_net_change={k:counts['new'][k]-counts['base'][k] for k in KINDS},
                added_FP_anchors=sum(len(r['added_FP_anchor_ids']) for r in rows),removed_FP_anchors=sum(len(r['removed_FP_anchor_ids']) for r in rows),
                gained_GT_ids=sum(len(r['newly_hit_target_ids']) for r in rows),lost_GT_ids=sum(len(r['lost_hit_target_ids']) for r in rows),
                all_prefix_groups_sorted_by_FP_delta=sorted(sensitivity['prefix_groups'],key=lambda g:(-g['FP_delta'],g['prefix_proxy'])),
                all_images_sorted_by_FP_delta=sorted(rows,key=lambda r:(-r['FP_delta'],r['image_id'])))
            cohorts[name]=sensitivity
        write(self.attempt/'robustness_diagnostics.json',{'cohorts':cohorts,'posthoc_diagnostic_only':True,'threshold':c['fixed_threshold'],
            'prefix_not_verified_scene':True,'new_acceptance_gate':False,'statistical_significance_claim':False,'scientific_acceptance':False})
        return cohorts

    def benchmark(self):
        c=self.contract;ext=Path(c['extension_root']);model=read(c['scorer_model']['path'])
        measurements=[]
        for j,member in enumerate(c['benchmark_members']):
            index=member['index'];folder=ext/'attempt_01/new_inference'/f'{index:04d}'
            original=read(folder/'predictions.json')['mse_r']['predictions']
            with np.load(folder/'raw.npz',allow_pickle=False) as d:raw=d['mse_r'].copy()
            with np.load(folder/'scorer_features.npz',allow_pickle=False) as d:features=d['features'].copy();p=d['original_score'].copy();expected=d['new_score'].copy()
            def score_only():return S.score_rows(features,p,model)
            def whole():
                x=S.visible_features(raw,original);scores=S.score_rows(x,p,model)
                order=sorted(range(len(original)),key=lambda k:(-float(scores[k]),original[k]['nms_input_order']))
                return x,scores,order
            score_only();whole()
            for repeat in range(3):
                start=time.perf_counter_ns();scores=score_only();score_ns=time.perf_counter_ns()-start
                require(np.array_equal(scores,expected),'Score-only benchmark output differs')
                start=time.perf_counter_ns();x,scores,order=whole();whole_ns=time.perf_counter_ns()-start
                require(np.array_equal(x,features) and np.array_equal(scores,expected),'Full scoring benchmark output differs')
                require(order==sorted(range(len(original)),key=lambda k:(-float(expected[k]),original[k]['nms_input_order'])),'Stable sort differs')
                measurements.append({'image_id':member['image_id'],'index':index,'repeat':repeat,'candidates':len(original),
                    'score_only_ns':score_ns,'features_score_stable_sort_ns':whole_ns})
            if (j+1)%32==0 or j==285:self.progress('MEASURE_FROZEN_CPU_SCORER_OVERHEAD',j+1,286)
        write(self.attempt/'CPU_timing_samples.json',measurements)
        result={'images':286,'warmup_per_image':1,'repeats_per_image':3,'sample_count':len(measurements),
            'score_only':summarize_timing([r['score_only_ns']/1e6 for r in measurements]),
            'features_score_stable_sort':summarize_timing([r['features_score_stable_sort_ns']/1e6 for r in measurements]),
            'candidate_counts':{'min':min(r['candidates'] for r in measurements),'median':float(np.median([r['candidates'] for r in measurements])),
                                'max':max(r['candidates'] for r in measurements)},
            'includes':['current22featureCPUimplementation','CPUfloat64frozenlinearresidualscoring','stablePythoncandidateindexsorting'],
            'excludes':['diskI/O','annotationread','imageDecode','detector','NMS','CUDAoutputtransfer','finalsubmissionSerialization'],
            'timing_clock':'time.perf_counter_ns','runtime':{'python':P.sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
                'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'torch_num_threads':torch.get_num_threads(),
                'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS')},
            'end_to_end_speedup_measured':False,'Jetson_latency_measured':False,'energy_measured':False,'model_forward_count':0,'parameter_updates':0}
        write(self.attempt/'CPU_overhead_summary.json',result);return result

    def run(self):
        P.configure();self.verify();cohorts=self.diagnose();timing=self.benchmark();self.verify()
        score_AP=read(Path(self.contract['extension_root'])/'attempt_01/evaluation/full414/frozen_score_base/metrics.json')['metrics_percent']['AP_all']
        fp32_AP=read(self.contract['historical_FP32_AP']['path'])['metrics_percent']['AP_all']
        write(self.attempt/'diagnostic_result.json',{'status':'COMPLETE_FROZEN_SCORER_ROBUSTNESS_COST_DIAGNOSTIC','cohorts':cohorts,
            'CPU_overhead':timing,'frozen_full414_AP_percent':score_AP,'historical_FP32_full414_AP_percent':fp32_AP,
            'AP_gap_to_historical_FP32_percentage_points':score_AP-fp32_AP,'new_model_fits':0,'new_detector_forwards':0,'new_full_AP_scores':0,
            'new_acceptance_gate':False,'scientific_acceptance':False,'actual_INT8':False,'Holdout_accessed':False,'Selection_accessed':False})
        files=[identity(p) for p in sorted(self.attempt.rglob('*')) if p.is_file() and p.name not in {'events.jsonl','live_status.json','manifest.json','completion_receipt.json'}]
        write(self.attempt/'manifest.json',{'files':files});self.progress('COMPLETE_FROZEN_SCORER_ROBUSTNESS_COST_DIAGNOSTIC',1,1)
        write(self.attempt/'completion_receipt.json',{**self.live,'finished_at_utc':now(),'elapsed_seconds':time.monotonic()-self.started,
            'manifest':identity(self.attempt/'manifest.json'),'result':identity(self.attempt/'diagnostic_result.json')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));args=parser.parse_args()
    if args.mode=='prepare':prepare();return
    worker=Worker()
    try:worker.run()
    except BaseException as error:
        worker.live.update(status='FAILED_ATTEMPT_PRESERVED',error_type=type(error).__name__,error=str(error),updated_at_utc=now())
        write(worker.attempt/'live_status.json',worker.live,once=False);write(worker.attempt/'failure_receipt.json',{**worker.live,'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
