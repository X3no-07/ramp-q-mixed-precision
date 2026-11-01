"""Post-worker audit; saved arrays only, no training/model/AP reruns."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import gate1_guarded_training_comparison as W
from audit_gate1_full_training_comparison import metric_aggregation
P, G = W.P, W.G
ROOT, OUT = W.ROOT, W.OUT
read, write, identity, require, now = W.read, W.write, W.identity, W.require, W.now


def check_guard(r):
    """Gram/KKT verification independent of full-vector active-set implementation."""
    g = np.asarray(r["gram_float64"], np.float64)
    require(g.shape == (4,4) and np.isfinite(g).all() and np.allclose(g,g.T,atol=1e-9,rtol=1e-12), "Invalid Gram")
    nb,nc,na,nu=np.sqrt(np.maximum(np.diag(g),0))
    require(nu<=.25*nb+1e-8*max(1.,nb),"Double norm cap differs")
    require(g[0,3]>=-G.DOUBLE_FEASIBILITY*max(1.,na)*nb-1e-9 and
            g[1,3]>=-G.DOUBLE_FEASIBILITY*max(1.,na)*nc-1e-9,"Double direction differs")
    castnorm=r['actual_FP32_auxiliary_norm']
    require(castnorm<=.25*nb+G.FLOAT32_FEASIBILITY*max(1.,nb),"FP32 norm cap differs")
    require(r['actual_FP32_dot_base']>=-G.FLOAT32_FEASIBILITY*max(1.,nb*castnorm) and
            r['actual_FP32_dot_classification']>=-G.FLOAT32_FEASIBILITY*max(1.,nc*castnorm),"FP32 direction differs")
    f=r['cap_factor']
    require(0<=f<=1 and r['rho']==.25,"Cap parameter differs")
    if r['active_set']=='zero_base_or_auxiliary':
        require(nb==0 or na==0,"Unexpected zero bypass")
        return
    # Let v be the uncapped Euclidean projection and d=v-a. KKT requires d
    # in the nonnegative span of the two normals, plus complementarity.
    div=f if f>0 else 1.
    rhs=np.array([g[0,3]/div-g[0,2],g[1,3]/div-g[1,2]])
    h=g[:2,:2]
    multipliers=np.linalg.lstsq(h,rhs,rcond=1e-12)[0]
    delta2=g[3,3]/(div*div)+g[2,2]-2*g[2,3]/div
    residual2=delta2-2*multipliers@rhs+multipliers@h@multipliers
    scale=max(1.,abs(delta2),g[2,2])
    require(abs(residual2)<=2e-6*scale,"Projection correction outside normal span")
    require(np.min(multipliers)>=-2e-6*max(1.,np.max(np.abs(multipliers))),"Negative KKT multiplier")
    require(max(abs(multipliers[0]*g[0,3]/div),abs(multipliers[1]*g[1,3]/div))<=2e-6*scale,"Complementarity differs")


def independent_controls(selected,pool,key,priority):
    quotas=Counter(key(x) for x in selected);groups=defaultdict(list)
    for x in pool:groups[key(x)].append(x)
    answer=[]
    for k in sorted(quotas):
        require(len(groups[k])>=quotas[k],"Insufficient control stratum")
        answer.extend(sorted(groups[k],key=priority)[:quotas[k]])
    return answer


def audit():
    P.configure();started=time.monotonic();c=read(OUT/'contract.json');attempt=OUT/'attempt_01'
    require(c['source_bindings']==W.source_bindings() and c['framework']==P.verify_framework_sources(),"Source/framework changed")
    terminal=read(OUT/'worker_terminal_receipt.json');completion=read(attempt/'completion_receipt.json')
    require(terminal['exit_code']==0 and terminal['exit_code_directly_observed'],"Worker exit0 absent")
    require(completion['status']=='COMPLETE_FIXED_GUARDED_THREE_ARM_DEVELOPMENT' and
            completion['forward_images']==42012 and completion['reconstruction_forward_images']==9,"Incomplete worker")
    manifest=read(attempt/'manifest.json')['files']
    for item in manifest:require(identity(item['path'])==item,"Output changed before audit")
    for key in ('candidate','architecture','formal_contract','train_manifest','training_val_manifest','evaluation_protocol',
                'training_annotation_assignment','ranges','scorer_model','previous_state','initial_candidate','prior_full_contract','proposed_plan','qualification'):
        require(identity(c[key]['path'])==c[key],"Input changed:"+key)
    for row in c['teacher_inputs']:
        for binding in [row['record'],row['raw'],*row['geometry_sources']]:
            require(identity(binding['path'])==binding,"Teacher input changed")
    for row in c['FP32_saved_reference_subset']:require(identity(row['raw']['path'])==row['raw'],"FP32 reference changed")
    controls=Counter()
    for i,source in enumerate(c['teacher_inputs']):
        r=read(attempt/'training_supervision'/f'{i:04d}.json');old=read(source['record']['path'])
        require(r['source']==source and r['image_id']==source['image_id'] and r['selection'][W.ARMS[2]]==old['native_intervention_correction'],"Treatment changed")
        pool=r['positive_pool'];indexed={x['gt_index']:x for x in pool}
        require(len(indexed)==len(pool),"Duplicate positive control candidate")
        selected=r['selection'][W.ARMS[2]]
        hardpos=independent_controls([indexed[i] for i in selected['positive_gt']],pool,lambda x:(x['category'],x['size_bin']),lambda x:(x['quality'],x['gt_index']))
        harderr=independent_controls(selected['errors'],r['error_pool'],lambda x:(x['category'],x['score_bin'],x['kind']),lambda x:(-x['score'],x['anchor']))
        require(r['selection'][W.ARMS[1]]=={'positive_gt':[x['gt_index'] for x in hardpos],'errors':harderr},"Matched hard control priority differs")
        require(r['R_counts']==old['r_counts'],"Teacher pool count changed")
        controls.update(images=1,positives=len(hardpos),errors=len(harderr))
    require(controls==Counter(images=4530,positives=5094,errors=7144),"Supervision budget differs")
    train=read(c['train_manifest']['path']);val=read(c['training_val_manifest']['path']);P.FT.validate_membership(train,val)
    member_count=0
    for binding,header in ((c['train_manifest'],train),(c['training_val_manifest'],val)):
        root=(Path(binding['path']).parent/header['data_root']).resolve(strict=True)
        for row in header['samples']:
            for key in ('image','labels','sidecar'):
                member=row[key];path=(root/member['path']).resolve(strict=True)
                require(path.is_relative_to(root) and path.stat().st_size==member['bytes'] and W.sha(path)==member['sha256'],"Data member changed")
                member_count+=1
    initial=torch.load(c['initial_candidate']['path'],map_location='cpu',weights_only=True)['state_dict']
    common_hash=W.VALUE_SHA(initial);orders=[];state_count=0;guard_count=0
    frozen={n:v for n,v in initial.items() if 'running_' in n or 'num_batches_tracked' in n or n.endswith('_scale') or '.dfl.' in n}
    for arm in W.ARMS:
        folder=attempt/'training'/arm;receipt=read(folder/'training_receipt.json')
        require(receipt['initial_state_value_sha256']==common_hash and receipt['epochs']==3 and receipt['batches']==3399 and receipt['optimizer_steps']==852 and receipt['CUDA_reconstruction_exact'],"Training budget/init differs")
        order=read(folder/'epoch_orders.json');orders.append(order)
        require(len(order)==3 and all(sorted(x)==list(range(4530)) for x in order),"Epoch members incomplete")
        trace=[json.loads(x) for x in (folder/'batch_trace.jsonl').read_text('utf-8').splitlines()]
        require(len(trace)==3399 and sum(x['update'] for x in trace)==852,"Training trace incomplete")
        for epoch in range(3):
            rows=trace[epoch*1133:(epoch+1)*1133]
            require([i for r in rows for i in r['indices']]==order[epoch],"Sample order differs")
            for b,r in enumerate(rows):
                width,update=W.W.accumulation_window(b,1133)
                require(r['accumulation_denominator']==width and r['update']==update,"Accumulation tail differs")
            if arm!=W.ARMS[0]:require(rows[0]['original_loss_parity'] is True,"Loss parity absent")
        guards=[json.loads(x) for x in (folder/'gradient_guard_trace.jsonl').read_text('utf-8').splitlines()]
        require(len(guards)==(0 if arm==W.ARMS[0] else 852),"Guard window count differs")
        for k,r in enumerate(guards):
            require(r['step']==k+1,"Guard sequence differs");check_guard(r);guard_count+=1
        for epoch in range(1,4):
            state=torch.load(folder/f'epoch_{epoch:02d}_fp32.pt',map_location='cpu',weights_only=True)
            require(state['epoch']==epoch and state['arm']==arm and state['contract_sha256']==completion['contract']['sha256'],"Checkpoint identity differs")
            values=state['state_dict'];require(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in values.values() if v.is_floating_point()),"Invalid FP32 checkpoint")
            require(all(torch.equal(values[n],v) for n,v in frozen.items()),"Frozen state differs")
            if epoch==3:require(W.VALUE_SHA(values)==receipt['final_state_value_sha256'],"Final state differs")
            state_count+=1
    require(orders[0]==orders[1]==orders[2] and state_count==9 and guard_count==1704,"Training audit incomplete")
    root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
    references={r['image_id']:r['raw'] for r in c['FP32_saved_reference_subset']}
    raw_count=pred_count=matrix_count=numeric_count=0
    for arm in W.ARMS:
        output=attempt/'evaluation'/arm;rows=read(output/'saved_predictions_and_targets.json');total=Counter();guards=[]
        with np.load(output/'prepared_matrices.npz',allow_pickle=False) as d:packed={k:d[k].copy() for k in d.files}
        require(len(rows)==414,"Evaluation length differs")
        for i,meta in enumerate(c['evaluation_members']):
            with np.load(output/'raw'/f'{i:04d}.npz',allow_pickle=False) as d:raw=d['raw'].copy()
            require(raw.shape==(1,14,8400) and raw.dtype==np.float32 and np.isfinite(raw).all(),"Invalid raw array")
            row=rows[i];require(row['image_id']==meta['image_id'] and row['frozen_confidence']==.25,"Evaluation identity differs")
            predictions=P.predictions(raw,row['letterbox'],meta['image_id'],arm)
            require(predictions==row['predictions'],"Saved raw/NMS replay differs");raw_count+=1
            ann=(root/meta['annotation_path']).resolve(strict=True)
            require(ann.is_relative_to(root) and W.sha(ann)==meta['annotation_sha256'],"Development annotation differs")
            gt=P.MATCH.parse_ground_truth(ann.read_text(encoding='utf-8-sig'),meta['image_id'])
            match=P.MATCH.match_project_targets(gt,predictions,meta['width'],meta['height'],confidence=.25,iou_threshold=.5)
            require(match==row['target_matching'],"Saved target replay differs");total.update(match['counts']);pred_count+=1
            gt8=np.array([g['bbox_xywh']+[g['score'],g['category'],g['truncation'],g['occlusion']] for g in gt],np.float64).reshape(-1,8)
            dt8=np.array([d['bbox_xywh']+[d['score'],d['category'],-1,-1] for d in predictions],np.float64).reshape(-1,8)
            prepared=P.AP.prepare_raw_image(gt8,dt8,width=meta['width'],height=meta['height'])
            require(np.array_equal(prepared['prepared_gt'],packed[f'gt_{i}']) and np.array_equal(prepared['filtered_predictions'],packed[f'dt_{i}']),"Zero-tolerance AP matrices differ")
            matrix_count+=2
            if meta['image_id'] in references:
                with np.load(references[meta['image_id']]['path'],allow_pickle=False) as d:ref=d['raw'].copy()
                guards.append({'image_id':meta['image_id'],'guard':P.EVAL.raw_guard(ref,raw)});numeric_count+=1
        require(guards==read(output/'all_anchor_saved128_guard.json')['checks'],"Original all-anchor guard replay differs")
        require(dict(total)==read(output/'project_metrics.json')['conf025_counts'],"Aggregated target counts differ")
        metric_aggregation(output);require(read(output/'AP_terminal.json')['exit_code']==0,"AP exit0 missing")
    require((raw_count,pred_count,matrix_count,numeric_count)==(1242,1242,2484,384),"Saved audit count incomplete")
    for item in manifest:require(identity(item['path'])==item,"Output changed after audit")
    result={'status':'PASS_SAVED_RESULTS_AND_GUARD_AUDIT_ONLY','at_utc':now(),'pid':os.getpid(),'parent_pid':os.getppid(),
            'output_members_hashed_before_and_after':len(manifest),'source_data_member_hashes':member_count,
            'matched_control_records':dict(controls),'FP32_checkpoints':state_count,'gradient_guard_windows_KKT_checked':guard_count,
            'full_raw_NMS_replays':raw_count,'target_replays':pred_count,'zero_tolerance_AP_matrix_replays':matrix_count,
            'all_anchor_saved128_guard_replays':numeric_count,'new_model_forwards':0,'AP_recomputed':False,
            'elapsed_seconds':time.monotonic()-started,'actual_INT8':False,'scientific_acceptance':False}
    write(OUT/'independent_saved_audit.json',result)
    comparison=read(attempt/'comparison_result.json');intervention=comparison['intervention_vs_ordinary'];hard=comparison['intervention_vs_matched_hard']
    decision={'status':'THREE_ARM_DEVELOPMENT_AUDITED_AWAIT_SUPERVISOR_TERMINAL','at_utc':now(),
              'intervention_vs_ordinary':intervention,'intervention_vs_matched_hard':hard,
              'meets_both_development_joint_rules':intervention['joint_rule_met'] and hard['joint_rule_met'],
              'automatic_extension':False,'actual_INT8':False,'scientific_acceptance':False,'Selection_payload_access':False,'Holdout_payload_access':False}
    write(OUT/'machine_decision.json',decision)
    lines=['# Gate1 Constrained Correction Three-Arm Training Results','',f'Generation time:{now()}（UTC）。','',
           'Three arms start from the same ordinary QAT state at iteration 10, each continuing for 3 iterations, with identical data, seed, and update steps; the candidate fixed arm is set at iteration 3. Two correction arms match the sample group count and employ the same gradient protection. The ordinary QAT computes backward pass once, while the constrained arms compute gradients for three components, and thus no claim is made regarding identical training FLOPs or wall-clock time.','',
           '| Branch | AP50–95(%) | Fixed detection threshold of 0.25 | Fixed false detection threshold of 0.25 |','|---|---:|---:|---:|']
    for arm in W.ARMS:
        m=comparison['metrics'][arm];t=m['project']['conf025_counts']
        lines.append(f"| {arm} | {m['official_source_python']['metrics_percent']['AP_all']:.9f} | {t['hits']} | {t['false_positive_detections']} |")
    lines += ['',f"Intervention group versus standard QAT joint condition:{intervention['joint_rule_met']}; joint condition for standard hard example group with same protection:{hard['joint_rule_met']}。",'',
              'Completed 9 instances of numerical/KKT validation for FP32 states and 1704 protection windows, 1242 full raw/NMS, 1242 prediction/object matching, 2484 AP input matrix zero-tolerance validations, and 3 AP job exits 0. Passing the validation does not equate to method acceptance.', '',
              'The protection constraint applies only to the current additional gradient; it does not guarantee outcomes for each ground-truth target, SGD momentum history, or AP after finite updates. The 414 training_val images are repeatedly reused for development. Single-seed continuation for 3 epochs is neither convergence nor independent scientific acceptance. HQOD has not been reproduced, and actual INT8/TensorRT/Orin execution has not been validated.', '',
              'The original all 8400-candidate numerical gate is rechecked only on the 128 images with old FP32 arrays. The other 286 images have no additional FP32 inference, so complete numerical equivalence over 414 images cannot be claimed. Selection/Holdout remain sealed, and historical failures are retained.', '',
              'This round does not automatically append additional training, thresholds, seeds, or candidates. The final official state must simultaneously read supervisor_terminal_receipt.json and delivery_validation.json; judgment based solely on this report or process termination is invalid.','']
    report=ROOT/'docs/89_gate1_guarded_three_arm_continuation_results.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    write(OUT/'audit_completion_receipt.json',{'at_utc':now(),'report':identity(report),'audit':identity(OUT/'independent_saved_audit.json'),'decision':identity(OUT/'machine_decision.json')})
    print(json.dumps({'status':result['status'],'both_rules':decision['meets_both_development_joint_rules']},ensure_ascii=True),flush=True)


if __name__=='__main__':
    try:audit()
    except BaseException as e:
        write(OUT/'audit_failure_receipt.json',{'at_utc':now(),'error':repr(e),'traceback':traceback.format_exc()});raise
