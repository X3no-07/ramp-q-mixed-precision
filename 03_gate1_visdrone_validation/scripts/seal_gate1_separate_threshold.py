"""Seal the only calibrated-threshold outcome and preserve all prior evidence."""
from datetime import datetime,timezone
from pathlib import Path
import json
from audit_gate1_separate_threshold import T,P,E,OUT,A
read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/'contract.json');decision=read(OUT/'machine_decision.json');audit=read(OUT/'saved_delivery_audit.json')
    terminal=read(OUT/'launcher_terminal_receipt.json');process=read(OUT/'terminal_validation.json')
    require(terminal['propagated_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'] and terminal['stderr_bytes']==0,'Worker did not finish normally')
    require(process['matching_active_pilot_count']==0,'Calibration process still active')
    require(read(A/'completion_receipt.json')['forward_images']==178 and decision['threshold_candidates']==1,'Budget differs')
    spec=read(OUT/'saved_audit_contract.json')
    for key in ('script','independent_helpers'):require(identity(spec[key]['path'])==spec[key],'Frozen audit changed')
    latest=P.ROOT/'artifacts/gate1/latest_execution_state.json';previous=read(latest)
    require(previous['snapshot']=='artifacts/gate1/current_execution_state_v37.json','Latest pointer advanced elsewhere')
    require(identity(P.ROOT/previous['snapshot'])['sha256']==previous['snapshot_sha256'],'Previous snapshot changed')
    expected={}
    def add(b):
        if b['path'] in expected:require(expected[b['path']]==b,'Conflicting file identity')
        expected[b['path']]=b
    for b in read(A/'manifest.json')['files']+c['reused_files']+list(c['source_bindings'].values()):add(b)
    for key in ('previous_contract','previous_delivery','previous_decision','scorer_model','candidate','architecture','ranges','evaluation_protocol','training_manifest','training_val_manifest','assignment_metadata'):add(c[key])
    root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
    for rows,kinds in ((c['calibration_metadata'],('image','annotation')),(c['evaluation_cohorts']['full414'],('annotation',))):
        for meta in rows:
            for kind in kinds:
                path=(root/meta[kind+'_path']).resolve(strict=True);require(path.is_relative_to(root),'Allowed input escaped')
                b=identity(path);require(b['sha256']==meta[kind+'_sha256'],'Allowed input changed');add(b)
    for path in sorted(OUT.rglob('*')):
        if path.is_file():add(identity(path))
    for path in (P.ROOT/previous['snapshot'],P.ROOT/'docs/74_gate1_scorer_separate_178_image_threshold_calibration_launch.md',P.ROOT/'docs/75_gate1_scorer_separate_threshold_calibration_results.md',
                 P.ROOT/'scripts/audit_gate1_separate_threshold.py',P.ROOT/'scripts/audit_gate1_frozen_box_scorer.py',
                 P.ROOT/'scripts/summarize_gate1_separate_threshold.py',Path(__file__)):
        add(identity(path))
    for b in expected.values():require(identity(b['path'])==b,'Delivery member differs: '+b['path'])
    manifest={'schema':'rampq.separate_threshold.delivery/v1','at_utc':P.now(),'files':[expected[k] for k in sorted(expected)],
        'mutable_latest_pointer_excluded':True,'one_threshold_outcome_only':True,
        'payload_scope':'Only explicit178originaltrain calibrationimage+annotation pairs and414existingtraining_val annotations/savedoutputs; Holdout/Selection sealed.'}
    write(OUT/'delivery_manifest.json',manifest)
    validation={'status':'PASS_SEPARATE_THRESHOLD_DELIVERY_INTEGRITY_ONLY','at_utc':P.now(),'manifest':identity(OUT/'delivery_manifest.json'),
        'members_verified':len(expected),'missing':0,'mismatched':0,'audit':identity(OUT/'saved_delivery_audit.json'),
        'machine_decision':identity(OUT/'machine_decision.json'),'report':identity(P.ROOT/'docs/75_gate1_scorer_separate_threshold_calibration_results.md'),
        'observed_worker_exit_code':0,'primary_official_rule_met':decision['primary_official_fixed_threshold_joint_rule_met'],
        'all_cohorts_rule_met':decision['all_three_cohorts_official_fixed_threshold_joint_rule_met'],
        'scientific_acceptance':False,'Gate1_overall_passed':False,'actual_INT8':False,'Holdout_accessed':False}
    write(OUT/'delivery_validation.json',validation)
    state={'schema':'rampq.gate1.execution_state/v38','status':'COMPLETE_SCORER_DISJOINT_THRESHOLD_CALIBRATION_DEVELOPMENT_ONLY',
        'created_at_utc':P.now(),'previous_snapshot':previous,'current_experiment':'artifacts/gate1/20261026_separate_threshold178_v1',
        'report':'docs/75_gate1_scorer_separate_threshold_calibration_results.md','delivery_validation':identity(OUT/'delivery_validation.json'),
        'machine_decision':identity(OUT/'machine_decision.json'),'contract_sha256':identity(OUT/'contract.json')['sha256'],
        'frozen_threshold':decision['frozen_threshold'],'calibration_members':178,'calibration_prefix_proxies':34,
        'threshold_candidates':1,'scorer_not_refit':True,'new_detector_forwards':178,'new_fullAP_scores':0,
        'primary_new286_official_rule_met':decision['primary_official_fixed_threshold_joint_rule_met'],
        'primary_new286_project_rule_met':decision['primary_project_joint_observation'],
        'all_three_cohorts_joint_rule_met':decision['all_three_cohorts_official_fixed_threshold_joint_rule_met'],
        'cohort_deltas':{name:{k:r[k] for k in ('official_TP_delta_vs_R','official_FP_delta_vs_R','project_TP_delta_vs_R','project_FP_delta_vs_R')} for name,r in decision['cohorts'].items()},
        'method_contribution_not_proven':True,'Gate1_overall_passed':False,'scientific_acceptance':False,'independent_test_set':False,'actual_INT8':False,
        'old_numerical_FAILs_preserved':True,'Selection_payload_access':False,'Holdout_payload_access':False,'no_active_worker':True,
        'heartbeat_recreated':False,'cloud_resources_created':False,
        'next_action':'One separately calibrated threshold budget complete and sealed. Read primary286 and failed47prefixsubcohort together; do not retune using its results. Paper contribution remains unproven versus generic scoring; next independent data and device phases require concrete frozen scope and real access/device evidence. No reruns of completed model/AP stages.'}
    snap=P.ROOT/'artifacts/gate1/current_execution_state_v38.json';write(snap,state)
    pointer={'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v38.json','snapshot_sha256':identity(snap)['sha256'],
        'delivery_validation':identity(OUT/'delivery_validation.json'),
        'note':'178scorer-fit-disjoint threshold calibration complete/audited. Fixedthreshold primary286 and full414improveTP/FP,but47prefix-new FPgrows. See report75;no contribution/independent/Gate1/INT8 acceptance,Holdout sealed.'}
    temp=latest.with_name('latest_execution_state.v38.pending.json');write(temp,pointer);temp.replace(latest)
    print(json.dumps({'status':validation['status'],'members_verified':len(expected),'latest':pointer['snapshot'],'snapshot_sha256':pointer['snapshot_sha256'],
        'primary_rule_met':validation['primary_official_rule_met'],'all_cohorts_rule_met':validation['all_cohorts_rule_met']},ensure_ascii=True))


if __name__=='__main__':main()
