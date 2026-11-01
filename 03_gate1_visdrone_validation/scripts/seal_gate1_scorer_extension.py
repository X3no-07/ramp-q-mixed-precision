"""Seal every extension cohort and update only the mutable latest pointer."""
from datetime import datetime,timezone
from pathlib import Path
import json
from audit_gate1_scorer_extension_v2 import E,P,OUT,A
read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/'contract.json');decision=read(OUT/'machine_decision.json');audit=read(OUT/'saved_delivery_audit.json')
    terminal=read(OUT/'launcher_terminal_receipt.json');process=read(OUT/'terminal_validation.json')
    require(terminal['propagated_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'] and terminal['stderr_bytes']==0,'Worker terminal failure')
    require(process['matching_active_pilot_count']==0,'Extension/AP process still active')
    require(read(A/'completion_receipt.json')['forward_images']==572 and decision['model_fits']==0,'Budget differs')
    spec=read(OUT/'saved_audit_contract_v2.json')
    require(identity(spec['script']['path'])==spec['script'] and identity(spec['independent_helpers']['path'])==spec['independent_helpers'],'Audit source changed')
    latest=P.ROOT/'artifacts/gate1/latest_execution_state.json';previous=read(latest)
    require(previous['snapshot']=='artifacts/gate1/current_execution_state_v35.json','Latest pointer advanced elsewhere')
    require(identity(P.ROOT/previous['snapshot'])['sha256']==previous['snapshot_sha256'],'Previous snapshot changed')
    expected={}
    def add(b):
        if b['path'] in expected:require(expected[b['path']]==b,'Conflicting source identity')
        expected[b['path']]=b
    for b in read(A/'manifest.json')['files']+c['reused_files']+c['training_threshold_inputs']+list(c['source_bindings'].values()):add(b)
    for key in ('previous_contract','previous_delivery','previous_decision','scorer_model','candidate','architecture','evaluation_protocol','training_manifest','training_val_manifest','ranges','QAT_candidate','QAT_training_receipt','training_threshold_calibration','training_threshold_arrays'):add(c[key])
    root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
    for meta in c['cohorts']['full414']:
        for kind in ('image','annotation'):
            path=(root/meta[kind+'_path']).resolve(strict=True)
            require(path.is_relative_to(root),'Allowed member path escaped')
            b=identity(path);require(b['sha256']==meta[kind+'_sha256'],'Allowed input differs');add(b)
    for path in sorted(OUT.rglob('*')):
        if path.is_file():add(identity(path))
    for path in (P.ROOT/previous['snapshot'],P.ROOT/'docs/72_gate1_frozen_scorer_286_image_extension_launch.md',
                 P.ROOT/'docs/73_gate1_frozen_scorer_286_image_extension_review.md',P.ROOT/'scripts/audit_gate1_scorer_extension.py',
                 P.ROOT/'scripts/audit_gate1_scorer_extension_v2.py',P.ROOT/'scripts/audit_gate1_frozen_box_scorer.py',P.ROOT/'scripts/summarize_gate1_scorer_extension.py',Path(__file__)):
        add(identity(path))
    for b in expected.values():require(identity(b['path'])==b,'Delivery binding differs: '+b['path'])
    manifest={'schema':'rampq.frozen_scorer_extension.delivery/v1','created_at_utc':P.now(),'files':[expected[k] for k in sorted(expected)],
        'mutable_latest_pointer_excluded':True,'all_predefined_cohorts_retained':True,'payload_scope':'Existing training caches and explicit414training_val image/annotation members; Selection and Holdout payloads not accessed.'}
    write(OUT/'delivery_manifest.json',manifest)
    validation={'status':'PASS_FROZEN_EXTENSION_DELIVERY_INTEGRITY_ONLY','at_utc':P.now(),'manifest':identity(OUT/'delivery_manifest.json'),
        'members_verified':len(expected),'missing':0,'mismatched':0,'audit':identity(OUT/'saved_delivery_audit.json'),
        'machine_decision':identity(OUT/'machine_decision.json'),'report':identity(P.ROOT/'docs/73_gate1_frozen_scorer_286_image_extension_review.md'),
        'observed_worker_exit_code':0,'all9AP_exit_codes_zero':True,'primary_new286_joint_rule_met':decision['primary_new286_joint_rule_met'],
        'all_three_cohorts_joint_rule_met':decision['all_three_overlapping_cohorts_joint_rule_met'],
        'Gate1_overall_passed':False,'scientific_acceptance':False,'actual_INT8':False,'Holdout_accessed':False}
    write(OUT/'delivery_validation.json',validation)
    state={'schema':'rampq.gate1.execution_state/v36','status':'COMPLETE_FROZEN_SCORER_286_EXTENSION_DEVELOPMENT_ONLY',
        'created_at_utc':P.now(),'previous_snapshot':previous,'current_experiment':'artifacts/gate1/20261026_scorer_extension_286_v1',
        'report':'docs/73_gate1_frozen_scorer_286_image_extension_review.md','delivery_validation':identity(OUT/'delivery_validation.json'),
        'machine_decision':identity(OUT/'machine_decision.json'),'contract_sha256':identity(OUT/'contract.json')['sha256'],
        'primary_new286_joint_rule_met':decision['primary_new286_joint_rule_met'],
        'all_three_overlapping_cohorts_joint_rule_met':decision['all_three_overlapping_cohorts_joint_rule_met'],
        'primary_fixed_training_threshold_joint_observation':decision['primary_frozen_operating_joint_observation'],
        'frozen_threshold':c['scorer_operating_threshold'],'scorer_was_not_refit':True,'new_detector_forwards':572,'AP_jobs_completed':9,
        'all_cohorts_and_branches_preserved':True,'Gate1_overall_passed':False,'scientific_acceptance':False,'actual_INT8':False,
        'new286_numerical_qualification':'NOT_RUN','old_numerical_FAILs_preserved':True,'Selection_payload_access':False,'Holdout_payload_access':False,
        'no_active_worker':True,'heartbeat_recreated':False,'cloud_resources_created':False,
        'next_action':'Registered286extension budget complete and sealed. Read all3cohort outcomes,ordinaryQAT contrast,and frozen-threshold actual counts together. Do not tune these results, rerun completed stages, or call overlapping developer cohorts independent acceptance. Freeze any next hypothesis/data-access/device budget separately before execution.'}
    snap=P.ROOT/'artifacts/gate1/current_execution_state_v36.json';write(snap,state)
    pointer={'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v36.json',
        'snapshot_sha256':identity(snap)['sha256'],'delivery_validation':identity(OUT/'delivery_validation.json'),
        'note':'Frozen scorer286extension complete/audited:3branches x286new/47prefix-new/414combined; no refit; read report73 for truthful AP,FP-budget and frozen-threshold outcomes. Not independent/Gate1/INT8 acceptance; Holdout sealed.'}
    temp=latest.with_name('latest_execution_state.v36.pending.json');write(temp,pointer);temp.replace(latest)
    print(json.dumps({'status':validation['status'],'members_verified':len(expected),'latest':pointer['snapshot'],'snapshot_sha256':pointer['snapshot_sha256'],
        'primary_rule_met':decision['primary_new286_joint_rule_met'],'all_three_rule_met':decision['all_three_overlapping_cohorts_joint_rule_met']},ensure_ascii=True))


if __name__=='__main__':main()
