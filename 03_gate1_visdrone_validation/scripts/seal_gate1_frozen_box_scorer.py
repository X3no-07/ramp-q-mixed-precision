"""Seal all registered scoring outcomes; update only the mutable latest pointer."""
from datetime import datetime,timezone
from pathlib import Path
import json
from audit_gate1_frozen_box_scorer import S,P,OUT,A

read,write,identity,require=P.read,P.write,P.identity,P.require


def main():
    c=read(OUT/'contract.json');decision=read(OUT/'machine_decision.json');audit=read(OUT/'saved_delivery_audit.json')
    terminal=read(OUT/'launcher_terminal_receipt.json');process=read(OUT/'terminal_validation.json')
    require(terminal['propagated_worker_exit_code']==0 and terminal['completion_receipt_exists'] and not terminal['failure_receipt_exists'],'Terminal failure')
    require(process['matching_active_pilot_count']==0,'Scorer worker still active')
    require(read(A/'completion_receipt.json')['new_detector_image_forwards']==0,'Unexpected model forwards')
    audit_spec=read(OUT/'saved_audit_contract.json')
    require(identity(audit_spec['script']['path'])==audit_spec['script'],'Audit source changed')
    latest=P.ROOT/'artifacts/gate1/latest_execution_state.json';previous=read(latest)
    require(previous['snapshot']=='artifacts/gate1/current_execution_state_v33.json','Latest pointer changed')
    require(identity(P.ROOT/previous['snapshot'])['sha256']==previous['snapshot_sha256'],'Previous snapshot changed')
    expected={}
    def add(b):
        if b['path'] in expected:require(expected[b['path']]==b,'Conflicting file identity')
        expected[b['path']]=b
    for b in read(A/'manifest.json')['files']+c['reused_files']+list(c['source_bindings'].values())+list(c['previous_source_bindings'].values()):add(b)
    for key in ('candidate','architecture','train_manifest','training_val_manifest','evaluation_protocol','previous_contract','previous_delivery','previous_comparison','training_annotation_assignment'):add(c[key])
    root=Path(read(c['evaluation_protocol']['path'])['prepared_root']).resolve(strict=True)
    for meta in c['training_annotation_metadata']+c['evaluation_members']:
        path=(root/meta['annotation_path']).resolve(strict=True)
        require(path.is_relative_to(root),'Annotation path escaped')
        b=identity(path);require(b['sha256']==meta['annotation_sha256'],'Allowed annotation changed');add(b)
    for path in sorted(OUT.rglob('*')):
        if path.is_file():add(identity(path))
    for path in (P.ROOT/previous['snapshot'],P.ROOT/'docs/70_gate1_frozen_box_scorer_validation_launch.md',
                 P.ROOT/'docs/71_gate1_frozen_box_scorer_results_and_false_positive_budget.md',P.ROOT/'scripts/launch_gate1_frozen_box_scorer.py',
                 P.ROOT/'scripts/audit_gate1_frozen_box_scorer.py',P.ROOT/'scripts/summarize_gate1_frozen_box_scorer.py',Path(__file__)):
        add(identity(path))
    for b in expected.values():require(identity(b['path'])==b,'Delivery file changed: '+b['path'])
    manifest={'schema':'rampq.frozen_box_scorer.delivery/v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
              'files':[expected[k] for k in sorted(expected)],'mutable_pointer_excluded':True,
              'input_payload_scope':'Only explicit existing training/output arrays and authorized512train+128training_val annotations; no image file payload read or Holdout access.'}
    write(OUT/'delivery_manifest.json',manifest)
    validation={'status':'PASS_SCORER_DELIVERY_INTEGRITY_ONLY','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'manifest':identity(OUT/'delivery_manifest.json'),'members_verified':len(expected),'missing':0,'mismatched':0,
        'audit':identity(OUT/'saved_delivery_audit.json'),'machine_decision':identity(OUT/'machine_decision.json'),
        'report':identity(P.ROOT/'docs/71_gate1_frozen_box_scorer_results_and_false_positive_budget.md'),'observed_worker_exit_code':0,
        'development_rule_met_candidates':decision['registered_candidates_meeting_development_joint_rule'],
        'Gate1_overall_passed':False,'scientific_acceptance':False,'actual_INT8':False,'Holdout_accessed':False}
    write(OUT/'delivery_validation.json',validation)
    state={'schema':'rampq.gate1.execution_state/v34','status':'COMPLETE_FROZEN_BOX_SCORER_PILOT_GENERIC_DEVELOPMENT_SIGNAL_ONLY',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'previous_snapshot':previous,
        'current_experiment':'artifacts/gate1/20261026_frozen_box_scorer_pilot_v1',
        'report':'docs/71_gate1_frozen_box_scorer_results_and_false_positive_budget.md','delivery_validation':identity(OUT/'delivery_validation.json'),
        'machine_decision':identity(OUT/'machine_decision.json'),'contract_sha256':identity(OUT/'contract.json')['sha256'],
        'development_rule_met_candidates':decision['registered_candidates_meeting_development_joint_rule'],
        'intervention_development_rule_met':decision['intervention_meets_development_joint_rule'],
        'intervention_advantage_over_both_controls':decision['intervention_descriptive_advantage_over_both_scorers'],
        'all3registered_candidates_preserved':True,'all_updates_complete':True,'new_detector_image_forwards':0,
        'Gate1_overall_passed':False,'scientific_acceptance':False,'independent_data_acceptance':False,
        'actual_INT8':False,'old_numerical_FAILs_preserved':True,'Selection_payload_access':False,'Holdout_payload_access':False,
        'no_active_worker':True,'heartbeat_recreated':False,'cloud_resources_created':False,
        'next_action':'All fixed3scorer pilot work and delivery complete. Generic score_base has a developer signal; intervention-specific value remains unsupported. Do not rename generic control as intervention method, rerun completed stages, pick favorable thresholds, or claim independent acceptance. Freeze any further replication/candidate/data-access budget before running it.'}
    snap=P.ROOT/'artifacts/gate1/current_execution_state_v34.json';write(snap,state)
    pointer={'schema':'rampq.gate1.mutable_latest_state_pointer/v1','snapshot':'artifacts/gate1/current_execution_state_v34.json',
        'snapshot_sha256':identity(snap)['sha256'],'delivery_validation':identity(OUT/'delivery_validation.json'),
        'note':'3fixed frozen-box scorers complete/audited. Generic scorer:AP+0.096968pp,37moreTP at same2605officialFP budget. Intervention fails AP joint rule; all candidates retained, no independent/Gate1/INT8 acceptance, Holdout sealed.'}
    temporary=latest.with_name('latest_execution_state.v34.pending.json');write(temporary,pointer);temporary.replace(latest)
    print(json.dumps({'status':validation['status'],'members_verified':len(expected),'snapshot':pointer['snapshot'],
        'snapshot_sha256':pointer['snapshot_sha256'],'development_rule_met':validation['development_rule_met_candidates']},ensure_ascii=True))


if __name__=='__main__':main()
