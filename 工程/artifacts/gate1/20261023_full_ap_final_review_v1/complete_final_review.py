"""Root completion of small metadata/link review; no detector or AP rerun."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'artifacts/gate1'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def ident(path):return {'bytes':path.stat().st_size,'sha256':sha(path)}


def main():
 target=HERE/'final_review.json';assert not target.exists()
 report=ROOT/'49_Gate1_AP聚合原生对照与忽略几何验证.md'
 state_path=BASE/'current_execution_state_v10.json'
 decision_path=BASE/'20261023_full_ap_delivery_v1/decision.json'
 assert sha(report)=='6097570f785d98b8a820bef76d276bc5d3dd3be4bffd03567fba8ae4aaccee46'
 assert sha(state_path)=='ae7b1e3e9819d39e7e64eba078fc51219b7647a9ed779b7d5984426738734936'
 assert sha(decision_path)=='ea4d3cfaf3d6552fa9bd619fc4761d302aefcba4faafc39e9709883f81bf83e1'
 source_review=read(HERE/'source_review.json')
 assert source_review['status']=='PASS_SOURCE_TRANSLATION_AND_RESTRICTED_GEOMETRY_PREPARATION_SCOPE_ONLY' and not source_review['issues']
 paths={Path(__file__),report,state_path,decision_path,HERE/'source_review.json'}
 for relative,expected in source_review['sources'].items():
  path=(ROOT/relative).resolve(strict=True)
  assert path.is_relative_to(ROOT) and ident(path)==expected,relative
  paths.add(path)
 decision,state=read(decision_path),read(state_path)
 for expected in decision['evidence']:
  path=ROOT/expected['path'];assert ident(path)=={k:expected[k] for k in ('bytes','sha256')}
  paths.add(path)
 result=read(BASE/'20261023_full_ap_oracle_audit_v1/comparison_attempt_v1.json')
 summary=read(BASE/'20261023_full_ap_oracle_audit_v1/audit_summary.json')
 assert result['status']==summary['status']=='PASS_DECLARED_SYNTHETIC_FULL_AP_NATIVE_COMPARISON_ONLY'
 matrices=[row for suite in result['suites'] for row in suite['matrices']]
 metrics=[row for suite in result['suites'] for row in suite['metrics']]
 assert len(matrices)==220 and all(row['exact'] and row['dtype']=='float64' for row in matrices)
 assert len(metrics)==161 and all(row['native_is_nan']==row['python_is_nan'] for row in metrics)
 assert sum(row['native_is_nan'] for row in metrics)==28
 finite=[row for row in metrics if not row['native_is_nan']]
 assert len(finite)==133
 for row in finite:
  assert all(math.isfinite(row[key]) for key in ('native','python','absolute_error','allowed_error'))
  assert abs(row['python']-row['native'])==row['absolute_error']
  assert row['allowed_error']==1e-10+1e-12*abs(row['native']) and row['absolute_error']<=row['allowed_error']
 assert max(row['absolute_error'] for row in finite)==7.105427357601002e-15
 assert [(s['matrix_count'],s['metric_count'],s['metric_nan_count']) for s in result['suites']]==[(166,77,14),(54,84,14)]
 assert summary['case_count']==32 and summary['aggregate_case_count']==23 and summary['image_count']==40
 assert result['reader']['dependency_verification']['verified_extracted_member_count']==1535
 assert decision['zero_tolerance_intermediate_matrices']==220 and decision['final_AP_AR_metric_count']==161
 assert decision['finite_metric_count']==133 and decision['matching_nan_metric_count']==28
 assert state['full_Python_AP_implemented'] and state['declared_synthetic_AP_native_agreement']
 assert state['same_backend_dtype_guard']=='FAIL_FIXED_TOLERANCES' and state['old_gate05_numeric_guard']=='FAILED_UNCHANGED'
 assert state['ignore_aware_training_pipeline_integrated'] is False and state['live_fp32_reference_hook_integrated'] is False
 assert state['arbitrary_input_AP_equivalence_certified'] is False and state['real_dataset_AP_computed'] is False and state['gate1_scientific_pass'] is False
 assert set(state['deprecated_ambiguous_fields_from_previous_snapshot'])=={'official_AP_implemented','official_oracle_verified','official_oracle_verified_field_scope','official_native_runtime_found_in_checked_locations'}
 ap=read(BASE/'20261023_full_ap_preparation_v1/preparation_receipt.json')
 geometry=read(BASE/'20261023_ignore_geometry_preparation_v1/final_tests_receipt.json')
 assert ap['test_count']==54 and ap['test_failures']==0 and geometry['counts']=={'tests':85,'failures':0,'errors':0,'skipped':0}
 tests=ET.parse(BASE/'20261023_full_ap_oracle_audit_v1/comparator_regression_v2.xml').getroot()
 suites=list(tests.iter('testsuite'));assert sum(int(s.get('tests','0')) for s in suites)==45
 assert all(int(s.get(k,'0'))==0 for s in suites for k in ('failures','errors','skipped'))
 links=[]
 for item in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
  path=Path(item);assert path.is_absolute() and path.is_file() and path.resolve().is_relative_to(ROOT),item
  links.append(path.relative_to(ROOT).as_posix())
 review={'schema':'rampq.gate1.full_ap_final_root_metadata_review/v1',
  'status':'PASS_ROOT_FINAL_METADATA_LINKS_AND_SCOPE_REVIEW_ONLY','at_utc':datetime.now(timezone.utc).isoformat(),
  'reviewer':'root','issues':[],'independent_source_review_completed':True,
  'independent_final_reviewer_reported_no_blockers_before_interruption':True,
  'independent_final_review_file_was_not_produced':'Reviewer stopped at its service usage limit after reporting source/numeric/scope checks; this final artifact is authored by root, not attributed to that reviewer.',
  'root_review_operation':'Small JSON/XML/link metadata, source/evidence identities and saved finite metric arithmetic only; no MAT decoding or AP/model rerun.',
  'local_links_checked':len(links),'local_links':links,
  'comparison':{'cases':32,'aggregate_cases':23,'images':40,'exact_matrices':220,'metrics':161,'finite':133,'matching_nan':28,'maximum_finite_absolute_error_percentage_points':7.105427357601002e-15},
  'synthetic_tests_verified':{'implementation':54,'comparator_negative':45,'geometry':85},
  'reader_count_scope':'Actual comparison receipt verifies1535 extracted files;1656 ZIP entries include121 directories. No new-versus-old file-count distinction is claimed.',
  'limits_verified':['Fixed synthetic domain and chosen runtime, not arbitrary-input or real-data AP certification.',
   'Old20 native references reused; new12 stress outputs acquired before comparison; no new independent datasets.',
   'Native internal AP/AR tables are not independently exported/compared.',
   'Geometry, quarantine/data contract and real Dataset/Trainer/criterion integration remain incomplete.',
   'Live FP32 atomic EMA capture, architecture reconstruction and provenance remain incomplete.',
   'No new training, model inference, Selection/Holdout use or backend/board/performance claim; original failed guards preserved.'],
  'sources':{p.relative_to(ROOT).as_posix():ident(p) for p in sorted(paths)},
  'model_or_real_data_executed_or_parsed':False,'gate1_scientific_pass':False}
 target.write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({'status':review['status'],'sources':len(paths),'links':len(links),'review_sha256':sha(target)}))


if __name__=='__main__':main()
