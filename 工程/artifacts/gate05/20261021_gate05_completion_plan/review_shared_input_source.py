"""Actual preparer source-only path plus static algorithm audit; zero image decode."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
source=ROOT/'scripts/prepare_gate05_shared_inputs.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
text=source.read_text(encoding='utf-8');tree=ast.parse(text)
checks=[]
def c(name,value):checks.append({'name':name,'passed':bool(value)})
c('pinned_source_and_manifest',sha(ROOT/'src/gate0.py')=='f3c4d80658b79ae685e569b37c5ea89af9e6e9dd4a63ce86aff2806a7fcc5adb' and sha(ROOT/'artifacts/gate05/data_v1/manifest.json')=='c4dac80ad7c59524194a9501f5ff8126cc0517e84f29f4106068c0c4b27a93aa')
c('old_letterbox_AST_extracted_not_reimplemented',"n.name=='letterbox'" in text and "exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)" in text)
c('same_cv2_decode_mode',"cv2.imread(str(image),cv2.IMREAD_COLOR)" in text)
c('same_RGB_CHW_conversion',"canvas[:,:,::-1].transpose(2,0,1)" in text)
c('same_FLOAT32_divide255_and_batch',"(np.ascontiguousarray(rgb,dtype=np.float32)/255.0)[None]" in text)
c('same_640_batch1_layout',"array.shape==(1,3,640,640)" in text and 'array.flags.c_contiguous' in text)
c('environment_bound_before_image_decode',"cv2.__version__==contract['environment']['opencv'] and np.__version__==contract['environment']['numpy']" in text)
c('no_model_imports',not any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(name in (getattr(n,'module','') or '') or any(name in a.name for a in getattr(n,'names',[])) for name in ('torch','onnx','onnxruntime','tensorrt','ultralytics')) for n in ast.walk(tree)))
c('exact15086564_counts',"manifest['evaluation_count']==150" in text and "manifest['evaluation_target_count']==865" in text and "manifest['calibration_count']==64" in text)
c('per_image_target_counts',"count==row['target_count']" in text)
c('blank_and_invalid_label_rows_not_silently_counted',"if not line.strip():continue" in text and "require(len(parts)==5" in text and 'math.isfinite' in text)
c('image_and_label_SHA_each',"sha(image)==row['image_sha256'] and sha(label)==row['label_sha256']" in text)
c('actual_calibration_identity_checked',"sha(image)==row['image_sha256'],'Calibration source image changed'" in text)
c('overlap_and_duplicates_checked','not seen.intersection' in text and "row['image_sha256'] not in seen" in text)
c('protected_source_before_after_all',"before=identities(protected)" in text and "after=identities(protected)" in text and "before==after" in text)
c('decode_mutability_checked',"'Image changed before decode: '" in text and "'Image changed during decode: '" in text)
c('no_shadow_source_import',not any((isinstance(n,ast.ImportFrom) and 'gate0' in (n.module or '')) or (isinstance(n,ast.Import) and any('gate0' in a.name for a in n.names)) for n in ast.walk(tree)))
c('fresh_safe_output_no_overwrite',"out.parent==ROOT/'artifacts/gate05'" in text and "not out.exists() and not out.with_suffix('.zip').exists()" in text)
c('zero_models_exact_image_budget',"contract['model_executions']==0 and contract['image_preprocessing_executions']==150" in text)
c('closed_zip_exact_inventory_and_SHA',"set(z.namelist())=={prefix+name for name in payload}" in text and "digest.hexdigest()==identity['sha256']" in text)
c('source_finally_rechecked_after_zip',"before==identities(protected),'Protected source changed during archive creation'" in text)
c('partial_failure_no_retry',"'no_auto_retry':True" in text and "'completed_images':len(result['images'])" in text)
# This is deliberately not a frozen execution protocol. --execute is omitted;
# the actual preparer validates 150/865/64 files and returns before cv2/numpy.
fixture=HERE/'shared_input_readonly_check_fixture.json'
if not fixture.exists():
    with fixture.open('x',encoding='utf-8') as f:json.dump({'schema':'rampq.gate05.shared_input_preparation_protocol/v1','status':'REVIEW_ONLY_NO_EXECUTION_FREEZE','preparer_sha256':sha(source),'model_executions':0,'image_preprocessing_executions':150},f)
modspec=importlib.util.spec_from_file_location('review_shared_input_only',source)
module=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(module)
output=ROOT/'artifacts/gate05/REVIEW_ONLY_NEVER_CREATED'
argv=sys.argv;stream=io.StringIO()
try:
    sys.argv=[str(source),'--protocol',str(fixture),'--protocol-sha256',sha(fixture),'--output',str(output)]
    with contextlib.redirect_stdout(stream):module.main()
finally:sys.argv=argv
actual=json.loads(stream.getvalue())
c('actual_preparer_source_only_branch',actual=={'status':'INPUT_PREPARATION_SOURCES_VERIFIED_NO_MODEL_OR_PREPROCESS_EXECUTION','images':150,'targets':865})
c('no_numpy_cv2_model_runtime_imports',not any(name in sys.modules for name in ('numpy','cv2','torch','onnx','onnxruntime','tensorrt','ultralytics')))
c('source_only_output_not_created',not output.exists())
report={'schema':'rampq.gate05.shared_input_source_review/v1','status':'SOURCE_ONLY_PREFLIGHT_AND_STATIC_PREPROCESSING_VERIFIED' if all(r['passed'] for r in checks) else 'BLOCKED','check_count':len(checks),'checks':checks,'blockers':[r['name'] for r in checks if not r['passed']],'preparer_sha256':sha(source),'zero_image_decodes_preprocessing_model_native_or_SSH':True,'actual_preparer_check_output':actual,'review_fixture_not_frozen_execution_protocol':str(fixture.relative_to(ROOT)).replace('\\','/'),'source_changes':['Protected150images/labels+64calibration/code/protocol before/after SHA identities and each decode surrounding checks.','Strict per-image label inventory against frozen counts; no silent invalid row count.','Fresh output restricted to a new direct child of Gate0.5 artifact root.','Closed ZIP exact inventory, member SHA/length and ZIP CRC verified at member EOF; final source reread.'],'unchanged_preprocessing':'Pinned old letterbox AST; original cv2.imread/BGR->RGB->CHW float32 division255->batch1. No mirrored image preprocessing executed.','environment_scope':'Frozen current OpenCV/Numpy must be recorded; identical code is not a claim of byte equality to untested older library versions.'}
report['local_review_revision']={'previous_report_sha256':sha(HERE/'shared_input_source_review.json'),'previous_false_string_check':'Comment containing never import gate0 tripped a raw substring check; actual import AST check now used. No image or model was executed in either check.'}
out=HERE/'shared_input_source_review_v2.json'
with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'checks':len(checks),'blockers':report['blockers'],'preparer_sha256':sha(source),'review_sha256':sha(out)}))
