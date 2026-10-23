"""Write a proposal only. No model imports, graph execution, device, or SSH."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
data_path = ROOT/'artifacts/gate05/data_v1/manifest.json'
data = read(data_path)
gap = read(HERE/'acceptance_gap_audit.json')
source_keys = ['original_gate_remaining','build_pass_field','execution_pass_field','not_a_pass_condition','dataset_gate_unfrozen','original_numeric_tolerances','original_quant_scheme']
sources = {k:gap['sources'][k] for k in source_keys}
sources['research_route'] = {'path':'研究状态与后续计划.md','sha256':sha(ROOT/'研究状态与后续计划.md'),'line_start':167,'line_end':187,'interpretation':'Original Gate question; five fixed configurations; G1 recovery as primary existence check; optional q_only G1 for bidirectional overlap; no outcome-based substitution.'}
for key, rel, lines in [
    ('preprocessing','src/gate0.py',[119,141]),
    ('postprocess','src/gate0.py',[394,420]),
    ('matching','src/gate0.py',[424,474]),
    ('target_identity','src/gate0.py',[216,245]),
    ('gate0_config','config/gate0.json',None),
    ('prior_reference_contract','artifacts/gate05/20261019_reference_contract/reference_contract.json',None),
    ('dataset','artifacts/gate05/data_v1/manifest.json',None),
    ('module_mapping','artifacts/gate05/20261021_qall_none_control/source_bundle/module_map.json',None),
    ('sass_static_review','artifacts/gate05/20261021_gate05_completion_plan/sass_static_independent_review.json',None),
    ('old_kernel_trace','artifacts/gate05/20261019_qall_kernel_diagnostic/analysis.json',None),
]:
    sources[key] = {'path':rel,'sha256':sha(ROOT/rel),'lines':lines}
configs = [
    {'name':'fp16_control','role':'CORE_NATIVE_BASELINE','quantized_groups':[],'fp16_groups':['G1','G2','G3','G4','G5','G6','G7','G8'],'expected_int8_source_convs':0,'expected_fp16_source_convs':88},
    {'name':'q_all','role':'CORE_DAMAGE_SIDE','quantized_groups':['G1','G2','G3','G4','G5','G6','G7','G8'],'fp16_groups':[],'expected_int8_source_convs':88,'expected_fp16_source_convs':0},
    {'name':'r_only_G1','role':'CORE_PRIMARY_RECOVERY','quantized_groups':['G2','G3','G4','G5','G6','G7','G8'],'fp16_groups':['G1'],'expected_int8_source_convs':82,'expected_fp16_source_convs':6},
    {'name':'r_only_G1_G5','role':'CORE_PREDECLARED_SECONDARY_RECOVERY','quantized_groups':['G2','G3','G4','G6','G7','G8'],'fp16_groups':['G1','G5'],'expected_int8_source_convs':74,'expected_fp16_source_convs':14},
    {'name':'r_only_G1_G2','role':'CORE_PREDECLARED_SECONDARY_RECOVERY','quantized_groups':['G3','G4','G5','G6','G7','G8'],'fp16_groups':['G1','G2'],'expected_int8_source_convs':77,'expected_fp16_source_convs':11},
    {'name':'q_only_G1','role':'PREDECLARED_AUXILIARY_WEAK_BIDIRECTIONAL_CONTROL','quantized_groups':['G1'],'fp16_groups':['G2','G3','G4','G5','G6','G7','G8'],'expected_int8_source_convs':6,'expected_fp16_source_convs':82},
]
for c in configs:
    c.update({'source_graph_sha256':None,'engine_sha256':None,'build_request_sha256':None,'native_precision_audit_sha256':None,'float_fusion_comparability_audit_sha256':None,'independent_ORT_reference_graph_sha256':None,'binding_status':'REQUIRES_ROOT_REVIEW_AND_BINDING_BEFORE_FREEZE'})
draft = {
 'schema':'rampq.gate05.dataset_acceptance_contract_draft/v1',
 'client_at':datetime.datetime.now().astimezone().isoformat(),
 'status':'DRAFT_FOR_ROOT_REVIEW_NOT_FROZEN_NOT_EXECUTION_AUTHORIZATION',
 'author_role':'Independent acceptance reviewer; proposal from existing original Gate purpose, not a retrospective pass amendment',
 'model_CPU_native_or_SSH_execution_by_this_draft':False,
 'original_documents_or_results_modified':False,
 'overall_gate05_current_status':'INCOMPLETE',
 'freeze_rule':{
   'must_freeze_before':'Any new six-configuration 150-image target/output dataset run',
   'root_must_review_this_draft':True,
   'all_null_artifact_identities_must_be_bound_or_explicitly_justified_before_execution':True,
   'separate_manifest_required':'Exact source graph/weights/ranges/module map/build flags/actual precision/float-fusion comparisons and evaluator hashes for all six; runtime arithmetic evidence may be collected in predeclared fixed-input preflight before dataset',
   'historical_holdout_results_known':True,
   'outcome_unknown_scope':'New matched native family 150-image target decisions only; this is not fresh independent research preregistration',
   'no_freeze_timestamp_or_approval_in_this_draft':True,
   'no_selective_family_or_parameter_revisions_after_dataset_outputs':'Preserve first preregistered outcomes. A separately justified subsequent family is a new diagnostic, cannot replace its result.'
 },
 'purpose_and_claim':{
   'question':'On a fixed real Orin deployment family, does a target hit by native FP16 disappear under verified W8A8 q_all and return for the same target under the predeclared G1 FP16 restoration?',
   'primary_estimand':'Existence of one or more target IDs in (F minus Q) intersect R_G1 on this fixed old Holdout',
   'primary_group_basis':'Original research route item 9 explicitly names G1; combined G1+G5/G1+G2 are required predeclared secondary summaries, not outcome-selected replacements.',
   'scope':'Local deployment migration and phenomenon authenticity, conditional on the tested compiler family and original data',
   'not_claimed':['robust superiority','unseen-data generalization','method novelty','G1-only causal mechanism','statistical significance','positive net benefit from any nonzero rescue','latency/FPS/power/energy/Pareto advantage','all-network INT8','original PyTorch/ORT numeric guard repaired'],
   'no_new_effect_size_threshold':'No arbitrary 99-percent agreement, minimum rescue-rate or significance threshold is invented to obtain PASS.'
 },
 'protected_history':{
   'original_complete_PyTorch_ORT_guard':'FAILED_UNCHANGED',
   'protected_preflight_path':'artifacts/gate05/20261018_int8/input_v1/preflight.json',
   'protected_preflight_sha256':'505d139b56c973196792ef60b7bba95e82bca612fab3b12b6183a53f7a815f21',
   'original_tolerances':{'box_rtol':0.005,'box_atol':0.5,'score_rtol':0.02,'score_atol':0.002,'changed':False},
   'failure_explanation_status':'Same-input quantizers and weight codes matched; small Conv floating differences reach activation rounding boundaries and propagate. Saved localization establishes this mechanism at inspected points, not an exhaustive explanation of every downstream difference.',
   'old_guard_is_not_relabelled':'A new independent full ONNX reference is a distinct deployment/export contract; no old row becomes PASS and no hybrid counterfactual becomes a reference.',
   'all_prior_build_numeric_and_precision_failures_retained':True,
   'old_missing_named_integer_tactic_statuses_unchanged':True,
   'old_single_image_and_slices':'Evidence reusable for provenance/localization only; not new 150-image performance or recovery evidence.'
 },
 'fixed_data_and_model':{
   'dataset_manifest_path':str(data_path.relative_to(ROOT)).replace('\\','/'),
   'dataset_manifest_sha256':sha(data_path),
   'evaluation_images':150,'evaluation_targets':865,'calibration_images':64,
   'image_content_overlap':0,
   'weights_sha256':data['weights_sha256'],
   'calibration_manifest_sha256':data['source_calibration_manifest_sha256'],
   'canonical_calibration_list_sha256':data['canonical_calibration_list_sha256'],
   'canonical_holdout_list_sha256':data['canonical_holdout_list_sha256'],
   'source_holdout_file_sha256':data['source_holdout_file_bytes_sha256'],
   'order':'Exact manifest.data.evaluation order, all images and all label-file SHA identities; no hard/easy subset selection',
   'legacy_holdout_scope':'Previously used Gate0 Holdout; diagnostic reuse only, not independent method confirmation',
   'fixed_smoke_image':'000020.jpg is the original Selection smoke image, not an added 151st Holdout image and not a substitute for the 150-image run',
   'frozen_calibration_semantics':{'activation':'symmetric per-tensor zero-point0, max(absmax/127,1e-12), narrow codes[-127,127]','weight':'symmetric per-output-channel axis0 zero-point0, max(per-channel absmax,1e-12)/127','convolution_output_quantizer_added':False,'recalibration_or_refitting':False},
   'group_counts':{'G1':6,'G2':5,'G3':10,'G4':19,'G5':8,'G6':15,'G7':8,'G8':17},
   'immutable_across_family':['trained source weights and biases','frozen calibration ranges/scales/zero points for the same remaining quantizer','source Conv identity and G1-G8 assignment','source nonquantized arithmetic except authenticated export-preserving rewrite','source image bytes, label bytes and target IDs','preprocessed RGB FP32 input array bytes per image','640 input, batch1, bindings and primary-output interpretation','confidence/NMS/IoU/matching/tie-breaking rules','common device/runtime/build policy declared before results'],
   'forbidden_outcome_adaptation':['replace images/labels','select new module combinations','recalibrate or tune ranges','alter match/confidence/NMS','take best of repeated builds or runs','discard images with no detections or inconvenient targets','select one reference or precision contract after outcomes']
 },
 'configurations':configs,
 'family_build_and_format_contract':{
   'recommended_common_policy':'Separate new matched family with runtime precision_constraints NONE; INT8+FP16 permitted for all six including unquantized FP16 control, TF32 false; no OBEY/PREFER; workspace1024MiB, avg_timing_iterations1, FLOAT LINEAR external bindings and detailed inspector. Root must confirm actual successful family rather than assuming q_all success transfers.',
   'same_flags_even_if_unused':'Do not vary INT8 enablement as an extra factor in FP16 control; absent quantizers/actual types, not the enable flag alone, define each source precision.',
   'code_clip_importer_correction':'For each remaining activation Q->INT8 Clip->DQ, record exact parser before/after alpha,beta and set to -127,+127; source/CPU equivalence and actual request audit required. Do not bypass weight QDQ as W8A8 repair.',
   'parser_requests':'Preserve intrinsic parsed precision/output requests; no hidden resets or manual per-layer precision/output setters. Record every layer before/after/build and actual get_flag values, not only requested string.',
   'binding':'Input images FLOAT LINEAR[1,3,640,640], primary output0 FLOAT LINEAR[1,14,8400]; common PE auxiliary declaration if retained must be identical across family and excluded from target inference.',
   'per_quantized_source_required':['unique source-to-native weighted operation mapping or verified in-group fusion accounting','native INT8 activation input provenance and INT8 weight type/count','correct source group, kernel weights/counts, activation quantizer/ranges','authenticated integer arithmetic evidence via unchanged named-tactic route or separately labelled new runtime+SASS route'],
   'quantized_output_format':'Do not require INT8 Conv output: FP16/FP32 output is permitted and must be recorded exactly under the existing audit rule.',
   'per_fp16_source_required':'Use existing restored-source audit: trace to original unquantized weight and activation, FP16 weights and actual FP16 convolution input/output formats, source weight-count/fusion checks. If compiler does not realize requested FP16 treatment, the intended configuration is ineligible, not scientific no-recovery.',
   'all_88_sources_accounted_per_configuration':True,
   'cross_group_fusion':'Apply original auditor rule; any unassignable mixed-group fusion or missing source identity is an eligibility gap, not omitted from counts.',
   'builder_success_alone_is_not_precision_or_numeric_pass':True,
   'input_extraneous_graph_output_or_policy_change':'Requires explicit new identity/comparability review before dataset; not silently interchangeable with historical engines.'
 },
 'floating_and_fusion_comparability':{
   'goal':'Make matched treatment claims honest by documenting shared preparation and actual floating computation; same flags/source alone do not prove identical realized arithmetic.',
   'require_exact_when_applicable':['source nonquantized node op/attribute/initializer bytes and primary-output indexing','source weights/bias identities, underlying tensor dimensions and constants','shared input array bytes and preprocessing transformations','remaining same quantizer scale/zero-point/axis/bounds','source Conv group identities and full source coverage','external bindings/layouts and independent evaluator implementation','same common builder flags/policy/device/software versions/power-mode request'],
   'actual_metadata_required_for_all_six':['raw detailed inspector layers including LayerType, Name, input/output tensor names/formats/dimensions, weight type/count and any reported compute precision','source-to-native mapping and all fused source members','raw tactic names and tactic IDs when supplied; missing names explicitly reported','nonquantized attention MatMul/Softmax, floating pointwise/activation, decode and shape paths: map relevant source ops to actual realized layers/fusions and record FP16/FP32 differences','all floating boundaries affected by QDQ insertion/removal, dequantization and output casts'],
   'must_match_or_be_bridged':'Unchanged nonquantized source operations with available mapping must have the same actual floating dtype/format across the comparison or an explicitly preregistered floating control/precision bridge. If metadata cannot establish this, preserve a comparability gap; do not invent equality from operator names.',
   'not_reasonable_to_require':['identical layer count','identical fused layer names or fusion partitions','identical tactics for INT8 and FP16 Conv','identical memory scheduling or summation order','INT8 Conv output matching FP16 Conv output dtype','bit-identical native tensors across different quantization treatments'],
   'quantization_necessary_changes':'QDQ propagation, boundary reformat/cast, legal quantization-aware fusion and INT8-vs-FP16 weighted kernels can change as part of the deployed treatment. Record and account for these; forcing identical fusion can remove the intervention itself.',
   'eligibility_labels':{
      'MATCHED_SOURCE_AND_FLOAT_METADATA_WITH_QUANTIZATION_LINKED_FUSION':'Required for primary authenticity eligibility; shared unrelated float path matched or controlled, remaining documented changes belong to quantization/restoration deployment realization.',
      'UNCONTROLLED_UNRELATED_FLOAT_PATH_DIFFERENCE':'Not eligible for primary authenticity verdict without a prespecified bridge/control; target counts may be descriptive but primary INCONCLUSIVE.',
      'MAPPING_OR_FORMAT_UNAVAILABLE':'Engineering INCONCLUSIVE.',
      'STRONG_G1_ONLY_CAUSAL_IDENTIFICATION':'NOT_ESTABLISHED by this experiment even when the primary existence result passes.'
   },
   'conditional_interpretation':'A returned target is associated with this audited deployed G1-restoration configuration, including necessary precision/fusion changes. It does not isolate a pure G1 arithmetic effect from every compiler consequence.',
   'no_post_outcome_control_search':'Any additional bridge is declared from source/metadata before new Holdout outputs. After outcomes, classify the missing control as a limitation; do not search until rescue appears.'
 },
 'independent_reference_contract':{
   'ORT_role':'Complete standalone exported-graph semantic reference, evaluated from original preprocessed image input for each configuration. Not a replacement PASS for the old PyTorch guard and not native arithmetic ground truth.',
   'provider_and_policy_to_bind':'CPUExecutionProvider, FLOAT32 input, ORT_DISABLE_ALL, four threads; bind ORT/ONNX versions and graph/hash. These match existing full rewrite diagnostic unless a separately justified common policy is frozen.',
   'complete_reference_for_six':'One complete corresponding ONNX graph per native configuration; no native intermediate cache, Conv/SiLU injection, shortened DFL fixture, mixed-backend stitched forward or candidate-dependent target selection.',
   'reuse':'Existing fixed000020 full original/code-Clip byte equality and source/static hash proof may be reused for that exact graph/input. It does not supply 150-image outputs, nor automatically prove every new restoration graph.',
   'per_config_semantics':'Statically authenticate allowed treatment changes, source weights/bias, frozen scales and group map. Authenticate the rewrite Q+codeClip[-127,127]+DQ versus floatClip+Q+DQ semantics; any sampled existing CPU byte equality remains explicitly input-bounded.',
   'two_distinct_comparisons':['original PyTorch FP32/QDQ vs exported ORT: historical simulation/export/backend sensitivity and unchanged failed guard','independent corresponding full ORT vs native TensorRT: deployed precision/backend/computation differences, raw and target-level descriptive reports'],
   'no_numeric_tolerance_relaxation':'No large all-position atol/rtol is introduced to bless CPU/native differences. Original tolerance check, if repeated descriptively, remains labelled with its original threshold and result; this new contract does not assert raw numeric equivalence.',
   'mandatory_discrepancy_outputs':['per image/config output shape/dtype/finite counts, raw hashes, coordinate/class-score max/mean/RMS absolute differences and exact mismatch count','same evaluator native-vs-ORT matched target sets with full gains/losses and marginal IoU/confidence values','native/ORT no-detection images and score/box ranges without censoring','FP32 full source export/ORT baseline vs native FP16 matched-target differences reported separately; backend and precision are confounded unless a native FP32 control exists'],
   'numeric_validity_rules':{
      'technical_ineligible':['nonfinite outputs','wrong tensor indexing/layout/dtype/shape or box coordinate transform','unauthenticated source/weight/range or incorrect Clip/importer semantics','precision/mapping/integer-evidence requirement unproven','independent full reference missing/corrupt or input provenance mismatch','proven implementation contract violation or unexplained nonquantized-path mismatch that prevents intended treatment identification'],
      'not_automatically_ineligible':['finite quantized result having fewer detections','large CPU/native raw disagreement alone','quantized target not returning','negative net recovery'],
      'interpretation':'Do not turn an unfavorable valid quantized outcome into an engineering exclusion. Conversely, a documented implementation/mapping mismatch cannot be declared scientific failure of recovery or numerical PASS.',
      'gross_collapse':'Always report full score/box and per-target discrepancy diagnostics. If contract integrity is established and no technical violation is found under the frozen checks, collapse is a valid unfavorable deployed result; if its cause prevents identifying the promised W8A8/FP16 treatment, primary INCONCLUSIVE with exact unresolved evidence. No ad hoc output threshold or error-tolerance increase.'
   },
   'original_PyTorch_FP32_role':'If same 150-image source-FP32 results exist with exact hashes/settings, reuse and label as historical. Otherwise bind a separate complete source-FP32 reference before execution. Do not mix its hit denominator into native F-minus-Q; no seventh native engine is silently added.',
   'reference_output_count_recommendation':'Six full ORT graphs x150 images=900 complete reference forwards; exact already authenticated saved outputs may be reused per matching identity. Bind budgets before freeze; no run is triggered by this draft.'
 },
 'integer_arithmetic_evidence':{
   'old_branch':'Original named integer tactic regex/proxy audit and its prior missing-seven statuses remain immutable.',
   'new_branch_label':'RUNTIME_AUTHENTICATED_KERNEL_PLUS_STATIC_SIGNED_INT8_MAC_EVIDENCE',
   'accept_either_branch_per_actual_quantized_operation':'Original verified integer tactic metadata OR the separately labelled authenticated runtime/SASS chain; do not rewrite old tactic result to named-tactic PASS.',
   'runtime_SASS_chain_required':['source Conv mapping to actual weighted layer with INT8 input/weights and correct source/group/weight counts','NVTX source/operation interval and exact CUDA API/kernel correlation from the same profiled process/run, with attribution not ambiguous under fusion','device identity and compute capability SM87 for that run','same-process /proc maps or equivalent loaded CUDA module identity plus exact library/cubin bytes/hash; path existence alone insufficient','exact mangled launched kernel symbol -> selected SM87 SASS body from that authenticated binary','signed8 dot/MAC operand provenance and integer accumulation for the weighted operation, distinguishing address IMAD and postprocessing FFMA; actual assembly and dataflow rationale retained'],
   'static43_review_use':'Existing four exact libnvinfer SM87 function bodies have IDP.4A.S8.S8 and integer-result conversion chains. This proves those static bodies contain signed integer dot arithmetic, not that the new engine ran them.',
   'not_required_or_claimed':['kernel function name containing IMMA','Tensor Core execution solely from a function name','per-instruction hardware trace as a blanket requirement','all GPU instructions integer','no legitimate FP32 epilogue'],
   'branch_ambiguity':'A runtime symbol that cannot be tied to the actual weighted source computation, loaded binary or relevant signed-MAC path stays UNVERIFIED/INCONCLUSIVE; never use unrelated library instruction hits.',
   'restored_FP16_evidence':'Actual restored-source format/mapping audit; do not demand integer arithmetic for a source intentionally restored to FP16.',
   'coverage':'Apply to every required quantized source/actual fused weighted operation in each of the five quantized configurations; same authenticated operation/binary body may be reused only with each configuration runtime identity/correlation proved.'
 },
 'common_input_postprocess_and_matching':{
   'input':{'batch':1,'shape':[1,3,640,640],'dtype':'float32','color':'RGB','normalization':'divide255','order':'NCHW C-contiguous','preprocess_source':'src/gate0.py:119-141','letterbox':'cv2.INTER_LINEAR; ratio=min(640/h,640/w); new dimensions=round; pad114; left/top=round(halfpad-0.1), right/bottom=round(halfpad+0.1)','per_image_input_hash':'One frozen preprocessed tensor reused byte-for-byte for all native and ORT configurations'},
   'NMS':{'candidate_confidence':0.001,'iou':0.7,'max_det':1000,'classes':10,'agnostic':False,'multi_label':False,'implementation_and_version_sha256':'MUST_BIND_BEFORE_FREEZE; use the existing Ultralytics implementation and same backend/device for all native/ORT outputs','time_limit_behavior':'Existing max_time_img2.0 semantics must be recorded. Any truncation/time-limit warning yields incomplete image/INCONCLUSIVE, not silently accepted fewer boxes.','ties':'Freeze candidate order and NMS runtime/device/version; preserve raw output and detection order. Cross-backend NMS tie differences must not be hidden as model differences.'},
   'smoke_threshold_warning':'The original smoke used candidate confidence0.25; the fixed dataset uses original Gate0 diagnostic candidate0.001 followed by primary0.25. They are different scopes, not an adaptive threshold change.',
   'coordinate_transform':'Subtract exact letterbox left/top, divide by recorded ratio, clip to original image width/height. Match in original pixel coordinates.',
   'target_id':'image_file_name + colon + original zero-based physical label-line index, as src/gate0.py:238; never renumber by successful match or new annotation parsing',
   'target_inventory':'All865 target IDs exactly once per configuration; original label bytes and manifest counts. Parse/count inconsistency is engineering incomplete, no target omission.',
   'match':{'class':'exact same class','primary_confidence_min':0.25,'IoU_min':0.5,'algorithm':'enumerate all eligible GT/detection pairs; tuples(IoU,confidence,gt_index,detection_index); sort reverse=True; greedily assign one detection and one GT once','decision_precision':'Use unrounded floats; CSV rounding only presentation','source':'src/gate0.py:424-474'},
   'no_reassociation':'One fixed target ID defines damage and rescue; a nearby different object or class cannot count as the same target.'
 },
 'target_sets_and_metrics':{
   'universe':'T=865 fixed target IDs',
   'sets':{'F':'native fp16_control hits','Q':'native q_all hits','R_G1':'native r_only_G1 hits','R_G1_G5':'native r_only_G1_G5 hits','R_G1_G2':'native r_only_G1_G2 hits','B_G1':'native q_only_G1 hits','A':'F minus Q; native quantization-associated added misses','S':'F intersect Q; stable native baseline hits'},
   'per_recovery_group':{'rescued':'A intersect R_group','still_missing':'A minus R_group','collateral':'S minus R_group','net_count':'|rescued| minus |collateral|','rescue_rate':'|rescued|/|A| only if |A|>0, otherwise null plus reason','collateral_rate':'|collateral|/|S| only if |S|>0, otherwise null plus reason','stable_retention':'|S intersect R_group|/|S| if |S|>0','outside_F':'R_group minus F reported separately, never pooled into A-rescue'},
   'auxiliary_G1':{'destruction':'D_G1=F minus B_G1','dual_overlap':'A intersect D_G1 intersect R_G1','overlap_count':'number of identical target IDs in dual_overlap','overlap_rate':'|dual_overlap|/|A| if |A|>0; also report all component denominators and conditional overlaps','weak_positive':'At least one dual_overlap target with all six eligible: weak same-target bidirectional pattern for this deployed family','weak_negative':'No dual_overlap on this set: no positive bidirectional pattern here; does not refute G1 causal relevance generally','not_identified':'Uncontrolled compiler/fusion consequences or G1 correlations prevent pure-G1 causal attribution in either direction; q_only G1 does not test q_only G1+G5.'},
   'mandatory_strata':'Class and existing Gate0 area-ratio/small-object definition only if exact pre-existing thresholds are pinned; otherwise continuous area or predeclared bins as descriptive. No outcome-selected subgroups.',
   'all_targets_recorded':'Six matched bits, all rescue/collateral/overlap bits, reference matched bits and supporting detection IDs/IoU/confidence, including negatives.',
   'no_robustness_inference':'One or more recovered targets establishes local existence only; no repeated-random-sample, multiple-seed, independent-confirmation or population effect claim.'
 },
 'eligibility_and_decision':{
   'engineering_eligibility_E':'Frozen identity+reference+float/fusion comparability+actual source formats/integer evidence+complete150x6 raw/detection/865x6 target records+saved evidence/cleanup chain all satisfied.',
   'primary_ordered_rules':[
     {'priority':1,'if':'E is false or unknown','verdict':'INCONCLUSIVE_ENGINEERING_OR_REFERENCE_GAP','meaning':'Mechanism/existence is not scientifically tested to the promised contract; list exact gap. Cannot claim Gate complete/pass merely because a diagnostic/checkpoint finished.'},
     {'priority':2,'if':'E true and |F|=0','verdict':'INCONCLUSIVE_NO_NATIVE_FP16_ELIGIBLE_TARGETS','meaning':'No usable native baseline hit universe on this data. Report full outcome and investigate; no recovery denominator.'},
     {'priority':3,'if':'E true and |F|>0 and |A|=0','verdict':'FAIL_PHENOMENON_NOT_REPRODUCED_NO_ADDED_MISS','meaning':'Quantization-associated disappearance was not observed on this fixed diagnostic set. Not a claim that quantization cannot cause misses elsewhere.'},
     {'priority':4,'if':'E true and |A|>0 and |A intersect R_G1|=0','verdict':'FAIL_PRIMARY_G1_SAME_TARGET_RECOVERY_NOT_REPRODUCED','meaning':'Primary G1 recovery not observed. Report combined-group recoveries without substituting them for primary success.'},
     {'priority':5,'if':'E true and |A intersect R_G1|>0','verdict':'PASS_LOCAL_G1_SAME_TARGET_RECOVERY_EXISTENCE','meaning':'At least one exact native FP16-hit target disappeared under verified q_all and returned under G1 restoration. This is deliberately narrow existence, not overall accuracy or net benefit.'}
   ],
   'collateral_does_not_disappear_on_PASS':'Always report collateral/net/retention. A nonzero rescue with negative net is still narrow existence PASS but explicitly NO_NET_RECOVERY_BENEFIT; cannot write beneficial mixed precision or superior deployment.',
   'combined_groups':'Complete independent tables for G1+G5 and G1+G2 on the same A/S denominators; not outcome-selected winner, no monotonic-rescue assumption.',
   'auxiliary_status_separate':'q_only_G1 is predeclared and must be completed for this six-config contract. Its dual-overlap result supplements, not redefines, primary existence PASS/FAIL. If only five core configurations are technically complete, report provisional core verdict plus overall contract incomplete; no bidirectional statement.',
   'completion_vs_success':'A technically valid scientific FAIL can finish the Gate0.5 authenticity investigation with a negative result. It is not permission to enter Gate1 on a positive-mechanism premise. Engineering INCONCLUSIVE is incomplete work, not scientific failure or success.',
   'preserve_raw_numeric_guard_status':'Every final verdict must also show original_complete_model_guard=FAILED_UNCHANGED and new_reference_role=independent_export_semantics; never collapse these into a single PASS.',
   'no_unjustified_percent_or_tolerance_added':True,
   'root_review_policy_decisions':['Confirm original item9 primary G1 reading, rather than any-of-three winner selection.','Confirm A=0 is a valid-set nonreproduction FAIL and F=0 is lack-of-baseline INCONCLUSIVE; do not choose after outcomes.','Confirm negative net can only support narrow existence PASS and never benefit.','Confirm all six are planned mandatory deliverables while dual-overlap is auxiliary to the core existence verdict.']
 },
 'planned_execution_and_stop_rules':{
   'native_primary_dataset_forwards':'6 configurations x150 images=900 primary scored image forwards, no execution authorized here. Bind exact batching/order/warmup and process budgets in final execution protocol.',
   'ordering':'All six configurations and full list order fixed before outputs. Each image uses common input. Record run order and environmental state; no best-run choice.',
   'repeat_policy':'No best-of-N or output-driven rerun. Any prespecified repeat check has its own explicit budget and all outputs preserved; cannot substitute a more favorable copy.',
   'stop_on_technical_issue':['missing/corrupt image/label/input/engine identity','nonfinite or wrong output schema','CUDA/driver/kernel/resource monitor failure','unexpected source precision/mapping or library identity','NMS timeout/truncated record inventory'],
   'scientific_unfavorable_is_not_stop':'Zero detections, no added misses, no rescue or collateral loss do not trigger tuning or alternative image/module selection.',
   'interruption':'Preserve partial raw/receipt and exact progress; final verdict remains INCONCLUSIVE until a separately authenticated continuation completes the original inventory without changing engines/criteria. Do not silently rerun failed build or overwrite outputs.',
   'environment':'Same Orin/software/power-mode request, fan active, boot/process IDs, memory/temperature logging, private process monitor/guard and cleanup. Preserve board clock discrepancy and nonprivileged EMC query warning.',
   'no_performance_claim':'Timing during guarded diagnostics is operational only; Gate2 latency/energy measurement is separate.'
 },
 'required_reviewable_outputs':[
   'Frozen acceptance contract and execution protocol hash before new150-image outputs',
   'Six-configuration artifact/engine/source/actual-format/floating-comparability eligibility matrix with all88 source identities',
   'Independent complete full-ORT reference provenance and raw outputs for the exact common inputs',
   'Authentic runtime SASS/tactic arithmetic branch evidence and exact dependency identities',
   'Native raw outputs/detections/transforms per image/config; all reference and source FP32 comparison results with limitations',
   'One master target CSV, 865rows with six native hit columns plus reference fields; optionally long form5190rows, both derive the same fixed IDs',
   'Per-config/per-group added-miss/rescue/collateral/net/retention/weak-dual sets with denominators, including empty sets',
   'A deterministic full case index; if visual cases displayed, include every recovered/collateral target or fixed target-ID ordering rather than favorable selection',
   'Exact saved receipt/archive hashes, process completion/cleanup and independent saved-data audit',
   'Final report with separate engineering eligibility, scientific verdict, old guard status, numeric discrepancies and narrow claim limits'
 ],
 'unbound_prerequisites_before_freeze':[
   'Bind all six actually eligible engines/build/control/reference graphs; only the new q_all build is currently reported successful, not yet independently audited in this draft.',
   'Define and audit exact actual floating-path comparison matrix; resolve unrelated differences before claiming matched recovery.',
   'Freeze concrete full-reference/evaluator/NMS versions and input/detection/target data schemas and run budgets.',
   'Complete same-process maps + exact correlated runtime kernels + authenticated SM87 SASS supplement or named tactic evidence for each quantized source.',
   'Root approve the explicit ordered PASS/FAIL/INCONCLUSIVE draft rules, then produce a separate immutable frozen contract before new150-image evaluation.'
 ],
 'sources':sources,
 'draft_construction':{'script_sha256':sha(Path(__file__)),'only_stdlib_reads_and_new_draft_write':True}
}
out = HERE/'dataset_acceptance_contract_draft.json'
with out.open('x',encoding='utf-8') as f:
    json.dump(draft,f,ensure_ascii=False,indent=2)
    f.write('\n')
print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'sha256':sha(out),'status':draft['status'],'configurations':len(configs)}))
