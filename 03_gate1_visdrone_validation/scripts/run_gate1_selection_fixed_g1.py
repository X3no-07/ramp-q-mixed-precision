"""One authorized report60 fixed-G1 Selection development attempt.

Preparation is metadata/source-only. Real Selection content remains behind all
109 strict debug reference checks, reviewed immutable sources and human receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gate1_w8a8_debug_pilot as P
import run_gate1_w8a8_debug_full as B

OUT = ROOT / "artifacts/gate1/20261024_selection_fixed_G1_execution_v1"
SEEDS = ["2026102201", "2026102202", "2026102203"]
CONFIGS = ["fp32", "q_all", "q_only:G1", "r_only:G1"]
SCHEMA = "rampq.gate1.Selection_fixed_G1_execution/v1"
AUTH_SHA = "5cd96e604cd7160dd3bb97709b1c0993215cf985cfb6ca3d65e36df524c9e72b"
RESERVE_BYTES = 20_000_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def budget():
    return {"debug_preflight_images": 109, "Selection_images": 548, "calibration_seeds": SEEDS,
        "calibration_images_per_seed": 64, "configurations": 4, "candidate_weights": 1,
        "forward_calls": 10713, "Selection_and_calibration_forwards": 10604,
        "forward_breakdown": {"debug_preflight": 109, "shared_Selection_FP32": 548,
            "per_seed": {"calibration": 64, "disabled_before": 548, "quantized": 1644,
                         "disabled_after": 548, "original_after": 548}, "per_seed_total": 3352},
        "unique_Selection_raw_files": 5480, "debug_preflight_raw_files": 109,
        "logical_Selection_arrays": 6576, "full_difference_arrays": 6576,
        "saved_submission_Python_AP_calls": 10, "timeout_seconds": 7200, "attempts": 1,
        "automatic_retry": False, "synthetic_construction_stride_forwards_not_real_image_calls": True,
        "disk_reserve_bytes": RESERVE_BYTES, "Holdout_access": False}


def expected_authorized_scope():
    return {"Selection": "ORIGINAL_548_IMAGES_AND_ORIGINAL_LABELS_FIXED_G1_DEVELOPMENT_ONLY",
        "Holdout": "SEALED_NO_ENUMERATION_IMAGES_LABELS_OR_OUTPUTS", "candidate_count": 1,
        "configurations": CONFIGS, "calibration_seeds": SEEDS, "calibration_images_per_seed": 64,
        "calibration_labels_access": False, "debug_strict_reference_preflight_before_Selection_content": 109,
        "original_Selection_order_and_548_member_units_retained": True,
        "duplicate_image_annotation_conflict_policy": "STOP_PRESERVE_REPORT_NO_AUTOMATIC_DEDUPLICATION_OR_REWEIGHTING",
        "real_image_forward_budget": 10713, "saved_submission_Python_AP_budget": 10, "attempts": 1,
        "timeout_seconds": 7200, "automatic_retry": False, "new_module_or_method_or_seed_search": False,
        "independent_scientific_acceptance": False, "native_INT8_or_device_acceptance": False}


def validate_authorization(receipt):
    P.require(receipt["schema"] == "rampq.gate1.Selection_fixed_G1_human_access_decision/v1" and
        receipt["decision"] == "AUTHORIZED_REPORT60_FIXED_G1_SELECTION_DEVELOPMENT_SCOPE" and
        receipt["human_message_verbatim"] == 'Continue' and receipt["authorized_scope"] == expected_authorized_scope(),
        "Human receipt does not authorize this exact scope")
    P.require(receipt["execution_requires_frozen_reviewed_sources_contract_and_tests"] is True,
        "Human decision cannot bypass frozen code/tests")


def read_authorization():
    path = OUT / "human_authorization.json"
    P.require(P.sha(path) == AUTH_SHA, "Only the root-recorded exact human authorization may be used")
    receipt = P.read(path); validate_authorization(receipt)
    for key in ("report60", "proposal", "previous_snapshot", "recorder"):
        P.require(P.identity(receipt[key]["path"]) == receipt[key], "Authorization context binding changed")
    return receipt


def AP_policy():
    return {"calls": 10, "order": ["fp32"] + [f"{seed}/{config.replace(':','_')}" for seed in SEEDS for config in CONFIGS[1:]],
        "source": "source-faithful Python mirror; original official code and prefilter transitively bound",
        "shared_FP32_scored_once": True, "input": "same saved ordered raw predictions and original publisher GT rows8",
        "publisher_score0_and_category0_preserved_before_official_filter": True,
        "first500_across_classes_before_class_filter": True, "metrics": ["AP_all","AP_50","AP_75","AR_1","AR_10","AR_100","AR_500"],
        "native_validation_scope": "Prior fixed414 FP32 only; no claim of native run for these ten Selection submissions",
        "Selection_AP_is_development_only": True, "scientific_PASS_threshold": None}


def postprocessing_policy():
    value=deepcopy(P.postprocessing_policy())
    value["submission"]="descending confidence stable NMS-order ties; original fixed548 Selection member order"
    value["official_AP"]="SOURCE_FAITHFUL_PYTHON_SELECTION_DEVELOPMENT_10_CALLS"
    return value


def output_schema():
    value = dict(B.output_schema())
    value.update(debug_preflight_raw="debug_preflight/raw/<stem>.npz:raw",
        debug_preflight_checks="debug_preflight/checks.json", debug_preflight_evidence="debug_preflight/evidence.json",
        content_preflight="content_preflight_receipt.json", ground_truth="ground_truth/<stem>.json",
        ground_truth_index="ground_truth/index.json", shared_reference_checks="shared_FP32/reference_receipt.json",
        AP_inputs="scores/<branch>/raw_inputs.json", AP_prepared_matrices="scores/<branch>/prepared_matrices.npz",
        AP_index="scores/<branch>/input_index.json", AP_metrics="scores/<branch>/metrics.json",
        AP_receipt="scores/AP_receipt.json", progress_stages=["DEBUG_STRICT_PREFLIGHT", "SELECTION_SHARED_FP32",
        "IMAGE_ONLY_CALIBRATION", "DISABLED_BEFORE_STRICT", "q_all", "q_only:G1", "r_only:G1",
        "DISABLED_AND_ORIGINAL_AFTER_STRICT", "SAVED_SUBMISSION_PYTHON_AP"])
    return value


def source_paths():
    paths = dict(B.source_paths())
    paths.update(runner=Path(__file__), tests=ROOT / "tests/test_gate1_selection_fixed_g1.py",
        debug_full_runner=ROOT / "scripts/run_gate1_w8a8_debug_full.py",
        members=ROOT / "scripts/gate1_selection_members.py", members_tests=ROOT / "tests/test_gate1_selection_members.py",
        member_metadata=OUT / "member_metadata.json",member_preparation_receipt=OUT / "member_preparation_receipt.json",
        member_tests_log=OUT / "member_tests_console.txt",member_tests_xml=OUT / "member_tests.xml",
        human_authorization=OUT / "human_authorization.json", AP_mirror=ROOT / "scripts/gate1_official_ap.py",
        AP_prefilter=ROOT / "scripts/gate1_ignore_aware_evaluation.py",
        AP_native_contract=ROOT / "artifacts/gate1/20261024_native_checkpointed_retry_v2/base_contract.json",
        AP_native_comparison=ROOT / "artifacts/gate1/20261024_native_checkpointed_retry_v2/comparison_result.json",
        AP_native_independent=ROOT / "artifacts/gate1/20261024_native_final_saved_audit_v1/result/audit_result.json",
        debug_full_contract=B.OUT / "contract.json",
        debug_full_saved_audit=ROOT / "artifacts/gate1/20261024_w8a8_debug_3seed_audit_v1/independent_saved_audit_result.json",
        report60=ROOT / "docs/60_gate1_selection_fixed_g1_development_reproduction_plan.md")
    return paths


def validate_member_aliases(p):
    for name in ("debug","selection","calibration","prepared_root","member_order_hashes","duplicate_policy"):
        P.require(p[name] == p["members_metadata"][name],"Top-level member alias diverges from frozen metadata: "+name)


def validate_contract(p):
    validate_member_aliases(p)
    P.require(p["schema"] == SCHEMA and p["seeds"] == SEEDS and p["configurations"] == CONFIGS,
        "Selection fixed intervention scope changed")
    P.require(p["budget"] == budget() and p["output_schema"] == output_schema() and p["AP"] == AP_policy(),
        "Budget/output/AP protocol changed")
    P.require(p["candidate"]["sha256"] == "4dd6627f2dd0f58982268c5fda1c7731f8d3f3b86c7b3e8ced5d8d7cd1b3ca27" and
        p["fusion"] == "BN_unfused_all_other_ops_FP32" and p["quantization"] == P.quantization_policy() and
        p["postprocessing"] == postprocessing_policy() and p["raw_tolerances"] == P.TOLERANCES,
        "Candidate/precision/postprocessing changed")
    P.require(p["input"] == {"size":640,"batch":1,"AMP":False,"device":"cuda:0","deterministic":True,"TF32":False},
        "Input backend changed")
    P.require(len(p["debug"]) == 109 and len(p["selection"]) == 548 and len({row["image_id"] for row in p["selection"]}) == 548 and
        list(p["calibration"]) == SEEDS and all(len(p["calibration"][seed]) == 64 for seed in SEEDS), "Cohort/seed count changed")
    P.require(p["Holdout"] == "SEALED" and p["scientific_acceptance"] is False and p["new_method_or_group_search"] is False,
        "Scientific/access scope escalated")
    P.require(p["authorization"]["sha256"] == AUTH_SHA, "Wrong human receipt identity")


def verify_sources(p):
    validate_member_aliases(p)
    read_authorization()
    P.require(p["source_bindings"] == {name:P.identity(path) for name,path in source_paths().items()}, "Frozen source inventory changed")
    for binding in [p[key] for key in ("candidate","architecture","state_manifest","state_receipt")]:
        P.require(P.identity(binding["path"]) == binding, "Candidate state identity changed")
    for binding in p["official_sources"]:
        observed = P.identity(ROOT / binding["path"])
        P.require(all(observed[key] == binding[key] for key in ("bytes","sha256")), "Official source changed")
    import gate1_selection_members as S
    P.require(S.prepare_metadata() == p["members_metadata"], "Metadata-only fixed membership changed")
    P.require(P.read(OUT / "member_metadata.json") == p["members_metadata"],"Stored member binding changed")
    member_tests=P.read(OUT / "member_preparation_receipt.json")
    P.require(member_tests["status"]=="PASS_PURE_SELECTION_MEMBER_BINDING_AND_CONTENT_GATE_TESTS_ONLY" and
        member_tests["tests"]==47 and member_tests["failures"]==member_tests["errors"]==0 and
        member_tests["helper"]==P.identity(source_paths()["members"]) and
        member_tests["tests_source"]==P.identity(source_paths()["members_tests"]) and
        member_tests["member_binding"]==P.identity(OUT / "member_metadata.json"),"Exact member helper tests/bindings not passed")
    native = P.read(p["source_bindings"]["AP_native_contract"]["path"])
    for binding in native["bindings"]:
        path = Path(binding["path"])
        if not path.is_absolute(): path = ROOT / path
        observed = P.identity(path)
        P.require(all(observed[key] == binding[key] for key in ("bytes","sha256")), "Native-verified transitive AP source changed")


def prepare():
    read_authorization()
    import gate1_selection_members as S
    import gate1_w8a8_entry as E
    members = S.prepare_metadata()
    formal = P.read(P.FORMAL / "protocol.json")
    native = P.read(source_paths()["AP_native_comparison"])
    P.require(native["status"] == "PASS_FIXED_REAL_414_TRAINING_VAL_FP32_NATIVE_PREFILTER_AND_AP_COMPARISON_ONLY", "Native AP prerequisite failed")
    old = P.read(P.FORMAL / "native_comparison_contract.json")
    p = {"schema":SCHEMA,"status":"DRAFT_AUTHORIZED_SCOPE_NOT_FROZEN_NOT_EXECUTED","prepared_at_utc":now(),
        "authorization":P.identity(OUT / "human_authorization.json"),"source_bindings":{name:P.identity(path) for name,path in source_paths().items()},
        "official_sources":old["official_sources"],"members_metadata":members,"prepared_root":members["prepared_root"],
        "debug":members["debug"],"selection":members["selection"],"calibration":members["calibration"],
        "member_order_hashes":members["member_order_hashes"],"duplicate_policy":members["duplicate_policy"],
        "candidate":P.identity(E.CAPTURE / "state/reference_state.pt"),"architecture":P.identity(E.CAPTURE / "architecture.json"),
        "state_manifest":P.identity(E.CAPTURE / "state/manifest.json"),"state_receipt":P.identity(E.CAPTURE / "state/snapshot_receipt.json"),
        "runtime_sources":E.capture_helpers().sources(),"seeds":SEEDS,"configurations":CONFIGS,"budget":budget(),
        "output_schema":output_schema(),"AP":AP_policy(),"fusion":"BN_unfused_all_other_ops_FP32","quantization":P.quantization_policy(),
        "postprocessing":postprocessing_policy(),"raw_tolerances":P.TOLERANCES,"preprocessing":formal["preprocessing"],
        "input":{"size":640,"batch":1,"AMP":False,"device":"cuda:0","deterministic":True,"TF32":False},
        "Holdout":"SEALED","calibration_labels_access":False,"scientific_acceptance":False,"new_method_or_group_search":False,
        "old_failures_preserved":["FAIL_FIXED_TOLERANCES","FAILED_UNCHANGED"]}
    validate_contract(p); verify_sources(p)
    P.write(OUT / "contract_draft.json",p)
    return {"status":p["status"],"draft":P.identity(OUT / "contract_draft.json"),"Selection_content_read":False,"real_forward_calls":0,"AP_calls":0}


def freeze(reviewed_sha):
    P.require(P.sha(OUT / "contract_draft.json") == reviewed_sha,"Explicit reviewed draft SHA required")
    p=P.read(OUT / "contract_draft.json");validate_contract(p);verify_sources(p)
    tests=P.read(OUT / "tests_receipt.json")
    P.require(tests["failures"] == tests["errors"] == 0 and tests["tests"] > 0 and
        tests["runner"] == P.identity(__file__) and tests["test_source"] == P.identity(source_paths()["tests"]),"Exact runner tests not passed")
    P.require(p["status"] == "DRAFT_AUTHORIZED_SCOPE_NOT_FROZEN_NOT_EXECUTED","Only reviewed new draft freezes")
    p.update(status="FROZEN_AUTHORIZED_REVIEWED_NOT_EXECUTED",frozen_at_utc=now(),reviewed_draft=P.identity(OUT / "contract_draft.json"),tests=P.identity(OUT / "tests_receipt.json"))
    P.write(OUT / "contract.json",p)
    return {"status":p["status"],"contract":P.identity(OUT / "contract.json")}


def annotation_payload(text,image_id):
    """Keep publisher rows and blank physical lines; never invert score for target/AP interchange."""
    import gate1_ignore_aware_evaluation as M
    parsed=M.parse_ground_truth(text,image_id)
    physical=text.lstrip("\ufeff").splitlines()
    rows=[]
    for position,line in enumerate(physical):
        if not line.strip():continue
        fields=line.split(",")
        if len(fields)==9 and not fields[-1].strip():fields.pop()
        P.require(len(fields)==8,"Annotation must retain original eight fields")
        values=[float(value.strip()) for value in fields]
        P.require(all(math.isfinite(value) for value in values),"Nonfinite original GT")
        P.require(all(math.isfinite(value) for value in (values[0]+values[2],values[1]+values[3],values[2]*values[3])),"Original GT geometry overflow")
        if values[4]==1 and 1<=values[5]<=10:
            P.require(values[2]>0 and values[3]>0,"Invalid evaluated original GT geometry")
        rows.append({"physical_line_zero_based":position,"row8":values})
    P.require([row["physical_line_zero_based"] for row in rows] == [row["physical_line_zero_based"] for row in parsed],"Physical GT identity lost")
    return {"original_annotation_text":text,"physical_lines":physical,"original_rows_with_physical_line":rows,
            "raw_ground_truth_rows8":[row["row8"] for row in rows],"parsed_project_ground_truth":parsed}


def prediction_rows8(predictions):
    expected=sorted(predictions,key=lambda item:(-item["score"],item["nms_input_order"]))
    P.require(predictions==expected,"Submission confidence/NMS tie order changed")
    return [item["bbox_xywh"]+[item["score"],item["category"],0.,0.] for item in predictions]


def debug_evidence(checks,debug_rows,forward_calls):
    P.require(len(checks)==len(debug_rows)==109 and forward_calls==109 and
        [item["image_id"] for item in checks]==[row["image_id"] for row in debug_rows] and
        all(item["strict_equal"] is True for item in checks),"All 109 exact debug checks must precede Selection content")
    return {"status":"PASS_ALL_109_DEBUG_STRICT_FP32_BEFORE_SELECTION_CONTENT","images":109,"all_strict_equal":True,
        "forward_calls":109,"Selection_content_reads_before_gate":0,"checks":checks}


def AP_jobs():
    return [("fp32",None,"fp32")]+[(f"{seed}/{config.replace(':','_')}",seed,config) for seed in SEEDS for config in CONFIGS[1:]]


def check_disk():
    P.require(shutil.disk_usage(OUT).free >= RESERVE_BYTES,"Disk below frozen 20GB reserve; preserve and stop")


def actual_worker_identity():
    own_pid=os.getpid()
    command=("[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); "
        f"Get-CimInstance Win32_Process -Filter 'ProcessId = {own_pid}' | "
        "Select-Object ProcessId,ParentProcessId,@{N='CreationTimeUTC';E={$_.CreationDate.ToUniversalTime().ToString('o')}},CommandLine | ConvertTo-Json -Compress")
    result=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",command],capture_output=True,
        timeout=20,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    P.require(result.returncode==0 and result.stdout.strip(),"Cannot bind actual owned worker PID/creation/command")
    actual=json.loads(result.stdout.decode("utf-8-sig"))
    P.require(actual["ProcessId"]==own_pid and str(Path(__file__).resolve()).casefold() in actual["CommandLine"].casefold(),"Worker process identity differs")
    return {"status":"BOUND_OWNED_SELECTION_WORKER_IDENTITY","observed_at_utc":now(),"actual_process":actual,
        "timeout_owner":"subprocess.run(timeout=7200) kills this owned worker handle and waits on TimeoutExpired; no automatic retry"}


def color_chw(image):
    P.require(isinstance(image,np.ndarray) and image.dtype==np.uint8 and image.ndim==3 and image.shape[2]==3,"Expected uint8 BGR image")
    return np.ascontiguousarray(image[...,::-1].transpose(2,0,1))


def image_input(p,row,torch,cv2,letterbox):
    """Same frozen decode/LetterBox/normalization, with new explicit Selection role support."""
    import gate1_selection_members as S
    role=row["assigned_split"]
    P.require(role in {"debug","selection","calibration_pool"},"Unsupported input role")
    P.require(row["source_split"] == ("val" if role=="selection" else "train"),"Unexpected role source")
    path=S.checked_member_path(p["members_metadata"],row,"image")
    image=cv2.imdecode(np.frombuffer(path.read_bytes(),dtype=np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    P.require(image is not None and image.shape==(row["height"],row["width"],3),"Unexpected image dimensions")
    params=letterbox.get_params({"img":image})
    padded=letterbox.apply_image({"img":image},params)["img"]
    tensor=torch.from_numpy(color_chw(padded)).unsqueeze(0).float().div(255).to("cuda:0")
    return tensor,params


def saved_AP(attempt,p,ground_truth,ap_progress):
    """Exactly ten official-mirror calls, no inference and no original label reread."""
    import gate1_official_ap as A
    scores=attempt / "scores";scores.mkdir()
    completed=[]
    for label,seed,config in AP_jobs():
        check_disk()
        output=scores / label;output.mkdir(parents=True)
        raw_inputs=[];prepared_gt=[];prepared_dt=[];prepared_arrays={};input_index=[]
        folder=attempt / "shared_FP32" if seed is None else attempt / seed
        for position,row in enumerate(p["selection"]):
            stem=row["image_id"].split("/")[1]
            filename=f"fp32__{stem}.json" if seed is None else f"{config.replace(':','_')}__{stem}.json"
            path=folder / "predictions" / filename
            record=P.read(path)
            P.require(record["image_id"]==row["image_id"] and record["configuration"]==config,"Wrong saved AP submission identity")
            rows=prediction_rows8(record["predictions"])
            raw_gt=ground_truth[row["image_id"]]["raw_ground_truth_rows8"]
            fixture={"image_id":row["image_id"],"width":row["width"],"height":row["height"],"raw_ground_truth":raw_gt,"raw_predictions":rows}
            raw_inputs.append(fixture)
            filtered=A.prepare_raw_image(np.asarray(raw_gt,dtype=np.float64).reshape(-1,8),np.asarray(rows,dtype=np.float64).reshape(-1,8),width=row["width"],height=row["height"])
            prepared_gt.append(filtered["prepared_gt"]);prepared_dt.append(filtered["filtered_predictions"])
            for key in ("filtered_gt","filtered_predictions","prepared_gt"):
                prepared_arrays[f"image_{position:04d}_{key}"]=filtered[key]
            input_index.append({"image_id":row["image_id"],"annotation_sha256":row["annotation_sha256"],"saved_GT":P.identity(attempt / "ground_truth" / f"{stem}.json"),
                "saved_predictions":P.identity(path),"GT_rows_after_filter":len(filtered["prepared_gt"]),"DT_rows_after_filter":len(filtered["filtered_predictions"])})
        P.write(output / "raw_inputs.json",{"role":"Selection","branch":label,"images":raw_inputs,"member_order_retained":True})
        np.savez_compressed(output / "prepared_matrices.npz",**prepared_arrays)
        # One and only one AP aggregation for this branch. No bootstrap/resampling/scientific threshold.
        result=A.calc_accuracy(prepared_gt,prepared_dt,capture_intermediates=False)
        P.require(all(math.isfinite(float(result["metrics"][name])) for name in A.METRIC_NAMES),"Nonfinite Selection AP result")
        np.savez_compressed(output / "AP_AR_arrays.npz",AP=result["AP"],AR=result["AR"],evalClass=result["evalClass"])
        P.write(output / "input_index.json",input_index)
        metrics={"status":"COMPLETED_OFFICIAL_MIRROR_SELECTION_DEVELOPMENT_AP_ONLY","role":"Selection","branch":label,"images":548,
            "metrics_percent":result["metrics"],"official_commit":A.OFFICIAL_COMMIT,"original_548_member_units_retained":True,
            "AP_aggregation_calls_this_branch":1,"fixed_job_index":len(completed),"total_frozen_AP_budget":10,
            "AP_mirror":p["source_bindings"]["AP_mirror"],"AP_prefilter":p["source_bindings"]["AP_prefilter"],
            "prior_native_comparison":p["source_bindings"]["AP_native_comparison"],
            "native_current_submission_run":False,"prior_native_scope":"fixed414 training_val FP32 only",
            "scientific_acceptance":False,"Holdout_accessed":False}
        P.write(output / "metrics.json",metrics)
        completed.append({"branch":label,"metrics":P.identity(output / "metrics.json"),"raw_inputs":P.identity(output / "raw_inputs.json"),
            "prepared_matrices":P.identity(output / "prepared_matrices.npz"),"input_index":P.identity(output / "input_index.json")})
        ap_progress(len(completed))
    P.require(len(completed)==10 and [row["branch"] for row in completed]==p["AP"]["order"],"Saved submission AP budget/order differs")
    receipt={"status":"COMPLETED_EXACT_TEN_SAVED_SUBMISSION_PYTHON_AP_CALLS","calls":10,"branches":completed,"contract":P.identity(OUT / "contract.json"),
        "AP_mirror":p["source_bindings"]["AP_mirror"],"AP_prefilter":p["source_bindings"]["AP_prefilter"],
        "official_AP_separate_from_target_matcher":True,"new_native_AP_calls":0,"scientific_acceptance":False}
    P.write(scores / "AP_receipt.json",receipt)
    return receipt


def worker():
    attempt=OUT / "attempt_01"
    P.require((attempt / "launcher_receipt.json").is_file(),"Only the reviewed bounded launcher may start worker")
    P.write(attempt / "worker_identity.json",actual_worker_identity())
    p=P.read(OUT / "contract.json");validate_contract(p);verify_sources(p)
    P.require(p["status"]=="FROZEN_AUTHORIZED_REVIEWED_NOT_EXECUTED","Reviewed frozen contract required")
    check_disk()
    import gate1_selection_members as S
    import gate1_w8a8_entry as E
    import evaluate_gate1_formal_candidate as F
    import qualify_gate1_float_baseline as Q
    import gate1_ignore_aware_evaluation as M
    torch,cv2=F.configure()
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import non_max_suppression
    capture=E.capture_helpers()
    P.require(capture.sources()==p["runtime_sources"],"Runtime changed")
    saved,metadata=capture.FORMAT.load_snapshot(E.CAPTURE / "state")
    architecture=P.read(p["architecture"]["path"])
    P.require(hashlib.sha256(capture.canonical(architecture).encode()).hexdigest()==metadata["architecture_sha256"],"Architecture drift")
    with torch.random.fork_rng(devices=[]):
        model=DetectionModel(deepcopy(architecture["cfg"]),ch=3,nc=10,verbose=False)
    model.load_state_dict(saved,strict=True);model.names=dict(enumerate(architecture["names"]))
    model.to("cuda:0").float().eval().requires_grad_(False)
    original_state=E.state_info(model);modules=E.legacy().list_conv_modules(model)
    inventory=P.read(P.ENTRY / "module_inventory.json")
    P.require([(name,group,list(conv.weight.shape)) for name,conv,group in modules]==
        [(row["name"],row["group"],row["weight_shape"]) for row in inventory["modules"]],"Actual Conv/group map drift")
    P.require(sum(isinstance(module,torch.nn.BatchNorm2d) for module in model.modules())==81,"BN fusion drift")
    letterbox=LetterBox((640,640),auto=False,scale_fill=False,scaleup=True,center=True,stride=32,padding_value=114,interpolation=cv2.INTER_LINEAR)
    forward_calls=0

    def progress(stage,index,total,seed=None):
        value={"stage":stage,"seed":seed,"images":index,"total":total,"forward_calls":forward_calls}
        P.write(attempt / "live_status.json",value,once=False);print(json.dumps(value),flush=True)

    def forward(current,tensor):
        nonlocal forward_calls
        forward_calls+=1
        P.require(forward_calls<=10713,"Real image forward budget exceeded")
        with torch.inference_mode(),torch.autocast("cuda",enabled=False):
            raw=current(tensor.to("cuda:0"))[0].detach().cpu().numpy().copy()
        P.check_raw(raw)
        return raw

    # No Selection image or annotation path is opened before every debug check passed.
    debug=attempt / "debug_preflight";debug.mkdir();(debug / "raw").mkdir();checks=[]
    for index,row in enumerate(p["debug"],1):
        tensor,_=image_input(p,row,torch,cv2,letterbox)
        raw=forward(model,tensor);old=row["saved_raw_reference"]
        P.require(P.identity(old["path"])==old,"Old sealed debug FP32 changed")
        with np.load(old["path"],allow_pickle=False) as arrays:old_raw=arrays["fp32"].copy()
        stem=row["image_id"].split("/")[1]
        np.savez_compressed(debug / "raw" / f"{stem}.npz",raw=raw)
        checks.append({"image_id":row["image_id"],"strict_equal":bool(np.array_equal(old_raw,raw)),"original_all_anchor_guard":Q.raw_guard(old_raw,raw)})
        P.write(debug / "checks.json",checks,once=False)
        P.strict_reference_check(old_raw,raw)
        progress("DEBUG_STRICT_PREFLIGHT",index,109)
        del tensor,raw,old_raw
    P.require(E.state_info(model)==original_state,"Debug state pollution")
    evidence=debug_evidence(checks,p["debug"],forward_calls)
    P.write(debug / "evidence.json",evidence)
    try:
        content=S.validate_content_after_debug_preflight(p["members_metadata"],evidence)
    except Exception as error:
        failure=getattr(error,"evidence",None)
        if failure is not None:
            P.write(attempt / "content_preflight_failure_evidence.json",failure)
        raise
    P.write(attempt / "content_preflight_receipt.json",content)
    check_disk()
    gt_dir=attempt / "ground_truth";gt_dir.mkdir();ground_truth={};gt_index=[]
    for row in p["selection"]:
        path=S.checked_member_path(p["members_metadata"],row,"annotation")
        payload=annotation_payload(path.read_text(encoding="utf-8-sig"),row["image_id"])
        payload.update(image_id=row["image_id"],width=row["width"],height=row["height"],annotation_source=P.identity(path))
        stem=row["image_id"].split("/")[1];P.write(gt_dir / f"{stem}.json",payload)
        ground_truth[row["image_id"]]=payload;gt_index.append({"image_id":row["image_id"],"saved_GT":P.identity(gt_dir / f"{stem}.json")})
    P.write(gt_dir / "index.json",gt_index)
    shared=attempt / "shared_FP32";shared.mkdir()
    for name in ("raw","predictions","targets"):(shared / name).mkdir()
    inputs,references={},{ }
    for index,row in enumerate(p["selection"],1):
        tensor,params=image_input(p,row,torch,cv2,letterbox)
        raw=forward(model,tensor);stem=row["image_id"].split("/")[1]
        np.savez_compressed(shared / "raw" / f"{stem}.npz",raw=raw)
        # Retain exactly one Selection input per member on CUDA, without a second CPU cache.
        # All debug inputs were already released; this bounded cache is about 2.7GB.
        inputs[row["image_id"]]=(tensor,params,E.tensor_info(tensor));references[row["image_id"]]=raw
        progress("SELECTION_SHARED_FP32",index,548)
        del tensor
    P.require(E.state_info(model)==original_state,"Fresh Selection reference polluted state")
    P.write(shared / "reference_receipt.json",{"images":548,"fresh_shared_FP32_once":True,"historical_Selection_reference_exists":False,
        "raw_bindings":[{"image_id":row["image_id"],"raw":P.identity(shared / "raw" / f"{row['image_id'].split('/')[1]}.npz")} for row in p["selection"]],
        "cached_inputs_device":"CUDA_SINGLE_CACHE_NO_CPU_DUPLICATE","no_debug_GPU_input_cache_retained":True})
    seed_receipts=[]
    for seed in SEEDS:
        check_disk();folder=attempt / seed;folder.mkdir()
        for name in ("raw","differences","predictions","targets"):(folder / name).mkdir()
        collector=P.ActivationRanges(modules)
        try:
            for index,row in enumerate(p["calibration"][seed],1):
                tensor,_=image_input(p,row,torch,cv2,letterbox)
                forward(model,tensor);del tensor
                progress("IMAGE_ONLY_CALIBRATION",index,64,seed)
        finally:collector.close()
        maxima=dict(collector.max_abs)
        P.require(set(maxima)=={name for name,_,_ in modules} and set(collector.observations)==set(maxima) and
            all(value==64 for value in collector.observations.values()),"Incomplete per-seed calibration coverage")
        P.require(E.state_info(model)==original_state,"Calibration state pollution")
        P.write(folder / "calibration_ranges.json",{"seed":seed,"max_abs":maxima,"activation_scale":{key:max(value/127.,1e-12) for key,value in maxima.items()},
            "observations_per_Conv":dict(collector.observations),"labels_read":False,"member_order_hash":p["member_order_hashes"]["calibration"][seed]})
        candidate=deepcopy(model)
        P.require(all(a.data_ptr()!=b.data_ptr() for a,b in zip(model.parameters(),candidate.parameters())),"Parameter storage aliases")
        wrappers=E.install(candidate,maxima);candidate.eval();wrapped_state=E.state_info(candidate)
        targets={key:{} for key in CONFIGS};counts={key:Counter() for key in CONFIGS};diffs=[];shared_refs=[];replay=[]

        def save_outputs(configuration,row,raw,params):
            stem=row["image_id"].split("/")[1];name=f"{configuration.replace(':','_')}__{stem}"
            if configuration=="fp32":
                shared_refs.append({"image_id":row["image_id"],"raw":P.identity(shared / "raw" / f"{stem}.npz"),"strict_disabled_recompute_passed":True})
            else:np.savez_compressed(folder / "raw" / f"{name}.npz",raw=raw)
            difference,summary=B.raw_difference(references[row["image_id"]],raw)
            np.savez_compressed(folder / "differences" / f"{name}.npz",difference_FP64=difference)
            diffs.append({"image_id":row["image_id"],"configuration":configuration,"full_difference":f"differences/{name}.npz","summary":summary,
                "original_all_anchor_guard_descriptive":Q.raw_guard(references[row["image_id"]],raw)})
            det,anchors=Q.guarded_nms(non_max_suppression,torch.from_numpy(raw))
            restored,checks_restore=Q.restored_predictions(det[0].numpy(),anchors[0].numpy(),params,row["image_id"],configuration)
            restored=sorted(restored,key=lambda item:(-item["score"],item["nms_input_order"]))
            matched=M.match_project_targets(ground_truth[row["image_id"]]["parsed_project_ground_truth"],restored,row["width"],row["height"],confidence=.25,iou_threshold=.5)
            saved_predictions={"image_id":row["image_id"],"configuration":configuration,"letterbox_params":params,
                "anchor_indices_in_NMS_order":anchors[0].numpy().reshape(-1).tolist(),"predictions":restored,
                "restoration_checks":checks_restore,"NMS_max_det_saturated":len(restored)==1000}
            P.write(folder / "predictions" / f"{name}.json",saved_predictions);P.write(folder / "targets" / f"{name}.json",matched)
            if configuration=="fp32" and seed==SEEDS[0]:
                P.write(shared / "predictions" / f"{name}.json",saved_predictions);P.write(shared / "targets" / f"{name}.json",matched)
            targets[configuration].update({item["target_id"]:bool(item["hit"]) for item in matched["targets"]});counts[configuration].update(matched["counts"])

        E.legacy().enable_configuration(wrappers,"fp32")
        for index,row in enumerate(p["selection"],1):
            tensor,params,_=inputs[row["image_id"]];raw=forward(candidate,tensor);P.strict_reference_check(references[row["image_id"]],raw)
            replay.append({"image_id":row["image_id"],"stage":"disabled_before","reference_sha256":hashlib.sha256(references[row["image_id"]].tobytes()).hexdigest(),"replay_sha256":hashlib.sha256(raw.tobytes()).hexdigest(),"strict_equal":True})
            save_outputs("fp32",row,references[row["image_id"]],params);progress("DISABLED_BEFORE_STRICT",index,548,seed)
        P.write(folder / "FP32_shared_references.json",shared_refs)
        for configuration in CONFIGS[1:]:
            E.legacy().enable_configuration(wrappers,configuration)
            P.require({wrapper.group for wrapper in wrappers if wrapper.enabled}==E.legacy().parse_configuration(configuration),"QDQ mask changed")
            for index,row in enumerate(p["selection"],1):
                tensor,params,_=inputs[row["image_id"]]
                save_outputs(configuration,row,forward(candidate,tensor),params);progress(configuration,index,548,seed)
                if index%32==0:check_disk()
            P.require(E.state_info(model)==original_state and E.state_info(candidate)==wrapped_state,"QDQ state pollution")
        E.legacy().enable_configuration(wrappers,"fp32")
        for index,row in enumerate(p["selection"],1):
            tensor,_,old_input=inputs[row["image_id"]]
            disabled,original=forward(candidate,tensor),forward(model,tensor)
            for stage,observed in (("disabled_after",disabled),("original_after",original)):
                P.strict_reference_check(references[row["image_id"]],observed)
                replay.append({"image_id":row["image_id"],"stage":stage,"reference_sha256":hashlib.sha256(references[row["image_id"]].tobytes()).hexdigest(),"replay_sha256":hashlib.sha256(observed.tobytes()).hexdigest(),"strict_equal":True})
            P.require(E.tensor_info(tensor)==old_input,"Cached Selection input mutation")
            progress("DISABLED_AND_ORIGINAL_AFTER_STRICT",index,548,seed)
        P.require(E.state_info(model)==original_state and E.state_info(candidate)==wrapped_state,"Final per-seed state pollution")
        P.write(folder / "strict_replay_evidence.json",{"checks":replay,"all_exact":True,"scope":"live recomputation hashes; these replay tensors are not separately saved"})
        P.write(folder / "state_fingerprint_evidence.json",{"original_before":original_state,"original_after":E.state_info(model),"wrapped_before":wrapped_state,"wrapped_after":E.state_info(candidate),
            "parameter_storage_disjoint":True,"source_tensor_check_scope":"live runner tensor descriptions, not separately saved state tensors"})
        P.write(folder / "raw_difference_checks.json",diffs);P.write(folder / "target_matrix.json",{"configurations":CONFIGS,"states":targets})
        P.write(folder / "target_summary.json",P.target_summary(targets)|{"seed":seed,"role":"Selection","original_548_member_units_retained":True,"branch_counts":{key:dict(value) for key,value in counts.items()}})
        seed_receipt={"seed":seed,"status":"COMPLETED_FIXED_SELECTION_SIMULATION_SEED_ONLY","forwards":3352,"calibration_images":64,"Selection_images":548,
            "raw_arrays_QDQ":1644,"FP32_shared_references":548,"full_difference_arrays":2192,"disabled_before_after_exact":True,"original_replay_exact":True,
            "states_and_inputs_unchanged":True,"scientific_acceptance":False}
        P.write(folder / "completion_receipt.json",seed_receipt);seed_receipts.append(seed_receipt)
        del candidate,wrappers
    P.require(forward_calls==10713 and E.state_info(model)==original_state,"Final inference budget/state failure")
    P.write(attempt / "inference_receipt.json",{"status":"COMPLETED_FIXED_SELECTION_INFERENCE_ONLY","forward_calls":forward_calls,"unique_Selection_raw_files":5480,
        "debug_preflight_raw_files":109,"logical_Selection_arrays":6576,"full_difference_arrays":6576,"seed_receipts":seed_receipts})
    ap=saved_AP(attempt,p,ground_truth,lambda done:progress("SAVED_SUBMISSION_PYTHON_AP",done,10))
    verify_sources(p);P.require(capture.sources()==p["runtime_sources"],"Runtime source changed during attempt")
    receipt={"status":"COMPLETED_AUTHORIZED_FIXED_G1_SELECTION_DEVELOPMENT_ONLY","contract":P.identity(OUT / "contract.json"),"forward_calls":10713,
        "unique_Selection_raw_files":5480,"debug_preflight_raw_files":109,"logical_Selection_arrays":6576,"full_difference_arrays":6576,"AP_calls":10,
        "seed_receipts":seed_receipts,"debug_strict_before_Selection_access":True,"fresh_shared_Selection_FP32_once":True,"original_FP32_and_quantized_states_unpolluted":True,
        "Selection_content_accessed_under_human_scope":True,"calibration_labels_read":False,"Holdout_accessed":False,"task_AP":True,"scientific_acceptance":False,
        "random_or_traditional_superiority":False,"deployment_INT8":False,"AP_receipt":P.identity(attempt / "scores/AP_receipt.json")}
    P.write(attempt / "completion_receipt.json",receipt)
    P.write(attempt / "manifest.json",{"files":[P.identity(path) for path in sorted(attempt.rglob("*")) if path.is_file() and path.name not in
        {"launcher_receipt.json","worker_console.txt","live_status.json"}]})
    return receipt


def run(contract_sha):
    P.require(P.sha(OUT / "contract.json")==contract_sha,"Explicit frozen Selection contract SHA required")
    p=P.read(OUT / "contract.json");validate_contract(p);verify_sources(p)
    P.require(p["status"]=="FROZEN_AUTHORIZED_REVIEWED_NOT_EXECUTED","Frozen contract missing")
    check_disk();attempt=OUT / "attempt_01";attempt.mkdir(exist_ok=False)
    receipt={"status":"RUNNING_BOUNDED_WORKER","contract":P.identity(OUT / "contract.json"),"started_at_utc":now(),"timeout_seconds":7200,"launcher_pid":os.getpid()}
    P.write(attempt / "launcher_receipt.json",receipt)
    try:
        with (attempt / "worker_console.txt").open("x",encoding="utf-8") as console:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),"worker"],cwd=ROOT,stdout=console,stderr=subprocess.STDOUT,
                timeout=7200,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        receipt.update(returncode=result.returncode,completion_receipt_present=(attempt / "completion_receipt.json").is_file())
        P.require(result.returncode==0 and receipt["completion_receipt_present"],"Worker failed; preserve attempt, no restart")
        receipt["status"]="COMPLETED_BOUNDED_WORKER"
    except subprocess.TimeoutExpired:
        receipt.update(status="TIMEOUT_PRESERVED_NO_RETRY",timeout=True);P.write(attempt / "timeout_receipt.json",receipt);raise
    except Exception as error:
        receipt.update(status="FAILED_PRESERVED_NO_RETRY",error=repr(error));raise
    finally:
        receipt["ended_at_utc"]=now();P.write(attempt / "launcher_receipt.json",receipt,once=False)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","freeze","run","worker"));parser.add_argument("--reviewed-draft-sha");parser.add_argument("--contract-sha")
    args=parser.parse_args()
    os.environ.update(YOLO_CONFIG_DIR=str(OUT / "settings"),YOLO_OFFLINE="true",YOLO_AUTOINSTALL="false",PYTHONDONTWRITEBYTECODE="1",CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if args.mode=="prepare":result=prepare()
    elif args.mode=="freeze":result=freeze(args.reviewed_draft_sha)
    elif args.mode=="run":result=run(args.contract_sha)
    else:
        try:result=worker()
        except Exception as error:
            attempt=OUT / "attempt_01"
            if attempt.is_dir() and not (attempt / "failure_receipt.json").exists():P.write(attempt / "failure_receipt.json",{"status":"FAILED_NO_AUTOMATIC_RETRY","error":repr(error)})
            raise
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
