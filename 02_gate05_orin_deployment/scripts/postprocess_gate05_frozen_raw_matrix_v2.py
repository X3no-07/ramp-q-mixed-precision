"""Reviewed v2 common Windows/CPU postprocessor for saved full-model outputs.

No model is loaded, built, or executed. Real NMS requires --execute and a frozen
acceptance contract. --self-test uses fake NMS outputs only. Raw files are read
without modification; only a fresh records directory and new matrix are written.
All warning/timeout evidence is retained and aborts this attempt without retry.
Preserves v1 unchanged. V2 binds every raw input row to the authenticated input
manifest, and binds the sole attempt store/matrix roots to the acceptance contract.
"""
import argparse
import ast
from collections import namedtuple
import contextlib
import copy
from datetime import datetime
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / 'scripts/evaluate_gate05_saved_target_matrix_v2.py'
EVALUATOR_SHA = 'b5ac60351e17761144a8897300ef8c8005f14e87e119b74dc1be2907be4cb895'
LEGACY = ROOT / 'src/gate0.py'
LEGACY_SHA = 'f3c4d80658b79ae685e569b37c5ea89af9e6e9dd4a63ce86aff2806a7fcc5adb'
SETTINGS = {'candidate_confidence': 0.001, 'nms_iou': 0.7, 'max_det': 1000,
            'nc': 10, 'max_time_img': 2.0, 'agnostic': False, 'multi_label': False}
CONFIGURATIONS = ('fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def contained(root, relative, directory=False):
    relative = Path(relative)
    path = (root/relative).resolve()
    require(not relative.is_absolute() and '..' not in relative.parts and root in path.parents and
            not path.is_symlink() and (path.is_dir() if directory else path.is_file()),
            'Missing or escaping raw input path: '+str(relative))
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), 'Symlink parent forbidden')
    return path


def evaluator_module():
    require(sha(EVALUATOR) == EVALUATOR_SHA, 'Frozen saved evaluator changed')
    spec = importlib.util.spec_from_file_location('_frozen_saved_evaluator', str(EVALUATOR))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_decode(nms_function):
    """Compile the original decode function without importing gate0/model code."""
    require(sha(LEGACY) == LEGACY_SHA, 'Original preprocessing/decode source changed')
    tree = ast.parse(LEGACY.read_text(encoding='utf-8'))
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'decode_detections')
    expected_kwargs = {'conf_thres': .001, 'iou_thres': .7, 'max_det': 1000, 'nc': 10, 'max_time_img': 2.0}
    def frozen_nms(prediction, **kwargs):
        require(kwargs == expected_kwargs, 'Original decode NMS call criteria changed')
        return nms_function(prediction, **dict(kwargs, agnostic=False, multi_label=False))
    namespace = {'non_max_suppression': frozen_nms,
                 'Detection': namedtuple('Detection', ('class_id', 'confidence', 'box'))}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              copy.deepcopy(original)], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(LEGACY)+':decode_only', 'exec'), namespace)
    return namespace['decode_detections']


def load_environment():
    """Import CPU postprocessing libraries and fingerprint them; does not call NMS."""
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_OFFLINE'] = 'true'
    import numpy as np
    import torch
    import torchvision
    import ultralytics
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils import LOGGER
    require(platform.system() == 'Windows', 'Frozen common postprocessing environment must be Windows CPU')
    versions = {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                'ultralytics': str(ultralytics.__version__), 'numpy': str(np.__version__)}
    modules = {'torch': torch, 'torchvision': torchvision, 'ultralytics': ultralytics, 'numpy': np}
    nms_source = Path(inspect.getsourcefile(non_max_suppression)).resolve()
    environment = {'schema': 'rampq.gate05.common_CPU_postprocess_environment/v1',
        'python': sys.version, 'executable': str(Path(sys.executable).resolve()), 'platform': platform.platform(),
        'machine': platform.machine(), 'library_versions': versions, 'device': 'cpu',
        'torch_threads': int(torch.get_num_threads()), 'torch_interop_threads': int(torch.get_num_interop_threads()),
        'module_files': {name: {'file': str(Path(module.__file__).resolve()), 'sha256': sha(Path(module.__file__))}
                         for name, module in modules.items()},
        'NMS_source_file': str(nms_source), 'NMS_source_file_sha256': sha(nms_source),
        'legacy_decode_source_sha256': LEGACY_SHA, 'evaluator_sha256': EVALUATOR_SHA,
        'torchvision_imported_before_NMS': True}
    identity = {'implementation_sha256': sha(Path(__file__).resolve()), 'environment_sha256': canonical_sha(environment),
        'device': 'cpu', 'library_versions': versions, 'nms_implementation_sha256': sha(nms_source),
        'legacy_decode_source_sha256': LEGACY_SHA}
    return np, torch, non_max_suppression, LOGGER, environment, identity


class WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []
        self.seen = set()
    def emit(self, record):
        if record in self.seen:
            return
        self.seen.add(record)
        self.records.append({'kind': 'logging', 'logger': record.name, 'level': record.levelname,
                             'message': record.getMessage()})


def captured_call(function, logger):
    """Capture warnings once; no retry, filtering, or independent timing tolerance."""
    capture, stdout, stderr = WarningCapture(), io.StringIO(), io.StringIO()
    loggers = [logger] if logger is logging.getLogger() else [logger, logging.getLogger()]
    for active_logger in loggers:
        active_logger.addHandler(capture)
    started = time.perf_counter()
    python_warnings = []
    try:
        with warnings.catch_warnings(record=True) as python_warnings, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            warnings.simplefilter('always')
            value = function()
        warning_rows = capture.records + [{'kind':'python_warning','category':row.category.__name__,
            'message':str(row.message)} for row in python_warnings]
        return value, {'complete':not warning_rows,'warnings':warning_rows,'stdout':stdout.getvalue(),
            'stderr':stderr.getvalue(),'elapsed_seconds_not_benchmark':time.perf_counter()-started,
            'library_max_time_img':2.0,'additional_elapsed_time_acceptance_threshold_defined':False}
    except BaseException as error:
        error.NMS_capture = {'complete':False,'warnings':capture.records + [
            {'kind':'python_warning','category':row.category.__name__,'message':str(row.message)} for row in python_warnings],
            'stdout':stdout.getvalue(),'stderr':stderr.getvalue(),
            'elapsed_seconds_not_benchmark':time.perf_counter()-started,'exception_type':type(error).__name__}
        raise
    finally:
        for active_logger in loggers:
            active_logger.removeHandler(capture)


def fixed_roots(policy, supplied_matrix_root):
    matrix = Path(policy['matrix_root'])
    attempts = Path(policy['attempt_root'])
    require(matrix.is_absolute() and attempts.is_absolute(),
            'Acceptance must pin absolute matrix_root and attempt_root')
    require(matrix.resolve() == supplied_matrix_root.resolve() and
            matrix.is_dir() and attempts.is_dir(),
            'Matrix root differs from acceptance authorization or attempt store is missing')
    return matrix.resolve(), attempts.resolve()


def match_frozen_input(row, frozen):
    direct = ('image_index', 'image', 'image_sha256', 'label', 'label_sha256',
              'input_file', 'input_sha256', 'original_shape', 'ratio', 'left', 'top')
    require(all(row[key] == frozen[key] for key in direct) and
            row['input_array_sha256'] == frozen['array_sha256'],
            'Raw row input file/payload/identity/transform differs from authenticated shared input row')


def verify_index(args, evaluator):
    root = args.matrix_root.resolve()
    index_path = args.raw_index.resolve()
    require(index_path.parent == root and sha(index_path) == args.raw_index_sha256,
            'Raw family index must be authenticated in the shared matrix root')
    contract_path = args.contract.resolve()
    contract = read(contract_path)
    evaluator.validate_frozen(contract,args.contract_sha256,contract_path,LEGACY)
    policy = contract['postprocess_execution']
    require(policy['maximum_NMS_calls'] == 1800 and policy['maximum_attempts'] == 1 and
            policy['automatic_retry'] is False and policy['warning_fail_closed'] is True,
            'Frozen acceptance contract must authorize only one warning-fail-closed 1800-call NMS attempt')
    root, attempt_root = fixed_roots(policy, root)
    _, metadata, _ = evaluator.verify_data(args.data_root.resolve(),contract)
    index = read(index_path)
    require(index['schema'] == 'rampq.gate05.frozen_raw_family_index/v1' and
            index['contract_sha256'] == args.contract_sha256 and index['dataset_manifest_sha256'] == evaluator.MANIFEST_SHA,
            'Raw family index has another acceptance contract or dataset')
    require(index['raw_contract_sha256'] == contract['raw_contract_sha256'],
            'Raw execution contract differs from the acceptance binding')
    inputs = contained(root,index['inputs_directory'],directory=True)
    require(attempt_root != inputs and inputs not in attempt_root.parents,
            'Contract attempt store must not modify the protected frozen input package')
    input_manifest = contained(inputs,index.get('input_manifest_file','manifest.json'))
    require(sha(input_manifest) == index['input_manifest_sha256'] == contract['input_manifest_sha256'],
            'Frozen shared input manifest differs')
    prepared = read(input_manifest)
    require(prepared['schema'] == 'rampq.gate05.shared_preprocessed_inputs/v1' and
            prepared['status'] == 'ALL150_COMMON_INPUTS_PREPARED_NO_MODEL_EXECUTION' and
            prepared['model_executions'] == 0 and len(prepared['images']) == 150 and
            prepared['data_manifest_sha256'] == evaluator.MANIFEST_SHA,
            'Shared input manifest is incomplete, model-selected, or bound to another dataset')
    frozen_rows = prepared['images']
    require([row['image_index'] for row in frozen_rows] == list(range(150)) and
            len({row['input_file'] for row in frozen_rows}) == 150,
            'Shared frozen input order or unique file identity differs')
    for frozen, (data, _, width, height) in zip(frozen_rows, metadata):
        require(all(frozen[key] == data[key] for key in ('image', 'image_sha256', 'label', 'label_sha256')) and
                frozen['original_shape'] == [height, width] and
                frozen['shape'] == [1, 3, 640, 640] and frozen['dtype'] == 'float32',
                'Shared input row differs from the frozen source image/dimensions')
    streams = index['streams']
    require(len(streams) == 12 and {(s['backend'],s['configuration']) for s in streams} ==
            {(backend,cfg) for backend in ('trt','ort') for cfg in CONFIGURATIONS}, 'Exactly six TRT and six ORT raw streams required')
    require([{'backend':s['backend'],'configuration':s['configuration']} for s in streams] == policy['stream_order'],
            'Postprocessing stream order differs from the pre-outcome contract')
    proof = index['engineering_evidence']
    proof_path = contained(root,proof['file'])
    require(sha(proof_path) == proof['sha256'], 'Engineering evidence identity differs')
    return root,index_path,contract,index,inputs,metadata,frozen_rows,attempt_root,input_manifest


def validate_raw_stream(root,stream,index,contract,metadata,frozen_rows):
    manifest_path = contained(root,stream['file'])
    require(sha(manifest_path) == stream['sha256'], 'Raw backend manifest differs')
    manifest = read(manifest_path)
    require(manifest['schema'] == 'rampq.gate05.frozen_raw_dataset_outputs/v1' and
            manifest['status'] == 'ALL150_RAW_OUTPUTS_SERIALIZED_NO_SCORING' and
            manifest['NMS_run'] is False and manifest['target_matching_run'] is False,
            'Incomplete or already-postprocessed backend raw stream')
    lineage = manifest['lineage']
    require(lineage['configuration'] == stream['configuration'] and lineage['backend'] == stream['backend'] and
            lineage['contract_sha256'] == index['raw_contract_sha256'] and
            lineage['input_manifest_sha256'] == index['input_manifest_sha256'] and
            lineage['source_data_manifest_sha256'] == index['dataset_manifest_sha256'], 'Raw backend lineage differs')
    binding = next(row for row in contract['configurations'] if row['name'] == stream['configuration'])
    require(lineage['graph_sha256'] == binding['source_graph_sha256' if stream['backend']=='trt' else
        'independent_ORT_reference_graph_sha256'], 'Raw graph differs from frozen reference/native graph')
    if stream['backend']=='trt':
        require(lineage['engine_sha256'] == binding['engine_sha256'], 'Raw native engine differs')
    rows=manifest['images']
    require(len(rows)==150 and manifest['state']['model_calls_attempted']==150 and
            manifest['state']['model_calls_completed']==150 and manifest['state']['serialized_images']==150,
            'Raw stream must contain the original 150 calls, not an output-selected subset')
    for image_index,(row,(data,_,width,height)) in enumerate(zip(rows,metadata)):
        match_frozen_input(row, frozen_rows[image_index])
        require(row['image_index']==image_index and row['image']==data['image'] and row['image_sha256']==data['image_sha256'] and
                row['label']==data['label'] and row['label_sha256']==data['label_sha256'] and
                row['original_shape']==[height,width], 'Raw image order/identity/coordinate frame differs')
        ratio=min(640/height,640/width)
        left=int(round((640-int(round(width*ratio)))/2-.1))
        top=int(round((640-int(round(height*ratio)))/2-.1))
        require(row['ratio']==ratio and row['left']==left and row['top']==top,
                'Exact original letterbox ratio/left/top differs; do not reconstruct with scale_boxes')
        require(row['all_outputs_finite'] is True, 'Nonfinite raw result preserved; postprocessing ineligible')
    return manifest_path,manifest


def execute(args):
    evaluator=evaluator_module()
    root,index_path,contract,index,inputs,metadata,frozen_rows,attempt_root,input_manifest=verify_index(args,evaluator)
    np,torch,nms,logger,environment,identity=load_environment()
    require(identity==contract['postprocess_identity'],'Current common CPU NMS code/environment differs from pre-outcome pin')
    require(Path(args.records_directory).name==args.records_directory and Path(args.matrix_name).name==args.matrix_name,
            'Records/matrix names must be plain basenames')
    directory=root/args.records_directory;matrix_path=root/args.matrix_name
    marker=attempt_root/('postprocess_attempt_'+args.contract_sha256+'.json')
    require(not directory.exists() and not matrix_path.exists() and not marker.exists(),
            'Exclusive contract-wide postprocessing attempt required; alternate folders do not authorize retry')
    manifests=[(stream,*validate_raw_stream(root,stream,index,contract,metadata,frozen_rows)) for stream in index['streams']]
    write(marker,{'status':'ONLY_FROZEN_POSTPROCESS_ATTEMPT_CLAIMED','contract_sha256':args.contract_sha256,
        'raw_contract_sha256':index['raw_contract_sha256'],'raw_index_sha256':args.raw_index_sha256,
        'maximum_NMS_calls':1800,'automatic_retry':False,'records_directory':args.records_directory,
        'matrix_root':str(root),'attempt_root':str(attempt_root)})
    directory.mkdir(exist_ok=False)
    write(directory/'attempt.json',{'status':'ONLY_FROZEN_COMMON_NMS_ATTEMPT_STARTED','contract_sha256':args.contract_sha256,
        'raw_index_sha256':args.raw_index_sha256,'maximum_NMS_calls':1800,'model_calls':0,'no_retry':True})
    write(directory/'environment.json',{'environment':environment,'identity':identity})
    decode=source_decode(nms);calls=0;output_streams=[];input_identities={};source_raw_identities={}
    try:
        for stream,manifest_path,manifest in manifests:
            backend='native' if stream['backend']=='trt' else 'ort'
            output_path=directory/(backend+'_'+stream['configuration']+'.jsonl')
            with output_path.open('x',encoding='utf-8',newline='\n') as handle:
                for row in manifest['images']:
                    image_index=row['image_index'];data,_,width,height=metadata[image_index]
                    input_path=contained(inputs,row['input_file']);input_bytes=input_path.read_bytes()
                    frozen = frozen_rows[image_index]
                    match_frozen_input(row, frozen)
                    require(len(input_bytes)==frozen['input_bytes'] and
                            hashlib.sha256(input_bytes).hexdigest()==frozen['input_sha256'],'Frozen input NPY bytes differ')
                    array=np.load(io.BytesIO(input_bytes),allow_pickle=False)
                    require(array.dtype==np.float32 and array.shape==(1,3,640,640) and array.flags.c_contiguous and
                            np.isfinite(array).all() and hashlib.sha256(array.tobytes(order='C')).hexdigest()==row['input_array_sha256'],
                            'Frozen shared input payload differs')
                    require(input_identities.setdefault(row['image'],row['input_sha256'])==row['input_sha256'],
                            'Native/ORT inputs differ for the same image')
                    raw_path=contained(manifest_path.parent,row['raw_file']);raw_bytes=raw_path.read_bytes()
                    require(len(raw_bytes)==row['raw_bytes'] and hashlib.sha256(raw_bytes).hexdigest()==row['raw_sha256'],
                            'Saved raw NPZ identity differs')
                    source_raw_identities[raw_path.relative_to(root).as_posix()]=row['raw_sha256']
                    keys=row['output_npz_keys']
                    require(keys=={'output0':'main_output','/model.10/m/m.0/attn/pe/conv/Conv_output_0':'pe_output'},'Raw output key binding differs')
                    with np.load(io.BytesIO(raw_bytes),allow_pickle=False) as raw:
                        require(set(raw.files)==set(keys.values()),'Unexpected raw output payload keys')
                        primary=raw['main_output'].copy();pe=raw['pe_output']
                        require(primary.dtype==pe.dtype==np.float32 and primary.shape==(1,14,8400) and
                                pe.shape==(1,128,20,20) and np.isfinite(primary).all() and np.isfinite(pe).all(),
                                'Wrong shape/dtype/nonfinite full raw output')
                    prediction=torch.from_numpy(primary)
                    require(str(prediction.device)=='cpu','NMS must use the pinned Windows CPU device')
                    require(calls < contract['postprocess_execution']['maximum_NMS_calls'], 'Frozen NMS call budget exhausted')
                    calls+=1
                    detections,capture=captured_call(lambda:decode(prediction,row['ratio'],row['left'],row['top'],width,height,
                        {'diagnostic_conf':.001,'nms_iou':.7,'max_det':1000}),logger)
                    output={'image':data['image'],'image_sha256':data['image_sha256'],'label_sha256':data['label_sha256'],
                        'original_width':width,'original_height':height,'input_file':input_path.relative_to(root).as_posix(),
                        'input_sha256':row['input_sha256'],'raw_output_file':raw_path.relative_to(root).as_posix(),
                        'raw_output_sha256':row['raw_sha256'],'raw_output_npz_key':'main_output',
                        'ratio':row['ratio'],'left':row['left'],'top':row['top'],
                        'nms_complete':capture['complete'],'nms_warnings':capture['warnings'],'NMS_capture':capture,
                        'detections':[{'class_id':d.class_id,'confidence':d.confidence,'box':list(d.box)} for d in detections]}
                    handle.write(json.dumps(output,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
                    require(capture['complete'],'NMS warning/timeout recorded; attempt incomplete, no retry')
                    write(directory/('progress_{:04d}.json'.format(calls)),{'NMS_calls_completed':calls,
                        'configuration':stream['configuration'],'backend':backend,'image_index':image_index,
                        'elapsed_NMS_seconds_not_benchmark':capture['elapsed_seconds_not_benchmark']})
            require(sha(manifest_path)==stream['sha256'],'Raw backend manifest changed during postprocessing')
            output_streams.append({'configuration':stream['configuration'],'backend':backend,
                'file':output_path.relative_to(root).as_posix(),'sha256':sha(output_path),
                'graph_sha256':manifest['lineage']['graph_sha256'],'engine_sha256':manifest['lineage'].get('engine_sha256'),
                'postprocess_settings':dict(SETTINGS),'postprocess_identity':identity})
        require(calls==1800 and len(output_streams)==12,'Incomplete frozen common postprocessing call inventory')
        require(sha(index_path)==args.raw_index_sha256 and sha(args.contract.resolve())==args.contract_sha256 and
                sha(Path(__file__).resolve())==identity['implementation_sha256'] and sha(EVALUATOR)==EVALUATOR_SHA and
                sha(LEGACY)==LEGACY_SHA and sha(input_manifest)==contract['input_manifest_sha256'],
                'Frozen index/code/contract/shared input manifest became stale')
        require(all(sha(root/name)==expected for name,expected in source_raw_identities.items()),'Original raw files changed')
        matrix={'schema':'rampq.gate05.saved_prediction_matrix/v1','contract_sha256':args.contract_sha256,
            'dataset_manifest_sha256':evaluator.MANIFEST_SHA,'streams':output_streams,
            'engineering_evidence':index['engineering_evidence'],'source_raw_index_sha256':args.raw_index_sha256,
            'NMS_calls':calls,'model_inference_calls':0,'target_matching_or_scoring_run':False}
        write(matrix_path,matrix)
        write(directory/'completion.json',{'status':'ALL1800_COMMON_CPU_NMS_OUTPUTS_SAVED_NO_TARGET_SCORING',
            'matrix_file':matrix_path.name,'matrix_sha256':sha(matrix_path),'NMS_calls':calls,
            'postprocess_identity':identity,'original_raw_files_unchanged':True,
            'original_complete_model_guard':'FAILED_UNCHANGED','accuracy_or_Gate_pass_claim':False})
        print(json.dumps({'status':'SAVED_COMMON_NMS_MATRIX_READY_NO_TARGET_SCORING','matrix':str(matrix_path),
            'matrix_sha256':sha(matrix_path),'NMS_calls':calls,'model_calls':0}))
    except BaseException as error:
        write(directory/'incomplete.json',{'status':'POSTPROCESS_INCOMPLETE_PRESERVE_NO_RETRY','NMS_calls_attempted':calls,
            'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
            'NMS_capture':getattr(error,'NMS_capture',None),
            'partial_records_preserved':True,'final_matrix_written':matrix_path.exists(),
            'original_complete_model_guard':'FAILED_UNCHANGED','target_scoring_run':False})
        raise


def self_test():
    """Fake NMS outputs and a fake logger only; no torch/numpy/real NMS imports."""
    class FakeTensor:
        def __init__(self,rows):self.rows=rows
        def detach(self):return self
        def cpu(self):return self
        def tolist(self):return self.rows
    observed=[]
    def fake_nms(prediction,**kwargs):
        observed.append(kwargs)
        return [FakeTensor([[0.,10.,300.,100.,.8,2],[12.,26.,32.,46.,.25,1]])]
    decode=source_decode(fake_nms)
    boxes=decode(object(),2.,10,20,100,50,{'diagnostic_conf':.001,'nms_iou':.7,'max_det':1000})
    require(len(observed)==1 and observed[0]=={'conf_thres':.001,'iou_thres':.7,'max_det':1000,'nc':10,
        'max_time_img':2.0,'agnostic':False,'multi_label':False},'Synthetic exact NMS call contract failed')
    require(boxes[0].box==(0.,0.,100.,40.) and boxes[1].box==(1.,3.,11.,13.) and
            [b.class_id for b in boxes]==[2,1],'Original subtract-pad/divide-ratio/clip/order semantics changed')
    logger=logging.getLogger('_synthetic_frozen_NMS_warning')
    logger.propagate=False
    def warn():
        logger.warning('synthetic NMS time limit exceeded')
        return []
    _,capture=captured_call(warn,logger)
    require(capture['complete'] is False and len(capture['warnings'])==1,'Synthetic warning was not marked incomplete')
    frozen={'image_index':0,'image':'synthetic.jpg','image_sha256':'a','label':'synthetic.txt',
        'label_sha256':'b','input_file':'inputs/frozen.npy','input_sha256':'c','array_sha256':'d',
        'original_shape':[10,20],'ratio':2.,'left':0,'top':1}
    row={key:value for key,value in frozen.items() if key!='array_sha256'}
    row['input_array_sha256']='d'
    match_frozen_input(row,frozen)
    for key in ('input_file','input_sha256','input_array_sha256'):
        wrong=dict(row);wrong[key]='changed'
        try:match_frozen_input(wrong,frozen)
        except ValueError:pass
        else:raise AssertionError('Unbound raw input accepted: '+key)
    policy={'matrix_root':str(ROOT),'attempt_root':str(ROOT)}
    require(fixed_roots(policy,ROOT)==(ROOT,ROOT),'Pinned synthetic absolute roots failed')
    try:fixed_roots(policy,ROOT/'copied_matrix')
    except ValueError:pass
    else:raise AssertionError('Alternate matrix root accepted under same contract')
    require(all(name not in sys.modules for name in ('numpy','torch','torchvision','ultralytics','onnxruntime','tensorrt')),
            'Synthetic-only test imported real inference/NMS libraries')
    print(json.dumps({'status':'SYNTHETIC_SCHEMA_GEOMETRY_AND_WARNING_CAPTURE_PASSED',
        'real_NMS_calls':0,'model_calls':0,'dataset_prediction_streams_read':0,
        'evaluator_sha256':EVALUATOR_SHA,'legacy_decode_sha256':LEGACY_SHA}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path);parser.add_argument('--contract-sha256')
    parser.add_argument('--raw-index',type=Path);parser.add_argument('--raw-index-sha256')
    parser.add_argument('--matrix-root',type=Path)
    parser.add_argument('--data-root',type=Path,default=ROOT/'artifacts/gate05/data_v1')
    parser.add_argument('--records-directory',default='postprocessed_detections')
    parser.add_argument('--matrix-name',default='saved_prediction_matrix.json')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--describe-environment',action='store_true')
    mode.add_argument('--check-input-only',action='store_true');mode.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    elif args.describe_environment:
        _,_,_,_,environment,identity=load_environment()
        print(json.dumps({'status':'ENVIRONMENT_IDENTITIES_ONLY_NO_NMS_EXECUTION','environment':environment,
            'postprocess_identity':identity,'real_NMS_calls':0,'model_calls':0},indent=2))
    else:
        require(all((args.contract,args.contract_sha256,args.raw_index,args.raw_index_sha256,args.matrix_root)),
                'Authenticated frozen contract/raw index/matrix root are required')
        if args.check_input_only:
            evaluator=evaluator_module()
            root,_,contract,index,_,metadata,frozen_rows,_,_=verify_index(args,evaluator)
            for stream in index['streams']:validate_raw_stream(root,stream,index,contract,metadata,frozen_rows)
            print(json.dumps({'status':'FROZEN_RAW_FAMILY_INDEX_VERIFIED_NO_NMS_EXECUTION','streams':12,
                'real_NMS_calls':0,'model_calls':0,'inference_runtime_imported':False}))
        else:execute(args)


if __name__=='__main__':
    main()
