"""Execute one independently audited matched control twice on the saved input.

Only a separately frozen smoke protocol and exact engine/audit hashes authorize
execution. Input checks are stdlib-only. No model is built, no CPU model is run,
and no new numerical tolerance, dataset, accuracy, speed or Gate pass is defined.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback
import zipfile


INPUT_SHA256 = '24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
FIXED_IMAGE_SHA256 = '494bbd9d09de29f3a992d434da03c252b4f6cf1f28099fd5a443d0c797f5ec52'
NMS = {'conf_thres': 0.25, 'iou_thres': 0.7, 'max_det': 1000, 'nc': 10}
NVTX_RANGE = 'RAMPQ_MATCHED_QDQ_FIXED_IMAGE'
SCHEMA = 'rampq.gate05.control_fixed_input_smoke/v1'
LIBRARY_NAME = re.compile(r'(?:libnvinfer|libcudnn|libcublas|libcuda(?:rt)?|libnvrtc|libnvJitLink|libtorch_cuda|libc10_cuda)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def contained(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe input path: ' + str(name))
    path = root / relative
    require(root in path.resolve().parents and not path.is_symlink(), 'Input escapes root or is symlink: ' + str(name))
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), 'Input has symlink parent: ' + str(name))
    return path


def verify_manifest(root):
    manifest = read(root / 'manifest.json')
    require(isinstance(manifest.get('files'), dict) and manifest['files'], 'Empty source manifest')
    for name, expected in manifest['files'].items():
        path = contained(root, name)
        require(path.is_file() and path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
                'Frozen manifest file changed: ' + str(path))
    return manifest


def binding_signature(rows):
    require(rows and len(rows) == len({row['name'] for row in rows}), 'Empty or duplicate binding contract')
    require(sum(row['input'] is True for row in rows) == 1, 'Expected one input')
    for row in rows:
        require(type(row['input']) is bool and row['dtype'] == 'DataType.FLOAT' and
                row['shape'] and all(type(value) is int and value > 0 for value in row['shape']),
                'External bindings require positive static FLOAT tensors')
    return sorted((row['name'], row['input'], tuple(row['shape']), row['dtype']) for row in rows)


def npy_header(stream):
    """Inspect only the NPY header with stdlib; never import NumPy for input checks."""
    require(stream.read(6) == b'\x93NUMPY', 'Not an NPY array')
    version = stream.read(2)
    require(version in (b'\x01\x00', b'\x02\x00', b'\x03\x00'), 'Unsupported NPY header version')
    length_size = 2 if version == b'\x01\x00' else 4
    encoded_length = stream.read(length_size)
    require(len(encoded_length) == length_size, 'Truncated NPY header length')
    length = struct.unpack('<H' if length_size == 2 else '<I', encoded_length)[0]
    require(0 < length <= 65536, 'Unreasonable NPY header size')
    encoded = stream.read(length)
    require(len(encoded) == length, 'Truncated NPY header')
    header = ast.literal_eval(encoded.decode('utf-8' if version == b'\x03\x00' else 'latin1').strip())
    require(isinstance(header, dict) and header['descr'] in ('<f4', '=f4') and
            header['fortran_order'] is False and isinstance(header['shape'], tuple) and
            all(type(value) is int and value > 0 for value in header['shape']), 'Expected C-contiguous FLOAT32 NPY')
    return header, 8 + length_size + length


def verify_requested_precision(audit, spec, diagnostic_formats_only):
    """Retain the original format gate and separate named-integer-tactic gaps."""
    requests = spec['requests']
    quantized = sum(row['precision'] == 'INT8' for row in requests)
    floating = sum(row['precision'] == 'FP16' for row in requests)
    require(len(requests) == len({row['onnx_node'] for row in requests}) and
            quantized + floating == len(requests) and
            (quantized, floating) == (spec['quantized_conv_count'], spec['fp16_conv_count']),
            'Frozen precision mask inventory differs')
    require(audit['status'] == 'REQUESTED_CONV_FORMATS_VERIFIED' and audit['requested_conv_formats_complete'] is True,
            'Actual requested Conv formats are incomplete')
    require(not audit['errors'] and not audit['unmapped_engine_convs'], 'Audit contains format errors or unmapped Convs')
    require(audit['source_convs'] == len(requests) and audit['quantized_conv_count'] == quantized and
            audit['fp16_conv_count'] == floating and audit['verified_int8_format_convs'] == quantized and
            audit['verified_fp16_format_convs'] == floating, 'Actual format counts differ from source requests')
    mappings = audit['mapping']
    require(len(mappings) == len({row['onnx_node'] for row in mappings}) == len(requests),
            'Actual audit must map every source Conv exactly once')
    mapped = {row['onnx_node']: row for row in mappings}
    require(set(mapped) == {row['onnx_node'] for row in requests}, 'Actual/source Conv identities differ')
    for request in requests:
        observed = mapped[request['onnx_node']]
        require(all(observed.get(key) == value for key, value in request.items()) and
                observed['requested_formats_verified'] is True, 'Requested precision mask or formats differ')
    strict_status = audit['strict_int8_status']
    strict_ok = strict_status == ('ALL_REQUESTED_INT8_TACTICS_VERIFIED' if quantized else 'NO_INT8_REQUESTED')
    if quantized:
        require(all(row['strict_int8_tactic_verified'] is True for row in mappings if row['precision'] == 'INT8') == strict_ok,
                'Named tactic aggregate disagrees with per-source evidence')
    if not strict_ok:
        require(quantized > 0 and strict_status == 'INTEGER_TACTIC_REVIEW_REQUIRED' and diagnostic_formats_only,
                'Named integer tactics incomplete; explicit --diagnostic-formats-only is required')
    gaps = [row for row in mappings if row['precision'] == 'INT8' and row['strict_int8_tactic_verified'] is not True]
    return strict_ok, gaps


def validate_inputs(args):
    root, build = args.root.resolve(), args.build.resolve()
    protocol_path, audit_path = args.protocol.resolve(), args.audit.resolve()
    require(root.is_dir() and root in build.parents and build.is_dir(), 'Build must be inside frozen control root')
    require(len(args.engine_sha256) == len(args.audit_sha256) == len(args.protocol_sha256) == 64,
            'Expected complete SHA256 pins')
    require(sha(protocol_path) == args.protocol_sha256 and sha(audit_path) == args.audit_sha256,
            'Frozen smoke protocol or precision audit identity differs')
    protocol = read(protocol_path)
    require(protocol['schema'] == 'rampq.gate05.control_fixed_input_protocol/v1', 'Wrong smoke protocol schema')
    control = read(root / 'control_spec.json')
    manifest = verify_manifest(root)
    source = contained(root, control['source_bundle_directory'] + '/spec.json').parent
    spec = read(source / 'spec.json')
    verify_manifest(source)
    require(protocol['control_spec_sha256'] == sha(root / 'control_spec.json') and
            protocol['root_manifest_sha256'] == sha(root / 'manifest.json') and
            protocol['source_spec_sha256'] == sha(source / 'spec.json') and
            protocol['source_graph_sha256'] == sha(source / 'model.onnx') == spec['graph_sha256'],
            'Protocol does not cover exact frozen control/source')
    require(protocol['runner_sha256'] == sha(Path(__file__).resolve()), 'Running unpinned fixed-input tool')
    require(protocol['engine_sha256'] == args.engine_sha256 and protocol['precision_audit_sha256'] == args.audit_sha256,
            'CLI engine/audit pins differ from frozen smoke protocol')
    require(protocol['executions'] == 2 and protocol['nms'] == NMS and protocol['nvtx_range'] == NVTX_RANGE,
            'Frozen execution count, NMS or NVTX scope differs')
    require(protocol['dataset_evaluation_run'] is False and protocol['new_CPU_model_executions'] == 0 and
            protocol['new_tolerances_defined'] is False and protocol['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
            protocol['gate05_status'] == 'INCOMPLETE', 'Protected smoke scope changed')
    require(not args.diagnostic_formats_only or protocol['diagnostic_formats_only_allowed'] is True,
            'Protocol does not permit diagnostic tactic-gap execution')
    image_shape = protocol['original_image_shape']
    require(len(image_shape) == 2 and all(type(value) is int and value > 0 for value in image_shape) and
            protocol['fixed_image_sha256'] == FIXED_IMAGE_SHA256,
            'Frozen original image geometry/identity missing')
    require(control['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and
            control['gate05_status'] == 'INCOMPLETE' and control['dataset_evaluation_run'] is False,
            'Control no longer preserves original failed guard')
    expected = spec['expected_binding_contract']
    require(len(expected) == 3 and len([row for row in expected if not row['input']]) == 2 and
            binding_signature(expected) == binding_signature(protocol['expected_binding_contract']),
            'Expected one input and two unchanged FLOAT outputs')
    require(spec['primary_output_name'] == protocol['primary_output_name'] == 'output0' and
            next(row for row in expected if row['input'])['shape'] == [1, 3, 640, 640] and
            next(row for row in expected if row['name'] == 'output0')['shape'] == [1, 14, 8400],
            'Frozen image/primary-output boundary differs')
    result_path, request_path = build / 'build_result.json', build / 'build_request.json'
    result, request = read(result_path), read(request_path)
    require(result['status'] == 'BUILT_AND_DESERIALIZED_BUILD_ONLY' and result['native_inference_executions'] == 0 and
            result['engine_execution_context_created'] is False, 'Engine not from an audited build-only attempt')
    engine_path = build / 'model.engine'
    require(sha(engine_path) == args.engine_sha256 == result['engine_sha256'], 'Built engine hash differs')
    require(sha(result_path) == protocol['build_result_sha256'] and
            sha(build / 'engine_layers.json') == protocol['engine_layers_sha256'], 'Build result/inspector identity differs')
    require(result['graph_sha256'] == spec['graph_sha256'] and
            result['tool_identity']['control_spec_sha256'] == sha(root / 'control_spec.json') and
            result['tool_identity']['outer_manifest_sha256'] == sha(root / 'manifest.json'), 'Build/source identity differs')
    require(binding_signature(result['bindings']) == binding_signature(expected) and
            all(row['format'] == 'TensorFormat.LINEAR' for row in result['bindings']), 'Built bindings violate FLOAT/LINEAR contract')
    audit = read(audit_path)
    for key, expected_hash in (('engine_sha256', args.engine_sha256), ('onnx_sha256', spec['graph_sha256']),
                               ('source_spec_sha256', sha(source / 'spec.json')),
                               ('control_spec_sha256', sha(root / 'control_spec.json')),
                               ('layer_file_sha256', sha(build / 'engine_layers.json')),
                               ('build_result_sha256', sha(result_path)), ('build_request_sha256', sha(request_path))):
        require(audit[key] == expected_hash, 'Precision audit lineage differs: ' + key)
    strict_ok, gaps = verify_requested_precision(audit, spec, args.diagnostic_formats_only)
    input_path = contained(root, protocol['input_file'])
    require(input_path == source / 'preprocessed_input.npy' and
            sha(input_path) == protocol['input_sha256'] == INPUT_SHA256, 'Use exact saved fixed-image preprocessing NPY')
    with input_path.open('rb') as stream:
        header, offset = npy_header(stream)
    require(header['shape'] == (1, 3, 640, 640) and input_path.stat().st_size == offset + 4 * math.prod(header['shape']),
            'Saved input NPY header or byte length differs')
    cpu_reference = protocol['cpu_reference']
    cpu_path = contained(root, cpu_reference['file'])
    require(sha(cpu_path) == cpu_reference['sha256'] and cpu_path.name.endswith('.npz'), 'Saved CPU raw reference changed')
    require(set(cpu_reference['output_npz_keys']) == {row['name'] for row in expected if not row['input']},
            'Saved CPU reference must map both named outputs')
    require(len(set(cpu_reference['output_npz_keys'].values())) == 2, 'CPU output keys must be distinct')
    expected_by_name = {row['name']: row for row in expected}
    with zipfile.ZipFile(cpu_path, 'r') as archive:
        for name, key in cpu_reference['output_npz_keys'].items():
            require(isinstance(key, str) and '/' not in key and '\\' not in key, 'Unsafe CPU NPZ key')
            member = key + '.npy'
            with archive.open(member) as stream:
                header, offset = npy_header(stream)
            require(list(header['shape']) == expected_by_name[name]['shape'] and
                    archive.getinfo(member).file_size == offset + 4 * math.prod(header['shape']),
                    'Saved CPU raw header or payload size differs: ' + name)
    require(protocol['input_file'] in manifest['files'] and cpu_reference['file'] in manifest['files'],
            'Frozen control manifest must cover saved input and CPU reference')
    return {'root': root, 'build_dir': build, 'protocol': protocol, 'control': control, 'spec': spec,
            'build_result': result, 'build_request': request, 'engine_path': engine_path, 'audit': audit,
            'input_path': input_path, 'cpu_path': cpu_path, 'strict_integer_tactics_verified': strict_ok,
            'strict_integer_gaps': gaps,
            'lineage': {'protocol_sha256': args.protocol_sha256, 'precision_audit_sha256': args.audit_sha256,
                        'engine_sha256': args.engine_sha256, 'control_spec_sha256': sha(root / 'control_spec.json'),
                        'root_manifest_sha256': sha(root / 'manifest.json'), 'source_spec_sha256': sha(source / 'spec.json'),
                        'source_graph_sha256': sha(source / 'model.onnx'), 'build_result_sha256': sha(result_path),
                        'build_request_sha256': sha(request_path), 'engine_layers_sha256': sha(build / 'engine_layers.json'),
                        'input_sha256': sha(input_path), 'cpu_reference_sha256': sha(cpu_path)}}


def maps_snapshot(output, label):
    text = Path('/proc/self/maps').read_text(encoding='utf-8')
    (output / ('maps_' + label + '.txt')).write_text(text, encoding='utf-8')
    return text


def build_id(path):
    executable = shutil.which('readelf')
    if executable is None:
        return {'available': False, 'reason': 'readelf not found'}
    try:
        result = subprocess.run([executable, '-n', str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, timeout=20)
        matches = re.findall(r'Build ID:\s*([0-9a-fA-F]+)', result.stdout)
        return {'available': bool(matches), 'command': [executable, '-n', str(path)],
                'returncode': result.returncode, 'build_ids': matches, 'output': result.stdout}
    except Exception as error:
        return {'available': False, 'error_type': type(error).__name__, 'error': str(error)}


def mapped_library_identity(maps_text):
    mapped = {}
    for line in maps_text.splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        raw_path = fields[5]
        if not LIBRARY_NAME.search(Path(raw_path).name):
            continue
        mapped.setdefault(raw_path, []).append({'address': fields[0], 'permissions': fields[1], 'offset': fields[2],
                                               'device': fields[3], 'inode': fields[4]})
    rows = []
    for raw_path, regions in sorted(mapped.items()):
        record = {'mapped_path': raw_path, 'mapping_regions': regions}
        if raw_path.endswith(' (deleted)'):
            record.update(hash_available=False, reason='Mapped inode marked deleted; do not substitute another file')
            rows.append(record)
            continue
        path = Path(raw_path)
        try:
            chain, current = [], path
            for _index in range(16):
                if not current.is_symlink():
                    break
                link = os.readlink(str(current))
                chain.append({'path': str(current), 'readlink': link})
                current = Path(link) if Path(link).is_absolute() else current.parent / link
            real = path.resolve(strict=True)
            stat = path.stat()
            record.update(real_path=str(real), readlink_chain=chain, bytes=stat.st_size,
                          inode=stat.st_ino, device_major=os.major(stat.st_dev), device_minor=os.minor(stat.st_dev))
            devices = {(int(row['device'].split(':')[0], 16), int(row['device'].split(':')[1], 16), int(row['inode']))
                       for row in regions}
            require(devices == {(os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)},
                    'Mapped inode/device differs from file currently at pathname')
            record.update(hash_available=True, sha256=sha(real), elf_build_id=build_id(real))
        except Exception as error:
            record.update(hash_available=False, error_type=type(error).__name__, error=str(error))
        rows.append(record)
    return {'libraries': rows, 'all_mapped_selected_library_hashes_available': bool(rows) and
            all(row.get('hash_available') for row in rows),
            'scope': 'Only selected shared objects present in this process maps; absent components are not guessed or imported for this report.'}


def array_stats(array, np):
    finite = np.isfinite(array)
    values = array[finite]
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'elements': int(array.size),
            'finite': bool(finite.all()), 'nonfinite_elements': int(np.count_nonzero(~finite)),
            'min_finite': float(values.min()) if values.size else None,
            'max_finite': float(values.max()) if values.size else None,
            'mean_finite': float(values.astype(np.float64).mean()) if values.size else None}


def describe_difference(actual, reference, np):
    require(actual.shape == reference.shape and actual.dtype == reference.dtype, 'Raw comparison shape/dtype differs')
    finite = np.isfinite(actual) & np.isfinite(reference)
    delta = np.abs(actual.astype(np.float64)[finite] - reference.astype(np.float64)[finite])
    return {'shape': list(actual.shape), 'dtype': str(actual.dtype), 'numeric_equal': bool(np.array_equal(actual, reference)),
            'byte_identical': actual.tobytes(order='C') == reference.tobytes(order='C'),
            'different_elements': int(np.count_nonzero(actual != reference)),
            'both_finite_elements': int(np.count_nonzero(finite)),
            'max_abs_finite': float(delta.max()) if delta.size else None,
            'mean_abs_finite': float(delta.mean()) if delta.size else None,
            'descriptive_only': True, 'tolerance_or_pass_threshold_defined': False}


def terminate_as_interrupt(_signum, _frame):
    raise KeyboardInterrupt('External monitor requested termination')


def execute(args, verified, output, state):
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_CONFIG_DIR'] = str(output / '.ultralytics')
    maps_snapshot(output, 'before_gpu_imports')
    import numpy as np
    import torch
    import tensorrt as trt
    require(trt.__version__.startswith('8.5.'), 'Expected TensorRT 8.5')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    environment = {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                   'tensorrt': trt.__version__, 'torch_CUDA': torch.version.cuda,
                   'torch_cuDNN_version': torch.backends.cudnn.version(), 'GPU': torch.cuda.get_device_name(0),
                   'CUDA_capability': list(torch.cuda.get_device_capability(0)),
                   'GPU_properties': {key: getattr(properties, key) for key in
                     ('name', 'major', 'minor', 'total_memory', 'multi_processor_count') if hasattr(properties, key)},
                   'GPU_properties_repr': repr(properties), 'device_index': 0,
                   'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pid': os.getpid(), 'pgid': os.getpgrp()}
    write(output / 'environment.json', environment)
    array = np.load(str(verified['input_path']), allow_pickle=False)
    require(array.shape == (1, 3, 640, 640) and array.dtype == np.float32 and
            array.flags.c_contiguous and bool(np.isfinite(array).all()), 'Invalid saved preprocessed input')
    expected = verified['spec']['expected_binding_contract']
    desired = {row['name']: row for row in expected}
    cpu_reference = verified['protocol']['cpu_reference']
    references = {}
    with np.load(str(verified['cpu_path']), allow_pickle=False) as data:
        for name, key in cpu_reference['output_npz_keys'].items():
            references[name] = data[key].copy()
            require(list(references[name].shape) == desired[name]['shape'] and references[name].dtype == np.float32 and
                    bool(np.isfinite(references[name]).all()), 'Invalid saved CPU raw reference: ' + name)
    logger = trt.Logger(trt.Logger.WARNING)
    require(trt.init_libnvinfer_plugins(logger, ''), 'Cannot initialize TensorRT plugins')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(verified['engine_path'].read_bytes())
    require(engine is not None, 'Cannot deserialize frozen engine')
    inspector_raw = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    (output / 'runtime_engine_layers.json').write_text(inspector_raw, encoding='utf-8')
    context = engine.create_execution_context()
    require(context is not None, 'Cannot create execution context')
    context.nvtx_verbosity = trt.ProfilingVerbosity.DETAILED
    descriptions = [{'name': engine.get_binding_name(index), 'input': bool(engine.binding_is_input(index)),
                     'shape': list(engine.get_binding_shape(index)), 'dtype': str(engine.get_binding_dtype(index)),
                     'format': str(engine.get_binding_format(index))} for index in range(engine.num_bindings)]
    write(output / 'bindings.json', {'bindings': descriptions})
    require(binding_signature(descriptions) == binding_signature(expected) and len(descriptions) == 3 and
            all(row['format'] == 'TensorFormat.LINEAR' for row in descriptions), 'Runtime named bindings differ')
    image_tensor = torch.from_numpy(array).to(device='cuda', dtype=torch.float32)
    # Separate output allocations retain both executions without device-to-host
    # copies or postprocessing inside the single NVTX execution capture range.
    buffers = []
    for _repeat in range(2):
        buffers.append([image_tensor if row['input'] else torch.empty(tuple(row['shape']), device='cuda', dtype=torch.float32)
                        for row in descriptions])
    torch.cuda.synchronize()
    before_maps = maps_snapshot(output, 'before_execution')
    write(output / 'loaded_libraries_before_execution.json', mapped_library_identity(before_maps))
    state.update(phase='EXECUTING', executions_attempted=0, executions_completed=0, native_inference_executions=0)
    write(output / 'execution_state.json', state)
    times, failure = [], None
    torch.cuda.nvtx.range_push(NVTX_RANGE)
    try:
        for repeat in range(2):
            torch.cuda.synchronize()
            started = time.perf_counter()
            state['executions_attempted'] += 1
            require(context.execute_v2([int(value.data_ptr()) for value in buffers[repeat]]), 'TensorRT execute_v2 returned false')
            torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000.0)
            state['executions_completed'] += 1
            state['native_inference_executions'] = state['executions_completed']
    except BaseException:
        failure = sys.exc_info()
    finally:
        torch.cuda.nvtx.range_pop()
    state['synchronized_call_ms_not_benchmark'] = times
    after_maps = maps_snapshot(output, 'after_execution')
    write(output / 'loaded_libraries_after_execution.json', mapped_library_identity(after_maps))
    saved, output_keys = {}, {}
    for index, row in enumerate(descriptions):
        if row['input']:
            continue
        output_number = len(output_keys)
        output_keys[row['name']] = {'first': 'output_{}_first'.format(output_number),
                                    'repeat': 'output_{}_repeat'.format(output_number)}
        for repeat in range(state['executions_completed']):
            label = 'first' if repeat == 0 else 'repeat'
            saved[output_keys[row['name']][label]] = buffers[repeat][index].detach().cpu().numpy().copy()
    if saved:
        np.savez_compressed(str(output / 'raw_outputs.npz'), **saved)
    write(output / 'execution_state.json', state)
    if failure is not None:
        raise failure[1].with_traceback(failure[2])
    require(state['executions_completed'] == 2, 'Expected exactly two completed engine executions')
    comparisons, stats, finite = {}, {}, {}
    for name, keys in output_keys.items():
        first, repeat = saved[keys['first']], saved[keys['repeat']]
        stats[name] = {'first': array_stats(first, np), 'repeat': array_stats(repeat, np)}
        finite[name] = stats[name]['first']['finite'] and stats[name]['repeat']['finite']
        comparisons[name] = {'repeat_vs_first': describe_difference(repeat, first, np),
                             'first_vs_saved_CPU': describe_difference(first, references[name], np),
                             'repeat_vs_saved_CPU': describe_difference(repeat, references[name], np)}
    # Reuse the existing frozen postprocessing parameters. No image decoding or
    # preprocessing rerun occurs; original image dimensions only restore XYXY.
    import ultralytics
    from ultralytics.utils import ops
    try:
        from ultralytics.utils.nms import non_max_suppression
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression
    detections, network_detections = {}, {}
    if all(finite.values()):
        for label in ('first', 'repeat'):
            primary = torch.from_numpy(saved[output_keys['output0'][label]]).clone()
            boxes = non_max_suppression(primary, **NMS)[0].detach().cpu()
            network_detections[label] = boxes.tolist()
            ops.scale_boxes((640, 640), boxes[:, :4], tuple(verified['protocol']['original_image_shape']))
            detections[label] = boxes.tolist()
    post_maps = maps_snapshot(output, 'after_postprocessing')
    write(output / 'loaded_libraries_after_postprocessing.json', mapped_library_identity(post_maps))
    validate_inputs(args)
    state.update(schema=SCHEMA, phase='COMPLETE', status='FIXED_INPUT_EXECUTED_DESCRIPTIVE_ONLY' if all(finite.values()) else
                 'FIXED_INPUT_EXECUTED_NONFINITE_OUTPUT',
          execution_success=True, configuration=verified['spec']['configuration'],
          expected_binding_contract=expected, bindings=descriptions, primary_output_name='output0',
          raw_outputs_sha256=sha(output / 'raw_outputs.npz'), output_npz_keys=output_keys,
          all_outputs_finite=finite, output_statistics=stats, comparisons_to_saved_CPU_and_repeat=comparisons,
          nms=NMS, detections=detections.get('first', []), repeat_detections=detections.get('repeat', []),
          network_coordinate_detections=network_detections, original_image_shape=verified['protocol']['original_image_shape'],
          image='000020.jpg', image_sha256=FIXED_IMAGE_SHA256, nvtx_capture_range=NVTX_RANGE,
          nvtx_capture_range_occurrences=1, nvtx_scope='Exactly two engine execute_v2 calls with device synchronization; no NMS or device-to-host output copy inside range',
          diagnostic_formats_only_opt_in=args.diagnostic_formats_only,
          all_requested_conv_formats_verified=True, precision_audit_status=verified['audit']['status'],
          strict_int8_status=verified['audit']['strict_int8_status'],
          strict_integer_tactics_verified=verified['strict_integer_tactics_verified'],
          strict_integer_gap_count=len(verified['strict_integer_gaps']),
          strict_integer_gaps_preserved=verified['strict_integer_gaps'], SASS_gap_closure_claim=False,
          source_CPU_reference_role='Saved CPU outputs only; no CPU model execution by this script',
          input_preprocessing_rerun=False, new_CPU_model_executions=0, ultralytics=ultralytics.__version__,
          original_complete_model_numerical_guard='FAILED_UNCHANGED', gate05_status='INCOMPLETE',
          new_tolerances_defined=False, numerical_migration_pass_claim=False, accuracy_pass_claim=False,
          performance_pass_claim=False, target_rescue_claim=False, causal_attribution_allowed=False,
          dataset_evaluation_run=False,
          limits='One frozen preprocessed Selection image, two executions. Raw and detection differences are descriptive; source floating-control comparison and runtime arithmetic attribution require separate saved-evidence review.')
    write(output / 'smoke_result.json', state)
    require(all(finite.values()), 'Nonfinite native outputs retained; do not claim successful numerical smoke')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--engine-sha256', required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--audit-sha256', required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-formats-only', action='store_true')
    parser.add_argument('--check-input-only', action='store_true')
    args = parser.parse_args()
    verified = validate_inputs(args)
    output = args.output.resolve()
    require(not output.exists() and output.parent.is_dir() and output not in verified['root'].parents and
            output != verified['root'] and verified['build_dir'] not in output.parents and
            verified['root'] / 'source_bundle' not in output.parents and
            verified['root'] / 'cpu_equivalence' not in output.parents, 'Choose a fresh output directory outside protected source/build inputs')
    if args.check_input_only:
        print(json.dumps({'status': 'FROZEN_CONTROL_FIXED_INPUTS_VERIFIED_NO_EXECUTION', 'native_runtime_imported': False,
              'engine_sha256': args.engine_sha256, 'lineage': verified['lineage'],
              'diagnostic_formats_only_opt_in': args.diagnostic_formats_only,
              'strict_integer_tactics_verified': verified['strict_integer_tactics_verified'],
              'strict_integer_gap_count': len(verified['strict_integer_gaps']), 'executions_planned': 2,
              'new_CPU_model_executions': 0, 'native_inference_executions': 0,
              'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE'}, indent=2), flush=True)
        return
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'Execution requires Jetson aarch64 Linux')
    # Replays need a separately frozen protocol. A different output directory
    # cannot silently turn this protocol's two calls into an additional run.
    marker = verified['root'] / ('fixed_input_attempt_' + args.protocol_sha256 + '.json')
    with marker.open('x', encoding='utf-8') as stream:
        json.dump({'protocol_sha256': args.protocol_sha256, 'engine_sha256': args.engine_sha256,
                   'pid': os.getpid(), 'output': str(output), 'maximum_engine_executions': 2,
                   'UTC': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.write('\n')
    output.mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM, terminate_as_interrupt)
    signal.signal(signal.SIGINT, terminate_as_interrupt)
    state = {'schema': SCHEMA, 'runner_sha256': sha(Path(__file__).resolve()), 'lineage': verified['lineage'],
             'started_UTC': datetime.now(timezone.utc).isoformat(), 'executions_attempted': 0,
             'executions_completed': 0, 'native_inference_executions': 0,
             'phase': 'INITIALIZING', 'dataset_evaluation_run': False,
             'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE'}
    write(output / 'execution_state.json', state)
    try:
        execute(args, verified, output, state)
    except BaseException as error:
        state.update(status='FIXED_INPUT_EXECUTION_FAILED', error_type=type(error).__name__, error=str(error),
                     traceback=traceback.format_exc(), numerical_migration_pass_claim=False,
                     accuracy_pass_claim=False, performance_pass_claim=False)
        write(output / 'smoke_failure.json', state)
        write(output / 'execution_state.json', state)
        raise
    finally:
        paths = [path for path in sorted(output.iterdir()) if path.is_file() and path.name != 'manifest.json']
        write(output / 'manifest.json', {'schema': 'rampq.gate05.control_fixed_input_files/v1',
              'files': {path.name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for path in paths},
              'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE'})
    print('CONTROL_FIXED_INPUT_TWO_EXECUTIONS_SAVED_NO_GATE_PASS_CLAIM', flush=True)


if __name__ == '__main__':
    main()
