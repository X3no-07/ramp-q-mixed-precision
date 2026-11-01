"""Serialize exactly 150 frozen full-model raw outputs; default is input check.

No warmup, build, NMS, target matching, accuracy scoring, reference reselection
or extra inference is implemented. Only --execute permits the chosen backend
to run. An interrupted attempt is preserved and cannot resume under the same
contract/configuration/backend namespace.
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


CONTRACT_SCHEMA = 'rampq.gate05.frozen_raw_dataset_contract/v1'
INPUT_SCHEMA = 'rampq.gate05.shared_preprocessed_inputs/v1'
RAW_SCHEMA = 'rampq.gate05.frozen_raw_dataset_outputs/v1'
PE_NAME = '/model.10/m/m.0/attn/pe/conv/Conv_output_0'
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


def terminate_as_interrupt(_signum, _frame):
    raise KeyboardInterrupt('External monitor requested termination')


def inspect_npy_file(path):
    with path.open('rb') as stream:
        header, offset = npy_header(stream)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    require(path.stat().st_size == offset + 4 * math.prod(header['shape']), 'NPY payload byte length differs')
    return header, digest.hexdigest()


def binding_signature(rows):
    require(len(rows) == len({row['name'] for row in rows}) == 3 and sum(row['input'] is True for row in rows) == 1,
            'Expected exactly one input and two uniquely named outputs')
    for row in rows:
        require(type(row['input']) is bool and row['dtype'] == 'DataType.FLOAT' and
                row['shape'] and all(type(value) is int and value > 0 for value in row['shape']), 'Expected static FLOAT bindings')
    required = {'images': (True, [1, 3, 640, 640]), 'output0': (False, [1, 14, 8400]),
                PE_NAME: (False, [1, 128, 20, 20])}
    require(set(required) == {row['name'] for row in rows} and
            all((row['input'], row['shape']) == required[row['name']] for row in rows), 'Full graph main/PE boundary differs')
    return sorted((row['name'], row['input'], tuple(row['shape']), row['dtype']) for row in rows)


def validate_inputs(args, permit_existing_output=False):
    contract_path, inputs = args.contract.resolve(), args.inputs.resolve()
    require(len(args.contract_sha256) == 64 and sha(contract_path) == args.contract_sha256, 'Final contract SHA256 differs')
    contract = read(contract_path)
    require(contract['schema'] == CONTRACT_SCHEMA and contract['runner_sha256'] == sha(Path(__file__).resolve()),
            'Wrong final raw contract or unpinned runner')
    require(args.configuration in contract['configurations'] and args.backend in ('ort', 'trt'), 'Unknown authorized configuration/backend')
    configuration = contract['configurations'][args.configuration]
    backend = configuration[args.backend]
    require(configuration['full_model'] is True and backend['authorized'] is True and backend['calls'] == 150,
            'This complete graph/backend is not authorized for exactly 150 calls')
    require(contract['maximum_attempts_per_backend_configuration'] == 1 and contract['warmup_calls'] == 0 and
            contract['extra_inference_calls'] == 0 and contract['new_builds'] == 0 and contract['no_NMS'] is True and
            contract['dataset_accuracy_evaluation_by_runner'] is False and contract['new_tolerances_defined'] is False and
            contract['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED' and contract['gate05_status'] == 'INCOMPLETE',
            'Frozen calls/build/postprocessing/acceptance scope differs')
    namespace = contract['attempt_namespace']
    require(isinstance(namespace, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', namespace), 'Unsafe or missing attempt namespace')
    require(args.graph.is_file() and not args.graph.is_symlink() and sha(args.graph) == configuration['graph_sha256'],
            'Exact full-model ONNX identity differs')
    if args.backend == 'trt':
        require(args.engine is not None and args.engine.is_file() and not args.engine.is_symlink() and
                sha(args.engine) == backend['engine_sha256'], 'Exact TensorRT engine identity differs')
        require(isinstance(backend['required_loaded_libraries'], list), 'Missing frozen loaded-library identity policy')
        for row in backend['required_loaded_libraries']:
            require(Path(row['basename']).name == row['basename'] and len(row['sha256']) == 64, 'Bad required library identity')
    else:
        require(args.engine is None and backend['provider'] == 'CPUExecutionProvider' and
                backend['graph_optimization'] == 'ORT_DISABLE_ALL' and backend['threads'] == 4,
                'ORT must execute the whole graph with CPUExecutionProvider/ORT_DISABLE_ALL/four threads')
    evidence_root = (args.evidence_root if args.evidence_root is not None else contract_path.parent).resolve()
    evidence = []
    for row in backend.get('required_evidence', []):
        path = contained(evidence_root, row['file'])
        require(path.is_file() and sha(path) == row['sha256'] and ('bytes' not in row or path.stat().st_size == row['bytes']),
                'Frozen backend qualification evidence differs: ' + row['file'])
        evidence.append({'file': row['file'], 'sha256': row['sha256']})
    expected = contract['bindings']
    binding_signature(expected)
    require(contract['output_npz_keys'] == {'output0': 'main_output', PE_NAME: 'pe_output'}, 'Frozen raw output names/keys differ')
    input_contract = contract['input_manifest']
    manifest_path = inputs / 'manifest.json'
    require(input_contract['schema'] == INPUT_SCHEMA and input_contract['count'] == 150 and
            sha(manifest_path) == input_contract['sha256'], 'Common prepared input manifest identity differs')
    manifest = read(manifest_path)
    require(manifest['schema'] == INPUT_SCHEMA and manifest['status'] == 'ALL150_COMMON_INPUTS_PREPARED_NO_MODEL_EXECUTION' and
            manifest['model_executions'] == 0 and len(manifest['images']) == 150, 'Common prepared inputs are incomplete or were model-selected')
    rows = manifest['images']
    require([row['image_index'] for row in rows] == list(range(150)) and
            [row['image'] for row in rows] == input_contract['ordered_ids'] and
            len({row['image'] for row in rows}) == len({row['input_file'] for row in rows}) ==
            len({row['image_sha256'] for row in rows}) == 150, 'Frozen 150-image order/uniqueness differs')
    for row in rows:
        path = contained(inputs, row['input_file'])
        require(path.is_file() and path.stat().st_size == row['input_bytes'] and sha(path) == row['input_sha256'] and
                row['shape'] == [1, 3, 640, 640] and row['dtype'] == 'float32', 'Prepared NPY identity differs: ' + row['image'])
        header, array_sha = inspect_npy_file(path)
        require(header['shape'] == (1, 3, 640, 640) and array_sha == row['array_sha256'], 'Prepared FLOAT32 payload differs: ' + row['image'])
        require(len(row['original_shape']) == 2 and all(type(value) is int and value > 0 for value in row['original_shape']),
                'Original image geometry missing from shared input row')
    state_root, output = args.state_root.resolve(), args.output.resolve()
    require((permit_existing_output or not output.exists()) and state_root != inputs and inputs not in state_root.parents and
            output != inputs and inputs not in output.parents and
            output not in args.graph.resolve().parents and (args.engine is None or output not in args.engine.resolve().parents),
            'Choose a fresh output and attempt store outside protected model/input files')
    # The immutable contract namespace, rather than the output directory, owns
    # one attempt for this configuration/backend. No resume or skip path exists.
    key = hashlib.sha256(args.configuration.encode('utf-8')).hexdigest()[:20] + '_' + args.backend
    attempt_directory = state_root / namespace
    require(attempt_directory != inputs and inputs not in attempt_directory.parents and output != attempt_directory,
            'Attempt markers must not modify the frozen input package or collide with output')
    marker = attempt_directory / (key + '_' + args.contract_sha256 + '.json')
    require(permit_existing_output or not marker.exists(), 'This frozen configuration/backend attempt was already claimed; no resume or retry')
    return {'contract': contract, 'configuration': configuration, 'backend': backend, 'rows': rows,
            'input_manifest': manifest, 'input_manifest_path': manifest_path, 'inputs': inputs,
            'state_root': state_root, 'output': output, 'attempt_directory': attempt_directory, 'marker': marker,
            'required_evidence': evidence,
            'lineage': {'contract_sha256': args.contract_sha256, 'runner_sha256': sha(Path(__file__).resolve()),
                        'configuration': args.configuration, 'backend': args.backend,
                        'graph_sha256': configuration['graph_sha256'],
                        'engine_sha256': backend.get('engine_sha256'), 'input_manifest_sha256': sha(manifest_path),
                        'source_data_manifest_sha256': manifest['data_manifest_sha256'],
                        'required_evidence': evidence}}


def capture_libraries(output, label):
    maps = Path('/proc/self/maps')
    if not maps.is_file():
        record = {'available': False, 'reason': '/proc/self/maps unavailable on this backend platform'}
    else:
        text = maps.read_text(encoding='utf-8')
        (output / ('maps_' + label + '.txt')).write_text(text, encoding='utf-8')
        record = dict(mapped_library_identity(text), available=True)
    write(output / ('loaded_libraries_' + label + '.json'), record)
    return record


def require_loaded_libraries(actual, required):
    for expected in required:
        matches = [row for row in actual.get('libraries', []) if
                   Path(row.get('real_path', row['mapped_path'])).name == expected['basename']]
        require(len(matches) == 1 and matches[0].get('hash_available') is True and
                matches[0]['sha256'] == expected['sha256'], 'Qualified loaded library missing/ambiguous/changed: ' + expected['basename'])


def actual_python_modules(prefixes):
    rows = []
    for name, module in sorted(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            continue
        value = getattr(module, '__file__', None)
        if value is None:
            continue
        path = Path(value)
        if path.is_file() and (path.suffix.lower() in ('.so', '.pyd', '.dll') or '.so.' in path.name):
            rows.append({'module': name, 'file': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    return rows


def check_environment_pin(actual, expected):
    for key, value in expected.items():
        require(actual.get(key) == value, 'Frozen backend environment differs: ' + key)


def make_ort_driver(args, verified, output, np):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.enable_profiling = False
    session = ort.InferenceSession(str(args.graph.resolve()), sess_options=options, providers=['CPUExecutionProvider'])
    require(session.get_providers() == ['CPUExecutionProvider'], 'Unexpected ORT provider/fallback')
    input_rows = [{'name': value.name, 'input': True, 'shape': list(value.shape), 'dtype': 'DataType.FLOAT'}
                  for value in session.get_inputs() if value.type == 'tensor(float)']
    output_rows = [{'name': value.name, 'input': False, 'shape': list(value.shape), 'dtype': 'DataType.FLOAT'}
                   for value in session.get_outputs() if value.type == 'tensor(float)']
    require(len(session.get_inputs()) == len(input_rows) == 1 and len(session.get_outputs()) == len(output_rows) == 2 and
            binding_signature(input_rows + output_rows) == binding_signature(verified['contract']['bindings']),
            'Actual full ORT model named bindings differ')
    names = [row['name'] for row in verified['contract']['bindings'] if not row['input']]
    environment = {'python': platform.python_version(), 'onnxruntime': ort.__version__, 'numpy': np.__version__,
                   'providers': session.get_providers(), 'graph_optimization': 'ORT_DISABLE_ALL',
                   'intra_op_threads': 4, 'inter_op_threads': 1, 'execution_mode': 'ORT_SEQUENTIAL',
                   'CPU_only_full_graph': True, 'runtime_modules': actual_python_modules(('onnxruntime',))}
    check_environment_pin(environment, verified['backend'].get('environment', {}))
    write(output / 'bindings.json', {'bindings': input_rows + output_rows})
    write(output / 'backend_environment.json', environment)

    def call(array):
        # Exactly one whole-graph session.run per input; no local reconstruction
        # or model-part/hybrid-provider execution is available.
        return dict(zip(names, session.run(names, {'images': array})))
    return call, environment


def make_trt_driver(args, verified, output, np):
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'TensorRT raw execution requires Jetson aarch64 Linux')
    import torch
    import tensorrt as trt
    require(torch.cuda.is_available() and trt.__version__.startswith('8.5.'), 'Expected CUDA/TensorRT8.5')
    torch.cuda.set_device(0)
    capability = list(torch.cuda.get_device_capability(0))
    require(capability == verified['backend'].get('compute_capability', [8, 7]), 'Qualified CUDA SM capability differs')
    properties = torch.cuda.get_device_properties(0)
    logger = trt.Logger(trt.Logger.WARNING)
    require(trt.init_libnvinfer_plugins(logger, ''), 'Cannot initialize TensorRT plugins')
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    require(engine is not None, 'Cannot deserialize frozen qualified engine')
    (output / 'runtime_engine_layers.json').write_text(
        engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON), encoding='utf-8')
    context = engine.create_execution_context()
    require(context is not None, 'Cannot create engine execution context')
    bindings = [{'name': engine.get_binding_name(index), 'input': bool(engine.binding_is_input(index)),
                 'shape': list(engine.get_binding_shape(index)), 'dtype': str(engine.get_binding_dtype(index)),
                 'format': str(engine.get_binding_format(index))} for index in range(engine.num_bindings)]
    require(binding_signature(bindings) == binding_signature(verified['contract']['bindings']) and
            all(row['format'] == 'TensorFormat.LINEAR' for row in bindings), 'Actual engine FLOAT/LINEAR names/shapes differ')
    write(output / 'bindings.json', {'bindings': bindings})
    buffers = [torch.empty(tuple(row['shape']), dtype=torch.float32, device='cuda') for row in bindings]
    input_indices = [index for index, row in enumerate(bindings) if row['input']]
    require(len(input_indices) == 1, 'Wrong engine input count')
    input_index = input_indices[0]
    torch.cuda.synchronize()
    environment = {'python': platform.python_version(), 'torch': torch.__version__, 'tensorrt': trt.__version__,
                   'numpy': np.__version__, 'torch_CUDA': torch.version.cuda, 'cuDNN': torch.backends.cudnn.version(),
                   'GPU': torch.cuda.get_device_name(0), 'compute_capability': capability,
                   'GPU_properties': repr(properties), 'runtime_modules': actual_python_modules(('torch', 'tensorrt'))}
    check_environment_pin(environment, verified['backend'].get('environment', {}))
    write(output / 'backend_environment.json', environment)

    def call(array):
        # Host input remains live through the blocking copy, and both stream
        # boundaries are synchronized before pointer reuse or host serialization.
        buffers[input_index].copy_(torch.from_numpy(array), non_blocking=False)
        torch.cuda.synchronize()
        require(context.execute_v2([int(value.data_ptr()) for value in buffers]), 'TensorRT execute_v2 returned false')
        torch.cuda.synchronize()
        return {row['name']: buffers[index].detach().cpu().numpy().copy()
                for index, row in enumerate(bindings) if not row['input']}
    return call, environment


def run(args, verified, output, state):
    import numpy as np
    capture_libraries(output, 'before_backend_initialization')
    if args.backend == 'trt':
        call, environment = make_trt_driver(args, verified, output, np)
    else:
        call, environment = make_ort_driver(args, verified, output, np)
    before_libraries = capture_libraries(output, 'before_first_model_call')
    if args.backend == 'trt':
        require_loaded_libraries(before_libraries, verified['backend']['required_loaded_libraries'])
    expected = {row['name']: row for row in verified['contract']['bindings'] if not row['input']}
    keys = verified['contract']['output_npz_keys']
    raw_directory = output / 'raw'
    raw_directory.mkdir(exist_ok=False)
    records, nonfinite_images = [], []
    state.update(phase='RUNNING', model_calls_attempted=0, model_calls_completed=0, serialized_images=0,
                 warmup_calls=0, extra_inference_calls=0, builds=0)
    write(output / 'run_state.json', state)
    started = time.monotonic()
    for row in verified['rows']:
        path = contained(verified['inputs'], row['input_file'])
        require(sha(path) == row['input_sha256'], 'Input became stale before its only model call')
        array = np.load(str(path), allow_pickle=False)
        require(array.dtype == np.float32 and array.shape == (1, 3, 640, 640) and array.flags.c_contiguous and
                bool(np.isfinite(array).all()) and hashlib.sha256(array.tobytes(order='C')).hexdigest() == row['array_sha256'],
                'Prepared input payload differs at execution')
        state.update(current_image_index=row['image_index'], current_image=row['image'])
        state['model_calls_attempted'] += 1
        write(output / 'run_state.json', state)
        call_started = time.perf_counter()
        arrays = call(array)
        call_ms = (time.perf_counter() - call_started) * 1000.0
        state['model_calls_completed'] += 1
        require(set(arrays) == set(expected), 'Backend returned unexpected main/PE output names')
        saved, statistics = {}, {}
        for name, value in arrays.items():
            require(value.dtype == np.float32 and list(value.shape) == expected[name]['shape'],
                    'Backend raw output shape/dtype differs: ' + name)
            saved[keys[name]] = np.ascontiguousarray(value)
            statistics[name] = dict(array_stats(value, np), raw_array_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
        raw_relative = 'raw/{:03d}.npz'.format(row['image_index'])
        raw_path = output / raw_relative
        with raw_path.open('xb') as stream:
            np.savez_compressed(stream, **saved)
        all_finite = all(value['finite'] for value in statistics.values())
        if not all_finite:
            nonfinite_images.append(row['image'])
        record = {'image_index': row['image_index'], 'image': row['image'], 'image_sha256': row['image_sha256'],
                  'label': row['label'], 'label_sha256': row['label_sha256'],
                  'input_file': row['input_file'], 'input_sha256': row['input_sha256'], 'input_array_sha256': row['array_sha256'],
                  'original_shape': row['original_shape'], 'ratio': row['ratio'], 'left': row['left'], 'top': row['top'],
                  'raw_file': raw_relative, 'raw_bytes': raw_path.stat().st_size, 'raw_sha256': sha(raw_path),
                  'output_npz_keys': keys, 'statistics': statistics, 'all_outputs_finite': all_finite,
                  'one_call_wall_ms_including_transfer_not_benchmark': call_ms}
        records.append(record)
        with (output / 'raw_progress.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        state['serialized_images'] += 1
        state['elapsed_seconds_not_benchmark'] = time.monotonic() - started
        write(output / 'run_state.json', state)
        write(output / 'raw_manifest.json', {'schema': RAW_SCHEMA, 'status': 'RAW_DATASET_IN_PROGRESS',
              'lineage': verified['lineage'], 'images': records, 'state': state})
        print('RAW_IMAGE_SAVED', row['image_index'], row['image'], 'calls', state['model_calls_completed'], flush=True)
    require(state['model_calls_attempted'] == state['model_calls_completed'] == state['serialized_images'] == len(records) == 150,
            'Exactly 150 frozen model calls and raw records are required')
    after_libraries = capture_libraries(output, 'after_last_model_call')
    if args.backend == 'trt':
        require_loaded_libraries(after_libraries, verified['backend']['required_loaded_libraries'])
    validate_inputs(args, permit_existing_output=True)
    state.update(phase='COMPLETE', status='ALL150_RAW_OUTPUTS_SERIALIZED_NO_SCORING',
                 nonfinite_images=nonfinite_images, all_outputs_finite=not nonfinite_images,
                 elapsed_seconds_not_benchmark=time.monotonic() - started,
                 accuracy_pass_claim=False, performance_pass_claim=False, new_tolerances_defined=False)
    write(output / 'run_state.json', state)
    write(output / 'raw_manifest.json', {'schema': RAW_SCHEMA, 'status': state['status'], 'lineage': verified['lineage'],
          'backend_environment': environment, 'bindings': verified['contract']['bindings'], 'output_npz_keys': keys,
          'images': records, 'state': state, 'nonfinite_images': nonfinite_images,
          'NMS_run': False, 'target_matching_run': False, 'dataset_accuracy_evaluated': False,
          'zero_detection_or_accuracy_PASS_predefined': False, 'new_tolerances_defined': False,
          'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
          'limits': 'Exactly one full-model call per frozen shared input. Raw tensors and descriptive finite statistics only. Nonfinite values are retained; no numerical/accuracy/performance/Gate PASS follows from serialization.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--backend', choices=('ort', 'trt'), required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--evidence-root', type=Path)
    parser.add_argument('--state-root', '--root', dest='state_root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check-input-only', action='store_true')
    args = parser.parse_args()
    verified = validate_inputs(args)
    if not args.execute:
        print(json.dumps({'status': 'FROZEN150_RAW_DATASET_INPUTS_VERIFIED_NO_MODEL_EXECUTION',
              'lineage': verified['lineage'], 'planned_calls': 150, 'model_calls': 0, 'native_runtime_imported': False,
              'new_builds': 0, 'NMS_run': False, 'accuracy_pass_claim': False, 'gate05_status': 'INCOMPLETE'}, indent=2), flush=True)
        return
    verified['attempt_directory'].mkdir(parents=True, exist_ok=True)
    with verified['marker'].open('x', encoding='utf-8', newline='\n') as stream:
        json.dump({'status': 'ONLY_FROZEN_BACKEND_CONFIGURATION_ATTEMPT_CLAIMED', 'lineage': verified['lineage'],
                   'maximum_calls': 150, 'output': str(verified['output']), 'pid': os.getpid(),
                   'UTC': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.write('\n')
    output = verified['output']
    output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, terminate_as_interrupt)
    signal.signal(signal.SIGINT, terminate_as_interrupt)
    boot = Path('/proc/sys/kernel/random/boot_id')
    state = {'schema': 'rampq.gate05.frozen_raw_dataset_state/v1', 'lineage': verified['lineage'],
             'phase': 'INITIALIZING', 'model_calls_attempted': 0, 'model_calls_completed': 0, 'serialized_images': 0,
             'pid': os.getpid(), 'platform': platform.platform(), 'machine': platform.machine(),
             'boot_id': boot.read_text().strip() if boot.is_file() else None,
             'started_UTC': datetime.now(timezone.utc).isoformat(), 'NMS_run': False,
             'accuracy_pass_claim': False, 'performance_pass_claim': False,
             'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE'}
    write(output / 'run_state.json', state)
    try:
        run(args, verified, output, state)
    except BaseException as error:
        state.update(phase='ABORTED_NO_RETRY', status='RAW_DATASET_ATTEMPT_INCOMPLETE_PRESERVED',
                     error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        write(output / 'run_failure.json', state)
        write(output / 'run_state.json', state)
        raise
    finally:
        files = [path for path in sorted(output.rglob('*')) if path.is_file() and path != output / 'manifest.json']
        write(output / 'manifest.json', {'schema': 'rampq.gate05.frozen_raw_dataset_files/v1',
              'files': {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path)} for path in files},
              'lineage': verified['lineage'], 'accuracy_pass_claim': False, 'gate05_status': 'INCOMPLETE'})
    print('FROZEN150_RAW_DATASET_SERIALIZED_NO_ACCURACY_OR_GATE_PASS_CLAIM', flush=True)


if __name__ == '__main__':
    main()
