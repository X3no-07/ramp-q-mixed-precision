"""Extract and CPU-check the exact two-Conv corrected q_all context.

Preparation and sealing are separate exclusive stages. Only the two small slices
execute on CPU; this tool never opens SSH or executes either complete detector.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
CORRECTED = ROOT / 'artifacts/gate05/20261019_corrected_qall_candidate'
ORIGINAL = ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles/q_all'
PROTOCOL = ROOT / 'artifacts/gate05/20261020_twoconv_context_protocol/protocol.json'
INVENTORY = ROOT / 'artifacts/gate05/20261019_corrected_qall_native/failure_boundary_inventory.json'
CHECKPOINT = ROOT / 'artifacts/gate05/20261019_corrected_qall_native/checkpoint.json'
OUTPUT = ROOT / 'artifacts/gate05/20261020_twoconv_context_probe'
ARCHIVE_ROOT = 'gate05_twoconv_context_probe_20261020'
PROTOCOL_SHA = '6ba4cae53e1ca115c2b49bd7fdac91d80c15ef1cc1f854c57f01cd09ae185fa0'
CORRECTED_SHA = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
CORRECTED_SPEC_SHA = 'f1df4d7353a83f97705ac0f049f7732e021e74dcc8e309fdb5adb0e67f8154bb'
CORRECTED_MANIFEST_SHA = 'f1679f4ebc7daa3e5589c28968785d48f31269cc5f83b0398bfa3bd7a91d861f'
ORIGINAL_SHA = '2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d'
ORIGINAL_MANIFEST_SHA = 'd7fb5fa24234ec38c11a88003a8cb5b7447257e4ca844e77e88247460524950b'
INPUT_SHA = '24240ec755e9a3c8b851b0702fbbbb14edcc55b52dad91633a7fc708cc4dbe94'
INVENTORY_SHA = 'a2780d83f1de24ab166553a8a3d341f04cf3c6f74ae41cabd5885124bc81d112'
CHECKPOINT_SHA = '66e149025bb0d777afe79a013797704c7878fdcad30140ac256aac305db48471'
PREFLIGHT_SHA = '505d139b56c973196792ef60b7bba95e82bca612fab3b12b6183a53f7a815f21'
CONVS = ['/model.0/conv/Conv', '/model.1/conv/Conv']
PREFIXES = ['rampq_model_0_conv', 'rampq_model_1_conv', 'rampq_model_2_cv1_conv']
CLIPS = [name + '_Clip' for name in PREFIXES]
OUTPUT_NAME = 'rampq_model_2_cv1_conv_a_dq'
CASE_ORDER = ['code_clip_none', 'code_clip_obey']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def message_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check_bundle(folder, manifest_sha):
    require(sha(folder / 'manifest.json') == manifest_sha, 'Frozen manifest differs: ' + str(folder))
    paths = [folder / 'manifest.json']
    for name, expected in read(folder / 'manifest.json')['files'].items():
        path = folder / name
        require(path.resolve().parent == folder.resolve() and not path.is_symlink() and
                path.is_file() and info(path) == expected, 'Frozen bundle differs: ' + str(path))
        paths.append(path)
    return paths


def source_hashes(paths):
    return {str(path.resolve().relative_to(ROOT)).replace('\\', '/'): info(path)
            for path in sorted(set(paths))}


def validate_recorded_sources(record):
    for relative, expected in record.items():
        path = ROOT / relative
        require(path.resolve().is_relative_to(ROOT) and path.is_file() and info(path) == expected,
                'Prepared source became stale: ' + relative)


def copied_value(model, name):
    values = {value.name: value for value in list(model.graph.input) +
              list(model.graph.value_info) + list(model.graph.output)}
    require(name in values, 'Missing original declared tensor: ' + name)
    return copy.deepcopy(values[name])


def extract(model, variant):
    import onnx
    from onnx import helper
    producers = {name: node for node in model.graph.node for name in node.output}
    initializers = {value.name: value for value in model.graph.initializer}
    needed, frontier = set(), [OUTPUT_NAME]
    while frontier:
        name = frontier.pop()
        if name in producers:
            node = producers[name]
            if node.name not in needed:
                needed.add(node.name)
                frontier.extend(item for item in node.input if item)
        else:
            require(name in initializers or name == 'images', 'Unexpected context dependency: ' + name)
    selected = [copy.deepcopy(node) for node in model.graph.node if node.name in needed]
    constant_names = {name for node in selected for name in node.input if name in initializers}
    constants = [copy.deepcopy(value) for value in model.graph.initializer if value.name in constant_names]
    require(len(selected) == 19 and len(constants) == 20 and
            [node.name for node in selected if node.op_type == 'Conv'] == CONVS and
            Counter(node.op_type for node in selected) ==
            Counter({'QuantizeLinear': 5, 'DequantizeLinear': 5, 'Clip': 3, 'Conv': 2, 'Sigmoid': 2, 'Mul': 2}),
            'Two-Conv context inventory changed')
    require('/model.2/cv1/conv/Conv' not in needed, 'Third Conv must be excluded')
    external, final = copied_value(model, 'images'), copied_value(model, OUTPUT_NAME)
    require(external.type.tensor_type.elem_type == final.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and
            [dim.dim_value for dim in external.type.tensor_type.shape.dim] == [1, 3, 640, 640] and
            [dim.dim_value for dim in final.type.tensor_type.shape.dim] == [1, 32, 160, 160],
            'Original tensor declarations differ')
    extracted = copy.deepcopy(model)
    extracted.graph.CopyFrom(helper.make_graph(selected, 'exact_twoconv_context_' + variant,
                                             [external], [final], constants))
    onnx.checker.check_model(extracted, full_check=True)
    source_nodes = {node.name: node for node in model.graph.node}
    require(all(node.SerializeToString() == source_nodes[node.name].SerializeToString()
                for node in extracted.graph.node), 'Extracted node bytes changed')
    require(all(value.SerializeToString() == initializers[value.name].SerializeToString()
                for value in extracted.graph.initializer), 'Extracted initializer bytes changed')
    return extracted


def check_rewrite(original, corrected):
    import numpy as np
    from onnx import numpy_helper
    old_nodes = {node.name: node for node in original.graph.node}
    new_nodes = {node.name: node for node in corrected.graph.node}
    old_init = {value.name: value for value in original.graph.initializer}
    new_init = {value.name: value for value in corrected.graph.initializer}
    require(old_nodes.keys() == new_nodes.keys() and old_init.keys() == new_init.keys(),
            'Original/corrected context names differ')
    boundary_details = []
    changed_nodes, changed_init = set(), set()
    for prefix in PREFIXES:
        q_name, clip_name, dq_name = prefix + '_ActivationQ', prefix + '_Clip', prefix + '_ActivationDQ'
        old_q, old_clip, old_dq = old_nodes[q_name], old_nodes[clip_name], old_nodes[dq_name]
        new_q, new_clip, new_dq = new_nodes[q_name], new_nodes[clip_name], new_nodes[dq_name]
        expected_q, expected_clip = copy.deepcopy(old_q), copy.deepcopy(old_clip)
        expected_q.input[0], expected_q.output[0] = old_clip.input[0], old_q.output[0] + '_before_code_clip'
        expected_clip.input[0], expected_clip.output[0] = expected_q.output[0], old_q.output[0]
        require(new_q.SerializeToString() == expected_q.SerializeToString() and
                new_clip.SerializeToString() == expected_clip.SerializeToString() and
                new_dq.SerializeToString() == old_dq.SerializeToString(), 'Unexpected activation rewrite')
        scale = numpy_helper.to_array(old_init[old_q.input[1]])
        zero = numpy_helper.to_array(old_init[old_q.input[2]])
        require(scale.dtype == np.float32 and scale.shape == () and zero.dtype == np.int8 and
                zero.shape == () and int(zero) == 0, 'Original scalar activation scale/zero differs')
        old_bounds = []
        for name, value in zip(old_clip.input[1:], (-127, 127)):
            old_array, new_array = numpy_helper.to_array(old_init[name]), numpy_helper.to_array(new_init[name])
            require(old_array.dtype == np.float32 and old_array.shape == () and
                    old_array.tobytes() == np.asarray(np.float32(value) * scale, dtype=np.float32).tobytes() and
                    new_array.dtype == np.int8 and new_array.shape == () and int(new_array) == value,
                    'Clip bounds differ from original frozen symmetric quantizer')
            changed_init.add(name)
            old_bounds.append(float(old_array))
        changed_nodes.update((q_name, clip_name))
        boundary_details.append({'prefix': prefix, 'clip_name': clip_name, 'q_name': q_name,
                                 'dq_name': dq_name, 'scale_initializer': old_q.input[1],
                                 'scale': float(scale), 'scale_serialized_sha256': message_sha(old_init[old_q.input[1]]),
                                 'zero_initializer': old_q.input[2], 'zero': int(zero),
                                 'original_float_clip_bounds': old_bounds, 'corrected_int8_clip_bounds': [-127, 127]})
    require(all(old_nodes[name].SerializeToString() == new_nodes[name].SerializeToString()
                for name in old_nodes if name not in changed_nodes), 'Unapproved non-boundary node change')
    require(all(old_init[name].SerializeToString() == new_init[name].SerializeToString()
                for name in old_init if name not in changed_init), 'Unapproved initializer change')
    require(len(changed_nodes) == len(changed_init) == 6, 'Unexpected rewrite scope')
    return boundary_details


def summary(array):
    import numpy as np
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'all_finite': bool(np.isfinite(array).all()),
            'min': float(array.min()), 'max': float(array.max()),
            'raw_array_bytes_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def prepare(output, protocol_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import onnx
    import onnxruntime as ort
    require(not output.exists() and not output.with_suffix('.zip').exists(), 'Preparation requires fresh paths')
    require(sha(protocol_path) == PROTOCOL_SHA, 'Root-frozen protocol changed')
    protocol = read(protocol_path)
    require(protocol['case_order'] == CASE_ORDER and protocol['batch_max_seconds'] == 900 and
            protocol['source_conv_names'] == CONVS and protocol['clip_names'] == CLIPS,
            'Protocol context or budget changed')
    paths = check_bundle(CORRECTED, CORRECTED_MANIFEST_SHA)
    paths.extend(check_bundle(ORIGINAL, ORIGINAL_MANIFEST_SHA))
    pinned = {CORRECTED / 'model.onnx': CORRECTED_SHA, CORRECTED / 'spec.json': CORRECTED_SPEC_SHA,
              CORRECTED / 'preprocessed_input.npy': INPUT_SHA, CORRECTED / 'preflight.json': PREFLIGHT_SHA,
              ORIGINAL / 'model.onnx': ORIGINAL_SHA, INVENTORY: INVENTORY_SHA, CHECKPOINT: CHECKPOINT_SHA,
              protocol_path: PROTOCOL_SHA}
    for path, expected in pinned.items():
        require(sha(path) == expected, 'Pinned context source differs: ' + str(path))
    paths.extend([*pinned, Path(__file__).resolve()])
    before = source_hashes(paths)
    models = {'original': extract(onnx.load(str(ORIGINAL / 'model.onnx')), 'original'),
              'code_clip': extract(onnx.load(str(CORRECTED / 'model.onnx')), 'code_clip')}
    boundaries = check_rewrite(models['original'], models['code_clip'])
    require(boundaries[-1]['scale'] == 1.5619523525238037, 'Third activation scale changed')
    output.mkdir(parents=True, exist_ok=False)
    copies = {CORRECTED / 'preprocessed_input.npy': 'preprocessed_input.npy',
              CORRECTED / 'model.onnx': 'source_model.onnx',
              CORRECTED / 'spec.json': 'parent_spec.json', CORRECTED / 'manifest.json': 'parent_manifest.json',
              CORRECTED / 'preflight.json': 'original_failed_preflight.json',
              ORIGINAL / 'spec.json': 'original_qall_spec.json', ORIGINAL / 'manifest.json': 'original_qall_manifest.json',
              INVENTORY: 'failure_inventory.json', CHECKPOINT: 'retained_qall_failure_checkpoint.json',
              protocol_path: 'protocol.json', Path(__file__).resolve(): Path(__file__).name}
    for path, name in copies.items():
        shutil.copy2(path, output / name)
    x = np.load(output / 'preprocessed_input.npy', allow_pickle=False)
    require(sha(output / 'preprocessed_input.npy') == INPUT_SHA and x.dtype == np.float32 and
            x.shape == (1, 3, 640, 640) and x.flags.c_contiguous and np.isfinite(x).all(), 'Invalid frozen input')
    arrays, variants, executions = {}, {}, {}
    for variant, model in models.items():
        filename = 'model_twoconv_' + variant + '.onnx'
        onnx.save(model, str(output / filename))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 4
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(output / filename), sess_options=options, providers=['CPUExecutionProvider'])
        require(session.get_providers() == ['CPUExecutionProvider'] and
                [value.name for value in session.get_inputs()] == ['images'] and
                [value.name for value in session.get_outputs()] == [OUTPUT_NAME], 'Unexpected CPU provider or I/O')
        result = session.run([OUTPUT_NAME], {'images': x})[0]
        require(result.dtype == np.float32 and result.shape == (1, 32, 160, 160) and np.isfinite(result).all(),
                'Invalid slice CPU output')
        key = variant + '_output'
        arrays[key] = result.copy()
        raw_name = 'cpu_actual_' + variant + '.npy'
        np.save(output / raw_name, result)
        executions[variant] = {'model_file': filename, 'providers': session.get_providers(),
                               'graph_optimization': 'ORT_DISABLE_ALL', 'runs': 1,
                               'raw_output_file': raw_name, 'raw_output_sha256': sha(output / raw_name),
                               'summary': summary(result)}
        variants[variant] = {'model_file': filename, 'model_sha256': sha(output / filename),
                             'expected_output_npz_keys': {OUTPUT_NAME: key}, 'nodes': 19, 'initializers': 20,
                             'selected_node_serialized_sha256': {node.name: message_sha(node) for node in model.graph.node},
                             'selected_initializer_serialized_sha256': {value.name: message_sha(value) for value in model.graph.initializer}}
        del session
    np.savez_compressed(output / 'cpu_expected_outputs.npz', **arrays)
    equal = arrays['original_output'].tobytes(order='C') == arrays['code_clip_output'].tobytes(order='C')
    difference = arrays['original_output'].astype(np.float64) - arrays['code_clip_output'].astype(np.float64)
    diagnostic = {'schema': 'rampq.gate05.twoconv_context_cpu/v1',
                  'status': 'SLICE_CPU_BYTE_EQUIVALENCE_PASSED' if equal else 'SLICE_CPU_BYTE_EQUIVALENCE_FAILED',
                  'slice_executions': 2, 'complete_detector_executions': 0,
                  'executions': executions, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__,
                  'both_outputs_byte_exact': equal, 'differing_elements': int(np.count_nonzero(difference)),
                  'max_absolute_difference': float(np.abs(difference).max()),
                  'source_corrected_graph_sha256': CORRECTED_SHA, 'source_original_graph_sha256': ORIGINAL_SHA,
                  'original_complete_model_guard': 'FAILED_UNCHANGED', 'native_executions': 0,
                  'dataset_evaluation_run': False, 'tolerances_changed': False}
    write(output / 'cpu_diagnostic.json', diagnostic)
    write(output / 'graph_identity_audit.json', {
        'schema': 'rampq.gate05.twoconv_context_graph_identity/v1',
        'status': 'EXACT_SOURCE_DEPENDENCY_CLOSURES_VERIFIED', 'conv_names': CONVS,
        'nodes_per_variant': 19, 'initializers_per_variant': 20,
        'all_extracted_nodes_and_initializers_exact_source_serialized_bytes': True,
        'corrected_vs_original_only_three_quantizer_rewrites': True,
        'source_graphs': {'original': ORIGINAL_SHA, 'code_clip': CORRECTED_SHA},
        'variant_identity': variants, 'activation_boundaries': boundaries,
        'third_conv_excluded': True, 'single_output_without_extra_taps': True,
        'original_complete_model_guard': 'FAILED_UNCHANGED'})
    spec = {'schema': 'rampq.gate05.twoconv_context_probe/v1',
            'prepared_client_at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
            'source_configuration': 'q_all', 'source_graph_sha256': CORRECTED_SHA,
            'source_graph_file': 'source_model.onnx',
            'source_original_qall_graph_sha256': ORIGINAL_SHA,
            'parent_spec_file': 'parent_spec.json', 'parent_spec_sha256': CORRECTED_SPEC_SHA,
            'parent_manifest_file': 'parent_manifest.json', 'parent_manifest_sha256': CORRECTED_MANIFEST_SHA,
            'protocol_file': 'protocol.json', 'protocol_sha256': PROTOCOL_SHA,
            'input_file': 'preprocessed_input.npy', 'input_file_sha256': INPUT_SHA,
            'input_binding': {'name': 'images', 'input': True, 'shape': [1, 3, 640, 640], 'dtype': 'DataType.FLOAT'},
            'output_bindings_in_order': [{'name': OUTPUT_NAME, 'input': False, 'shape': [1, 32, 160, 160],
                                          'dtype': 'DataType.FLOAT', 'role': 'third_conv_input_dequantized_activation', 'kind': 'primary'}],
            'conv_names': CONVS, 'source_conv_names': CONVS, 'excluded_third_conv_name': '/model.2/cv1/conv/Conv',
            'clip_names': CLIPS, 'source_activation_boundaries': boundaries,
            'variants': {'code_clip': variants['code_clip']}, 'cpu_control_variants': variants,
            'cpu_expected_outputs_file': 'cpu_expected_outputs.npz',
            'cpu_expected_outputs_sha256': sha(output / 'cpu_expected_outputs.npz'),
            'cpu_diagnostic_file': 'cpu_diagnostic.json', 'cpu_diagnostic_sha256': sha(output / 'cpu_diagnostic.json'),
            'graph_identity_audit_file': 'graph_identity_audit.json',
            'graph_identity_audit_sha256': sha(output / 'graph_identity_audit.json'),
            'cases': copy.deepcopy(protocol['cases']), 'case_order': CASE_ORDER,
            'max_seconds': 900, 'max_build_seconds': 300, 'maximum_attempts_per_case': 1,
            'maximum_native_builds': 2, 'maximum_native_inferences': 4, 'executions_per_successful_case': 2,
            'native_importer_clip_attribute_correction': {
                'variant': 'code_clip', 'layer_names': CLIPS, 'activation_alpha': -127.0, 'activation_beta': 127.0,
                'when': 'After native ONNX parsing and before build; retain all parser precision/output constraints.',
                'reason': 'Preserve actual frozen INT8 Clip scalar values in the TensorRT 8.5 importer.'},
            'native_build_policy': copy.deepcopy(protocol['common_native_settings']),
            'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
            'corrected_full_qall_failure': 'FAILED_TENSORRT_FORMAT_SELECTION_NO_ENGINE_UNCHANGED',
            'retained_qall_failure_checkpoint_sha256': CHECKPOINT_SHA,
            'failure_boundary_inventory_sha256': INVENTORY_SHA,
            'original_tolerances_changed': False, 'calibration_recomputed': False, 'weights_changed': False,
            'dataset_evaluation_run': False, 'complete_detector_executions': 0,
            'native_build_or_execution_run_by_preparer': False,
            'limits': protocol['interpretation']}
    write(output / 'spec_draft.json', spec)
    write(output / 'source_hashes_before.json', before)
    after = source_hashes(paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'Frozen source changed during slice preparation')
    require(equal, 'Slice CPU byte equivalence failed; raw evidence retained and sealing forbidden')
    (output / 'README.md').write_text('# Precise context probes at two Conv and three activation boundary locations\n\nExtract the dependency closure of 19 nodes and 20 initializer dependencies from a prior correction of q_all that failed.\nInput images FLOAT[1,3,640,640], unique output is the original third Conv input\nrampq_model_2_cv1_conv_a_dq FLOAT[1,32,160,160];contains the first two Conv layers and the original SiLU\nNo third Conv, no addition of intermediate output taps. All selected nodes and constants retain their original graph serialization bytes.\n\nThe matching original q_all exact slice is used only as a new CPU semantic control; both slices use the fixed 000020\ninput for one CPUExecutionProvider/ORT_DISABLE_ALL run each and must produce byte-identical outputs.\nNative execution builds only the code_clip topology, comparing none/obey cases; allow one attempt per case, limited to 300 seconds,\nwith a batch limit of 900 seconds. After success, repeat execution twice with the fixed input. Correct only the three original named Clip alpha/beta values\nto -127/+127, without setting or clearing parser layer precision/output type constraints.\n\nThis probe adds the second Conv and third activation boundaries for local context validation based on log evidence;\nfailure cannot be assumed to vary monotonically with prefix length, and added nodes cannot be identified as causes. The old full guardrail remains\nFAILED_UNCHANGED: previous complete q_all construction failure is retained; no re-construction of the full model is performed;\nno dataset evaluation is conducted, thus no Gate 0.5 pass or proof of accuracy/performance is established;\n', encoding='utf-8')
    write(output / 'prepared_content_manifest.json', {
        'schema': 'rampq.gate05.twoconv_context_prepared_content/v1',
        'status': 'PREPARED_CPU_EQUIVALENT_NOT_SEALED',
        'files': {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}})
    print(json.dumps({'status': 'PREPARED_CPU_EQUIVALENT_NOT_SEALED', 'output': str(output),
                      'source_bytes_unchanged': True, 'cpu_byte_exact': True,
                      'variants': {name: {'model_file': value['model_file'], 'model_sha256': value['model_sha256']}
                                   for name, value in variants.items()}, 'cpu_diagnostic': diagnostic}, indent=2))


def seal(output, native_paths):
    require(output.is_dir() and not output.with_suffix('.zip').exists() and
            not (output / 'manifest.json').exists() and not (output / 'spec.json').exists(),
            'Sealing requires prepared, previously unsealed paths')
    content = read(output / 'prepared_content_manifest.json')
    require(content['status'] == 'PREPARED_CPU_EQUIVALENT_NOT_SEALED', 'Wrong preparation status')
    for name, expected in content['files'].items():
        path = output / name
        require(path.resolve().parent == output and info(path) == expected, 'Prepared content changed: ' + name)
    sources = read(output / 'source_hashes_before.json')
    require(sources == read(output / 'source_hashes_after.json'), 'Preparation source records differ')
    validate_recorded_sources(sources)
    require(read(output / 'cpu_diagnostic.json')['both_outputs_byte_exact'], 'CPU equivalence must pass')
    require(sha(output / 'protocol.json') == PROTOCOL_SHA, 'Frozen protocol changed')
    require(len(native_paths) == 2 and len({path.name for path in native_paths}) == 2 and
            all(path.parent == ROOT / 'scripts' and path.is_file() and not path.is_symlink() for path in native_paths),
            'Native scripts must be two separate scripts in scripts/')
    native_before = source_hashes(native_paths)
    for path in native_paths:
        shutil.copy2(path, output / path.name)
    spec = read(output / 'spec_draft.json')
    spec['native_runner_file'], spec['native_builder_file'] = [path.name for path in native_paths]
    spec['native_runner_sha256'], spec['native_builder_sha256'] = [sha(path) for path in native_paths]
    spec['sealed_client_at'] = datetime.now().astimezone().isoformat()
    write(output / 'spec.json', spec)
    write(output / 'seal_native_source_hashes_before.json', native_before)
    native_after = source_hashes(native_paths)
    write(output / 'seal_native_source_hashes_after.json', native_after)
    require(native_before == native_after, 'Native source changed while sealing')
    validate_recorded_sources(sources)
    files = {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}
    require(len(files) == len(list(output.iterdir())), 'Unexpected bundle subdirectory')
    write(output / 'manifest.json', {'schema': 'rampq.gate05.twoconv_context_probe_bundle/v1', 'files': files})
    sums = dict(files)
    sums['manifest.json'] = info(output / 'manifest.json')
    (output / 'SHA256SUMS').write_text(''.join(value['sha256'] + '  ' + name + '\n'
                                            for name, value in sorted(sums.items())), encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, ARCHIVE_ROOT + '/' + path.name)
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, 'ZIP CRC failed')
        for path in output.iterdir():
            require(hashlib.sha256(archive.read(ARCHIVE_ROOT + '/' + path.name)).hexdigest() == sha(path),
                    'ZIP payload bytes changed: ' + path.name)
    print(json.dumps({'status': 'TWOCONV_CONTEXT_STATIC_AND_CPU_BUNDLE_SEALED', 'output': str(output),
                      'archive': str(archive_path), 'archive_bytes': archive_path.stat().st_size,
                      'archive_sha256': sha(archive_path), 'spec_sha256': sha(output / 'spec.json'),
                      'manifest_sha256': sha(output / 'manifest.json'), 'protocol_sha256': PROTOCOL_SHA,
                      'native_runner_sha256': spec['native_runner_sha256'],
                      'native_builder_sha256': spec['native_builder_sha256'],
                      'source_bytes_unchanged': True, 'complete_detector_executions': 0}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--prepare-only', action='store_true')
    stage.add_argument('--seal-only', action='store_true')
    parser.add_argument('--native-runner', type=Path, default=ROOT / 'scripts/jetson_twoconv_context_probe.py')
    parser.add_argument('--native-builder', type=Path, default=ROOT / 'scripts/jetson_build_twoconv_context_probe.py')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / 'artifacts/gate05').resolve()), 'Unsafe artifact destination')
    if args.prepare_only:
        prepare(output, args.protocol.resolve())
    else:
        seal(output, [args.native_runner.resolve(), args.native_builder.resolve()])


if __name__ == '__main__':
    main()
