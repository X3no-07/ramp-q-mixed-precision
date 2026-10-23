"""Statically prepare original DFL topology controls from authenticated old bytes.

No CPU inference, fixture generation, board connection, or native execution is
performed. Preparation and sealing have separate exclusive output states.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'artifacts/gate05/20261020_dfl_context_probe'
ORIGINAL = ROOT / 'artifacts/gate05/20261019_precision_runtime_bundles/q_all'
CHECKPOINT = ROOT / 'artifacts/gate05/20261020_dfl_context_native/checkpoint.json'
PARENT_NATIVE_ARCHIVE = ROOT / 'artifacts/gate05/20261020_dfl_context_native/native_evidence.zip'
PARENT_INPUT_ARCHIVE = ROOT / 'artifacts/gate05/20261020_dfl_context_probe.zip'
PROTOCOL = ROOT / 'artifacts/gate05/20261020_dfl_original_protocol/protocol.json'
OUTPUT = ROOT / 'artifacts/gate05/20261020_dfl_original_probe'
ARCHIVE_ROOT = 'gate05_dfl_original_probe_20261020'
PROTOCOL_SHA = '81200cd4900a35f0533c45c7542f6dacc353203e8338a09b47cd24465cc0f6ed'
PARENT_SPEC_SHA = '934e075aa58dde1829b600d3d011f5ad51c4f1da1af78f7791944f492781b14d'
PARENT_MANIFEST_SHA = '3836e82ed53afb14fe16d3b9431864c0c4a45fd0d3418c7c06a4f6fcd46ed1d0'
PARENT_CHECKPOINT_SHA = '28edf81c0be1f72065a8952f84dc31330ba95709d50ed13aafc16a5bb56c602b'
PARENT_NATIVE_ARCHIVE_SHA = 'a53750401ccdbb7c008c14e75cadec80a586a05f2baa32f5e985abdff527bd24'
PARENT_INPUT_ARCHIVE_SHA = '05b9b50e4ddc294151121044facef5e1711ac368c793ca5232c1c1a557c42b9f'
ORIGINAL_SHA = '2feb721523cb8d960ceb5d74bd6689fd2f33a58f7d5761587b2d346d0e96eb8d'
ORIGINAL_MANIFEST_SHA = 'd7fb5fa24234ec38c11a88003a8cb5b7447257e4ca844e77e88247460524950b'
CORRECTED_SHA = '755b27a8f135e50adbf626e2f77ccd6745302b3577a0ba2873c10792dbfa08ee'
MODEL_SHA = 'b60be2c2d196ed1fb4e962446f374d9e53d066476172cfa1db415194b2306d90'
FIXTURE_SHA = 'cb1e437ef9c67af95ab66dfe7e8db5f28ebbfc437835502f1da1ac816e414dab'
CPU_NPZ_SHA = 'fe6e4048681a5ac252b430b850cd27129061f8f1602835a8723ca7d827d59511'
CPU_RAW_SHA = 'fd4bcdcee24cd53651fdd7a508423c802f3b432c759647204c7b748f22abd50d'
FIXTURE_LINEAGE_SHA = 'ecd2bb92223c5e3d820cd70c0bacc5f1b9fc42b40a545f0552b0c247e45e4365'
INPUT_NAME = '/model.23/dfl/Transpose_output_0'
OUTPUT_NAME = '/model.23/dfl/Reshape_1_output_0'
CONVS = ['/model.23/dfl/conv/Conv']
CLIPS = ['rampq_model_23_dfl_conv_Clip']
CASE_ORDER = ['original_none', 'original_obey']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def info(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def message_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def check_bundle(folder, expected_manifest_sha):
    require(sha(folder / 'manifest.json') == expected_manifest_sha, 'Frozen source manifest changed')
    paths = [folder / 'manifest.json']
    for name, expected in read(folder / 'manifest.json')['files'].items():
        path = folder / name
        require(path.resolve().parent == folder.resolve() and path.is_file() and
                not path.is_symlink() and info(path) == expected, 'Frozen source differs: ' + str(path))
        paths.append(path)
    return paths


def source_hashes(paths):
    return {str(path.resolve().relative_to(ROOT)).replace('\\', '/'): info(path)
            for path in sorted(set(paths))}


def validate_recorded_sources(record):
    for name, expected in record.items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT) and path.is_file() and info(path) == expected,
                'Prepared source became stale: ' + name)


def audit_exact_original_tail(source, tail):
    import onnx
    from onnx import numpy_helper
    import numpy as np
    onnx.checker.check_model(tail, full_check=True)
    nodes = {node.name: node for node in source.graph.node}
    producers = {name: node for node in source.graph.node for name in node.output}
    initializers = {value.name: value for value in source.graph.initializer}
    needed, frontier = set(), [OUTPUT_NAME]
    while frontier:
        name = frontier.pop()
        if name == INPUT_NAME or name in initializers:
            continue
        require(name in producers, 'Unexpected original tail dependency: ' + name)
        node = producers[name]
        if node.name not in needed:
            needed.add(node.name)
            frontier.extend(value for value in node.input if value)
    selected = [node for node in source.graph.node if node.name in needed]
    constant_names = {name for node in selected for name in node.input if name in initializers}
    require(len(selected) == 9 and len(constant_names) == 7 and
            [node.name for node in selected] == [node.name for node in tail.graph.node] and
            Counter(node.op_type for node in selected) == Counter({
                'Softmax': 1, 'QuantizeLinear': 2, 'DequantizeLinear': 2,
                'Clip': 1, 'Conv': 1, 'Constant': 1, 'Reshape': 1}), 'Original exact closure inventory differs')
    require([node.name for node in tail.graph.node if node.op_type == 'Conv'] == CONVS and
            [node.name for node in tail.graph.node if node.op_type == 'Clip'] == CLIPS,
            'Unexpected original DFL Conv or Clip')
    require(all(node.SerializeToString() == nodes[node.name].SerializeToString() for node in tail.graph.node),
            'Copied original tail node bytes differ from original full q_all')
    require({value.name for value in tail.graph.initializer} == constant_names and
            all(value.SerializeToString() == initializers[value.name].SerializeToString()
                for value in tail.graph.initializer), 'Copied original initializer bytes differ from original full q_all')
    source_values = {value.name: value for value in list(source.graph.input) +
                     list(source.graph.value_info) + list(source.graph.output)}
    require(len(tail.graph.input) == len(tail.graph.output) == 1 and
            tail.graph.input[0].name == INPUT_NAME and tail.graph.output[0].name == OUTPUT_NAME and
            tail.graph.input[0].SerializeToString() == source_values[INPUT_NAME].SerializeToString() and
            tail.graph.output[0].SerializeToString() == source_values[OUTPUT_NAME].SerializeToString(),
            'Original tail input/output declarations differ from original full q_all')
    require([d.dim_value for d in tail.graph.input[0].type.tensor_type.shape.dim] == [1, 16, 4, 8400] and
            [d.dim_value for d in tail.graph.output[0].type.tensor_type.shape.dim] == [1, 4, 8400], 'Tail shape differs')
    clip, q, dq = (nodes['rampq_model_23_dfl_conv' + suffix] for suffix in ('_Clip', '_ActivationQ', '_ActivationDQ'))
    require(q.input[0] == clip.output[0] and dq.input[0] == q.output[0] and
            clip.input[0] == '/model.23/dfl/Softmax_output_0', 'Expected original FLOAT Clip -> Q -> DQ')
    bounds = [numpy_helper.to_array(initializers[name]) for name in clip.input[1:]]
    scale = numpy_helper.to_array(initializers[q.input[1]])
    require(all(value.dtype == np.float32 and value.shape == () for value in bounds) and
            scale.dtype == np.float32 and scale.shape == () and float(scale) == 0.007873943075537682 and
            all(value.tobytes() == np.asarray(np.float32(code) * scale, dtype=np.float32).tobytes()
                for value, code in zip(bounds, (-127, 127))), 'Original frozen FLOAT Clip bounds changed')
    return {'schema': 'rampq.gate05.dfl_original_source_identity/v1',
        'status': 'REUSED_ORIGINAL_TAIL_EXACT_DEPENDENCY_CLOSURE_VERIFIED',
        'source_original_full_graph_sha256': ORIGINAL_SHA, 'reused_original_tail_sha256': MODEL_SHA,
        'nodes': 9, 'initializers': 7, 'conv_names': CONVS, 'clip_names': CLIPS,
        'node_serialized_sha256': {node.name: message_sha(node) for node in tail.graph.node},
        'initializer_serialized_sha256': {value.name: message_sha(value) for value in tail.graph.initializer},
        'all_selected_nodes_initializers_and_io_declarations_exact_source_bytes': True,
        'topology': 'ORIGINAL_FLOAT_CLIP_Q_DQ', 'activation_scale': float(scale),
        'original_float_clip_bounds': [float(value) for value in bounds],
        'native_clip_attribute_mutation_permitted': False, 'new_CPU_executions': 0}


def summary(array):
    import numpy as np
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'all_finite': bool(np.isfinite(array).all()),
            'min': float(array.min()), 'max': float(array.max()),
            'raw_array_bytes_sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def prepare(output, protocol_path):
    import numpy as np
    import onnx
    require(not output.exists() and not output.with_suffix('.zip').exists(), 'Fresh paths required; preserve old packages')
    require(sha(protocol_path) == PROTOCOL_SHA, 'Root-frozen protocol changed')
    protocol = read(protocol_path)
    require(protocol['schema'] == 'rampq.gate05.dfl_original_protocol/v1' and
            protocol['case_order'] == CASE_ORDER and protocol['batch_max_seconds'] == 900 and
            protocol['source_conv_names'] == CONVS and protocol['clip_names'] == CLIPS and
            protocol['CPU_execution_runs_this_stage'] == 0, 'Protocol scope differs')
    expected_cases = [{'id': name, 'variant': 'original',
        'precision_constraint_flag': 'none' if name.endswith('_none') else 'obey',
        'max_build_seconds': 300, 'iterations': 1, 'apply_native_clip_attribute_correction': False}
        for name in CASE_ORDER]
    require(protocol['cases'] == expected_cases, 'Frozen original cases changed')
    source_paths = check_bundle(PARENT, PARENT_MANIFEST_SHA)
    source_paths.extend(check_bundle(ORIGINAL, ORIGINAL_MANIFEST_SHA))
    pinned = {PARENT / 'spec.json': PARENT_SPEC_SHA, PARENT / 'model_dfl_original.onnx': MODEL_SHA,
              PARENT / 'dfl_transpose_fixture.npy': FIXTURE_SHA, PARENT / 'cpu_expected_outputs.npz': CPU_NPZ_SHA,
              PARENT / 'cpu_actual_original.npy': CPU_RAW_SHA, PARENT / 'cpu_actual_code_clip.npy': CPU_RAW_SHA,
              PARENT / 'fixture_lineage.json': FIXTURE_LINEAGE_SHA,
              ORIGINAL / 'model.onnx': ORIGINAL_SHA, CHECKPOINT: PARENT_CHECKPOINT_SHA,
              PARENT_NATIVE_ARCHIVE: PARENT_NATIVE_ARCHIVE_SHA,
              PARENT_INPUT_ARCHIVE: PARENT_INPUT_ARCHIVE_SHA, protocol_path: PROTOCOL_SHA}
    checkpoint = read(CHECKPOINT)
    for relative, expected_sha in checkpoint['protected_identities'].items():
        pinned[ROOT / relative] = expected_sha
    for path, expected in pinned.items():
        require(sha(path) == expected, 'Pinned historical source differs: ' + str(path))
    source_paths.extend([*pinned, Path(__file__).resolve()])
    before = source_hashes(source_paths)
    tail = onnx.load(str(PARENT / 'model_dfl_original.onnx'))
    audit = audit_exact_original_tail(onnx.load(str(ORIGINAL / 'model.onnx')), tail)
    fixture = np.load(PARENT / 'dfl_transpose_fixture.npy', allow_pickle=False)
    require(fixture.dtype == np.float32 and fixture.shape == (1, 16, 4, 8400) and
            fixture.flags.c_contiguous and np.isfinite(fixture).all(), 'Frozen reused fixture invalid')
    with np.load(PARENT / 'cpu_expected_outputs.npz', allow_pickle=False) as arrays:
        require(set(arrays.files) == {'original_output', 'code_clip_output'}, 'Expected CPU NPZ keys changed')
        original_output, code_clip_output = arrays['original_output'], arrays['code_clip_output']
    require(original_output.dtype == code_clip_output.dtype == np.float32 and
            original_output.shape == code_clip_output.shape == (1, 4, 8400) and
            np.isfinite(original_output).all() and np.isfinite(code_clip_output).all() and
            original_output.tobytes(order='C') == code_clip_output.tobytes(order='C'),
            'Saved original/corrected same-fixture CPU outputs not byte identical')
    for name in ('cpu_actual_original.npy', 'cpu_actual_code_clip.npy'):
        saved = np.load(PARENT / name, allow_pickle=False)
        require(saved.dtype == original_output.dtype and saved.shape == original_output.shape and
                saved.tobytes(order='C') == original_output.tobytes(order='C'), 'Saved CPU NPY differs from NPZ')
    parent_spec = read(PARENT / 'spec.json')
    lineage = read(PARENT / 'fixture_lineage.json')
    require(parent_spec['input_file_sha256'] == FIXTURE_SHA and
            lineage['fixture_sha256'] == FIXTURE_SHA and lineage['source_graph_sha256'] == CORRECTED_SHA and
            lineage['ancestor_cpu_executions'] == 1 and lineage['ancestor_conv_count'] == 72,
            'Authenticated inherited fixture origin changed')
    output.mkdir(parents=True, exist_ok=False)
    copy_map = {PARENT / 'model_dfl_original.onnx': 'model_dfl_original.onnx',
        PARENT / 'dfl_transpose_fixture.npy': 'dfl_transpose_fixture.npy',
        PARENT / 'cpu_expected_outputs.npz': 'cpu_expected_outputs.npz',
        PARENT / 'cpu_actual_original.npy': 'cpu_actual_original.npy',
        PARENT / 'cpu_actual_code_clip.npy': 'cpu_actual_code_clip.npy',
        PARENT / 'cpu_diagnostic.json': 'parent_cpu_diagnostic.json',
        PARENT / 'fixture_lineage.json': 'fixture_lineage.json',
        PARENT / 'spec.json': 'parent_dfl_spec.json', PARENT / 'manifest.json': 'parent_dfl_manifest.json',
        CHECKPOINT: 'parent_dfl_checkpoint.json', PARENT / 'protocol.json': 'parent_dfl_protocol.json',
        ORIGINAL / 'model.onnx': 'source_original_full_model.onnx',
        ORIGINAL / 'manifest.json': 'original_qall_manifest.json', ORIGINAL / 'spec.json': 'original_qall_spec.json',
        PARENT / 'original_failed_preflight.json': 'original_failed_preflight.json',
        protocol_path: 'protocol.json', Path(__file__).resolve(): Path(__file__).name}
    for path, name in copy_map.items():
        shutil.copy2(path, output / name)
    write(output / 'graph_identity_audit.json', audit)
    reuse_audit = {'schema': 'rampq.gate05.dfl_original_static_reuse_audit/v1',
        'status': 'AUTHENTICATED_REUSE_STATIC_CHECKS_PASSED', 'new_CPU_executions': 0,
        'new_ORT_executions': 0, 'new_Torch_executions': 0, 'new_upstream_executions': 0,
        'new_native_executions': 0, 'new_fixture_generated': False, 'fixture': summary(fixture),
        'reused_fixture_sha256': FIXTURE_SHA, 'reused_cpu_npz_sha256': CPU_NPZ_SHA,
        'reused_original_code_clip_outputs_byte_identical': True,
        'saved_npy_outputs_equal_npz_bytes': True, 'reused_output': summary(original_output),
        'output_meaning': 'DFL_DISTANCES_NOT_DECODED_BOXES',
        'inherited_corrected_upstream_CPU_runs': 1, 'inherited_corrected_upstream_conv_count': 72,
        'inheritance_is_not_original_corrected_upstream_or_native_distribution_equality': True,
        'original_complete_model_guard': 'FAILED_UNCHANGED'}
    write(output / 'static_reuse_audit.json', reuse_audit)
    variant = copy.deepcopy(parent_spec['cpu_control_variants']['original'])
    require(variant['model_sha256'] == MODEL_SHA and variant['nodes'] == 9 and variant['initializers'] == 7 and
            variant['expected_output_npz_keys'] == {OUTPUT_NAME: 'original_output'}, 'Parent original variant differs')
    spec = {'schema': 'rampq.gate05.dfl_original_probe/v1',
        'prepared_client_at': datetime.now().astimezone().isoformat(), 'diagnostic_only': True,
        'source_configuration': 'q_all', 'source_graph_file': 'source_original_full_model.onnx',
        'source_graph_sha256': ORIGINAL_SHA, 'source_original_qall_graph_sha256': ORIGINAL_SHA,
        'corrected_source_graph_sha256': CORRECTED_SHA,
        'parent_dfl_spec_file': 'parent_dfl_spec.json', 'parent_dfl_spec_sha256': PARENT_SPEC_SHA,
        'parent_dfl_manifest_file': 'parent_dfl_manifest.json', 'parent_dfl_manifest_sha256': PARENT_MANIFEST_SHA,
        'parent_dfl_checkpoint_file': 'parent_dfl_checkpoint.json', 'parent_dfl_checkpoint_sha256': PARENT_CHECKPOINT_SHA,
        'parent_dfl_native_archive_sha256': PARENT_NATIVE_ARCHIVE_SHA,
        'protocol_file': 'protocol.json', 'protocol_sha256': PROTOCOL_SHA,
        'input_file': 'dfl_transpose_fixture.npy', 'input_file_sha256': FIXTURE_SHA,
        'input_binding': copy.deepcopy(parent_spec['input_binding']),
        'output_bindings_in_order': copy.deepcopy(parent_spec['output_bindings_in_order']),
        'conv_names': CONVS, 'source_conv_names': CONVS, 'clip_names': CLIPS,
        'variants': {'original': variant}, 'cases': copy.deepcopy(protocol['cases']), 'case_order': CASE_ORDER,
        'cpu_expected_outputs_file': 'cpu_expected_outputs.npz', 'cpu_expected_outputs_sha256': CPU_NPZ_SHA,
        'fixture_lineage_file': 'fixture_lineage.json', 'fixture_lineage_sha256': FIXTURE_LINEAGE_SHA,
        'graph_identity_audit_file': 'graph_identity_audit.json',
        'graph_identity_audit_sha256': sha(output / 'graph_identity_audit.json'),
        'static_reuse_audit_file': 'static_reuse_audit.json',
        'static_reuse_audit_sha256': sha(output / 'static_reuse_audit.json'),
        'max_seconds': 900, 'max_build_seconds': 300, 'maximum_attempts_per_case': 1,
        'maximum_native_builds': 2, 'maximum_native_inferences': 4, 'executions_per_successful_case': 2,
        'native_build_policy': copy.deepcopy(protocol['common_native_settings']),
        'requires_explicit_native_importer_clip_attribute_correction': False,
        'native_clip_attribute_mutation_permitted': False,
        'original_complete_model_numerical_guard': 'FAILED_UNCHANGED',
        'corrected_full_qall_failure': 'FAILED_TENSORRT_FORMAT_SELECTION_NO_ENGINE_UNCHANGED',
        'parent_corrected_dfl_status': 'NONE_BUILT_AND_EXECUTED;OBEY_FORMAT_SELECTION_FAILURE_NO_ENGINE',
        'calibration_recomputed': False, 'weights_changed': False, 'original_tolerances_changed': False,
        'dataset_evaluation_run': False, 'complete_detector_executions': 0,
        'new_CPU_execution_runs_this_stage': 0, 'native_build_or_execution_run_by_preparer': False,
        'limits': protocol['interpretation']}
    write(output / 'spec_draft.json', spec)
    write(output / 'source_hashes_before.json', before)
    after = source_hashes(source_paths)
    write(output / 'source_hashes_after.json', after)
    require(before == after, 'A historical source changed during static preparation')
    (output / 'README.md').write_text('''# 原始 FLOAT Clip DFL 局部对照

原样复用上一DFL阶段的original tail、内部logits fixture、CPU NPZ与两份原始
CPU NPY。只静态读取验证字节，不运行新的CPU推理、上游或fixture生成。
original tail为原始完整q_all的精确9节点7initializer依赖闭包，输入为原
DFL Transpose FLOAT[1,16,4,8400]，输出为DFL distance FLOAT[1,4,8400]，
不是解码检测框；保留FLOAT Clip→Q→DQ，不校正或修改原生Clip属性。

fixture继承上一阶段修正ONNX的72Conv上游CPU一次输出，不是原始/修正上游
等价或native分布证明。原始与修正tail在该fixture上的旧CPU输出字节相同。
本轮仅original_none/original_obey各一次有界native构建，300秒/case、900秒
batch，成功才固定输入重复执行两次。保留解析器约束，不设置或重置精度、
输出类型，不要求Conv输出INT8，失败无引擎不得执行。

旧corrected DFL NONE成功/OBEY失败记录仅用于跨阶段对照，不重跑旧case。
拓扑与导入修正属于组合因素，不能拆成单独原因；局部结果不证明完整图
唯一根因、取消OBEY能修完整图或Gate通过。原完整护栏FAILED_UNCHANGED。
''', encoding='utf-8')
    write(output / 'prepared_content_manifest.json', {
        'schema': 'rampq.gate05.dfl_original_prepared_content/v1',
        'status': 'STATIC_AUTHENTICATED_REUSE_PREPARED_NOT_SEALED',
        'files': {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}})
    print(json.dumps({'status': 'STATIC_AUTHENTICATED_REUSE_PREPARED_NOT_SEALED', 'output': str(output),
        'model_sha256': MODEL_SHA, 'fixture_sha256': FIXTURE_SHA, 'cpu_expected_outputs_sha256': CPU_NPZ_SHA,
        'source_bytes_unchanged': True, 'new_CPU_executions': 0, 'native_executions': 0,
        'original_tail_nodes_initializers': [9, 7], 'saved_cpu_outputs_byte_identical': True}, indent=2))


def seal(output, native_paths):
    require(output.is_dir() and not output.with_suffix('.zip').exists() and
            not (output / 'manifest.json').exists() and not (output / 'spec.json').exists(),
            'Choose an existing, prepared, previously unsealed destination')
    prepared = read(output / 'prepared_content_manifest.json')
    require(prepared['status'] == 'STATIC_AUTHENTICATED_REUSE_PREPARED_NOT_SEALED', 'Incorrect preparation status')
    for name, expected in prepared['files'].items():
        path = output / name
        require(path.resolve().parent == output and info(path) == expected, 'Prepared file changed: ' + name)
    sources = read(output / 'source_hashes_before.json')
    require(sources == read(output / 'source_hashes_after.json'), 'Preparation source records differ')
    validate_recorded_sources(sources)
    require(sha(output / 'protocol.json') == PROTOCOL_SHA, 'Protocol changed before seal')
    require(len(native_paths) == 2 and len({path.name for path in native_paths}) == 2 and
            all(path.parent == ROOT / 'scripts' and path.is_file() and not path.is_symlink() for path in native_paths),
            'Native scripts must be two separate workspace files')
    native_before = source_hashes(native_paths)
    for path in native_paths:
        shutil.copy2(path, output / path.name)
    spec = read(output / 'spec_draft.json')
    spec['native_runner_file'], spec['native_builder_file'] = [path.name for path in native_paths]
    spec['native_runner_sha256'], spec['native_builder_sha256'] = [sha(path) for path in native_paths]
    spec['sealed_client_at'] = datetime.now().astimezone().isoformat()
    write(output / 'spec.json', spec)
    write(output / 'seal_native_source_hashes_before.json', native_before)
    after = source_hashes(native_paths)
    write(output / 'seal_native_source_hashes_after.json', after)
    require(native_before == after, 'Native source changed while sealing')
    validate_recorded_sources(sources)
    files = {path.name: info(path) for path in sorted(output.iterdir()) if path.is_file()}
    require(len(files) == len(list(output.iterdir())), 'Unexpected subdirectory in bundle')
    write(output / 'manifest.json', {'schema': 'rampq.gate05.dfl_original_probe_bundle/v1', 'files': files})
    checksums = dict(files)
    checksums['manifest.json'] = info(output / 'manifest.json')
    (output / 'SHA256SUMS').write_text(''.join(value['sha256'] + '  ' + name + '\n'
        for name, value in sorted(checksums.items())), encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            archive.write(path, ARCHIVE_ROOT + '/' + path.name)
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, 'ZIP CRC check failed')
        for path in output.iterdir():
            require(hashlib.sha256(archive.read(ARCHIVE_ROOT + '/' + path.name)).hexdigest() == sha(path),
                    'ZIP payload bytes differ: ' + path.name)
    print(json.dumps({'status': 'DFL_ORIGINAL_STATIC_REUSE_BUNDLE_SEALED', 'output': str(output),
        'archive': str(archive_path), 'archive_bytes': archive_path.stat().st_size,
        'archive_sha256': sha(archive_path), 'spec_sha256': sha(output / 'spec.json'),
        'manifest_sha256': sha(output / 'manifest.json'), 'protocol_sha256': PROTOCOL_SHA,
        'native_runner_sha256': spec['native_runner_sha256'], 'native_builder_sha256': spec['native_builder_sha256'],
        'source_bytes_unchanged': True, 'new_CPU_executions': 0, 'native_executions': 0}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--prepare-only', action='store_true')
    stage.add_argument('--seal-only', action='store_true')
    parser.add_argument('--native-runner', type=Path, default=ROOT / 'scripts/jetson_dfl_original_probe.py')
    parser.add_argument('--native-builder', type=Path, default=ROOT / 'scripts/jetson_build_dfl_original_probe.py')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / 'artifacts/gate05').resolve()), 'Unsafe artifact destination')
    if args.prepare_only:
        prepare(output, args.protocol.resolve())
    else:
        seal(output, [args.native_runner.resolve(), args.native_builder.resolve()])


if __name__ == '__main__':
    main()
