"""Static source-only DFL output boundary review; never executes any graph."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[3]
REVIEW = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    output = REVIEW / 'source_readonly_review.json'
    assert not output.exists()
    checks = []

    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
        if not passed:
            raise AssertionError(name)

    protocol_path = ROOT / 'artifacts/gate05/20261020_dfl_conv_output_protocol/protocol.json'
    protocol = read(protocol_path)
    check('frozen_protocol_sha', sha(protocol_path) == '8f6a55f640915b84d7300d1595ebb895092ccf0374e40a27b6aaf812f406e09b')
    old = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_probe'
    native = ROOT / 'artifacts/gate05/20261020_dfl_softmax_cut_native'
    check('stage35_checkpoint_identity', sha(native / 'checkpoint.json') == protocol['parent_softmax_cut_checkpoint_sha256'])
    check('stage35_next_plan_identity', sha(native / 'next_control_plan.json') == protocol['next_plan_sha256'])
    check('stage35_graph_identity', sha(old / 'model_dfl_softmax_cut_code_clip.onnx') == protocol['parent_softmax_cut_graph_sha256'])
    graph = onnx.load(old / 'model_dfl_softmax_cut_code_clip.onnx')
    producers = {name: node for node in graph.graph.node for name in node.output}
    nodes = {node.name: node for node in graph.graph.node}
    needed = set()
    init_names = {i.name for i in graph.graph.initializer}
    selected_init = set()

    def visit(name):
        if name in init_names:
            selected_init.add(name)
        elif name in producers:
            node = producers[name]
            if node.name not in needed:
                needed.add(node.name)
                for value in node.input:
                    if value:
                        visit(value)
        else:
            check('closure_external_input:' + name, name == protocol['input_name'])

    visit(protocol['output_name'])
    check('closure_exact_six_nodes', len(needed) == protocol['graph_node_count'] == 6)
    check('closure_exact_seven_initializers', len(selected_init) == protocol['initializer_count'] == 7 and selected_init == init_names)
    check('exact_two_nodes_removed', set(nodes) - needed == set(protocol['removed_node_names']) ==
          {'/model.23/dfl/Constant_1', '/model.23/dfl/Reshape_1'})
    check('one_Conv_no_Softmax_or_Reshape', sum(nodes[name].op_type == 'Conv' for name in needed) == 1 and
          not any(nodes[name].op_type in ('Softmax', 'Reshape') for name in needed))
    check('terminal_reshape_directly_uses_Conv_output', list(nodes['/model.23/dfl/Reshape_1'].input) ==
          [protocol['output_name'], nodes['/model.23/dfl/Constant_1'].output[0]])
    check('unchanged_input_FLOAT_shape', graph.graph.input[0].name == protocol['input_name'] and
          [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim] == protocol['input_shape'] == [1, 16, 4, 8400] and
          graph.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT)
    source = onnx.load(ROOT / protocol['source_graph'])
    check('full_failed_source_identity', sha(ROOT / protocol['source_graph']) == protocol['source_graph_sha256'])
    values = {v.name: v for v in list(source.graph.value_info) + list(source.graph.output) + list(source.graph.input)}
    boundary = values.get(protocol['output_name'])
    if boundary is None:
        inferred = onnx.shape_inference.infer_shapes(source)
        values = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.output) + list(inferred.graph.input)}
        boundary = values[protocol['output_name']]
    check('declared_Conv_output_FLOAT_4D', boundary.type.tensor_type.elem_type == onnx.TensorProto.FLOAT and
          [d.dim_value for d in boundary.type.tensor_type.shape.dim] == protocol['output_shape'] == [1, 1, 4, 8400])
    probability = np.load(old / protocol['input_file'], allow_pickle=False)
    previous_cpu = np.load(old / 'cpu_actual_code_clip.npy', allow_pickle=False)
    check('probability_file_identity', sha(old / protocol['input_file']) == protocol['saved_probability_fixture_sha256'])
    check('saved_probability_finite_FLOAT_shape', probability.dtype == np.float32 and probability.shape == (1, 16, 4, 8400) and np.isfinite(probability).all())
    check('saved_3D_reference_file_identity', sha(old / 'cpu_actual_code_clip.npy') == protocol['saved_corrected_CPU_npy_sha256'])
    check('saved_3D_reference_explicit_reshape_preserves_rawbytes', previous_cpu.dtype == np.float32 and previous_cpu.shape == (1, 4, 8400) and
          previous_cpu.size == 33600 and previous_cpu.tobytes(order='C') == previous_cpu.reshape(1, 1, 4, 8400).tobytes(order='C'))
    check('CPU_budget_exact_one', protocol['cpu_composition_check']['total_new_CPU_runs'] == protocol['cpu_composition_check']['cut_CPU_runs'] == 1 and
          protocol['cpu_composition_check']['new_Softmax_72Conv_old_DFL_or_full_detector_runs'] == 0)
    check('native_cases_and_budgets', protocol['case_order'] == ['code_clip_none', 'code_clip_obey'] and
          protocol['maximum_builds'] == 2 and protocol['maximum_executions'] == 4 and protocol['attempts_each'] == 1 and
          protocol['per_case_max_build_seconds'] == 300 and protocol['batch_max_seconds'] == 900)
    check('no_precision_resets_policy', protocol['common_native_settings']['layer_precision_or_output_type_setters_or_resets'] is False)
    report = {'schema': 'rampq.gate05.dfl_conv_output_source_review/v1',
        'reviewed_client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'SOURCE_OUTPUT_BOUNDARY_PROTOCOL_STATIC_REVIEW_PASSED', 'checks': checks, 'check_count': len(checks),
        'selected_node_names': [n.name for n in graph.graph.node if n.name in needed],
        'selected_initializer_names': sorted(selected_init), 'source_old_CPU_raw_array_sha256': hashlib.sha256(previous_cpu.tobytes()).hexdigest(),
        'launch_blockers_at_source_review': [], 'sealed_bundle_pre_native_review_still_required': True,
        'CPU_OR_native_graph_execution_by_reviewer': False, 'SSH_by_reviewer': False,
        'scope_limit': 'Removing Constant/Reshape also changes native output boundary and format selection; no unique removed-node or full-model cause follows.'}
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': sha(output)}))


if __name__ == '__main__':
    main()
