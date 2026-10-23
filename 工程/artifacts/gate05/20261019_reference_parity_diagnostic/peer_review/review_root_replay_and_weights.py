"""Verify saved root replay and isolate signed-zero weight-QDQ differences.

This loads original ONNX initializers and executes only 88 weight Q/DQ pairs.
No full PyTorch model, image inference, native inference, or dataset is loaded.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    import onnxruntime as ort
    import torch
    torch.set_num_threads(4)
    full = OUT.parent / 'full_replay'
    prefix = OUT.parent / 'first_boundary'
    original = ROOT / 'artifacts/gate05/20261018_int8/input_v1'
    output = OUT / 'root_replay_and_weight_review.json'
    assert not output.exists()
    sources = [full/'manifest.json', full/'analysis.json', full/'all_88_conv_replay.json',
               full/'first_two_conv_arrays.npz', full/'raw_primary_outputs.npz',
               full/'diagnose_gate05_reference_replay.py', full/'diagnostic_recorded_conv_output_tail.onnx',
               prefix/'raw_outputs.npz', prefix/'analysis.json', original/'model_q_all.onnx',
               original/'module_map.json', original/'preflight.json']
    before = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    manifest = read(full/'manifest.json')
    for name, rec in manifest['files'].items():
        path = full/name
        assert path.stat().st_size == rec['bytes'] and sha(path) == rec['sha256']
    assert read(full/'source_hashes_before.json') == read(full/'source_hashes_after.json')
    for name, rec in read(full/'source_hashes_after.json').items():
        path = ROOT/name
        assert path.stat().st_size == rec['bytes'] and sha(path) == rec['sha256']
    a, rows = read(full/'analysis.json'), read(full/'all_88_conv_replay.json')
    assert len(rows) == 88
    assert all(row['ort_replay_same_ort_input_vs_original_ort_conv']['byte_identical'] and
               row['ort_replay_same_ort_input_vs_original_ort_conv']['different'] == 0 for row in rows)
    same_pt_max = max(row['ort_replay_same_pytorch_input_vs_original_pytorch_conv']['max_abs'] for row in rows)
    free_max = max(row['full_backend_conv_output_difference']['max_abs'] for row in rows)
    assert same_pt_max == a['largest_same_pytorch_input_conv_output_difference'] == 2.86102294921875e-5
    assert free_max == a['largest_full_backend_conv_output_difference'] == 3.26055645942688
    assert all(row['weight_qdq']['different'] == 0 for row in rows)
    with np.load(full/'first_two_conv_arrays.npz', allow_pickle=False) as f, np.load(prefix/'raw_outputs.npz', allow_pickle=False) as p:
        comparisons = {}
        for tag, first_key, prefix_key in [
            ('first_conv_pytorch','pytorch_0_output','conv_pt'),
            ('first_conv_ort','ort_0_output','conv_ort_prefix'),
            ('second_activation_dq_pytorch','pytorch_1_input_qdq','conv_pt_silu_pt_dq_pt'),
            ('second_activation_dq_ort','ort_1_input_qdq','prefix_dq1')]:
            x,y = f[first_key],p[prefix_key]
            bit_mask = x.view(np.uint32) != y.view(np.uint32)
            comparisons[tag] = {'numeric_equal':bool(np.array_equal(x,y)),
                'byte_identical':x.tobytes() == y.tobytes(), 'bit_different':int(bit_mask.sum()),
                'bit_differences_signed_zero_only':bool(np.all((x[bit_mask] == 0) & (y[bit_mask] == 0)))}
        assert all(v['numeric_equal'] and v['bit_differences_signed_zero_only'] for v in comparisons.values())
        coords = np.argwhere(f['pytorch_1_input_qdq'] != f['ort_1_input_qdq']).tolist()
        assert coords == p['actual_q1_disagreement_indices'].tolist()
        boundary_signed_zero, boundary_raw = {}, {}
        for c in ('pt','ort'):
            for s in ('pt','ort'):
                tag = f'conv_{c}_silu_{s}'
                value = torch.from_numpy(p[tag+'_post'].copy())
                scale = torch.tensor(float(p['scale1']), dtype=torch.float32)
                original_torch_dq = (torch.round(value/scale).clamp(-127,127)*scale).numpy()
                ort_dq = p[tag+'_dq_ort']
                mask = original_torch_dq.view(np.uint32) != ort_dq.view(np.uint32)
                assert np.array_equal(original_torch_dq,ort_dq)
                assert np.all((original_torch_dq[mask] == 0) & (ort_dq[mask] == 0))
                boundary_signed_zero[tag] = {'numeric_different':0,'bit_different':int(mask.sum()),
                    'bit_differences_signed_zero_only':True}
                boundary_raw[tag+'_original_torch_float_qdq'] = original_torch_dq
                if tag == 'conv_pt_silu_pt':
                    assert original_torch_dq.tobytes() == f['pytorch_1_input_qdq'].tobytes()
        np.savez_compressed(OUT/'boundary_signed_zero_supplement.npz',**boundary_raw)
    tail = onnx.load(str(full/'diagnostic_recorded_conv_output_tail.onnx'))
    assert len(tail.graph.node) == 36 and not any(n.op_type == 'Conv' for n in tail.graph.node)
    expected_cuts = ['/model.23/cv3.0/cv3.0.2/Conv_output_0', '/model.23/cv3.1/cv3.1.2/Conv_output_0',
                     '/model.23/cv3.2/cv3.2.2/Conv_output_0', '/model.23/dfl/conv/Conv_output_0']
    assert sorted(v.name for v in tail.graph.input) == expected_cuts == a['tail_cut']['external_recorded_conv_outputs']
    with np.load(full/'raw_primary_outputs.npz', allow_pickle=False) as raw:
        pt, ort_full, ort_tail = raw['pytorch'], raw['ort'], raw['recorded_pytorch_conv_output_ort_tail']
        tail_boxes = float(np.max(np.abs(pt[:, :4].astype(np.float64) - ort_tail[:, :4].astype(np.float64))))
        tail_scores = float(np.max(np.abs(pt[:, 4:].astype(np.float64) - ort_tail[:, 4:].astype(np.float64))))
        assert tail_boxes == 0.0 and tail_scores == 1.3146405919428616e-7
        guard_counts = [int(np.count_nonzero(~np.isclose(ort_full[:, slc], pt[:, slc], rtol=rtol, atol=atol)))
                        for slc, rtol, atol in [(slice(None,4),.005,.5),(slice(4,None),.02,.002)]]
        assert guard_counts == [14204, 47]
    graph = onnx.load(str(original/'model_q_all.onnx'))
    mapping = read(original/'module_map.json')['mapping']
    nodes = {n.name:n for n in graph.graph.node}
    init = {v.name:v for v in graph.graph.initializer}
    selected_nodes, used, output_names = [], set(), []
    for row in mapping:
        for suffix in ('WeightQ','WeightDQ'):
            node = nodes[row['qdq_prefix']+'_'+suffix]
            selected_nodes.append(copy.deepcopy(node))
            used.update(node.input)
        output_names.append(row['qdq_prefix']+'_w_dq')
    weight_graph = copy.deepcopy(graph)
    del weight_graph.graph.node[:]
    weight_graph.graph.node.extend(selected_nodes)
    del weight_graph.graph.initializer[:]
    weight_graph.graph.initializer.extend(copy.deepcopy(v) for v in graph.graph.initializer if v.name in used)
    del weight_graph.graph.input[:]
    del weight_graph.graph.output[:]
    weight_graph.graph.output.extend(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT,
        list(numpy_helper.to_array(init[row['weight_input']]).shape)) for row, name in zip(mapping,output_names))
    del weight_graph.graph.value_info[:]
    onnx.checker.check_model(weight_graph, full_check=True)
    assert len(weight_graph.graph.node) == 176
    onnx.save(weight_graph, str(OUT/'diagnostic_weight_only_qdq.onnx'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    weight_session = ort.InferenceSession(weight_graph.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
    observed = weight_session.run(None,{})
    weight_rows, weight_raw = [], {}
    for i, (row, actual) in enumerate(zip(mapping,observed)):
        original_weight = numpy_helper.to_array(init[row['weight_input']]).copy()
        frozen_scale = numpy_helper.to_array(init[row['qdq_prefix']+'_w_scale'])
        weight = torch.from_numpy(original_weight)
        dims = tuple(range(1,weight.ndim))
        scale = weight.abs().amax(dim=dims,keepdim=True).clamp_min(1e-12)/127.0
        assert scale.numpy().reshape(-1).tobytes() == frozen_scale.tobytes()
        expected = (torch.round(weight/scale).clamp(-127,127)*scale).numpy()
        assert np.array_equal(actual,expected)
        actual_bits, expected_bits = actual.view(np.uint32), expected.view(np.uint32)
        bit_diff = actual_bits != expected_bits
        assert np.all((actual[bit_diff] == 0) & (expected[bit_diff] == 0))
        assert np.all(actual_bits[bit_diff] == 0) and np.all(expected_bits[bit_diff] == 0x80000000)
        assert np.all(np.signbit(original_weight[bit_diff]))
        weight_rows.append({'module':row['pytorch_module'],'group':row['group'],'elements':int(actual.size),
            'numeric_different':int(np.count_nonzero(actual != expected)), 'bit_different':int(np.count_nonzero(bit_diff)),
            'ort_positive_zero_vs_torch_negative_zero':int(np.count_nonzero(bit_diff)),
            'original_weight_negative_signbit_at_every_bit_disagreement':True,
            'strictly_negative_source_weight_count':int(np.count_nonzero(original_weight[bit_diff] < 0)),
            'original_negative_zero_source_weight_count':int(np.count_nonzero(original_weight[bit_diff] == 0)),
            'frozen_scale_bytes_equal_original_gate0_scale_formula':True})
        weight_raw[f'layer_{i:03d}_pytorch_qdq'] = expected
        weight_raw[f'layer_{i:03d}_ort_qdq'] = actual
        weight_raw[f'layer_{i:03d}_bit_disagreement_flat_indices'] = np.flatnonzero(bit_diff)
    np.savez_compressed(OUT/'weight_only_qdq_arrays.npz',**weight_raw)
    after = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    assert before == after
    result = {'schema':'rampq.gate05.root_replay_and_weight_peer_review/v1',
        'status':'ROOT_REPLAY_AND_SIGNED_ZERO_WEIGHT_REVIEW_VERIFIED',
        'sources_before_and_after_identical':True,'source_hashes':before,
        'root_replay_manifest_files_verified':len(manifest['files']),
        'root_code_review':{'conv_replay':'Preserves every original weight Q/DQ, scale, zero point, bias and Conv attribute. Only activation DQ tensor is externalized.',
            'same_ort_input_control':'All 88 saved exact statistics verify the isolated replay retains original full-ORT Conv execution for these inputs.',
            'tail_cut':'Four supplied recorded Conv outputs include the DFL Conv; boxes bypass the preceding DFL/softmax/attention computations. Only remaining 36 non-Conv nodes are tested.',
            'copied_script_rerun_note':'The archived root script is a source snapshot; its ROOT=parents[1] assumes it is launched from the original scripts directory, not directly from the archive.'},
        'first_boundary_prefix_full_raw_equality':comparisons,
        'first_boundary_actual_float_qdq_signed_zero_supplement':boundary_signed_zero,
        'same_three_qdq_disagreement_coordinates':coords,
        'all_88_same_ort_input_conv_replay_byte_identical':True,
        'maximum_same_pt_input_conv_error':same_pt_max,
        'maximum_free_full_backend_conv_error':free_max,
        'tail_recorded_inputs':expected_cuts,'tail_remaining_nodes':36,'tail_remaining_conv_nodes':0,
        'tail_box_max_abs_error':tail_boxes,'tail_score_max_abs_error':tail_scores,
        'original_full_guard_violation_counts':{'boxes':guard_counts[0],'scores':guard_counts[1]},
        'weight_only_graph_nodes':176,'weight_only_forward_runs':1,'full_pytorch_model_loaded':False,
        'weight_elements_total':sum(r['elements'] for r in weight_rows),
        'weight_numeric_disagreements_total':sum(r['numeric_different'] for r in weight_rows),
        'weight_bit_disagreements_total':sum(r['bit_different'] for r in weight_rows),
        'weight_layers_with_bit_disagreement':sum(r['bit_different']>0 for r in weight_rows),
        'all_weight_bit_disagreements_only_signed_zero':True,
        'first_boundary_analysis_field_correction':{
            'unchanged_file':'first_boundary/analysis.json',
            'overstrong_original_field':'source_qdq_input_and_weight_bitexact=true',
            'reason':'Original np.testing.assert_array_equal checks numeric equality and accepts +0/-0; it is not a byte-equality assertion.',
            'precise_input0_status':'Numeric equality in prefix experiment and byte equality in independently verified saved full-replay layer0 input.',
            'precise_weight0_status':'Numeric equality, byte inequality from signed zero only.',
            'weight0_signed_zero_bit_disagreement_count':weight_rows[0]['bit_different'],
            'original_torch_second_dq_comparator_correction':'first_boundary/raw Torch second-DQ arrays were reconstructed from INT8 codes and normalize -0 to +0. Supplementary arrays here recompute the original floating round/clamp/DQ expression. They numerically match ORT and the full model; original full-model Torch negative-zero signs are retained only in the supplementary arrays.',
            'actual_float_second_dq_matches_root_full_replay_bytes':True,
            'correction_does_not_modify_original_raw_arrays_or_analysis':True},
        'strictly_negative_source_weights_among_bit_disagreements':sum(r['strictly_negative_source_weight_count'] for r in weight_rows),
        'original_negative_zero_weights_among_bit_disagreements':sum(r['original_negative_zero_source_weight_count'] for r in weight_rows),
        'signed_zero_direction':'Original negative weights round to floating -0 under Torch round/clamp; existing original -0 values also retain their sign. Multiplication by positive scale retains -0. ONNX INT8 quantization stores integer code 0 and dequantizes to +0.',
        'weight_per_module':weight_rows,
        'original_complete_model_guard':'FAILED_UNCHANGED','new_tolerance_defined':False,
        'dataset_or_native_execution':False,'limits':['Read-only fixed-image saved outputs plus initializer-only weight QDQ experiment.',
            'All-88 Conv controls are verified against saved per-layer statistics and code; only first two layers have saved arrays independently recomputed here.',
            'No result passes the original guard or proves all non-Conv exported semantics. Signed zero is numeric equality, not bit identity.'],
        'derived_files':{name:{'sha256':sha(OUT/name),'bytes':(OUT/name).stat().st_size} for name in
                         ['diagnostic_weight_only_qdq.onnx','weight_only_qdq_arrays.npz','boundary_signed_zero_supplement.npz',
                          'review_root_replay_and_weights.py']}}
    write(output,result)
    print(json.dumps({k:result[k] for k in ('status','same_three_qdq_disagreement_coordinates','weight_elements_total',
        'weight_bit_disagreements_total','weight_layers_with_bit_disagreement','all_weight_bit_disagreements_only_signed_zero')},indent=2))


if __name__ == '__main__':
    main()
