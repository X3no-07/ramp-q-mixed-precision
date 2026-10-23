"""CPU fixed-image operator-backend counterfactuals; never a replacement guard.

Every source graph and source reference remains unchanged on disk. The original
PyTorch QDQConv2d objects are temporarily rebound in memory to an original ONNX
Conv with its original weight Q/DQ. Activation quantization remains PyTorch.
"""
import argparse
import contextlib
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def inventory(paths):
    return {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/gate05/20261019_reference_parity_diagnostic/hybrid_reference')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_CONFIG_DIR'] = str(out/'.ultralytics')
    original = ROOT/'artifacts/gate05/20261018_int8/input_v1'
    replay = out.parent/'full_replay'
    source = original/'model_q_all.onnx'
    weights = ROOT/'artifacts/jetson_fp16_smoke_20261016_v1/best.pt'
    image = weights.parent/'000020.jpg'
    sources = [Path(__file__).resolve(), ROOT/'src/gate0.py', source, weights, image,
               original/'module_map.json', original/'calibration.json', original/'preflight.json',
               original/'protocol.json', original/'local_raw_outputs.npz',
               replay/'diagnostic_tapped_q_all.onnx', replay/'diagnostic_same_input_convs.onnx',
               replay/'preprocessed_input.npy', replay/'raw_primary_outputs.npz', replay/'analysis.json']
    before = inventory(sources)
    write(out/'source_hashes_before.json', before)
    shutil.copy2(Path(__file__), out/Path(__file__).name)
    old_guard = read(original/'preflight.json')
    assert old_guard['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert sha(source) == old_guard['onnx_sha256']
    assert sha(image) == old_guard['image_sha256']
    protocol = {
        'schema': 'rampq.gate05.hybrid_reference_protocol/v1',
        'client_at': datetime.now().astimezone().isoformat(),
        'input': 'Original fixed 000020.jpg only; CPU FP32; no dataset or native execution',
        'sessions': '88 independent original WeightQ, WeightDQ, Conv subgraphs; all original Conv attributes and initializer bytes retained; activation DQ is explicit input',
        'controls': 'Every independent session must be byte exact with both original tapped full ORT and original all-88 disconnected ORT replay, using full ORT activation DQ inputs',
        'case_1': 'Original PyTorch QDQ activation round/clamp/multiply; ORT original WeightQ/DQ + Conv; original remaining PyTorch model',
        'case_2': 'Case 1 plus temporary unique nn.SiLU instance forwards rebound to ORT Sigmoid then Mul; all other PyTorch arithmetic remains unchanged',
        'threads': {'torch': 4, 'full_ort': 4, 'independent_conv_ort': 1, 'silu_ort': 1},
        'guard': read(original/'protocol.json')['local_preflight'],
        'tolerances_changed': False, 'original_complete_model_guard': 'FAILED_UNCHANGED',
        'deployment_reference_replaced': False, 'deployment_graph_change': False,
        'interpretation': 'Operator-backend counterfactuals only. Even passing original numerical slices in a counterfactual does not pass or replace the original full PyTorch/ORT guard.'}
    write(out/'protocol.json', protocol)
    with (out/'console.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import numpy as np
        import onnx
        from onnx import helper
        import onnxruntime as ort
        import torch
        import ultralytics
        sys.path.insert(0, str(ROOT/'src'))
        import gate0
        torch.set_num_threads(4)

        def session(graph, threads=1):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            return ort.InferenceSession(graph.SerializeToString(), sess_options=opts, providers=['CPUExecutionProvider'])

        def stats(a, b):
            assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
            return {'shape': list(a.shape), 'elements': int(a.size), 'different': int(np.count_nonzero(a != b)),
                    'byte_identical': a.tobytes() == b.tobytes(), 'max_abs': float(delta.max()),
                    'mean_abs': float(delta.mean()), 'rmse': float(np.sqrt(np.mean(delta*delta)))}

        def guard(a, b):
            result = {}
            for name, slc, rtol, atol in (('boxes', slice(None,4), .005, .5), ('scores', slice(4,None), .02, .002)):
                aa, bb = a[:,slc,:], b[:,slc,:]
                mask = ~np.isclose(aa, bb, rtol=rtol, atol=atol)
                result[name] = {'rtol': rtol, 'atol': atol, 'violating_elements': int(mask.sum()),
                                'passing_elements': int(mask.size-mask.sum()), 'total_elements': int(mask.size),
                                'within_original_tolerance': not bool(mask.any()),
                                'max_abs': float(np.max(np.abs(aa.astype(np.float64)-bb.astype(np.float64))))}
            result['both_slices_within_original_tolerance'] = all(result[n]['within_original_tolerance'] for n in ('boxes','scores'))
            return result

        mapping = read(original/'module_map.json')['mapping']
        calibration = read(original/'calibration.json')
        assert len(mapping) == 88
        x, *_ = gate0.image_tensor(image, 640, torch.device('cpu'))
        saved_x = np.load(replay/'preprocessed_input.npy', allow_pickle=False)
        assert x.numpy().tobytes() == saved_x.tobytes()
        graph = onnx.load(str(source))
        by_name = {n.name: n for n in graph.graph.node}
        init = {v.name: v for v in graph.graph.initializer}
        tapped = onnx.load(str(replay/'diagnostic_tapped_q_all.onnx'))
        tapped_session = session(tapped, 4)
        actual_names = [v.name for v in tapped.graph.output]
        actual = dict(zip(actual_names, tapped_session.run(None, {'images': saved_x})))
        ort_primary = actual[graph.graph.output[0].name]
        with np.load(replay/'raw_primary_outputs.npz', allow_pickle=False) as saved:
            pt_primary = saved['pytorch'].copy()
            saved_ort = saved['ort'].copy()
        assert ort_primary.tobytes() == saved_ort.tobytes()
        baseline_guard = guard(ort_primary, pt_primary)
        assert not baseline_guard['both_slices_within_original_tolerance']

        combined_graph = onnx.load(str(replay/'diagnostic_same_input_convs.onnx'))
        combined_feeds = {'replay_input_'+str(i): actual[r['qdq_prefix']+'_a_dq'] for i,r in enumerate(mapping)}
        combined_outputs = session(combined_graph, 4).run(None, combined_feeds)
        sessions, controls = {}, []
        for i, row in enumerate(mapping):
            prefix, module = row['qdq_prefix'], row['pytorch_module']
            conv_original = by_name[row['onnx_node']]
            nodes = [copy.deepcopy(by_name[prefix+'_WeightQ']), copy.deepcopy(by_name[prefix+'_WeightDQ']), copy.deepcopy(conv_original)]
            nodes[-1].input[0] = 'activation_dq_input'
            needed_init = {name for node in nodes for name in node.input if name in init}
            a, expected = actual[prefix+'_a_dq'], actual[conv_original.output[0]]
            single = copy.deepcopy(graph)
            del single.graph.node[:]; single.graph.node.extend(nodes)
            del single.graph.initializer[:]
            single.graph.initializer.extend(copy.deepcopy(v) for v in graph.graph.initializer if v.name in needed_init)
            del single.graph.input[:]
            single.graph.input.append(helper.make_tensor_value_info('activation_dq_input',onnx.TensorProto.FLOAT,list(a.shape)))
            del single.graph.output[:]
            single.graph.output.append(helper.make_tensor_value_info(conv_original.output[0],onnx.TensorProto.FLOAT,list(expected.shape)))
            del single.graph.value_info[:]
            onnx.checker.check_model(single, full_check=True)
            assert list(nodes[-1].attribute) == list(conv_original.attribute)
            assert all(v.SerializeToString() == init[v.name].SerializeToString() for v in single.graph.initializer)
            sess = session(single, 1)
            got = sess.run(None, {'activation_dq_input': a})[0]
            row_control = {'module': module, 'group': row['group'], 'onnx_node': row['onnx_node'],
                           'all_original_conv_attributes_identical': True, 'all_original_initializer_bytes_identical': True,
                           'independent_session_model_sha256': hashlib.sha256(single.SerializeToString()).hexdigest(),
                           'against_full_ort_conv': stats(got, expected),
                           'against_all_88_replay_conv': stats(got, combined_outputs[i])}
            controls.append(row_control)
            sessions[module] = sess
        write(out/'all_88_independent_conv_controls.json', controls)
        assert all(r['against_full_ort_conv']['byte_identical'] and r['against_all_88_replay_conv']['byte_identical'] for r in controls), 'Independent-session controls failed; no hybrid interpretation allowed'
        del combined_outputs, combined_graph, combined_feeds, tapped_session

        model = gate0.load_fused_model(weights, torch.device('cpu'))
        wrappers = gate0.install_qdq_wrappers(model, calibration['activation_max_abs'])
        gate0.enable_configuration(wrappers, 'q_all')
        assert {m.name for m in wrappers} == set(sessions)
        by_module = {r['pytorch_module']: r for r in mapping}
        silu_aliases = {}
        for name, mod in model.named_modules(remove_duplicate=False):
            if isinstance(mod, torch.nn.SiLU):
                silu_aliases.setdefault(id(mod), {'module':mod, 'aliases':[]})['aliases'].append(name)
        silu_inventory = [{'aliases':r['aliases'], 'alias_count':len(r['aliases']), 'inplace':bool(r['module'].inplace)} for r in silu_aliases.values()]
        write(out/'silu_unique_instance_inventory.json', {'unique_instances':len(silu_inventory),
              'named_aliases':sum(r['alias_count'] for r in silu_inventory), 'instances':silu_inventory})
        silu_graph = helper.make_model(helper.make_graph([
            helper.make_node('Sigmoid',['x'],['sigmoid'],name='diagnostic_Sigmoid'),
            helper.make_node('Mul',['x','sigmoid'],['y'],name='diagnostic_Mul')], 'diagnostic_original_silu_ops',
            [helper.make_tensor_value_info('x',onnx.TensorProto.FLOAT,None)],
            [helper.make_tensor_value_info('y',onnx.TensorProto.FLOAT,None)]), opset_imports=graph.opset_import)
        silu_graph.ir_version = graph.ir_version
        silu_session = session(silu_graph,1)
        runtime_rows, silu_calls = {}, 0
        def hybrid_conv(mod, tensor):
            assert mod.enabled and tensor.device.type == 'cpu' and tensor.dtype == torch.float32
            scale = mod.activation_scale.to(dtype=tensor.dtype, device=tensor.device)
            q = torch.round(tensor/scale).clamp(-127,127)*scale
            a = q.detach().numpy()
            value = sessions[mod.name].run(None, {'activation_dq_input':a})[0]
            row = by_module[mod.name]
            prefix = row['qdq_prefix']
            assert mod.name not in runtime_rows, 'Expected one call per Conv'
            runtime_rows[mod.name] = {'module':mod.name, 'input_qdq_vs_full_ort':stats(a,actual[prefix+'_a_dq']),
                                      'conv_output_vs_full_ort':stats(value,actual[by_name[row['onnx_node']].output[0]])}
            return torch.from_numpy(value)
        def hybrid_silu(mod, tensor):
            nonlocal silu_calls
            silu_calls += 1
            assert tensor.device.type == 'cpu' and tensor.dtype == torch.float32
            return torch.from_numpy(silu_session.run(None, {'x':tensor.detach().numpy()})[0])
        saved_forwards = [(m, m.forward) for m in wrappers]
        saved_silu_forwards = [(r['module'],r['module'].forward) for r in silu_aliases.values()]
        results, arrays = {}, {'original_pytorch':pt_primary,'original_full_ort':ort_primary}
        try:
            for mod in wrappers: mod.forward = types.MethodType(hybrid_conv, mod)
            for case in ('ort_conv_torch_other', 'ort_conv_and_ort_silu_torch_other'):
                if case == 'ort_conv_and_ort_silu_torch_other':
                    for mod,_ in saved_silu_forwards: mod.forward = types.MethodType(hybrid_silu, mod)
                runtime_rows, silu_calls = {}, 0
                with torch.inference_mode():
                    pred = model(x)
                    primary = (pred[0] if isinstance(pred,(tuple,list)) else pred).numpy().copy()
                assert len(runtime_rows) == 88
                ordered = [runtime_rows[r['pytorch_module']] for r in mapping]
                write(out/(case+'_all_88_boundaries.json'),ordered)
                first_input = next((r['module'] for r in ordered if not r['input_qdq_vs_full_ort']['byte_identical']),None)
                first_output = next((r['module'] for r in ordered if not r['conv_output_vs_full_ort']['byte_identical']),None)
                arrays[case] = primary
                results[case] = {'vs_original_full_ort':stats(primary,ort_primary),
                                 'vs_original_pytorch':stats(primary,pt_primary),
                                 'original_guard_slices_counterfactual_only':guard(primary,ort_primary),
                                 'quantized_conv_calls':len(runtime_rows), 'ort_silu_calls':silu_calls,
                                 'first_activation_qdq_byte_difference_from_ort':first_input,
                                 'first_conv_output_byte_difference_from_ort':first_output,
                                 'unchanged_orthogonal_arithmetic':'All operations other than Conv (and SiLU in case 2) retain original PyTorch execution'}
        finally:
            for mod, forward in saved_forwards: mod.forward = forward
            for mod, forward in saved_silu_forwards: mod.forward = forward
        np.savez_compressed(out/'raw_primary_outputs.npz',**arrays)
        summary = {'schema':'rampq.gate05.hybrid_reference_result/v1',
                   'status':'FIXED_IMAGE_CPU_OPERATOR_BACKEND_COUNTERFACTUALS_COMPLETE',
                   'versions':{'torch':torch.__version__,'onnx':onnx.__version__,'ort':ort.__version__,
                               'ultralytics':ultralytics.__version__,'numpy':np.__version__},
                   'thread_settings':protocol['threads'],
                   'original_full_ort_primary_saved_replay_byte_exact':True,
                   'input_byte_exact_with_saved_replay':True,
                   'all_88_independent_conv_sessions_byte_exact_with_full_ort_and_replay':True,
                   'all_88_original_initializer_bytes_and_conv_attributes_preserved':True,
                   'original_full_model_guard_rechecked':baseline_guard,
                   'silu_unique_instances':len(silu_inventory),'silu_named_aliases':sum(r['alias_count'] for r in silu_inventory),
                   'cases':results,'original_complete_model_guard':'FAILED_UNCHANGED',
                   'tolerances_changed':False,'calibration_changed':False,'weights_changed':False,
                   'source_graph_changed':False,'deployment_reference_replaced':False,
                   'dataset_evaluation_run':False,'native_execution_run':False,'cuda_used':False,
                   'accuracy_pass_claim':False,'performance_pass_claim':False,'numerical_migration_pass_claim':False,
                   'limits':'One original fixed image only. In-memory operator backend substitutions change the reference implementation and are causal localization counterfactuals, never a guard pass or replacement deployment reference. Torch quantization and all unmodified arithmetic may still differ from ONNX runtime.'}
    after = inventory(sources)
    assert before == after, 'Original source bytes changed'
    write(out/'source_hashes_after.json',after)
    summary['all_original_source_hashes_unchanged'] = True
    write(out/'analysis.json',summary)
    files = sorted(p for p in out.rglob('*') if p.is_file())
    write(out/'manifest.json',{'schema':'rampq.gate05.hybrid_reference_files/v1',
                              'files':{p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}})
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
