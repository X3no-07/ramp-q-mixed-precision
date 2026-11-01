"""Fixed-image CPU counterfactuals for the retained PyTorch/ORT QDQ failure.

The original graph, reference, calibration, and guard remain unchanged. Replay
graphs cut dependencies and are diagnostics, never deployment candidates.
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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def inventory(paths):
    return {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'artifacts/gate05/20261019_reference_parity_diagnostic/full_replay')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_CONFIG_DIR'] = str(out/'.ultralytics')
    original = ROOT/'artifacts/gate05/20261018_int8/input_v1'
    source = original/'model_q_all.onnx'
    weights = ROOT/'artifacts/jetson_fp16_smoke_20261016_v1/best.pt'
    image = weights.parent/'000020.jpg'
    sources = [Path(__file__).resolve(), ROOT/'src/gate0.py', source, weights, image,
               original/'module_map.json', original/'calibration.json', original/'preflight.json',
               original/'protocol.json', original/'local_raw_outputs.npz']
    before = inventory(sources)
    write(out/'source_hashes_before.json', before)
    shutil.copy2(Path(__file__), out/Path(__file__).name)
    old_guard = read(original/'preflight.json')
    assert old_guard['status'] == 'LOCAL_QDQ_PREFLIGHT_FAILED'
    assert sha(source) == old_guard['onnx_sha256']
    assert sha(image) == old_guard['image_sha256']
    protocol = {
        'schema': 'rampq.gate05.reference_replay_protocol/v1',
        'client_at': datetime.now().astimezone().isoformat(),
        'input': 'Original fixed 000020.jpg only; CPU FP32; ORT_DISABLE_ALL; four threads',
        'baseline': 'Reproduce original q_all full PyTorch and ORT primary outputs and both original guard slices',
        'conv_replay': '88 disconnected original Conv+weightQ/DQ subgraphs; activation DQ values are explicit inputs',
        'conv_control': 'Feed full-ORT activation DQ and compare original full-ORT Conv output, before interpreting PyTorch-input replay',
        'conv_counterfactual': 'Feed recorded PyTorch activation DQ; compare original PyTorch Conv output on exactly those inputs',
        'head_replay': 'Cut all grouped Conv outputs to explicit recorded PyTorch inputs; backwards-prune to primary output',
        'head_limit': 'Tests only the surviving downstream non-Conv arithmetic, not the full model',
        'guard': read(original/'protocol.json')['local_preflight'],
        'tolerances_changed': False, 'original_guard': 'FAILED_UNCHANGED',
        'dataset_inference': False, 'native_execution': False, 'deployment_graph_change': False,
        'interpretation': 'Counterfactual cuts can localize amplification; their closeness does not pass the original guard.'}
    write(out/'protocol.json', protocol)
    with (out/'console.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        import onnxruntime as ort
        import torch
        import ultralytics
        sys.path.insert(0, str(ROOT/'src'))
        import gate0
        torch.set_num_threads(4)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        def session(graph):
            return ort.InferenceSession(graph.SerializeToString(), sess_options=options,
                                        providers=['CPUExecutionProvider'])

        def stats(a, b):
            assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
            return {'shape': list(a.shape), 'elements': int(a.size),
                    'different': int(np.count_nonzero(a != b)), 'byte_identical': a.tobytes() == b.tobytes(),
                    'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
                    'rmse': float(np.sqrt(np.mean(delta*delta)))}

        def original_guard(a, b):
            result = {}
            for name, slc, rtol, atol in (('boxes', slice(None,4), .005, .5),
                                          ('scores', slice(4,None), .02, .002)):
                aa, bb = a[:,slc,:], b[:,slc,:]
                mask = ~np.isclose(aa, bb, rtol=rtol, atol=atol)
                result[name] = {'rtol': rtol, 'atol': atol, 'violating_elements': int(mask.sum()),
                                'total_elements': int(mask.size), 'within_original_tolerance': not bool(mask.any()),
                                'max_abs': float(np.max(np.abs(aa.astype(np.float64)-bb.astype(np.float64))))}
            result['both_slices_within_original_tolerance'] = all(result[n]['within_original_tolerance'] for n in ('boxes','scores'))
            return result

        mapping = read(original/'module_map.json')['mapping']
        calibration = read(original/'calibration.json')
        assert len(mapping) == 88
        x, *_ = gate0.image_tensor(image, 640, torch.device('cpu'))
        np.save(out/'preprocessed_input.npy', x.numpy())
        model = gate0.load_fused_model(weights, torch.device('cpu'))
        wrappers = gate0.install_qdq_wrappers(model, calibration['activation_max_abs'])
        gate0.enable_configuration(wrappers, 'q_all')
        recorded, handles = {}, []
        def record(mod, inputs, output):
            with torch.inference_mode():
                q = torch.round(inputs[0]/mod.activation_scale).clamp(-127,127)*mod.activation_scale
            recorded[mod.name] = {'input_qdq': q.detach().numpy().copy(),
                                  'output': output.detach().numpy().copy(),
                                  'weight_qdq': mod.quantized_weight.numpy().copy()}
        for mod in wrappers:
            handles.append(mod.register_forward_hook(record))
        try:
            with torch.inference_mode():
                pred = model(x)
                pt = (pred[0] if isinstance(pred,(tuple,list)) else pred).numpy().copy()
        finally:
            for handle in handles: handle.remove()
        graph = onnx.load(str(source))
        by_name = {n.name:n for n in graph.graph.node}
        init = {v.name:v for v in graph.graph.initializer}
        tapped = copy.deepcopy(graph)
        names = [graph.graph.output[0].name]
        for row in mapping:
            names.extend([by_name[row['onnx_node']].output[0], row['qdq_prefix']+'_a_dq', row['qdq_prefix']+'_w_dq'])
        existing_outputs = {v.name for v in tapped.graph.output}
        for name in names:
            if name not in existing_outputs:
                tapped.graph.output.append(helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,None))
                existing_outputs.add(name)
        # The diagnostic taps are serialized separately; source bytes remain untouched.
        onnx.save(tapped, str(out/'diagnostic_tapped_q_all.onnx'))
        ort_tapped = session(tapped)
        actual = dict(zip(names, ort_tapped.run(names, {'images':x.numpy()})))
        ort_raw = actual[names[0]]
        with np.load(original/'local_raw_outputs.npz', allow_pickle=False) as old:
            historical = {'pytorch': stats(pt, old['pytorch_qdq']), 'ort': stats(ort_raw, old['onnx_qdq'])}
        full_baseline = original_guard(ort_raw, pt)
        # Keep every original weight Q/DQ node and initializer byte unchanged.
        replay = copy.deepcopy(graph)
        replay_nodes, replay_initializers, replay_inputs, replay_outputs = [], {}, [], []
        feeds_pt, feeds_ort, all_rows = {}, {}, []
        for i, row in enumerate(mapping):
            prefix = row['qdq_prefix']
            for suffix in ('WeightQ','WeightDQ'):
                node = by_name[prefix+'_'+suffix]
                replay_nodes.append(copy.deepcopy(node))
                for name in node.input:
                    if name in init: replay_initializers[name] = init[name]
            conv = copy.deepcopy(by_name[row['onnx_node']])
            input_name = 'replay_input_'+str(i)
            conv.input[0] = input_name
            for name in conv.input:
                if name in init: replay_initializers[name] = init[name]
            replay_nodes.append(conv)
            a = recorded[row['pytorch_module']]['input_qdq']
            replay_inputs.append(helper.make_tensor_value_info(input_name,onnx.TensorProto.FLOAT,list(a.shape)))
            replay_outputs.append(helper.make_tensor_value_info(conv.output[0],onnx.TensorProto.FLOAT,
                                                               list(recorded[row['pytorch_module']]['output'].shape)))
            feeds_pt[input_name] = a
            feeds_ort[input_name] = actual[prefix+'_a_dq']
        del replay.graph.node[:]; replay.graph.node.extend(replay_nodes)
        del replay.graph.initializer[:]; replay.graph.initializer.extend(replay_initializers.values())
        del replay.graph.input[:]; replay.graph.input.extend(replay_inputs)
        del replay.graph.output[:]; replay.graph.output.extend(replay_outputs)
        del replay.graph.value_info[:]
        onnx.checker.check_model(replay, full_check=True)
        onnx.save(replay,str(out/'diagnostic_same_input_convs.onnx'))
        conv_session = session(replay)
        replay_names = [v.name for v in replay_outputs]
        control = conv_session.run(None, feeds_ort)
        counterfactual = conv_session.run(None, feeds_pt)
        for i, row in enumerate(mapping):
            prefix, module = row['qdq_prefix'], row['pytorch_module']
            expected = recorded[module]
            full_conv = actual[replay_names[i]]
            all_rows.append({'module':module,'group':row['group'],
                             'weight_qdq':stats(actual[prefix+'_w_dq'],expected['weight_qdq']),
                             'full_backend_input_qdq_difference':stats(actual[prefix+'_a_dq'],expected['input_qdq']),
                             'full_backend_conv_output_difference':stats(full_conv,expected['output']),
                             'ort_replay_same_ort_input_vs_original_ort_conv':stats(control[i],full_conv),
                             'ort_replay_same_pytorch_input_vs_original_pytorch_conv':stats(counterfactual[i],expected['output'])})
        write(out/'all_88_conv_replay.json', all_rows)
        # The output cut isolates ONNX decoding/non-Conv tail; it is not a full inference reference.
        cut = copy.deepcopy(graph)
        cuts = {by_name[r['onnx_node']].output[0]:r for r in mapping}
        producers = {name:n for n in graph.graph.node for name in n.output}
        needed_nodes, needed_cuts, needed_initializers = set(), set(), set()
        def visit(name):
            if not name: return
            if name in cuts:
                needed_cuts.add(name); return
            if name in init:
                needed_initializers.add(name); return
            node = producers.get(name)
            assert node is not None, 'Unexpected surviving external input: '+name
            if node.name in needed_nodes: return
            needed_nodes.add(node.name)
            for value in node.input: visit(value)
        visit(graph.graph.output[0].name)
        del cut.graph.node[:]
        cut.graph.node.extend(copy.deepcopy(n) for n in graph.graph.node if n.name in needed_nodes)
        del cut.graph.initializer[:]
        cut.graph.initializer.extend(copy.deepcopy(v) for v in graph.graph.initializer if v.name in needed_initializers)
        del cut.graph.input[:]
        cut_feeds = {}
        for name in sorted(needed_cuts):
            value = recorded[cuts[name]['pytorch_module']]['output']
            cut.graph.input.append(helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,list(value.shape)))
            cut_feeds[name] = value
        del cut.graph.output[1:]
        del cut.graph.value_info[:]
        onnx.checker.check_model(cut, full_check=True)
        onnx.save(cut,str(out/'diagnostic_recorded_conv_output_tail.onnx'))
        tail = session(cut).run(None,cut_feeds)[0]
        np.savez_compressed(out/'raw_primary_outputs.npz', pytorch=pt, ort=ort_raw,
                            recorded_pytorch_conv_output_ort_tail=tail)
        # First-layer arrays remain available for detailed independent boundary review.
        first_arrays={}
        for i in range(2):
            row=mapping[i]; prefix=row['qdq_prefix']; rec=recorded[row['pytorch_module']]
            for key in ('input_qdq','output'): first_arrays['pytorch_'+str(i)+'_'+key]=rec[key]
            first_arrays['ort_'+str(i)+'_input_qdq']=actual[prefix+'_a_dq']
            first_arrays['ort_'+str(i)+'_output']=actual[replay_names[i]]
            first_arrays['same_pytorch_input_ort_'+str(i)+'_output']=counterfactual[i]
        np.savez_compressed(out/'first_two_conv_arrays.npz',**first_arrays)
        control_exact = all(r['ort_replay_same_ort_input_vs_original_ort_conv']['byte_identical'] for r in all_rows)
        summary={'schema':'rampq.gate05.reference_replay_result/v1',
                 'status':'FIXED_IMAGE_CPU_COUNTERFACTUALS_COMPLETE',
                 'versions':{'torch':torch.__version__,'onnx':onnx.__version__,'ort':ort.__version__,
                             'ultralytics':ultralytics.__version__,'numpy':np.__version__},
                 'historical_full_output_reproduction':historical,
                 'full_original_tolerances_rechecked_both_slices':full_baseline,
                 'conv_replay_same_ort_inputs_all_88_bit_exact':control_exact,
                 'all_88_weight_qdq_byte_identical':all(r['weight_qdq']['byte_identical'] for r in all_rows),
                 'largest_full_backend_conv_output_difference':max(r['full_backend_conv_output_difference']['max_abs'] for r in all_rows),
                 'largest_same_pytorch_input_conv_output_difference':max(r['ort_replay_same_pytorch_input_vs_original_pytorch_conv']['max_abs'] for r in all_rows),
                 'first_two_conv_rows':all_rows[:2],
                 'tail_cut':{'external_recorded_conv_outputs':sorted(needed_cuts), 'remaining_nodes':len(cut.graph.node),
                             'remaining_ops':sorted(set(n.op_type for n in cut.graph.node)),
                             'output_difference_from_pytorch':stats(tail,pt),
                             'original_tolerance_diagnostic_only':original_guard(tail,pt)},
                 'original_complete_model_guard':'FAILED_UNCHANGED','new_tolerances_defined':False,
                 'dataset_evaluation_run':False,'native_execution_run':False,'cuda_used':False,
                 'accuracy_pass_claim':False,'performance_pass_claim':False,'numerical_migration_pass_claim':False,
                 'limits':'One existing fixed Selection image. Replay graphs cut all Conv dependencies. Exact same-ORT-input control must be checked before interpreting synchronized Conv differences. The tail cut only tests surviving decoding arithmetic. No counterfactual result passes or replaces the original guard.'}
    after = inventory(sources)
    assert before == after, 'Original source bytes changed'
    write(out/'source_hashes_after.json',after)
    summary['all_original_source_hashes_unchanged']=True
    write(out/'analysis.json',summary)
    files=sorted(p for p in out.rglob('*') if p.is_file())
    write(out/'manifest.json',{'schema':'rampq.gate05.reference_replay_files/v1',
                             'files':{p.relative_to(out).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}})
    print(json.dumps({k:summary[k] for k in ('status','full_original_tolerances_rechecked_both_slices',
          'conv_replay_same_ort_inputs_all_88_bit_exact','largest_same_pytorch_input_conv_output_difference',
          'largest_full_backend_conv_output_difference','tail_cut')},ensure_ascii=False))


if __name__ == '__main__':
    main()
