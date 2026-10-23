"""Describe frozen saved TensorRT-versus-ORT arrays; never infer or run NMS.

All six configurations and all 150 ordered inputs are mandatory. Shape, input,
graph/engine and byte identities fail closed. No tolerance or numeric PASS is
defined. --self-test uses only synthetic NumPy arrays and standard-library files.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys

sys.dont_write_bytecode = True
ORDER = ['fp16_control', 'q_all', 'r_only_G1', 'r_only_G1_G5', 'r_only_G1_G2', 'q_only_G1']
PE = '/model.10/m/m.0/attn/pe/conv/Conv_output_0'
KEYS = {'output0': 'main_output', PE: 'pe_output'}
SHAPES = {'output0': (1, 14, 8400), PE: (1, 128, 20, 20)}
RUNNER_SHA = 'ef7e2780203c04442ee3378b7cea31093dafa40eb29c4cea9c012cdb0cd4d26b'
DATASET_SHA = 'c4dac80ad7c59524194a9501f5ff8126cc0517e84f29f4106068c0c4b27a93aa'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def contained(root, name, directory=False):
    rel = PurePosixPath(name)
    require(str(rel) == name and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,
            'Unsafe saved path: ' + str(name))
    path = root.joinpath(*rel.parts)
    require(root in path.resolve().parents and not path.is_symlink(), 'Saved path escapes root: ' + name)
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), 'Saved path has symlink parent: ' + name)
    require(path.is_dir() if directory else path.is_file(), 'Missing saved path: ' + name)
    return path


def npy_identity(path):
    """Check shared FLOAT32 payload without importing or executing any runtime."""
    with path.open('rb') as stream:
        require(stream.read(6) == b'\x93NUMPY', 'Not NPY')
        version = stream.read(2)
        require(version in (b'\x01\x00', b'\x02\x00', b'\x03\x00'), 'Unsupported NPY version')
        size = 2 if version == b'\x01\x00' else 4
        encoded = stream.read(size)
        require(len(encoded) == size, 'Truncated NPY length')
        count = struct.unpack('<H' if size == 2 else '<I', encoded)[0]
        require(0 < count <= 65536, 'Unreasonable NPY header')
        header_bytes = stream.read(count)
        require(len(header_bytes) == count, 'Truncated NPY header')
        header = ast.literal_eval(header_bytes.decode('utf-8' if version == b'\x03\x00' else 'latin1').strip())
        require(header['descr'] in ('<f4', '=f4') and header['fortran_order'] is False and
                header['shape'] == (1, 3, 640, 640), 'Wrong common input shape/dtype/layout')
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    require(path.stat().st_size == 8 + size + count + 4 * 3 * 640 * 640, 'Wrong common input payload size')
    return digest.hexdigest()


def match_frozen_input(row, frozen):
    keys = ('image_index', 'image', 'image_sha256', 'label', 'label_sha256', 'input_file',
            'input_sha256', 'original_shape', 'ratio', 'left', 'top')
    require(all(row[k] == frozen[k] for k in keys) and row['input_array_sha256'] == frozen['array_sha256'],
            'Raw row input identity/order/letterbox differs')


def validate(args):
    index_path, contract_path = args.raw_index.resolve(), args.contract.resolve()
    require(sha(index_path) == args.raw_index_sha256 and sha(contract_path) == args.contract_sha256,
            'Index or frozen acceptance SHA256 differs')
    root, index, acceptance = index_path.parent, read(index_path), read(contract_path)
    require(acceptance['schema'] == 'rampq.gate05.dataset_acceptance_contract/v1' and
            acceptance['status'] == 'FROZEN_BEFORE_DATASET_INFERENCE' and
            [r['name'] for r in acceptance['configurations']] == ORDER and
            acceptance['protected_history']['original_complete_PyTorch_ORT_guard'] == 'FAILED_UNCHANGED',
            'Wrong or incomplete frozen acceptance contract')
    frozen_root = Path(acceptance['postprocess_execution']['matrix_root'])
    require(frozen_root.is_absolute() and frozen_root.resolve() == frozen_root and
            root == frozen_root and frozen_root.is_dir() and not frozen_root.is_symlink(),
            'Raw index parent must equal the frozen absolute canonical matrix_root')
    raw_path = contract_path.parent / 'raw_contract.json'
    require(sha(raw_path) == acceptance['raw_contract_sha256'], 'Frozen raw contract sibling differs')
    raw = read(raw_path)
    require(raw['schema'] == 'rampq.gate05.frozen_raw_dataset_contract/v1' and
            raw['status'] == 'FROZEN_BEFORE_DATASET_INFERENCE' and raw['runner_sha256'] == RUNNER_SHA and
            list(raw['configurations']) == ORDER and raw['no_NMS'] is True and
            raw['warmup_calls'] == raw['extra_inference_calls'] == raw['new_builds'] == 0 and
            raw['original_complete_model_numerical_guard'] == 'FAILED_UNCHANGED', 'Wrong frozen raw scope')
    require(index['schema'] == 'rampq.gate05.frozen_raw_family_index/v1' and
            index['contract_sha256'] == args.contract_sha256 and index['raw_contract_sha256'] == sha(raw_path) and
            index['dataset_manifest_sha256'] == acceptance['fixed_data_and_model']['dataset_manifest_sha256'] == DATASET_SHA,
            'Index dataset or contract lineage differs')
    streams = index['streams']
    expected = [{'configuration': cfg, 'backend': backend} for backend in ('trt', 'ort') for cfg in ORDER]
    require(len(streams) == 12 and [{'configuration': r['configuration'], 'backend': r['backend']} for r in streams] ==
            acceptance['postprocess_execution']['stream_order'] == expected, 'Exactly frozen twelve-stream order required')
    inputs = contained(root, index['inputs_directory'], directory=True)
    input_manifest = contained(inputs, index.get('input_manifest_file', 'manifest.json'))
    require(sha(input_manifest) == index['input_manifest_sha256'] == acceptance['input_manifest_sha256'] ==
            raw['input_manifest']['sha256'], 'Frozen shared input manifest differs')
    prepared = read(input_manifest)
    frozen = prepared['images']
    require(prepared['schema'] == raw['input_manifest']['schema'] == 'rampq.gate05.shared_preprocessed_inputs/v1' and
            prepared['status'] == 'ALL150_COMMON_INPUTS_PREPARED_NO_MODEL_EXECUTION' and
            prepared['model_executions'] == 0 and prepared['data_manifest_sha256'] == DATASET_SHA and
            len(frozen) == raw['input_manifest']['count'] == 150 and
            [r['image_index'] for r in frozen] == list(range(150)) and
            [r['image'] for r in frozen] == raw['input_manifest']['ordered_ids'] and
            len({r['input_file'] for r in frozen}) == len({r['image'] for r in frozen}) == 150,
            'Wrong shared input order/inventory')
    protected = {str(p): sha(p) for p in (index_path, contract_path, raw_path, input_manifest, Path(__file__).resolve())}
    for row in frozen:
        path = contained(inputs, row['input_file'])
        require(path.stat().st_size == row['input_bytes'] and sha(path) == row['input_sha256'] and
                row['shape'] == [1, 3, 640, 640] and row['dtype'] == 'float32' and
                npy_identity(path) == row['array_sha256'], 'Frozen input bytes/payload differ')
        protected[str(path)] = row['input_sha256']
    evidence = index['engineering_evidence']; path = contained(root, evidence['file'])
    require(sha(path) == evidence['sha256'], 'Engineering evidence receipt differs')
    protected[str(path)] = evidence['sha256']
    bindings = {row['name']: row for row in acceptance['configurations']}
    validated = {}
    output = args.output.resolve()
    require(not output.exists() and output.parent.is_dir() and not output.is_symlink() and
            inputs not in output.parents, 'Fresh result outside frozen shared inputs required')
    for stream in streams:
        path = contained(root, stream['file']); digest = sha(path)
        require(digest == stream['sha256'] and path.parent not in output.parents, 'Raw manifest changed or output modifies raw package')
        manifest = read(path); cfg, backend = stream['configuration'], stream['backend']; lineage = manifest['lineage']
        policy = raw['configurations'][cfg][backend]
        require(manifest['schema'] == 'rampq.gate05.frozen_raw_dataset_outputs/v1' and
                manifest['status'] == 'ALL150_RAW_OUTPUTS_SERIALIZED_NO_SCORING' and
                manifest['NMS_run'] is False and manifest['target_matching_run'] is False and
                manifest['dataset_accuracy_evaluated'] is False and manifest['output_npz_keys'] == raw['output_npz_keys'] == KEYS,
                'Incomplete/wrong raw stream or postprocessing already included')
        require(manifest['bindings'] == raw['bindings'], 'Raw external binding contract differs')
        require(lineage['configuration'] == cfg and lineage['backend'] == backend and lineage['runner_sha256'] == RUNNER_SHA and
                lineage['contract_sha256'] == index['raw_contract_sha256'] and
                lineage['input_manifest_sha256'] == index['input_manifest_sha256'] and lineage['source_data_manifest_sha256'] == DATASET_SHA,
                'Raw stream lineage differs')
        graph = bindings[cfg]['source_graph_sha256' if backend == 'trt' else 'independent_ORT_reference_graph_sha256']
        require(lineage['graph_sha256'] == graph == raw['configurations'][cfg]['graph_sha256'] and
                lineage['engine_sha256'] == (bindings[cfg]['engine_sha256'] if backend == 'trt' else None), 'Graph/engine source differs')
        require(policy['authorized'] is True and policy['calls'] == 150 and
                (backend != 'trt' or policy['engine_sha256'] == lineage['engine_sha256']), 'Wrong authorized raw source')
        rows, state = manifest['images'], manifest['state']
        require(state['phase'] == 'COMPLETE' and state['status'] == manifest['status'] and
                state['model_calls_attempted'] == state['model_calls_completed'] == state['serialized_images'] == len(rows) == 150,
                'Raw stream must include all150 original calls')
        require(state['lineage'] == lineage and [r['image_index'] for r in rows] == list(range(150)), 'State lineage/order differs')
        paths = []
        for row, reference in zip(rows, frozen):
            match_frozen_input(row, reference)
            require(row['output_npz_keys'] == KEYS, 'Wrong output keys/layout')
            tensor = contained(path.parent, row['raw_file'])
            require(tensor.stat().st_size == row['raw_bytes'] and sha(tensor) == row['raw_sha256'], 'Saved tensor file identity differs')
            protected[str(tensor)] = row['raw_sha256']; paths.append(tensor)
        require(len(set(paths)) == 150, 'Raw stream repeats a saved tensor file path')
        protected[str(path)] = digest
        validated[(cfg, backend)] = {'path': path, 'sha256': digest, 'manifest': manifest, 'paths': paths}
    return root, index, acceptance, raw, frozen, validated, protected


def load_pair_array(path, row, np):
    payload = path.read_bytes()
    require(len(payload) == row['raw_bytes'] and hashlib.sha256(payload).hexdigest() == row['raw_sha256'], 'Tensor changed before comparison')
    arrays = {}
    require(set(row['statistics']) == set(KEYS), 'Wrong raw statistics output inventory')
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        require(len(saved.files) == 2 and set(saved.files) == set(KEYS.values()), 'Wrong raw NPZ array inventory')
        for name, key in KEYS.items():
            value = saved[key]
            require(value.dtype == np.float32 and value.shape == SHAPES[name] and value.flags.c_contiguous,
                    'Saved raw tensor shape/dtype/layout differs: ' + name)
            statistics = row['statistics'][name]
            digest = hashlib.sha256(value.tobytes(order='C')).hexdigest()
            require(statistics['raw_array_sha256'] == digest and statistics['shape'] == list(value.shape) and
                    statistics['dtype'] == 'float32', 'Saved raw array payload identity differs: ' + name)
            finite = np.isfinite(value)
            require(statistics['finite'] is bool(finite.all()) and statistics['nonfinite_elements'] == int(np.count_nonzero(~finite)),
                    'Saved raw finite statistics do not match actual array')
            arrays[name] = value
    require(row['all_outputs_finite'] is all(bool(np.isfinite(v).all()) for v in arrays.values()), 'Row finite flag differs from tensors')
    return arrays


def describe(native, ort, np):
    require(native.shape == ort.shape and native.dtype == ort.dtype == np.float32, 'Comparison shape/dtype mismatch')
    nf, of = np.isfinite(native), np.isfinite(ort); paired = nf & of
    count, nfinite, ofinite = int(native.size), int(nf.sum()), int(of.sum())
    difference = np.abs(native[paired].astype(np.float64) - ort[paired].astype(np.float64))
    absolute_sum = float(difference.sum(dtype=np.float64)); square_sum = float(np.square(difference).sum(dtype=np.float64))
    pairs = int(difference.size); mismatch = int(np.count_nonzero(native != ort))
    def stats(array, finite):
        values = array[finite].astype(np.float64)
        return {'finite_count': int(values.size), 'nonfinite_count': int(array.size-values.size),
                'min_finite': float(values.min()) if values.size else None,
                'max_finite': float(values.max()) if values.size else None,
                'mean_finite': float(values.mean()) if values.size else None}
    return {'shape': list(native.shape), 'dtype': 'float32', 'elements': count,
            'native': stats(native, nf), 'ORT': stats(ort, of), 'paired_finite_count': pairs,
            'paired_nonfinite_count': count-pairs, 'numeric_equal': mismatch == 0, 'exact_mismatch_count': mismatch,
            'byte_identical': native.tobytes(order='C') == ort.tobytes(order='C'),
            'max_abs_difference_finite_pairs': float(difference.max()) if pairs else None,
            'mean_abs_difference_finite_pairs': absolute_sum/pairs if pairs else None,
            'RMS_difference_finite_pairs': math.sqrt(square_sum/pairs) if pairs else None,
            'absolute_difference_sum_finite_pairs': absolute_sum, 'squared_difference_sum_finite_pairs': square_sum,
            'no_finite_pair_reason': None if pairs else 'NO_PAIRED_FINITE_ELEMENTS',
            'all_outputs_finite': nfinite == ofinite == count}


def aggregate(rows):
    count = sum(row['elements'] for row in rows); pairs = sum(row['paired_finite_count'] for row in rows)
    absolute_sum = sum(row['absolute_difference_sum_finite_pairs'] for row in rows)
    square_sum = sum(row['squared_difference_sum_finite_pairs'] for row in rows)
    maxima = [row['max_abs_difference_finite_pairs'] for row in rows if row['paired_finite_count']]
    return {'images': len(rows), 'elements': count,
            'native_finite_count': sum(row['native']['finite_count'] for row in rows),
            'ORT_finite_count': sum(row['ORT']['finite_count'] for row in rows), 'paired_finite_count': pairs,
            'paired_nonfinite_count': count-pairs, 'exact_mismatch_count': sum(row['exact_mismatch_count'] for row in rows),
            'all_numeric_equal': all(row['numeric_equal'] for row in rows), 'all_byte_identical': all(row['byte_identical'] for row in rows),
            'max_abs_difference_finite_pairs': max(maxima) if maxima else None,
            'mean_abs_difference_finite_pairs': absolute_sum/pairs if pairs else None,
            'RMS_difference_finite_pairs': math.sqrt(square_sum/pairs) if pairs else None,
            'all_outputs_finite': all(row['all_outputs_finite'] for row in rows),
            'aggregation': 'Element-weighted sums across all150images; never average image RMS or suppress unfavorable images.'}


def execute(args):
    _, index, acceptance, raw, frozen, streams, protected = validate(args)
    import numpy as np
    configurations = []; finite = True
    for cfg in ORDER:
        native, ort = streams[(cfg, 'trt')], streams[(cfg, 'ort')]
        images = []
        for position, reference in enumerate(frozen):
            nrow, orow = native['manifest']['images'][position], ort['manifest']['images'][position]
            na = load_pair_array(native['paths'][position], nrow, np); oa = load_pair_array(ort['paths'][position], orow, np)
            parts = {'main_coordinates_4': describe(na['output0'][:, :4, :], oa['output0'][:, :4, :], np),
                     'class_scores_10': describe(na['output0'][:, 4:14, :], oa['output0'][:, 4:14, :], np),
                     'PE_output': describe(na[PE], oa[PE], np)}
            finite = finite and all(row['all_outputs_finite'] for row in parts.values())
            images.append({'image_index': position, 'image': reference['image'], 'image_sha256': reference['image_sha256'],
                  'input_sha256': reference['input_sha256'], 'input_array_sha256': reference['array_sha256'],
                  'native_raw_file': str(native['paths'][position]), 'native_raw_sha256': nrow['raw_sha256'],
                  'ORT_raw_file': str(ort['paths'][position]), 'ORT_raw_sha256': orow['raw_sha256'],
                  'native_output_array_sha256': {k: v['raw_array_sha256'] for k, v in nrow['statistics'].items()},
                  'ORT_output_array_sha256': {k: v['raw_array_sha256'] for k, v in orow['statistics'].items()}, 'comparisons': parts})
        configurations.append({'configuration': cfg, 'actual_build_policy': raw['configurations'][cfg]['actual_build_policy'],
              'auxiliary_different_policy': cfg == 'q_only_G1', 'native_manifest': str(native['path']),
              'native_manifest_sha256': native['sha256'], 'ORT_manifest': str(ort['path']), 'ORT_manifest_sha256': ort['sha256'],
              'native_graph_sha256': native['manifest']['lineage']['graph_sha256'],
              'ORT_graph_sha256': ort['manifest']['lineage']['graph_sha256'], 'native_engine_sha256': native['manifest']['lineage']['engine_sha256'],
              'aggregates': {key: aggregate([row['comparisons'][key] for row in images]) for key in ('main_coordinates_4', 'class_scores_10', 'PE_output')},
              'images': images})
    require(protected == {name: sha(Path(name)) for name in protected}, 'Protected saved inputs changed during comparison')
    result = {'schema': 'rampq.gate05.saved_raw_reference_numeric_comparison/v1',
              'status': 'SAVED_RAW_DESCRIPTIVE_COMPARISON_COMPLETE' if finite else 'TECHNICAL_INELIGIBLE_NONFINITE_SAVED_OUTPUTS_REPORTED',
              'client_at_UTC': datetime.now(timezone.utc).isoformat(), 'tool_sha256': sha(Path(__file__).resolve()),
              'raw_index_sha256': args.raw_index_sha256, 'acceptance_contract_sha256': args.contract_sha256,
              'raw_contract_sha256': acceptance['raw_contract_sha256'], 'input_manifest_sha256': index['input_manifest_sha256'],
              'dataset_manifest_sha256': DATASET_SHA, 'numpy': np.__version__, 'configuration_order': ORDER,
              'configurations': configurations, 'images_per_configuration': 150, 'raw_streams': 12, 'all_outputs_finite': finite,
              'protected_file_hashes_before_and_after': protected, 'model_execution_calls': 0, 'NMS_calls': 0,
              'GPU_runtime_imports': 0, 'configuration_selection_performed': False, 'new_numeric_tolerances': False,
              'numeric_equivalence_PASS_claim': False, 'original_PyTorch_ORT_guard_retested': False,
              'original_complete_model_numerical_guard': 'FAILED_UNCHANGED', 'gate05_status': 'INCOMPLETE',
              'migration_or_accuracy_or_performance_PASS_claim': False,
              'limits': 'Saved complete corresponding full ORT versus deployed TensorRT arrays only. Coordinates are rawnetworkxywh channels0:4, classes channels4:14, PE separate. No decode/NMS/targetmatching, tolerance waiver, original PyTorch guard repair, pureG1 causal attribution or GatePASS.'}
    with args.output.resolve().open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()), 'output_sha256': sha(args.output.resolve()),
          'configurations': 6, 'images_each': 150, 'all_outputs_finite': finite, 'model_calls': 0, 'NMS_calls': 0}), flush=True)


def self_test():
    import numpy as np
    first = np.array([0., 1., 2., -4.], dtype=np.float32)
    second = np.array([-0., 2., 0., -4.], dtype=np.float32)
    result = describe(first, second, np)
    require(result['exact_mismatch_count'] == 2 and result['byte_identical'] is False and
            result['max_abs_difference_finite_pairs'] == 2. and result['mean_abs_difference_finite_pairs'] == .75 and
            result['RMS_difference_finite_pairs'] == math.sqrt(1.25), 'Synthetic difference statistics wrong')
    zero = describe(np.array([0.], np.float32), np.array([-0.], np.float32), np)
    require(zero['numeric_equal'] is True and zero['byte_identical'] is False, 'Signed zero byte/numeric distinction lost')
    nonfinite = describe(np.array([np.nan, np.inf, 1.], np.float32), np.array([0., np.inf, 3.], np.float32), np)
    require(nonfinite['paired_finite_count'] == 1 and nonfinite['paired_nonfinite_count'] == 2 and
            nonfinite['all_outputs_finite'] is False and nonfinite['mean_abs_difference_finite_pairs'] == 2., 'Nonfinite accounting wrong')
    combined = aggregate([result, zero])
    require(combined['elements'] == 5 and combined['mean_abs_difference_finite_pairs'] == .6 and
            combined['RMS_difference_finite_pairs'] == 1., 'Element-weighted aggregate wrong')
    frozen = {'image_index': 0, 'image': 'evaluation/images/a.jpg', 'image_sha256': 'a', 'label': 'a.txt',
              'label_sha256': 'b', 'input_file': 'inputs/a.npy', 'input_sha256': 'c', 'original_shape': [3, 4],
              'ratio': 1., 'left': 0, 'top': 0, 'array_sha256': 'd'}
    saved = dict(frozen, input_array_sha256='d'); match_frozen_input(saved, frozen)
    saved['input_sha256'] = 'changed'
    try:
        match_frozen_input(saved, frozen)
    except ValueError:
        pass
    else:
        raise AssertionError('Input mismatch accepted')
    require(not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules
                    for prefix in ('torch', 'tensorrt', 'onnxruntime', 'onnx', 'ultralytics')), 'Forbidden model/NMS runtime imported')
    print(json.dumps({'status': 'SYNTHETIC_SAVED_ARRAY_SELF_TESTS_PASS', 'tests': 6, 'model_calls': 0,
          'NMS_calls': 0, 'real_dataset_outputs_read': False, 'numpy': np.__version__}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-index', type=Path); parser.add_argument('--raw-index-sha256')
    parser.add_argument('--contract', type=Path); parser.add_argument('--contract-sha256')
    parser.add_argument('--output', type=Path); parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        require(all((args.raw_index, args.raw_index_sha256, args.contract, args.contract_sha256, args.output)),
                'Exact frozen index/contract identities and fresh output path required')
        execute(args)


if __name__ == '__main__':
    main()
