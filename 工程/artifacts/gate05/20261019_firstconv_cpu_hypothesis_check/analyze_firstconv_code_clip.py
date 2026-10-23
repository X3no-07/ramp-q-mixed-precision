"""CPU-only hypothesis check against the saved isolated native first Conv.

No original model, calibration, input, output, or acceptance criterion is changed.
The intentionally wrong code-domain Clip is a diagnostic prediction only.
"""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PROBE = ROOT / 'artifacts/gate05/20261019_firstconv_numeric_probe'
NATIVE = ROOT / 'artifacts/gate05/20261019_firstconv_interim_native'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require(test, message):
    if not test:
        raise RuntimeError(message)


def main():
    import numpy as np
    import onnx
    from onnx import numpy_helper
    import torch
    import torch.nn.functional as functional

    require(not (OUT / 'analysis.json').exists(), 'Refuse to overwrite prior derived results')
    sources = [PROBE / name for name in ('model_firstconv_qdq.onnx', 'preprocessed_input.npy',
               'spec.json', 'manifest.json', 'cpu_expected_outputs.npz', 'cpu_expectations.json')]
    sources += [NATIVE / name for name in ('result.json', 'raw_outputs.npz', 'engine_layers.json')]
    sources += [Path(__file__).resolve()]
    before = {str(p.relative_to(ROOT)).replace('\\', '/'): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sources}
    spec, result = read(PROBE / 'spec.json'), read(NATIVE / 'result.json')
    require(sha(PROBE / 'model_firstconv_qdq.onnx') == result['onnx_sha256'] == spec['variants']['qdq']['model_sha256'], 'Graph SHA differs')
    require(sha(PROBE / 'preprocessed_input.npy') == spec['input_file_sha256'], 'Frozen input SHA differs')
    require(sha(NATIVE / 'raw_outputs.npz') == result['raw_outputs_sha256'], 'Native output SHA differs')
    require(result['case'] == 'qdq_fp16_no_tf32', 'Unexpected saved native case')
    graph = onnx.load(str(PROBE / 'model_firstconv_qdq.onnx'))
    constants = {value.name: numpy_helper.to_array(value).copy() for value in graph.graph.initializer}
    by_type = {kind: [node for node in graph.graph.node if node.op_type == kind] for kind in ('Conv', 'QuantizeLinear', 'DequantizeLinear', 'Clip')}
    require(len(by_type['Conv']) == 1 and len(by_type['Clip']) == 1, 'Not the isolated frozen first Conv')
    conv, clip = by_type['Conv'][0], by_type['Clip'][0]
    producers = {name: node for node in graph.graph.node for name in node.output}
    adq, wdq = producers[conv.input[0]], producers[conv.input[1]]
    aq, wq = producers[adq.input[0]], producers[wdq.input[0]]
    require(aq.op_type == wq.op_type == 'QuantizeLinear' and adq.op_type == wdq.op_type == 'DequantizeLinear', 'Unexpected Q/DQ graph')
    require({a.name: onnx.helper.get_attribute_value(a) for a in wq.attribute}.get('axis') == 0, 'Weight Q axis differs')
    require({a.name: onnx.helper.get_attribute_value(a) for a in wdq.attribute}.get('axis') == 0, 'Weight DQ axis differs')
    require(aq.input[1:] == adq.input[1:] and wq.input[1:] == wdq.input[1:], 'Q/DQ constants differ')
    require(np.count_nonzero(constants[aq.input[2]]) == np.count_nonzero(constants[wq.input[2]]) == 0, 'Nonzero zero point')
    x = np.load(PROBE / 'preprocessed_input.npy', allow_pickle=False)
    a_scale = constants[aq.input[1]]
    w_scale = constants[wq.input[1]]
    weight, bias = constants[wq.input[0]], constants[conv.input[2]]
    lo, hi = constants[clip.input[1]], constants[clip.input[2]]
    require(x.dtype == np.float32 and x.shape == (1, 3, 640, 640), 'Wrong fixed input')
    require(float(lo) == -1.0 and float(hi) == 1.0 and x.min() >= lo and x.max() <= hi, 'First Clip not identity on fixed input')
    require(np.all(w_scale > 0) and float(a_scale) > 0, 'Invalid scales')
    a_codes = np.rint(np.clip(x, lo, hi) / a_scale).clip(-128, 127).astype(np.int8)
    # Deliberate hypothesis: float Clip bounds are wrongly applied to raw INT8 codes.
    wrong_codes = np.clip(a_codes, int(lo), int(hi)).astype(np.int8)
    w_codes = np.rint(weight / w_scale[:, None, None, None]).clip(-128, 127).astype(np.int8)
    w_dequant = w_codes.astype(np.float32) * w_scale[:, None, None, None]
    torch.set_num_threads(4)
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in conv.attribute}
    require(attrs['pads'] == [1, 1, 1, 1] and attrs['strides'] == [2, 2] and attrs.get('group', 1) == 1, 'Unexpected Conv geometry')

    def stats(a):
        a = np.asarray(a)
        flat = a.astype(np.float64).ravel()
        return {'shape': list(a.shape), 'dtype': str(a.dtype), 'all_finite': bool(np.isfinite(a).all()),
                'min': float(a.min()), 'max': float(a.max()), 'mean': float(flat.mean()),
                'rms': float(np.sqrt(np.mean(flat * flat))),
                'raw_bytes_sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}

    def diff(a, b):
        delta = a.astype(np.float64) - b.astype(np.float64)
        return {'bit_exact': a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(),
                'equal_elements': int(np.count_nonzero(a == b)), 'elements': a.size,
                'max_abs': float(np.abs(delta).max()), 'mean_abs': float(np.abs(delta).mean()),
                'rms': float(np.sqrt(np.mean(delta * delta))),
                'abs_percentiles_50_90_99_999': np.percentile(np.abs(delta), [50, 90, 99, 99.9]).tolist()}

    def convolution(codes):
        inp = torch.from_numpy(codes.astype(np.float32) * a_scale)
        with torch.inference_mode():
            pre = functional.conv2d(inp, torch.from_numpy(w_dequant), torch.from_numpy(bias), stride=2, padding=1)
            post = pre * torch.sigmoid(pre)
        return pre.numpy().copy(), post.numpy().copy()

    correct_pre, correct_post = convolution(a_codes)
    wrong_pre, wrong_post = convolution(wrong_codes)
    with np.load(NATIVE / 'raw_outputs.npz', allow_pickle=False) as saved:
        native_pre, native_post = saved['first_qdq_pre_silu'].copy(), saved['first_qdq_post_silu'].copy()
    with np.load(PROBE / 'cpu_expected_outputs.npz', allow_pickle=False) as saved:
        ort_pre, ort_post = saved['qdq_pre_silu'].copy(), saved['qdq_post_silu'].copy()

    # Independent exact integer accumulation, then alternative FP32 rescale orders.
    padded = np.pad(wrong_codes.astype(np.int64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))[:, :, ::2, ::2]
    accum = np.einsum('nchwkl,ockl->nohw', windows, w_codes.astype(np.int64), dtype=np.int64, optimize=True)
    af = accum.astype(np.float32)
    product_scale = np.float32(a_scale * w_scale)[None, :, None, None]
    bias_view = bias[None, :, None, None]
    integer_variants = {
        'integer_accum_scale_product_separate_add': np.float32(af * product_scale + bias_view),
        'integer_accum_scale_product_single_round_fma_model': np.float32(af.astype(np.float64) * product_scale.astype(np.float64) + bias_view.astype(np.float64)),
        'integer_accum_activation_then_weight_scale': np.float32(np.float32(af * a_scale) * w_scale[None, :, None, None] + bias_view),
    }
    fits = []
    for channel in range(16):
        target = native_pre[0, channel].astype(np.float64) - float(bias[channel])
        row = {'channel': channel, 'bias': float(bias[channel]), 'weight_scale': float(w_scale[channel]),
               'native_pre_min': float(native_pre[0, channel].min()), 'native_pre_max': float(native_pre[0, channel].max()), 'fits_of_y_minus_frozen_bias': {}}
        for name, pred in [('correct_qdq', correct_pre), ('wrong_code_clip', wrong_pre)]:
            feature = pred[0, channel].astype(np.float64) - float(bias[channel])
            slope = float(np.dot(feature.ravel(), target.ravel()) / np.dot(feature.ravel(), feature.ravel()))
            residual = target - slope * feature
            var = float(np.sum((target - target.mean()) ** 2))
            row['fits_of_y_minus_frozen_bias'][name] = {'fixed_zero_intercept_slope': slope,
                'rms_residual': float(np.sqrt(np.mean(residual ** 2))), 'max_abs_residual': float(np.abs(residual).max()),
                'r_squared': float(1 - np.sum(residual ** 2) / var) if var else None}
        fits.append(row)

    def counts(a):
        values, counts_ = np.unique(a, return_counts=True)
        return {'unique_count': len(values), 'values_and_counts': [[int(v), int(c)] for v, c in zip(values, counts_)]}

    arrays = {'correct_pre': correct_pre, 'correct_post': correct_post, 'wrong_code_clip_pre': wrong_pre,
              'wrong_code_clip_post': wrong_post, 'correct_activation_codes': a_codes, 'wrong_activation_codes': wrong_codes,
              'weight_codes': w_codes, 'wrong_integer_accumulator': accum}
    arrays.update(integer_variants)
    np.savez_compressed(OUT / 'predictions.npz', **arrays)
    after = {str(p.relative_to(ROOT)).replace('\\', '/'): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sources}
    require(before == after, 'A source file changed during analysis')
    write(OUT / 'source_hashes_before.json', before)
    write(OUT / 'source_hashes_after.json', after)
    report = {'schema': 'rampq.gate05.firstconv_code_domain_clip_cpu_hypothesis/v1',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'status': 'CPU_DIAGNOSTIC_COMPLETE',
        'hypothesis': 'Correct first activation INT8 codes incorrectly clipped by original float bounds [-1,+1], then dequantized with unchanged a_scale',
        'source_case': result['case'], 'source_engine_sha256': result['engine_sha256'], 'source_native_raw_sha256': sha(NATIVE / 'raw_outputs.npz'),
        'native_evidence_receipt_status': 'Interim native result/raw SHA verified here; complete external receipt pending parent collection',
        'activation_scale': float(a_scale), 'float_clip_bounds': [float(lo), float(hi)],
        'input_stats': stats(x), 'correct_activation_codes': dict(stats(a_codes), **counts(a_codes)),
        'wrong_activation_codes': dict(stats(wrong_codes), **counts(wrong_codes)), 'weight_codes': dict(stats(w_codes), **counts(w_codes)),
        'output_stats': {name: stats(a) for name, a in [('native_pre', native_pre), ('native_post', native_post),
            ('correct_cpu_pre', correct_pre), ('correct_cpu_post', correct_post), ('wrong_code_clip_pre', wrong_pre), ('wrong_code_clip_post', wrong_post)]},
        'comparisons_to_native': {'correct_torch_pre': diff(correct_pre, native_pre), 'correct_torch_post': diff(correct_post, native_post),
            'saved_correct_ort_pre': diff(ort_pre, native_pre), 'saved_correct_ort_post': diff(ort_post, native_post),
            'wrong_code_clip_torch_pre': diff(wrong_pre, native_pre), 'wrong_code_clip_torch_post': diff(wrong_post, native_post),
            **{name: diff(a, native_pre) for name, a in integer_variants.items()}},
        'correct_torch_to_saved_ort': {'pre': diff(correct_pre, ort_pre), 'post': diff(correct_post, ort_post)},
        'per_channel_fits': fits, 'predictions_sha256': sha(OUT / 'predictions.npz'),
        'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__, 'onnx': onnx.__version__,
        'cuda_used': False, 'native_execution_run': False, 'full_detector_run': False, 'cpu_convolutions_run': 2,
        'calibration_or_source_changed': False, 'all_source_hashes_unchanged': before == after,
        'new_tolerance_or_pass_criterion': False, 'original_complete_model_preflight': 'FAILED_UNCHANGED',
        'limits': ['This only predicts the saved isolated first Conv; it does not establish the complete engine root cause.',
            'Exact element equality and descriptive numerical differences are reported without a new acceptance threshold.',
            'Pre-SiLU output tap has removed SiLU fusion; actual saved Conv is already discrepant before SiLU.',
            'Integer-rescale variants model floating evaluation order; they do not inspect native instructions or packed scales.']}
    write(OUT / 'analysis.json', report)
    (OUT / 'README.md').write_text('CPU-only diagnostic of the saved isolated first Conv. Original graph/input/results remain unchanged. See analysis.json for exact comparisons, source hashes, and evidence limits. No new threshold, complete-model run, GPU execution, or board access.\n', encoding='utf-8')
    files = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('manifest.json', 'SHA256SUMS')}
    write(OUT / 'manifest.json', {'schema': 'rampq.gate05.cpu_diagnostic_manifest/v1', 'files': files})
    (OUT / 'SHA256SUMS').write_text(''.join('{}  {}\n'.format(info['sha256'], name) for name, info in sorted(files.items())) + '{}  manifest.json\n'.format(sha(OUT / 'manifest.json')), encoding='utf-8')
    print(json.dumps({'analysis_sha256': sha(OUT / 'analysis.json'), 'comparisons_to_native': report['comparisons_to_native']}, indent=2))


if __name__ == '__main__':
    main()
