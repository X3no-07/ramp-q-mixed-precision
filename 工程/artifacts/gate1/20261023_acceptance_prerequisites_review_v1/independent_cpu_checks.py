"""Small invented CPU checks; no model, image, labels, or historical tensors."""
from pathlib import Path
import importlib.util
import json
import os
import hashlib

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    loss_source = ROOT / 'scripts/gate1_ignore_negative_loss.py'
    format_source = ROOT / 'scripts/gate1_fp32_reference_state.py'
    loss = module('review_ignore_primitive', loss_source)
    fmt = module('review_fp32_snapshot', format_source)
    # Two logits share one parameter. Retained logit gradients are unchanged,
    # while removing an ignored negative changes the shared parameter gradient.
    shared = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    logits = (shared * torch.tensor([1.0, 2.0], dtype=torch.float64)).reshape(1, 2, 1)
    targets = torch.tensor([[[1.0], [0.0]]], dtype=torch.float64)
    fg = torch.tensor([[True, False]])
    result = loss.masked_classification_bce(
        logits, targets, fg, torch.tensor([[1.0, 1.0], [3.0, 3.0]], dtype=torch.float64),
        [torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float64)], (5, 5))
    ordinary = F.binary_cross_entropy_with_logits(logits, targets, reduction='sum')
    masked_logits_gradient = torch.autograd.grad(result['loss'], logits, retain_graph=True)[0]
    ordinary_logits_gradient = torch.autograd.grad(ordinary, logits, retain_graph=True)[0]
    assert masked_logits_gradient[0, 0, 0] == ordinary_logits_gradient[0, 0, 0]
    assert masked_logits_gradient[0, 1, 0] == 0
    masked_shared = torch.autograd.grad(result['loss'], shared, retain_graph=True)[0]
    ordinary_shared = torch.autograd.grad(ordinary, shared)[0]
    assert masked_shared.item() == -0.5 and ordinary_shared.item() == 0.5

    # Bit-pattern integrity covers signed zero, a noncontiguous view, a tiny
    # finite value and nonfloating state, without any architecture/model load.
    source = torch.tensor([-0.0, 0.0, 1.00012362, torch.finfo(torch.float32).tiny], dtype=torch.float32)
    state = {'view': source[::2], 'matrix': source.reshape(2, 2).t(),
             'counter': torch.tensor(9, dtype=torch.int64), 'flag': torch.tensor(True)}
    assert not state['matrix'].is_contiguous()
    metadata = {'training_contract_sha256': 'c' * 64, 'architecture_sha256': 'd' * 64,
                'epoch_one_based': 1, 'source_stage': fmt.STAGE}
    snapshot = HERE / 'synthetic_bitpattern_snapshot'
    receipt = fmt.write_snapshot(state, snapshot, metadata)
    restored, restored_metadata = fmt.load_snapshot(snapshot)
    assert fmt.describe_state(restored) == fmt.describe_state(state)
    assert restored_metadata == metadata and torch.signbit(restored['view'][0])
    assert not receipt['training_origin_independently_proven']
    # An already lifted input is numerically valid FP32 and cannot reveal its
    # lost provenance from dtype; only a future live capture can establish it.
    lifted = torch.tensor([1.234567], dtype=torch.float32).half().float()
    frozen = fmt.freeze_state({'w': lifted})
    assert torch.equal(frozen['w'], lifted)
    record = {
        'status': 'PASS_SMALL_INDEPENDENT_SYNTHETIC_CPU_CHECKS_ONLY',
        'fixed_positive_logit_gradient_unchanged': True,
        'ignored_negative_logit_gradient_zero': True,
        'ordinary_shared_parameter_gradient': ordinary_shared.item(),
        'masked_shared_parameter_gradient': masked_shared.item(),
        'retained_logit_gradient_does_not_prove_shared_parameter_gradient_invariance': True,
        'signed_zero_noncontiguous_and_integer_state_roundtrip_bytes_exact': True,
        'already_lifted_half_values_pass_dtype_validation_without_origin_proof': True,
        'synthetic_metadata_only': True,
        'real_model_or_data_loaded': False,
        'gpu_used': False,
        'loss_integrated': False,
        'scientific_reference_accepted': False,
        'sources': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
                    for p in (loss_source, format_source)},
        'script_sha256': digest(Path(__file__).resolve()),
    }
    destination = HERE / 'independent_cpu_checks.json'
    assert not destination.exists()
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(record['status'])


if __name__ == '__main__':
    main()
