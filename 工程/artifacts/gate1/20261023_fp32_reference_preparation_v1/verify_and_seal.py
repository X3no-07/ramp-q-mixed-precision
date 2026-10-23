"""Produce a synthetic precision-format example and seal only new files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import re

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ident(path):
    return {'path':path.relative_to(ROOT).as_posix(), 'bytes':path.stat().st_size, 'sha256':sha(path)}


def new_json(path, value):
    assert not path.exists(),path
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    assert not (HERE/'manifest.json').exists()
    log=(HERE/'format_tests_v2.log').read_text(encoding='utf-8-sig')
    assert re.search(r'Ran 10 tests',log) and re.search(r'\nOK\s*$',log) and 'FAILED' not in log
    source=ROOT/'scripts/gate1_fp32_reference_state.py'
    spec=importlib.util.spec_from_file_location('synthetic_fp32_format_example',source)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    values=torch.tensor([1.00012362,-0.34567893,0.00001356789],dtype=torch.float32)
    assert not torch.equal(values, values.half().float())
    receipt=helper.write_snapshot({'synthetic_weight':values,'synthetic_counter':torch.tensor(7)},
        HERE/'synthetic_example', {'training_contract_sha256':'a'*64,'architecture_sha256':'b'*64,
        'epoch_one_based':1,'source_stage':helper.STAGE})
    restored, _=helper.load_snapshot(HERE/'synthetic_example')
    assert torch.equal(values,restored['synthetic_weight'])
    new_json(HERE/'verification_receipt.json', {
        'schema':'rampq.gate1.fp32_reference_format_preparation/v1','status':'PASS_SYNTHETIC_FORMAT_ONLY',
        'at_utc':datetime.now(timezone.utc).isoformat(),'synthetic_cpu_tests':10,
        'final_test_log':ident(HERE/'format_tests_v2.log'),
        'initial_test_attempt':'format_tests.log retained; sandbox TEMP access was denied before snapshot execution, so tests now use verified workspace temporary paths',
        'source':ident(source),'tests':ident(ROOT/'tests/test_gate1_fp32_reference_state.py'),
        'synthetic_example_receipt':ident(HERE/'synthetic_example/snapshot_receipt.json'),
        'float32_values_exact_after_serialization':True,'half_roundtrip_counterexample_differs':True,
        'no_live_training_hook_integrated':True,'concurrent_update_atomicity_proven':False,
        'full_precision_training_origin_independently_proven':False,
        'old_fp16_checkpoint_recovered':False,'accepted_scientific_reference':False,
        'model_or_real_data_loaded':False,'GPU_executed':False,'selection_or_holdout_accessed':False})
    readme='''# FP32参考状态保存组件

本轮仅实现并测试独立的FP32 state_dict保存格式：10项CPU合成测试通过。人工构造的、不能由FP16精确表示的数值，保存后通过受限weights_only加载逐元素、dtype及字节摘要一致；半精度往返的对照会改变这些数值。示例中的架构/训练合同哈希均为明确的合成占位符，不是训练证据。

组件拒绝原本是FP16、BF16或FP64的浮点输入，以及非有限值；它不自动提升类型，不保存优化器，也不是可恢复训练的完整checkpoint。给定声明的source_stage不证明数据确实来自未被降精度的训练EMA，测试专门保留了这一反例。

旧best/last原本已经按FP16 EMA序列化，本组件没有读取、修改或恢复它们。既有混合精度原型及原全FP16失败记录不变；Gate1科学参考仍未验收。

下一步必须另接经审查的live-training hook：暂停更新或原子捕获EMA、固定架构重建与完整来源、绑定训练合同、逐项核对序列化前后。caller_state_unchanged只能检查调用期间稳定的普通状态，不能证明并发训练的原子快照。当前完整性清单也不是独立来源认证。

初次单元测试受sandbox临时目录ACL影响，在snapshot写入前失败；原日志保留。改用已检查绝对路径的工程内临时目录后，10项全部通过。没有实际训练、真实数据、模型权重、GPU、Selection或Holdout访问。
'''
    path=HERE/'README.md'; assert not path.exists();path.write_text(readme,encoding='utf-8')
    files={path.relative_to(HERE).as_posix():{'bytes':path.stat().st_size,'sha256':sha(path)}
        for path in sorted(HERE.rglob('*')) if path.is_file() and '__pycache__' not in path.parts}
    new_json(HERE/'manifest.json',{'schema':'rampq.gate1.fp32_reference_format_manifest/v1','files':files})
    print(json.dumps({'status':'PASS_SYNTHETIC_FORMAT_ONLY','tests':10,'members':len(files),'manifest_sha256':sha(HERE/'manifest.json')}))


if __name__=='__main__':main()
