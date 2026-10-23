"""Independently inspect authenticated saved SASS text; no device access."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
RAW = BASE / 'sass_read_evidence'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_gate05_evidence_archive import verify_archive


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    output = BASE / 'sass_static_independent_review.json'
    assert not output.exists()
    initial = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in RAW.iterdir()}
    checks = []

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})
        if not value:
            raise AssertionError(name)

    archive = verify_archive(BASE / 'sass_read_evidence.zip',
        '79511a927d8820943022d3d9f274836fa703dcd4b7f740441dddf7659c64c667', 83856)
    receipt = read(RAW / 'remote_receipt.json')
    check('exact_six_payload_files_plus_receipt', len(receipt['files']) == 6 and set(initial) == set(receipt['files']) | {'remote_receipt.json'})
    for name, info in receipt['files'].items():
        check('exact_receipt:' + name, initial[name] == info)
    protocol = read(BASE / 'sass_read_protocol.json')
    check('frozen_protocol_pin', sha(BASE / 'sass_read_protocol.json') == 'e531d9df3218c50b1428b08d824f26e30d5acd1be774c0fe42a5dde2f9007716')
    source = ROOT / protocol['source_analysis']
    check('saved_launch_analysis_pin', sha(source) == protocol['source_analysis_sha256'] == '6cab40a4da8cbc90329fcf12feea10771360bbed2c48a588abeec5d99a1bb634')
    trace = read(source)
    trace_symbols = {k['mangled_name_text'] for row in trace['layers'] for c in row['correlations'] for k in c['kernel_rows_with_exact_correlation_id']}
    request, result = read(RAW / 'request.json'), read(RAW / 'result.json')
    check('saved_request_exact_symbols', set(request['symbols']) == trace_symbols and len(request['symbols']) == 4 and result['request'] == request)
    check('architecture_and_no_inference', request['architecture'] == protocol['gpu_architecture'] == 'sm_87' and
          request['new_native_build_or_inference'] is False and protocol['new_builds'] == protocol['new_inference'] == 0)
    libraries = result['libraries']
    check('exact_two_libraries_read_once_each', [r['library'] for r in libraries] == protocol['libraries'] and len(libraries) == 2)
    for index, row in enumerate(libraries):
        check('library_stdout_identity:' + str(index), row['stdout_sha256'] == sha(RAW / ('library' + str(index) + '.sass.txt')))
        check('library_stderr_identity:' + str(index), row['stderr_sha256'] == sha(RAW / ('library' + str(index) + '.stderr.txt')))
        check('bounded_read_argv:' + str(index), row['argv'] == [protocol['tool'], '--dump-sass', '--gpu-architecture', 'sm_87',
              '--function', ','.join(request['symbols']), row['library']] and row['seconds'] < protocol['max_seconds_per_library'])
    check('libnvinfer_static_read_success', libraries[0]['returncode'] == 0 and libraries[0]['status'] == 'COMPLETE' and
          libraries[0]['library_sha256'] == '4536788450d9a11f98701aa05021116ef13efd99f9318a03c01a2bc014d1c6ca')
    check('builder_resource_no_device_code_failure_preserved', libraries[1]['returncode'] == 255 and libraries[1]['stdout_bytes'] == 0 and
          'does not contain device code' in (RAW / 'library1.stderr.txt').read_text(encoding='utf-8'))
    lines = (RAW / 'library0.sass.txt').read_text(encoding='utf-8').splitlines()
    starts = [index for index, line in enumerate(lines) if re.search(r'Function\s*:', line)]
    check('all_four_unique_exact_functions_found', len(starts) == 4 and
          {lines[i].split(':', 1)[1].strip() for i in starts} == trace_symbols == set(libraries[0]['functions']))
    functions = []
    for number, begin in enumerate(starts):
        end = starts[number + 1] if number + 1 < len(starts) else len(lines)
        symbol = lines[begin].split(':', 1)[1].strip()
        arch_markers = [(i, line.strip()) for i, line in enumerate(lines[:begin]) if re.search(r'code for sm_\d+', line)]
        check('function_SM87:' + symbol, arch_markers and arch_markers[-1][1] == 'code for sm_87')
        instructions = []
        for line_index in range(begin, end):
            match = re.search(r'/\*([0-9a-f]+)\*/\s*(.*?)\s*;', lines[line_index])
            if match:
                instructions.append({'line': line_index + 1, 'pc': '0x' + match.group(1), 'text': match.group(2).strip()})
        idp = [r for r in instructions if 'IDP.4A.S8.S8' in r['text']]
        expected_count = 144 if 'Specialized' in symbol else 36 if 'Generic' in symbol else 96
        check('signed8_dotproduct_count:' + symbol, len(idp) == expected_count)
        chains, converted = [], []
        # Linear local def/use observations only, explicitly not a CFG proof.
        last_idp_dest = {}
        for row in instructions:
            match = re.search(r'IDP\.4A\.S8\.S8\s+(R\d+),\s*([^,]+),\s*([^,]+),\s*([^;]+)', row['text'])
            if match:
                dest, left, right, accum = [part.strip().replace('.reuse', '') for part in match.groups()]
                if accum in last_idp_dest:
                    chains.append({'producer': last_idp_dest[accum], 'consumer': row, 'accumulator_register': accum})
                last_idp_dest[dest] = row
                continue
            conversion = re.search(r'I2FP\.F32\.S32\s+(R\d+),\s*(R\d+)', row['text'])
            if conversion and conversion.group(2) in last_idp_dest:
                converted.append({'IDP_producer': last_idp_dest[conversion.group(2)], 'signed32_to_float_conversion': row})
        check('repeated_IDP_accumulation_observed:' + symbol, bool(chains))
        check('IDP_result_signed32_to_FLOAT_observed:' + symbol, bool(converted))
        ffma = [row for row in instructions if re.search(r'\bFFMA', row['text'])]
        loads = [row for row in instructions if re.search(r'\bLDG|\bLDS', row['text'])]
        check('loads_and_FLOAT_epilogue_present:' + symbol, bool(ffma) and bool(loads))
        windows = []
        for anchor in [idp[0]['line'], idp[-1]['line'], converted[0]['signed32_to_float_conversion']['line']]:
            lower, upper = max(begin + 1, anchor - 8), min(end, anchor + 12)
            windows.append({'line_start': lower, 'line_end': upper, 'text': '\n'.join(f'{i + 1}:{lines[i]}' for i in range(lower - 1, upper))})
        functions.append({'symbol': symbol, 'function_line': begin + 1, 'architecture_marker': arch_markers[-1],
            'signed8_IDP4A_count': len(idp), 'IMMA_opcode_count': sum(bool(re.search(r'\bIMMA\b', row['text'])) for row in instructions),
            'FFMA_count': len(ffma), 'integer_dotproduct_instructions': idp,
            'local_accumulator_def_use_examples': chains[:4], 'signed32_accumulator_to_FLOAT_examples': converted[:4],
            'representative_original_line_windows': windows,
            'interpretation': 'Static function has genuine signed8 dotproduct/accumulation dataflow and signed32-to-float conversion, beyond address IMAD. Floating operations coexist, consistent with scaling/activation/epilogue, not integer-only kernel.',
            'full_weight_activation_operand_role_mapping_and_CFG_path_proof': False,
            'runtime_loadedbinary_identity_proven': False, 'runtime_executed_integer_path_proven': False})
    check('old_strict_audit_unchanged', result['strict_precision_status_changed'] is False and trace['strict_precision_audit_changed'] is False)
    check('raw_unchanged_by_review', initial == {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in RAW.iterdir()})
    report = {'schema': 'rampq.gate05.sass_static_independent_review/v1',
        'client_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status': 'AUTHENTICATED_SM87_STATIC_SIGNED8_MAC_CHAINS_OBSERVED_RUNTIME_CHAIN_PENDING',
        'checks': checks, 'check_count': len(checks), 'archive_verification': archive,
        'libraries': [{key: value for key, value in row.items() if key != 'integer_mac_lines'} for row in libraries],
        'functions': functions, 'source_analysis_sha256': sha(source),
        'caution_on_earlier_summary': 'result.integer_mac_lines also includes address IMAD and is truncated; collection first3lines are not arithmetic proof. This report instead scans complete authenticated function text for IDP.4A.S8.S8.',
        'warnings_retained': 'library0 has per-fatbin function-not-found warnings despite all4 requested SM87 function bodies being emitted. builder_resource rc255/noDeviceCode is a preserved unsuccessful lookup, not a successful disassembly.',
        'remaining_identity_and_execution_gaps': ['No toolversion/binarybuildID authentication recorded beyond path/librarySHA.',
            'Saved historical launch symbol does not itself bind that run to this exact current librarySHA/cubin; same-process maps/module identity and architecture chain remain required.',
            'Local linear def/use examples are not full control-flow/predicate or runtime operand-role proof; branches and exceptional paths remain unverified.',
            'New candidate engine must map its own layers/launches/exact symbols/module identity, not inherit oldtrace as executed evidence.'],
        'strict_integer_pass_claim': False, 'old_tactic_name_filled_in': False,
        'TensorCore_only_claim': False, 'full_qall_numerical_correctness_claim': False,
        'new_CPU_native_SSH_or_disassembly_by_reviewer': False, 'saved_raw_unchanged': True}
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': sha(output),
                      'counts': [row['signed8_IDP4A_count'] for row in functions]}))


if __name__ == '__main__':
    main()
