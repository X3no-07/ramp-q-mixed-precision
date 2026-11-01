"""One supervisor. Worker exit, then saved-array audit; never restart a stage."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/gate1/20261027_full_training_comparison_v1'


def now(): return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with path.open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    if (OUT / 'attempt_01').exists() or (OUT / 'worker_launcher_receipt.json').exists():
        raise ValueError('Existing comparison attempt; duplicate execution prohibited')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with (OUT / 'console_stdout.txt').open('x', encoding='utf-8') as stdout, (OUT / 'console_stderr.txt').open('x', encoding='utf-8') as stderr:
        argv = [sys.executable, '-B', '-X', 'utf8', '-u', str(ROOT / 'scripts/gate1_full_training_comparison.py'), 'run']
        started = now(); child = subprocess.Popen(argv, cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=flags)
        write(OUT / 'worker_launcher_receipt.json', {'created_at_utc': started, 'launcher_pid': os.getpid(),
              'launched_pid': child.pid, 'argv': argv, 'automatic_retry': False})
        code = child.wait()
    write(OUT / 'worker_terminal_receipt.json', {'at_utc': now(), 'pid': child.pid, 'exit_code': code,
          'exit_code_directly_observed': True, 'completion_receipt_exists': (OUT / 'attempt_01/completion_receipt.json').is_file(),
          'failure_receipt_exists': (OUT / 'attempt_01/failure_receipt.json').is_file(), 'automatic_retry': False})
    audit_code = None
    if code == 0:
        with (OUT / 'audit_console.txt').open('x', encoding='utf-8') as log:
            argv = [sys.executable, '-B', '-X', 'utf8', '-u', str(ROOT / 'scripts/audit_gate1_full_training_comparison.py')]
            auditor = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
            write(OUT / 'audit_launcher_receipt.json', {'at_utc': now(), 'pid': auditor.pid, 'parent_pid': os.getpid(), 'argv': argv})
            audit_code = auditor.wait()
    record = {'at_utc': now(), 'launcher_pid': os.getpid(), 'worker_pid': child.pid, 'worker_exit_code': code,
              'audit_exit_code': audit_code, 'delivery_exists': (OUT / 'delivery_validation.json').is_file(), 'automatic_retry': False}
    write(OUT / 'launcher_terminal_receipt.json', record)
    print(json.dumps(record), flush=True)
    return code if code else (audit_code or 0)


if __name__ == '__main__': raise SystemExit(main())
