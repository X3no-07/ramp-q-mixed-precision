"""One-shot calibration supervisor with actual propagated exit receipt."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/gate1/20261026_separate_threshold178_v1'


def write(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)


def main():
    if (OUT/'attempt_01').exists() or (OUT/'worker_launcher_receipt.json').exists():raise ValueError('Existing calibration attempt; no duplicate')
    with (OUT/'console_stdout.txt').open('x',encoding='utf-8') as stdout,(OUT/'console_stderr.txt').open('x',encoding='utf-8') as stderr:
        argv=[sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts/gate1_separate_threshold.py'),'run']
        child=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        write(OUT/'worker_launcher_receipt.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'launched_pid':child.pid,'argv':argv,'automatic_retry':False})
        code=child.wait()
    result={'at_utc':datetime.now(timezone.utc).isoformat(),'launched_pid':child.pid,'propagated_worker_exit_code':code,
        'exit_code_directly_observed':True,'completion_receipt_exists':(OUT/'attempt_01/completion_receipt.json').is_file(),
        'failure_receipt_exists':(OUT/'attempt_01/failure_receipt.json').is_file(),'stdout_bytes':(OUT/'console_stdout.txt').stat().st_size,
        'stderr_bytes':(OUT/'console_stderr.txt').stat().st_size,'automatic_retry':False}
    write(OUT/'launcher_terminal_receipt.json',result);print(json.dumps(result),flush=True);return code


if __name__=='__main__':raise SystemExit(main())
