"""Single owned subprocess, no retry; directly observed terminal receipt."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/gate1/20261026_local_visual_information_probe_v1'


def write(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)


def main():
    if (OUT/'attempt_01').exists() or (OUT/'worker_launcher_receipt.json').exists():raise ValueError('Existing visualprobe;no duplicate')
    argv=[sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts/gate1_visual_evidence_probe.py'),'run']
    with (OUT/'console_stdout.txt').open('x',encoding='utf-8') as stdout,(OUT/'console_stderr.txt').open('x',encoding='utf-8') as stderr:
        proc=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        write(OUT/'worker_launcher_receipt.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'pid':proc.pid,'argv':argv,'automatic_retry':False})
        code=proc.wait()
    record={'at_utc':datetime.now(timezone.utc).isoformat(),'pid':proc.pid,'observed_worker_exit_code':code,'exit_code_directly_observed':True,
        'completion_receipt_exists':(OUT/'attempt_01/completion_receipt.json').is_file(),'failure_receipt_exists':(OUT/'attempt_01/failure_receipt.json').is_file(),
        'stdout_bytes':(OUT/'console_stdout.txt').stat().st_size,'stderr_bytes':(OUT/'console_stderr.txt').stat().st_size,'automatic_retry':False}
    write(OUT/'launcher_terminal_receipt.json',record);print(json.dumps(record),flush=True);return code


if __name__=='__main__':raise SystemExit(main())
