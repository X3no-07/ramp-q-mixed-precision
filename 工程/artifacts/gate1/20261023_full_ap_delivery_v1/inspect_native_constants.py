"""Inspect Octave colon constants without any detections, labels or AP output."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import hashlib
import json
import struct

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    executable=ROOT/'artifacts/gate1/20261023_native_oracle_runtime_v1/runtime/octave-11.3.0-w64/mingw64/bin/octave-cli.exe'
    assert sha(executable)=='387ac823fb5b01b3407fb34edda6ab1cdba0c9619841a0f6dee323d05a850f2d'
    output=HERE/'native_constants_probe.json';assert not output.exists()
    expression="v=0.5:0.05:0.95;p=struct('version',version,'thresholds',v,'hex',num2hex(v));disp(jsonencode(p));"
    command=[str(executable),'--no-gui','--quiet','--no-history','--no-init-file','--no-site-file','--eval',expression]
    result=subprocess.run(command,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
    (HERE/'native_constants_console.txt').write_bytes(result.stdout)
    (HERE/'native_constants_stderr.txt').write_bytes(result.stderr)
    assert result.returncode==0,result.stderr.decode('utf-8',errors='replace')
    native=json.loads(result.stdout.decode('utf-8').strip())
    python=[.5+.05*i for i in range(10)]
    receipt={'schema':'rampq.gate1.native_arithmetic_constants_probe/v1','status':'OBSERVED_CONSTANTS_ONLY',
        'at_utc':datetime.now(timezone.utc).isoformat(),'native':native,
        'python_direct_formula_values':python,'python_direct_formula_hex':[struct.pack('>d',v).hex() for v in python],
        'command':command,'executable_sha256':sha(executable),'model_or_real_data_read':False,
        'AP_or_matcher_outputs_read':False,'existing_runtime_modified':False}
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
