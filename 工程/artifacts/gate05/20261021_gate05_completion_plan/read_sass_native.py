import datetime,hashlib,json,subprocess,time,zipfile,re
from pathlib import Path
root=Path('/home/nvidia/rampq/gate05_sass_read_20261021')
root.mkdir(exist_ok=False)
symbols=['_ZN11cuDepthwise31depthwiseConvINT8GenericKernelFILi3ELi3EEEvPKiPfN8nvinfer12rt14ReducedDivisorEiiiS6_iiiiiiS6_S2_PK6float4S9_ffbbfiiii', '_ZN11cuDepthwise34depthwiseConvINT8SpecializedKernelILi3ELi3ELi1ELi1EEEvPKiP5char4N8nvinfer12rt14ReducedDivisorEiiS7_iiiS7_S2_PK6float4SA_ffbbfiii', '_ZN8nvinfer15sdcrr15kernelFrameworkINS0_13TraitsWrapperINS0_18TraitsR3S3U1V1BaseILi8ELi1ELi4EEEEENS0_20MathModuleDSConvIMMAIS5_EELb1EEEvNS0_12LaunchParamsEi', '_ZN8nvinfer15sdcrr15kernelFrameworkINS0_13TraitsWrapperINS0_18TraitsR3S3U1V1BaseILi8ELi1ELi4EEEEENS0_29MathModuleDSConvIMMAReusePRMTIS5_EELb1EEEvNS0_12LaunchParamsEi']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
request=dict(board_time=datetime.datetime.now().astimezone().isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),symbols=symbols,architecture='sm_87',new_native_build_or_inference=False)
(root/'request.json').write_text(json.dumps(request,indent=2))
records=[]
for index,name in enumerate(['/usr/lib/aarch64-linux-gnu/libnvinfer.so.8.5.2','/usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource.so.8.5.2']):
 library=Path(name);stdout=root/('library'+str(index)+'.sass.txt');stderr=root/('library'+str(index)+'.stderr.txt');start=time.monotonic()
 argv=['/usr/local/cuda/bin/cuobjdump','--dump-sass','--gpu-architecture','sm_87','--function',','.join(symbols),str(library)]
 status='COMPLETE';rc=None
 with stdout.open('xb') as out,stderr.open('xb') as err:
  try:rc=subprocess.run(argv,stdout=out,stderr=err,timeout=120).returncode
  except subprocess.TimeoutExpired:status='TIMEOUT_PRESERVED'
 text=stdout.read_text(errors='replace')
 record=dict(library=str(library),library_bytes=library.stat().st_size,library_sha256=sha(library),argv=argv,status=status,returncode=rc,seconds=time.monotonic()-start,stdout_bytes=stdout.stat().st_size,stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),functions=re.findall(r'Function\s*:\s*(.*)',text),integer_mac_lines=[line.strip() for line in text.splitlines() if re.search(r'\b(IMMA|IDP\.4A|DP4A|IMAD)',line)][:250])
 records.append(record)
(root/'result.json').write_text(json.dumps(dict(request=request,libraries=records,strict_precision_status_changed=False,limits='Static code presence only; a runtime loadedbinary identity and active accumulation path must still be authenticated.'),indent=2))
files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.iterdir() if p.is_file()}
receipt=dict(schema='rampq.remote_file_receipt/v1',files=files)
archive=root.with_suffix('.zip')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for name in sorted(files):z.write(root/name,name)
 z.writestr('remote_receipt.json',json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),libraries=[dict(library=x['library'],status=x['status'],returncode=x['returncode'],functions=len(x['functions']),stdout_bytes=x['stdout_bytes'],integer_mac_excerpt=x['integer_mac_lines'][:3]) for x in records])))
