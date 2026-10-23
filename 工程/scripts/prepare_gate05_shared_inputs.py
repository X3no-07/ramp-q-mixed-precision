"""Prepare exact common150 FP32 inputs; no model import or execution."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import zipfile

DATA_SHA='c4dac80ad7c59524194a9501f5ff8126cc0517e84f29f4106068c0c4b27a93aa'
SOURCE_SHA='f3c4d80658b79ae685e569b37c5ea89af9e6e9dd4a63ce86aff2806a7fcc5adb'
ROOT=Path(__file__).resolve().parents[1]

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,obj):
    with p.open('x',encoding='utf-8') as out:json.dump(obj,out,indent=2,ensure_ascii=False,allow_nan=False);out.write('\n')

def identities(paths):
    return {str(p.resolve()):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(paths),key=str)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();contract=read(a.protocol);source=ROOT/'src/gate0.py';data=ROOT/'artifacts/gate05/data_v1'
    require(sha(a.protocol)==a.protocol_sha256 and contract['schema']=='rampq.gate05.shared_input_preparation_protocol/v1','Protocol differs')
    require(contract['preparer_sha256']==sha(Path(__file__)) and contract['model_executions']==0 and contract['image_preprocessing_executions']==150,'Budget/code differs')
    require(sha(data/'manifest.json')==DATA_SHA and sha(source)==SOURCE_SHA,'Historical source/data changed')
    manifest=read(data/'manifest.json');rows=manifest['data']['evaluation']
    require(len(rows)==manifest['evaluation_count']==150 and manifest['evaluation_target_count']==865 and
            len(manifest['data']['calibration'])==manifest['calibration_count']==64,'Data count differs')
    protected=[a.protocol.resolve(),source,data/'manifest.json',Path(__file__).resolve()]
    targets=0;seen=set()
    for row in rows:
        image=data/row['image'];label=data/row['label']
        require(data in image.resolve().parents and data in label.resolve().parents,'Unsafe data path')
        require(sha(image)==row['image_sha256'] and sha(label)==row['label_sha256'],'Image/label changed')
        require(row['image_sha256'] not in seen,'Duplicate evaluation image');seen.add(row['image_sha256'])
        protected.extend((image,label))
        count=0
        for physical_line,line in enumerate(label.read_text(encoding='utf-8-sig').splitlines()):
            if not line.strip():continue
            parts=line.split();require(len(parts)==5,'Malformed label row: '+row['label']+':'+str(physical_line))
            values=[float(value) for value in parts]
            require(all(math.isfinite(value) for value in values) and values[0].is_integer() and 0<=values[0]<10 and
                    all(0<=value<=1 for value in values[1:]) and values[3]>0 and values[4]>0,
                    'Invalid fixed YOLO target: '+row['label']+':'+str(physical_line))
            count+=1
        require(count==row['target_count'],'Per-image target count differs: '+row['label'])
        targets+=count
    for row in manifest['data']['calibration']:
        image=data/row['image']
        require(data in image.resolve().parents and sha(image)==row['image_sha256'],'Calibration source image changed')
        protected.append(image)
    require(targets==865 and not seen.intersection(x['image_sha256'] for x in manifest['data']['calibration']),'Targets or split overlap differ')
    before=identities(protected)
    if not a.execute:
        print(json.dumps({'status':'INPUT_PREPARATION_SOURCES_VERIFIED_NO_MODEL_OR_PREPROCESS_EXECUTION','images':150,'targets':865}));return
    out=a.output.resolve();require(out.parent==ROOT/'artifacts/gate05' and not out.exists() and not out.with_suffix('.zip').exists(),'Fresh direct Gate0.5 child output needed')
    # Compile only the pinned old pure letterbox function; never import gate0's
    # torch/YOLO model runtime or silently use a newer preprocessing helper.
    import cv2
    import numpy as np
    require(cv2.__version__==contract['environment']['opencv'] and np.__version__==contract['environment']['numpy'],'Frozen preprocessing environment differs')
    tree=ast.parse(source.read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='letterbox'];require(len(nodes)==1,'Letterbox source missing')
    scope={'cv2':cv2,'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    out.mkdir();(out/'inputs').mkdir();write(out/'attempt.json',{'protocol_sha256':a.protocol_sha256,'image_preprocessing_budget':150,'model_executions':0})
    write(out/'source_hashes_before.json',before)
    result={'schema':'rampq.gate05.shared_preprocessed_inputs/v1','protocol_sha256':a.protocol_sha256,'preparer_sha256':sha(Path(__file__)),'source_gate0_sha256':SOURCE_SHA,'data_manifest_sha256':DATA_SHA,'environment':contract['environment'],'images':[],'model_executions':0,'created_client_at':datetime.now().astimezone().isoformat()}
    try:
        for index,row in enumerate(rows):
            image=data/row['image']
            require(sha(image)==row['image_sha256'],'Image changed before decode: '+row['image'])
            bgr=cv2.imread(str(image),cv2.IMREAD_COLOR);require(bgr is not None,'Cannot decode '+row['image'])
            require(sha(image)==row['image_sha256'],'Image changed during decode: '+row['image'])
            canvas,ratio,left,top=scope['letterbox'](bgr,640)
            rgb=canvas[:,:,::-1].transpose(2,0,1)
            array=(np.ascontiguousarray(rgb,dtype=np.float32)/255.0)[None]
            require(array.shape==(1,3,640,640) and array.dtype==np.float32 and array.flags.c_contiguous and np.isfinite(array).all(),'Input array contract failed')
            name='inputs/{:03d}_{}.npy'.format(index,Path(row['image']).stem);np.save(str(out/name),array,allow_pickle=False)
            result['images'].append({'image_index':index,'image':row['image'],'image_sha256':row['image_sha256'],'label':row['label'],'label_sha256':row['label_sha256'],'input_file':name,'input_sha256':sha(out/name),'input_bytes':(out/name).stat().st_size,'array_sha256':hashlib.sha256(array.tobytes()).hexdigest(),'shape':[1,3,640,640],'dtype':'float32','original_shape':list(bgr.shape[:2]),'ratio':ratio,'left':left,'top':top})
        after=identities(protected);write(out/'source_hashes_after.json',after)
        require(before==after and sha(data/'manifest.json')==DATA_SHA and sha(source)==SOURCE_SHA,'Protected image/label/calibration/code/protocol changed during preparation')
        result['all_protected_sources_unchanged']=True
        result['status']='ALL150_COMMON_INPUTS_PREPARED_NO_MODEL_EXECUTION';write(out/'manifest.json',result)
        payload={path.relative_to(out).as_posix():{'bytes':path.stat().st_size,'sha256':sha(path)} for path in sorted(out.rglob('*')) if path.is_file()}
        with zipfile.ZipFile(out.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for path in sorted(out.rglob('*')):
                if path.is_file():z.write(path,'gate05_shared_inputs_20261021/'+path.relative_to(out).as_posix())
        # One read of each closed archive member verifies ZIP CRC at EOF and
        # exact SHA/length against the actual files, without recomputing images.
        prefix='gate05_shared_inputs_20261021/'
        with zipfile.ZipFile(out.with_suffix('.zip'),'r') as z:
            require(set(z.namelist())=={prefix+name for name in payload},'ZIP inventory differs')
            for name,identity in payload.items():
                digest=hashlib.sha256();length=0
                with z.open(prefix+name) as member:
                    while True:
                        block=member.read(1024*1024)
                        if not block:break
                        digest.update(block);length+=len(block)
                require(length==identity['bytes'] and digest.hexdigest()==identity['sha256'],'ZIP payload bytes differ: '+name)
        require(before==identities(protected),'Protected source changed during archive creation')
        write(out/'transport_receipt.json',{'archive_sha256':sha(out.with_suffix('.zip')),'archive_bytes':out.with_suffix('.zip').stat().st_size,'manifest_sha256':sha(out/'manifest.json'),'zip_contains_transport_receipt':False,'archive_CRC_and_member_SHA256_verified':True,'payload_files':len(payload)})
        print(json.dumps(read(out/'transport_receipt.json')))
    except BaseException as exc:
        write(out/'preparation_failure.json',{'error':repr(exc),'completed_images':len(result['images']),'model_executions':0,'no_auto_retry':True});raise

if __name__=='__main__':main()
