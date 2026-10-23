"""Download private release assets with gh and verify each restored result."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def safe_target(root,rel):
    pure=PurePosixPath(rel)
    if pure.is_absolute() or '..' in pure.parts or ':' in rel or '\\' in rel:raise ValueError('Unsafe result member')
    p=(root/Path(*pure.parts)).resolve()
    if not p.is_relative_to(root):raise ValueError('Result escaped target root')
    return p
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--download',action='store_true')
    args=ap.parse_args();root=args.root.resolve();index=root/'工程/发布清单'
    release=json.loads((index/'release_assets.json').read_text(encoding='utf-8'))
    rows=[json.loads(v) for v in (index/'included_files.jsonl').read_text(encoding='utf-8').splitlines()]
    downloads=root/'.release-downloads';downloads.mkdir(exist_ok=True)
    for asset in release['assets']:
        p=downloads/asset['name']
        if not p.exists() and args.download:
            subprocess.run(['gh','release','download',release['release_tag'],'--repo',release['repository'],
                            '--pattern',asset['name'],'--dir',str(downloads)],check=True)
        if not p.is_file():raise FileNotFoundError(f'Download missing: {p}; use --download with gh authenticated')
        if p.stat().st_size!=asset['bytes'] or sha(p)!=asset['sha256']:raise ValueError('Release archive identity differs')
        targets=[r for r in rows if r.get('storage')=='release' and r['asset']==asset['name']]
        with zipfile.ZipFile(p) as z:
            for row in targets:
                dest=safe_target(root,row['path'])
                if dest.exists():
                    if dest.stat().st_size!=row['bytes'] or sha(dest)!=row['sha256']:raise FileExistsError(f'Existing result differs: {dest}')
                    continue
                member=row['archive_member'];safe_target(root,member)
                dest.parent.mkdir(parents=True,exist_ok=True)
                temp=dest.with_name(dest.name+'.restoring')
                if temp.exists():raise FileExistsError(temp)
                with z.open(member) as source,temp.open('xb') as out:shutil.copyfileobj(source,out)
                if temp.stat().st_size!=row['bytes'] or sha(temp)!=row['sha256']:raise ValueError('Restored result identity differs')
                temp.rename(dest)
        print(f'Verified and restored: {asset["name"]}')
if __name__=='__main__':main()
