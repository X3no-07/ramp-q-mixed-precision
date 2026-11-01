"""Static checks for this source/report export. Does not run model code."""
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv'}
DENIED_EXT = {'.pt', '.pth', '.onnx', '.engine', '.plan', '.npy', '.npz', '.zip', '.exe', '.dll', '.pem', '.key', '.ppk', '.pfx', '.p12'}
PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9_]{25,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'cloud_access_key': re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
    'openai_key': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}'),
    'slack_token': re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,}'),
    'huggingface_token': re.compile(r'\bhf_[A-Za-z0-9]{25,}'),
    'literal_api_credential': re.compile(r'''(?im)\b(?:api_key|access_token|client_secret|authorization)\s*["']?\s*[:=]\s*["'](?:Bearer\s+)?[A-Za-z0-9_./+=-]{24,}["']'''),
    'literal_password': re.compile(r'''(?im)\b(?:password|passwd)\s*["']?\s*[:=]\s*["'][^"'\s<>\[\]]{4,}["']'''),
    'local_user_path': re.compile(r'(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\s"\x27]+'),
    'private_network': re.compile(r'\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})\b'),
    'ssh_public_key': re.compile(r'ssh-(?:rsa|ed25519)\s+[A-Za-z0-9+/]{30,}'),
}

def all_files():
    for current, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for directory in dirs:
            path = Path(current)/directory
            if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0)&1024:
                raise ValueError('Repository must contain materialized directories: '+str(path.relative_to(ROOT)))
        for name in names:
            path=Path(current)/name
            if name.endswith('.local.json') or name.endswith('.local.txt'):
                continue
            yield path

def main():
    issues=[]; python_count=0; local_links=0; members=[]
    for path in all_files():
        relative=path.relative_to(ROOT).as_posix()
        if path.is_symlink() or getattr(path.lstat(),'st_file_attributes',0)&1024:
            issues.append({'file':relative,'kind':'link_in_repository'})
        size=path.stat().st_size
        members.append((relative,size))
        if size>2*1024*1024:issues.append({'file':relative,'kind':'large_file'})
        if path.suffix.lower() in DENIED_EXT:issues.append({'file':relative,'kind':'excluded_file_type'})
        try:text=path.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:
            issues.append({'file':relative,'kind':'unexpected_binary'});continue
        for name,pattern in PATTERNS.items():
            match=pattern.search(text)
            if match:issues.append({'file':relative,'kind':name,'line':text[:match.start()].count('\n')+1})
        if path.suffix=='.py':
            python_count+=1
            try:ast.parse(text,filename=relative)
            except SyntaxError as error:issues.append({'file':relative,'kind':'python_syntax','line':error.lineno})
        if path.suffix=='.md':
            for match in re.finditer(r'!?\[[^\]\n]*\]\(([^\n]*?)\)',text):
                target=match.group(1).strip().strip('<>')
                if not target or target.startswith('#') or re.match(r'^(?:https?|mailto):',target,re.I):continue
                target=unquote(target.split('#',1)[0])
                local_links+=1
                destination=path.parent/target
                if not destination.exists():issues.append({'file':relative,'kind':'broken_local_link','target':target})
    manifest=ROOT/'export/SOURCE_FILES.jsonl'
    if manifest.exists():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            row=json.loads(line); path=ROOT/row['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['export_sha256']:
                issues.append({'file':row['path'],'kind':'export_digest_mismatch'})
    result={'status':'PASS' if not issues else 'FAIL','files_checked':len(members),'bytes':sum(s for _,s in members),
            'python_files_parsed':python_count,'local_links_checked':local_links,'issues':issues,
            'scope':'Static repository packaging checks; no unit tests, model training, inference, AP or device validation.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if not issues else 1

if __name__=='__main__':raise SystemExit(main())
