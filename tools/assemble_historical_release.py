import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from download_historical_assets import restore

REPOSITORY = 'X3no-07/ramp-q-mixed-precision'
TAG = 'snapshot-2026-10-23'


def api(endpoint, token):
    request = urllib.request.Request('https://api.github.com/' + endpoint, headers={
        'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'RAMP-Q-release-assembly'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def asset_matches(asset, size, digest):
    return asset.get('state') == 'uploaded' and asset['size'] == size and asset.get('digest') == 'sha256:' + digest


def assets_for(release, token):
    assets = []
    page = 1
    while True:
        batch = api(f"repos/{REPOSITORY}/releases/{release['id']}/assets?per_page=100&page={page}", token)
        assets.extend(batch)
        if len(batch) < 100:
            return {asset['name']: asset for asset in assets}
        page += 1


def main():
    parser = argparse.ArgumentParser(description='Reconstruct verified release segments and publish the original ZIP on a GitHub-hosted runner.')
    parser.add_argument('--parts', required=True, help='Space-separated original archive part numbers.')
    args = parser.parse_args()
    parts = {int(value) for value in args.parts.split()}
    if not parts or not parts <= set(range(1, 17)):
        parser.error('Archive parts must be in the range 1-16')
    if os.environ.get('GITHUB_REPOSITORY') != REPOSITORY or os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('This publishing entry point is restricted to the intended repository workflow')
    token = os.environ['GH_TOKEN']
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'export/HISTORICAL_ASSEMBLY_PLAN.json').read_text(encoding='utf-8'))
    release = api(f'repos/{REPOSITORY}/releases/tags/{TAG}', token)
    remote = assets_for(release, token)
    for archive in plan['archives']:
        if archive['part'] not in parts:
            continue
        name = archive['name']
        prior = remote.get(name)
        if prior:
            if not asset_matches(prior, archive['bytes'], archive['sha256']):
                raise ValueError('Existing original ZIP differs; refusing to overwrite: ' + name)
            print(json.dumps({'event': 'original_zip_already_verified', 'name': name}), flush=True)
            continue
        if archive['download_mode'] != 'segmented':
            raise ValueError('Missing original direct archive is outside this assembly operation')
        segments = []
        for segment in archive['segments']:
            actual = remote.get(segment['name'])
            if actual is None or not asset_matches(actual, segment['bytes'], segment['sha256']):
                raise ValueError('Required segment unavailable or different: ' + segment['name'])
            segments.append({**segment, 'download_url': actual['browser_download_url']})
        record = {'name': name, 'part': archive['part'], 'expected_bytes': archive['bytes'],
                  'expected_sha256_from_historical_manifest': archive['sha256'], 'download_mode': 'segmented', 'segments': segments}
        # The runner owns this temporary directory; no existing research files are modified.
        with tempfile.TemporaryDirectory(prefix='rampq-release-assembly-') as folder:
            output = Path(folder)
            restore(record, output)
            original = output / name
            upload_url = release['upload_url'].split('{', 1)[0] + '?' + urllib.parse.urlencode({'name': name})
            if not upload_url.startswith('https://uploads.github.com/repos/' + REPOSITORY + '/'):
                raise ValueError('Unexpected upload destination')
            # The token is supplied in the runner environment and never written to argv or logs.
            headers = ('Authorization: Bearer ' + token + '\nAccept: application/vnd.github+json\nContent-Type: application/zip\n').encode()
            response_path = output / 'upload-response.json'
            result = subprocess.run(['curl', '--fail-with-body', '--request', 'POST', '--header', '@-',
                                     '--upload-file', str(original), '--output', str(response_path),
                                     '--connect-timeout', '30', '--max-time', '1200', upload_url],
                                    input=headers, check=False)
            # An uncertain transport result must be resolved by querying identity before any retry.
            current = assets_for(release, token)
            actual = current.get(name)
            if actual is None or not asset_matches(actual, archive['bytes'], archive['sha256']):
                raise RuntimeError(f'Original ZIP not server-verified after upload (curl exit {result.returncode}): ' + name)
            remote[name] = actual
            print(json.dumps({'event': 'original_zip_server_sha256_verified', 'name': name,
                              'bytes': actual['size'], 'digest': actual['digest'], 'asset_id': actual['id']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
