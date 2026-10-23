"""Fresh read-only process/file observation after a bounded native probe."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, universal_newlines=True, timeout=15)
    return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with zipfile.ZipFile(args.archive) as archive:
        receipt_bytes = archive.read('remote_receipt.json')
    receipt = json.loads(receipt_bytes)
    launch = read(root / 'launch.json')
    run = Path(read(root / 'latest_probe.json')['run'])
    summary = read(run / 'summary.json')
    pids = [launch['monitor_pid'], launch['private_group'], launch['guard_pid']]
    pids.extend(read(run / name / 'child_cleanup.json')['pid'] for name in summary['requested_case_order'])
    known = {str(pid): dict(exists=(Path('/proc') / str(pid)).exists()) for pid in pids}
    assert not any(item['exists'] for item in known.values()), known
    remote = {}
    for relative, expected in receipt['files'].items():
        path = (root / relative).resolve()
        assert root in path.parents
        remote[relative] = path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
    assert all(remote.values())
    tasks = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            text = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
            if any(value in text for value in ('jetson_build_', 'trtexec', 'jetson_dfl_')):
                tasks.append(dict(pid=int(process.name), command=text))
        except OSError:
            pass
    assert not tasks, tasks
    qonly = Path('/home/nvidia/rampq/gate05_corrected_qonly_20261019/gate05_corrected_qonly_candidate_20261019/results/build_20261016T170532_870381Z/model.engine')
    previous = Path('/home/nvidia/rampq/gate05_dfl_conv_output_20261020/gate05_dfl_conv_output_probe_20261020')
    previous_run = Path(read(previous / 'latest_probe.json')['run'])
    qall = Path('/home/nvidia/rampq/gate05_corrected_qall_20261019')
    result = dict(board_time=datetime.datetime.now().astimezone().isoformat(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), known_processes=known,
        task_processes=tasks, archive_sha256=sha(args.archive), receipt_file_count=len(receipt['files']),
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(), all_remote_receipt_files_still_match=True,
        remote_checks=remote, qonly_preserved_engine=dict(bytes=qonly.stat().st_size, sha256=sha(qonly)),
        corrected_qall_engine_paths=[str(path) for path in qall.rglob('*.engine')],
        parent_conv_output_none_engine_sha256=sha(previous_run / 'code_clip_none/model.engine'),
        parent_conv_output_obey_engine_present=(previous_run / 'code_clip_obey/model.engine').exists(),
        fan=command(['systemctl', 'is-active', 'nvfancontrol.service']),
        jtop=command(['systemctl', 'show', 'jtop.service', '-p', 'ActiveState', '-p', 'SubState', '-p', 'NRestarts']),
        memory=command(['free', '-m']), power=command(['nvpmodel', '-q']))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
