"""Supervise one build on the Jetson and retain crash diagnostics."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, data):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def resources():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        parts = line.split()
        if parts[0].rstrip(':') in ('MemAvailable', 'SwapFree', 'SwapTotal'):
            memory[parts[0].rstrip(':') + '_kib'] = int(parts[1])
    temperatures = {}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            temperatures[(zone / 'type').read_text().strip()] = int((zone / 'temp').read_text()) / 1000
        except (OSError, ValueError):
            pass
    return dict(memory, temperatures_c=temperatures)


def stop_group(process):
    # The build and its children have a private process group. Never target other jobs.
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=10)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=1800)
    parser.add_argument('--build-script', default='build_fp16.py')
    args = parser.parse_args()
    root = args.root.resolve()
    build_script = (root / args.build_script).resolve()
    if build_script.parent != root or not build_script.is_file():
        raise RuntimeError('Build script must be an existing file directly inside root')
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    lock = (root / '.monitored_build.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = subprocess.run(['pgrep', '-x', 'trtexec'], capture_output=True, text=True)
    if active.returncode != 1:
        raise RuntimeError('trtexec already running, or process check failed: ' + active.stdout + active.stderr)
    initial = resources()
    if initial['MemAvailable_kib'] < 3 * 1024 * 1024:
        raise RuntimeError('Less than 3 GiB available before build')
    if subprocess.run(['systemctl', 'is-active', '--quiet', 'nvfancontrol.service']).returncode:
        raise RuntimeError('Fan service is not active')
    started = time.monotonic()
    identity = {'monitor_pid': os.getpid(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'root': str(root),
                'max_seconds': args.max_seconds, 'stop_temperature_c': 85, 'build_script': str(build_script),
                'low_memory_stop': 'MemAvailable < 384 MiB for 3 consecutive 2-second samples',
                'note': 'Supervision is best effort; cannot guarantee recovery from a kernel hang.'}
    save(run / 'request.json', dict(identity, initial_resources=initial))
    children = []
    handles = []
    build = None
    outcome = 'FAILED'
    reason = None
    build_code = None
    try:
        for name, command in [('kernel.log', ['journalctl', '-k', '-b', '-f', '-n', '0', '-o', 'short-monotonic', '--no-pager']),
                              ('tegrastats.log', ['tegrastats', '--interval', '1000'])]:
            handle = (run / name).open('wb', buffering=0)
            handles.append(handle)
            children.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True))
        time.sleep(1)
        if any(p.poll() is not None for p in children):
            raise RuntimeError('A diagnostic collector exited before the build started')
        console = (run / 'build_console.log').open('wb', buffering=0)
        handles.append(console)
        build = subprocess.Popen([sys.executable, '-u', str(build_script)], cwd=str(root),
                                 stdout=console, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        identity['build_pid'] = build.pid
        low_memory = 0
        with (run / 'resources.jsonl').open('a') as observations:
            while True:
                sample = dict(resources(), elapsed_seconds=round(time.monotonic() - started, 2))
                observations.write(json.dumps(sample) + '\n')
                observations.flush()
                os.fsync(observations.fileno())
                save(run / 'status.json', dict(identity, status='RUNNING', latest=sample))
                build_code = build.poll()
                if build_code is not None:
                    outcome = 'BUILD_SCRIPT_OK' if build_code == 0 else 'BUILD_SCRIPT_FAILED'
                    break
                kernel = (run / 'kernel.log').read_text(errors='replace')
                low_memory = low_memory + 1 if sample['MemAvailable_kib'] < 384 * 1024 else 0
                if any(marker in kernel for marker in ('controller error detected', 'HC died;', 'unregister_netdevice: waiting', 'Out of memory:', 'oom-kill:', 'Kernel panic')):
                    reason = 'New kernel USB failure, OOM, or panic record'
                elif max(sample['temperatures_c'].values(), default=0) >= 85:
                    reason = 'Preventive temperature limit reached (85 C)'
                elif low_memory >= 3:
                    reason = 'Available memory stayed below 384 MiB'
                elif sample['elapsed_seconds'] >= args.max_seconds:
                    reason = 'Build supervision time limit reached'
                elif any(p.poll() is not None for p in children):
                    reason = 'A diagnostic collector exited'
                if reason:
                    outcome = 'STOPPED_BY_MONITOR'
                    stop_group(build)
                    build_code = build.poll()
                    break
                time.sleep(2)
    except BaseException as exc:
        reason = repr(exc)
        raise
    finally:
        if build is not None and build.poll() is None:
            stop_group(build)
        for child in children:
            stop_group(child)
        for handle in handles:
            handle.close()
        final = dict(identity, status=outcome, reason=reason, build_returncode=build_code,
                     elapsed_seconds=round(time.monotonic() - started, 2), final_resources=resources())
        save(run / 'status.json', final)
        print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
