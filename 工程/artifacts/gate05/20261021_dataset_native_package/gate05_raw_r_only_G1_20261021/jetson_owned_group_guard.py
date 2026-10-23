"""Supplement one running Jetson monitor without changing its files (Python 3.8).

Launch in a new session, for example with setsid. --check-only performs only
read-only identity checks and prints the observed private build group.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


TERMINAL_STATES = {'BUILD_SCRIPT_OK', 'BUILD_SCRIPT_FAILED', 'STOPPED_BY_MONITOR', 'FAILED'}
POLL_SECONDS = 2.0
STALE_SECONDS = 15.0
BUDGET_GRACE_SECONDS = 20.0


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def process_info(pid):
    """Read Linux process identity; a vanished process is an ordinary race."""
    folder = Path('/proc') / str(pid)
    try:
        raw = (folder / 'stat').read_text()
        tail = raw[raw.rfind(')') + 2:].split()
        result = {'pid': pid, 'state': tail[0], 'ppid': int(tail[1]),
                  'pgid': int(tail[2]), 'sid': int(tail[3]), 'start_ticks': int(tail[19])}
        result['argv'] = (folder / 'cmdline').read_bytes().decode('utf-8', errors='replace').rstrip('\0').split('\0')
        # A second stat read prevents attaching argv from a reused PID.
        after = (folder / 'stat').read_text()
        require(after[after.rfind(')') + 2:].split()[19] == str(result['start_ticks']),
                'Process PID changed while reading identity')
        return result
    except (FileNotFoundError, ProcessLookupError):
        return None


def read_status(path):
    with path.open('rb') as handle:
        raw = handle.read()
        metadata = os.fstat(handle.fileno())
    value = json.loads(raw.decode('utf-8'))
    elapsed = value.get('elapsed_seconds', value.get('latest', {}).get('elapsed_seconds'))
    require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and elapsed >= 0,
            'Monitor status has no valid elapsed_seconds')
    require(value.get('status') == 'RUNNING' or value.get('status') in TERMINAL_STATES,
            'Unknown monitor status')
    return value, {'sha256': sha_bytes(raw), 'mtime_ns': metadata.st_mtime_ns,
                   'mtime_wall_seconds': metadata.st_mtime, 'elapsed_seconds': elapsed}


def verify_status(value, frozen, expected_boot):
    require(boot_id() == expected_boot == value.get('boot_id'), 'Boot identity changed')
    for key in ('root', 'monitor_pid', 'build_pid', 'max_seconds', 'build_script'):
        require(value.get(key) == frozen[key], 'Monitor identity changed: ' + key)


class OwnedGroup:
    def __init__(self, group, expected_boot, build_script):
        self.group = group
        self.expected_boot = expected_boot
        self.build_script = str(build_script)
        self.leader_start_ticks = None
        self.leader_absent_at_start = False
        members = self.snapshot()
        leader = next((row for row in members if row['pid'] == group), None)
        if leader is not None and leader['state'] not in ('Z', 'X'):
            require(self.build_script in leader['argv'], 'Private group leader is not the recorded build script')
            self.leader_start_ticks = leader['start_ticks']
        else:
            self.leader_absent_at_start = leader is None
            if leader is not None:
                self.leader_start_ticks = leader['start_ticks']
            # An orphaned builder must still name a script inside this root.
            root = str(build_script.parent) + '/'
            active = live_members(members)
            require(not active or any(any(arg.startswith(root) and arg.endswith('.py') for arg in row['argv'])
                                      for row in active), 'Cannot authenticate orphaned private group')
        self.initial_members = members

    def snapshot(self):
        require(boot_id() == self.expected_boot, 'Boot identity changed')
        require(self.group > 1 and self.group != os.getpgrp() and self.group != os.getsid(0),
                'Refuse to inspect or signal our own process group/session')
        result = subprocess.run(['ps', '-eo', 'pid=,ppid=,pgid=,sid=,stat=,args=', '-ww'],
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        members = []
        for line in result.stdout.splitlines():
            columns = line.split(None, 5)
            if len(columns) < 5 or int(columns[2]) != self.group:
                continue
            require(int(columns[3]) == self.group, 'Group is not a private PGID=SID session')
            info = process_info(int(columns[0]))
            if info is None or info['pgid'] != self.group:
                continue
            require(info['sid'] == self.group, 'Group member changed session')
            info['ps'] = {'ppid': int(columns[1]), 'pgid': int(columns[2]), 'sid': int(columns[3]),
                          'stat': columns[4], 'args': columns[5] if len(columns) == 6 else ''}
            if info['pid'] == self.group:
                require(not self.leader_absent_at_start, 'A new leader appeared after the original leader exited')
                require(self.leader_start_ticks is None or info['start_ticks'] == self.leader_start_ticks,
                        'Private group leader PID was reused')
            members.append(info)
        return sorted(members, key=lambda row: row['pid'])


def live_members(members):
    return [row for row in members if row['state'] not in ('Z', 'X')]


def atomic_save(path, value):
    require(not path.is_symlink(), 'Guard output became a symlink')
    temporary = path.with_name(path.name + '.guard-' + str(os.getpid()) + '.tmp')
    require(not temporary.is_symlink(), 'Guard temporary path is a symlink')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def terminate_owned(group, record, save):
    """Escalate according to group members, even after its leader exits."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        before = group.snapshot()
        if not live_members(before):
            return before
        action = {'signal': sig.name, 'owned_pgid': group.group, 'before': before,
                  'board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'dispatch': 'PENDING'}
        record['termination_actions'].append(action)
        save()
        # Recheck the recorded private group identity immediately before killpg.
        action['verified_immediately_before_signal'] = group.snapshot()
        if not live_members(action['verified_immediately_before_signal']):
            action['dispatch'] = 'GROUP_ALREADY_EMPTY'
            save()
            return action['verified_immediately_before_signal']
        try:
            os.killpg(group.group, sig)
            action['dispatch'] = 'SENT'
        except ProcessLookupError:
            action['dispatch'] = 'GROUP_ALREADY_GONE'
        save()
        deadline = time.monotonic() + 5.0
        while True:
            members = group.snapshot()
            action['after'] = members
            if not live_members(members) or time.monotonic() >= deadline:
                save()
                break
            time.sleep(0.5)
    return group.snapshot()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--monitor-status', type=Path)
    parser.add_argument('--expected-monitor-pid', type=int, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    require(sys.platform.startswith('linux'), 'Linux /proc and process groups required')
    root = args.root.resolve(strict=True)
    status_path = (args.monitor_status or root / 'monitor/status.json').resolve(strict=True)
    output = (args.output or root / 'supplemental_group_guard.json').resolve()
    require(root in status_path.parents and output.parent == root and output != status_path,
            'Status must be inside root and guard output directly inside root')
    require(output.name == 'supplemental_group_guard.json', 'Use the dedicated supplemental guard output file')
    initial, stamp = read_status(status_path)
    current_boot = boot_id()
    require(initial.get('root') == str(root) and initial.get('boot_id') == current_boot,
            'Monitor root or boot differs')
    require(initial.get('monitor_pid') == args.expected_monitor_pid, 'Unexpected monitor PID')
    require(isinstance(initial.get('build_pid'), int) and initial['build_pid'] > 1, 'Missing private build PID')
    require(isinstance(initial.get('max_seconds'), int) and initial['max_seconds'] > 0, 'Invalid monitor budget')
    build_script = Path(initial['build_script']).resolve(strict=True)
    require(build_script.parent == root, 'Recorded build script is outside root')
    frozen = {key: initial[key] for key in ('root', 'monitor_pid', 'build_pid', 'max_seconds', 'build_script')}
    group = OwnedGroup(initial['build_pid'], current_boot, build_script)
    monitor = process_info(initial['monitor_pid'])
    if monitor is not None:
        require('--root' in monitor['argv'] and str(root) in monitor['argv'], 'Live monitor argv has another root')
    record = {'schema': 'rampq.gate05.supplemental_owned_group_guard/v1', 'status': 'CHECK_ONLY' if args.check_only else 'WATCHING',
              'boot_id': current_boot, 'root': str(root), 'guard_pid': os.getpid(),
              'guard_pgid': os.getpgrp(), 'guard_sid': os.getsid(0), 'owned_pgid': group.group, 'owned_sid': group.group,
              'guard_script_sha256': sha_bytes(Path(__file__).read_bytes()), 'monitor_status_path': str(status_path),
              'monitor_identity': frozen, 'initial_monitor_process': monitor, 'initial_monitor_status': initial,
              'initial_status_stamp': stamp, 'initial_group_members': group.initial_members,
              'leader_start_ticks': group.leader_start_ticks, 'started_board_time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'poll_seconds': POLL_SECONDS, 'stale_seconds': STALE_SECONDS, 'budget_grace_seconds': BUDGET_GRACE_SECONDS,
              'termination_actions': [], 'check_only': args.check_only,
              'original_monitor_status_and_tools_modified': False}
    if args.check_only:
        print(json.dumps(record, indent=2, allow_nan=False), flush=True)
        return 0
    require(os.getsid(0) == os.getpid() == os.getpgrp(), 'Launch guard in an independent new session with setsid')
    require(not output.exists(), 'Refuse to overwrite an existing guard evidence file')
    # First creation is exclusive; subsequent writes atomically replace this new file.
    with output.open('x', encoding='utf-8') as handle:
        handle.write('{}\n')
    save = lambda: atomic_save(output, record)
    started = time.monotonic()
    initial_age = max(0.0, time.time() - stamp['mtime_wall_seconds'])
    last_update = started - initial_age
    budget_deadline = started + max(0.0, initial['max_seconds'] + BUDGET_GRACE_SECONDS - stamp['elapsed_seconds'] - initial_age)
    last_stamp, last_status = stamp, initial
    save()
    try:
        while True:
            verify_status(last_status, frozen, current_boot)
            try:
                observed, observed_stamp = read_status(status_path)
            except (OSError, ValueError, RuntimeError) as exc:
                record['latest_status_read_error'] = repr(exc)
            else:
                verify_status(observed, frozen, current_boot)
                require(observed_stamp['elapsed_seconds'] >= last_stamp['elapsed_seconds'], 'Monitor elapsed time moved backwards')
                if (observed_stamp['mtime_ns'], observed_stamp['sha256']) != (last_stamp['mtime_ns'], last_stamp['sha256']):
                    last_update = time.monotonic()
                last_status, last_stamp = observed, observed_stamp
            members = group.snapshot()
            now = time.monotonic()
            record.update(last_monitor_status=last_status, last_status_stamp=last_stamp, last_group_members=members,
                          status_update_age_seconds=round(now - last_update, 3), guard_elapsed_seconds=round(now - started, 3))
            reason = None
            if last_status['status'] in TERMINAL_STATES:
                reason = 'ORIGINAL_MONITOR_TERMINAL'
            elif now - last_update > STALE_SECONDS:
                reason = 'ORIGINAL_MONITOR_STATUS_STALE_OVER_15_SECONDS'
            elif now >= budget_deadline:
                reason = 'ORIGINAL_MONITOR_BUDGET_PLUS_20_SECONDS_REACHED'
            save()
            if reason is not None:
                record['cleanup_trigger'] = reason
                final_members = terminate_owned(group, record, save)
                record.update(status='COMPLETE' if not live_members(final_members) else 'FAILED_OWNED_MEMBERS_REMAIN',
                              intervention_performed=any(row['dispatch'] == 'SENT' for row in record['termination_actions']),
                              final_group_members=final_members, owned_group_non_zombie_empty=not live_members(final_members),
                              owned_group_fully_empty=not final_members,
                              finished_board_time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                save()
                print(json.dumps({'status': record['status'], 'output': str(output),
                                  'owned_pgid': group.group, 'owned_group_non_zombie_empty': record['owned_group_non_zombie_empty'],
                                  'intervention_performed': record['intervention_performed']}), flush=True)
                return 0 if not live_members(final_members) else 2
            time.sleep(POLL_SECONDS)
    except BaseException as exc:
        record.update(status='GUARD_FAILED_NO_FURTHER_SIGNALS', error=repr(exc),
                      finished_board_time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        save()
        raise


if __name__ == '__main__':
    sys.exit(main())
