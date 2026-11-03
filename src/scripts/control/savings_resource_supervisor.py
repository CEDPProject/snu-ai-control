"""기존 savings worker를 보존하며 자원 제한·동시성만 바꾸는 supervisor."""

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_info(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'state': text[0], 'ppid': int(text[1]), 'start_ticks': int(text[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(pid, start_ticks=None):
    info = process_info(pid)
    return bool(info and info['state'] not in ('Z', 'X') and
                (start_ticks is None or info['start_ticks'] == start_ticks))


def worker_matches(pid, config):
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        index = args.index(b'--worker-config')
        return Path(os.fsdecode(args[index+1])).resolve() == config.resolve()
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return False


def limit_process(pid, cpus, nice):
    count = 0
    for task in Path(f'/proc/{pid}/task').glob('*'):
        try:
            tid = int(task.name)
            os.sched_setaffinity(tid, cpus)
            os.setpriority(os.PRIO_PROCESS, tid, nice)
            if os.sched_getaffinity(tid) != set(cpus):
                raise RuntimeError(f'Affinity verification failed: {tid}')
            count += 1
        except ProcessLookupError:
            pass
    return count


def load_base(root):
    path = root/'source_snapshot/src/scripts/control/savings_improvement.py'
    spec = importlib.util.spec_from_file_location('frozen_savings_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_artifacts(root, manifest, base):
    """Science snapshot뿐 아니라 별도 고정 분석 코드와 재사용 결과도 검증한다."""
    base.validate(root, manifest)
    for relative, expected in manifest.get('additional_frozen_files', {}).items():
        if base.digest(root/relative) != expected:
            raise ValueError(f'Frozen file changed: {relative}')
    for case in manifest['cases']:
        for relative, expected in case.get('reused_artifact_sha256', {}).items():
            path = root/case['relative_directory']/relative
            if base.digest(path) != expected:
                raise ValueError(f'Reused artifact changed: {path}')


def supervisor_matches(pid, root):
    """--root와 허용된 진입점을 대조하여 다른 실행의 PID를 중지하지 않는다."""
    try:
        args = [os.fsdecode(a) for a in
                Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if a]
        index = args.index('--root')
        permitted = {'savings_improvement.py', 'savings_expansion.py',
                     'savings_resource_supervisor.py'}
        return (Path(args[index+1]).resolve() == root.resolve()
                and any(Path(arg).name in permitted for arg in args[:index]))
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return False


def final_analysis(root, manifest, base, env):
    """Expansion 인계 시 기존 wrapper 대신 고정 최종 분석까지 책임진다."""
    if manifest.get('experiment_kind') != 'm3_full_scope_expansion':
        return 0
    validate_artifacts(root, manifest, base)
    with (root/'final_analysis.log').open('a') as out:
        code = subprocess.call(
            [sys.executable, str(root/'savings_improvement_final_analysis.py'), str(root)],
            cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT)
    manifest['final_analysis_exit_code'] = code
    base.write_json(root/'manifest.json', manifest)
    return code


def run(root, workers, cpus, nice, memory, takeover):
    base = load_base(root)
    manifest = json.loads((root/'manifest.json').read_text())
    validate_artifacts(root, manifest, base)
    if manifest.get('experiment_kind') == 'm3_full_scope_expansion' and not os.environ.get('TMUX'):
        raise RuntimeError('Expansion scheduler must run inside tmux')
    active = {}
    old_pid = manifest.get('supervisor_pid')
    if old_pid and alive(old_pid) and old_pid != takeover:
        raise RuntimeError(f'Supervisor {old_pid} is alive; explicit matching takeover required')
    if takeover:
        if takeover != old_pid or not alive(takeover):
            raise RuntimeError('Takeover PID does not match live manifest supervisor')
        if not supervisor_matches(takeover, root):
            raise RuntimeError('Unexpected supervisor identity')
        # Stop only scheduling, never signal the workers. Freeze before reading
        # the final manifest to eliminate a launch/queue race.
        os.kill(takeover, signal.SIGSTOP)
        killed = False
        try:
            for _ in range(100):
                info = process_info(takeover)
                if info and info['state'] in ('T', 't'):
                    break
                time.sleep(.01)
            else:
                raise RuntimeError('Supervisor did not stop')
            manifest = json.loads((root/'manifest.json').read_text())
            known = set(manifest.get('active_pids', []))
            children = {
                int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit()
                and (info := process_info(int(p.name))) and info['ppid'] == takeover
                and info['state'] not in ('Z', 'X')
            }
            if not children <= known:
                raise RuntimeError(f'Unrecorded child processes; retry later: {children-known}')
            for case in manifest['cases']:
                directory = root/case['relative_directory']
                state = base.read_status(directory)
                pid = state.get('pid')
                if pid in known and alive(pid):
                    if not worker_matches(pid, directory/'configs.yaml'):
                        raise RuntimeError(f'Worker identity mismatch: {pid}')
                    active[pid] = {'case': case, 'process': None, 'handle': None,
                                   'start_ticks': process_info(pid)['start_ticks'], 'adopted': True}
            if not children <= set(active):
                raise RuntimeError('Some live children could not be adopted')
            if len(active) > workers:
                raise RuntimeError('Requested concurrency smaller than existing workers')
            handoff = {
                'at': base.now(), 'old_supervisor_pid': takeover, 'new_supervisor_pid': os.getpid(),
                'adopted_worker_pids': list(active), 'workers': workers, 'logical_cpu_ids': cpus,
                'reason': f'user requested CPU{len(cpus)}/worker{workers}; preserve all running cases',
                'adopted_worker_identity': {
                    str(pid): {'start_ticks': item['start_ticks'],
                               'relative_directory': item['case']['relative_directory']}
                    for pid, item in active.items()},
                'old_supervisor_only_signal': 'SIGSTOP then SIGKILL; no worker termination',
                'previous_manifest': manifest,
            }
            base.write_json(root/f'supervisor_handoff_{takeover}_{os.getpid()}.json', handoff)
            # SIGTERM would invoke the old handler and terminate its workers.
            # SIGKILL only this already-stopped supervisor instead; children keep running.
            os.kill(takeover, signal.SIGKILL)
            killed = True
            for _ in range(200):
                if not alive(takeover):
                    break
                time.sleep(.01)
            else:
                raise RuntimeError('Old supervisor has not released its process resources')
        finally:
            if not killed and alive(takeover):
                os.kill(takeover, signal.SIGCONT)
    else:
        # Recover surviving workers after an interrupted coordinator too.
        for case in manifest['cases']:
            directory = root/case['relative_directory']
            state = base.read_status(directory)
            pid = state.get('pid')
            if state['status'] == 'running' and pid and alive(pid):
                if not worker_matches(pid, directory/'configs.yaml'):
                    raise RuntimeError(f'Unexpected live PID: {pid}')
                active[pid] = {'case': case, 'process': None, 'handle': None,
                               'start_ticks': process_info(pid)['start_ticks'], 'adopted': True}

    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        limit_process(os.getpid(), cpus, nice)
        for pid in active:
            limit_process(pid, cpus, nice)
        active_dirs = {item['case']['relative_directory'] for item in active.values()}
        pending = []
        for case in manifest['cases']:
            directory = root/case['relative_directory']
            if case['relative_directory'] in active_dirs or base.read_status(directory)['status'] == 'completed':
                continue
            # Match the original attempt-preservation contract.
            if any(p.name not in ('configs.yaml', 'scope_audit.json') for p in directory.iterdir()):
                archive = directory.with_name(directory.name+'_attempt_'+
                                              datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                directory.rename(archive)
                directory.mkdir()
                import shutil
                for name in ('configs.yaml', 'scope_audit.json'):
                    shutil.copy2(archive/name, directory/name)
            pending.append(case)
        manifest.setdefault('initial_resource_configuration', {
            'workers': manifest.get('workers'),
            'logical_cpu_ids': manifest.get('logical_cpu_ids'),
            'nice': manifest.get('nice'),
        })
        manifest.update(status='running', supervisor_pid=os.getpid(), workers=workers,
                        parallel_workers=workers, logical_cpu_ids=cpus, nice=nice)
        manifest.setdefault('launches', []).append({
            'at': base.now(), 'pid': os.getpid(), 'workers': workers,
            'logical_cpu_ids': cpus, 'nice': nice, 'min_available_memory_gb': memory,
            'adopted_worker_pids': list(active), 'superseded_supervisor_pid': takeover,
            'scheduler_path': str(Path(__file__).resolve()), 'scheduler_sha256': base.digest(__file__),
        })
        previous_limits = json.loads((root/'resource_limits.json').read_text()) \
            if (root/'resource_limits.json').exists() else None
        base.write_json(root/'resource_limits.json', {
            'applied_at': base.now(), 'supervisor_pid': os.getpid(), 'workers': workers,
            'logical_cpu_ids': cpus, 'aggregate_max_logical_cpus': len(cpus), 'nice': nice,
            'scope': 'supervisor, adopted and new worker threads, inherited analysis processes',
            'previous_limits': previous_limits, 'snapshot_and_case_configs_unchanged': True,
        })
        (root/'supervisor.pid').write_text(str(os.getpid())+'\n')
        snapshot = root/'source_snapshot/src'
        env = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            # Stop dispatching but let active cases finish: no lost windows.
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(f'{base.now()} ADOPTED {list(active)} CPU={cpus} WORKERS={workers}', flush=True)
        last_completed = -1
        while True:
            available = base.available_memory_gb()
            if pending and not stopping and len(active) < workers and (
                    available is not None and available >= memory):
                case = pending.pop(0)
                directory = root/case['relative_directory']
                handle = (directory/'process.log').open('a')
                proc = subprocess.Popen([
                    sys.executable, str(snapshot/'scripts/control/full_control_matrix.py'),
                    '--worker-config', str(directory/'configs.yaml'), '--method', case['method'],
                ], cwd=root/'source_snapshot', env=env, stdout=handle, stderr=subprocess.STDOUT)
                active[proc.pid] = {'case': case, 'process': proc, 'handle': handle,
                                    'start_ticks': process_info(proc.pid)['start_ticks'], 'adopted': False}
                print(f'{base.now()} START {proc.pid} {case["relative_directory"]}', flush=True)
            for pid, item in list(active.items()):
                proc = item['process']
                code = proc.poll() if proc else None
                if (proc is not None and code is None) or (
                        proc is None and alive(pid, item['start_ticks'])):
                    continue
                if item['handle']:
                    item['handle'].close()
                directory = root/item['case']['relative_directory']
                state = base.read_status(directory)
                # An adopted process has no waitpid exit code. Never invent one.
                if state['status'] != 'completed' or (code is not None and code != 0):
                    state.update(status='failed', exit_code=code, finished_at=base.now())
                    base.write_json(directory/'status.json', state)
                print(f'{base.now()} EXIT {pid} code={code} adopted={item["adopted"]} '
                      f'{item["case"]["relative_directory"]}', flush=True)
                del active[pid]
            states = [{**case, **base.read_status(root/case['relative_directory'])}
                      for case in manifest['cases']]
            counts = Counter(s['status'] for s in states)
            manifest.update(updated_at=base.now(), counts=dict(counts), active_pids=list(active),
                            parallel_workers=workers, logical_cpu_ids=cpus, nice=nice)
            base.write_json(root/'progress.json', {'updated_at': base.now(), 'cases': states})
            base.write_json(root/'manifest.json', manifest)
            if counts['completed'] != last_completed:
                with (root/'analysis.log').open('a') as out:
                    manifest['analysis_exit_code'] = subprocess.call([
                        sys.executable, str(root/'savings_improvement_analysis.py'), str(root),
                    ], cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT)
                last_completed = counts['completed']
            if not active and (not pending or stopping):
                break
            time.sleep(5)
        manifest.update(status='cancelled' if stopping and pending else
                        'completed_with_failures' if counts['failed'] else 'completed',
                        finished_at=base.now(), active_pids=[], supervisor_pid=None)
        manifest['input_hashes_unchanged_at_end'] = all(
            base.digest(i['path']) == i['sha256'] for i in manifest['input_sources'])
        base.write_json(root/'manifest.json', manifest)
        if manifest['status'] == 'completed':
            return final_analysis(root, manifest, base, env)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--cpus', default='0,1,2,3,4,5,6,7,8,9,10,11')
    parser.add_argument('--nice', type=int, default=19)
    parser.add_argument('--min-available-memory-gb', type=float, default=24)
    parser.add_argument('--takeover-pid', type=int)
    args = parser.parse_args()
    cpus = sorted(set(map(int, args.cpus.split(','))))
    if not cpus or not set(cpus) <= os.sched_getaffinity(0):
        parser.error('CPUs must be a nonempty subset of the launch process affinity')
    if args.workers < 1 or not 0 <= args.nice <= 19 or args.min_available_memory_gb < 0:
        parser.error('Invalid worker, nice or memory limit')
    return run(args.root.resolve(), args.workers, cpus, args.nice,
               args.min_available_memory_gb, args.takeover_pid)


if __name__ == '__main__':
    sys.exit(main())
