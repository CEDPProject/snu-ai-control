"""절감률 개선 1차 실험. 원본 root는 읽기 전용이며 별도 snapshot에서 실행한다."""

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from omegaconf import OmegaConf
from runners.full_control_benchmark_runner import (
    available_memory_gb, digest, mpc_fingerprint, now, scope_audit, write_json,
)

REPO = Path(__file__).resolve().parents[3]
CELLS = [('v3', 'room2'), ('v4', 'livingroom'), ('v5', 'livingroom')]
SAFETY = 'MPC.evaluation.action_model_screen.state_safety.out_of_band_policy'
M3, M6 = 'm3_physics_residual', 'm6_decision_focused'
ARMS = {
    'm3_original': (M3, {SAFETY: 'reject'}),
    'm3_b1': (M3, {SAFETY: 'non_worsening'}),
    'm3_b1_cem128x5': (M3, {
        SAFETY: 'non_worsening',
        'CONTROL_BENCHMARK.planner.cem_population': 128,
        'CONTROL_BENCHMARK.planner.cem_iterations': 5,
    }),
    'm6_original': (M6, {SAFETY: 'reject'}),
    'm6_b1': (M6, {SAFETY: 'non_worsening'}),
}


def case_config(source, root, overrides):
    cfg = OmegaConf.load(source)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    cfg.RESULT.savedir = str(root/'post_evaluation')
    # All arms omit only the expensive auxiliary report-input sweep; the main
    # factual/counterfactual reports and full policy evaluation remain enabled.
    cfg.MPC.counterfactual_report_input_ablation.run_after_mpc = False
    cfg.CONTROL_BENCHMARK.resume_from = None
    return cfg


def prepare(source_root, root, *, cells=None, arms=None, snapshot_src=None):
    cells = CELLS if cells is None else cells
    arms = ARMS if arms is None else arms
    snapshot_src = REPO/'src' if snapshot_src is None else Path(snapshot_src)
    root.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads((source_root/'manifest.json').read_text())
    inputs = source_manifest['input_sources']
    # Validate all copied input CSVs before using any original config.
    for item in inputs:
        if digest(item['path']) != item['sha256']:
            raise ValueError(f"Input mismatch: {item['path']}")
    hashes = {}
    for path in snapshot_src.rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.yaml', '.sh'):
            relative = path.relative_to(snapshot_src)
            target = root/'source_snapshot'/'src'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(relative)] = digest(target)
    cases = []
    # Complete matched arms on each cell before proceeding to the next cell.
    for group, environment in cells:
        original_scope = None
        for arm, (method, overrides) in arms.items():
            source = source_root/group/environment/'seed_42'/method/'configs.yaml'
            directory = root/arm/group/environment/'seed_42'/method
            directory.mkdir(parents=True)
            cfg = case_config(source, root, overrides)
            audit = scope_audit(cfg)
            original_audit = json.loads((source.parent/'scope_audit.json').read_text())
            for key in ('window_starts', 'expected_policy_steps', 'train_end_exclusive',
                        'test_start', 'counterfactual_report_rows'):
                if audit[key] != original_audit[key]:
                    raise ValueError(f'Original evaluation scope changed: {arm}/{group}/{key}')
            if original_scope is not None and audit['window_starts'] != original_scope:
                raise ValueError('Methods have different window cohorts')
            original_scope = audit['window_starts']
            OmegaConf.save(cfg, directory/'configs.yaml')
            write_json(directory/'scope_audit.json', audit)
            cases.append({
                'arm': arm, 'group': group, 'environment': environment, 'seed': 42,
                'method': method, 'relative_directory': str(directory.relative_to(root)),
                'source_config': str(source), 'source_config_sha256': digest(source),
                'config_sha256': digest(directory/'configs.yaml'), 'overrides': overrides,
                'mpc_config_sha256': mpc_fingerprint(cfg),
                'expected_policy_steps': audit['expected_policy_steps'],
            })
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'scikit-learn', 'torch', 'pandas', 'omegaconf')}
    manifest = {
        'status': 'prepared', 'created_at': now(), 'source_root': str(source_root),
        'case_count': len(cases), 'cases': cases, 'input_sources': inputs,
        'source_sha256': hashes, 'python': sys.version, 'packages': versions,
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'expected_policy_steps_all_cases': sum(c['expected_policy_steps'] for c in cases),
        'execution_source': 'immutable source_snapshot/src; all arms retrain in this run',
        'protocol': 'full original policy/factual/counterfactual scope; auxiliary post sweep disabled',
        'interpretation': 'selected discovery cells, seed42, surrogate only; not deployment validation',
    }
    write_json(root/'manifest.json', manifest)
    # Preserve the independent paired analysis code alongside frozen source.
    analysis = REPO/'src/reporting/savings_improvement_analysis.py'
    shutil.copy2(analysis, root/'savings_improvement_analysis.py')
    manifest['analysis_sha256'] = digest(root/'savings_improvement_analysis.py')
    write_json(root/'manifest.json', manifest)
    return manifest


def validate(root, manifest):
    for relative, expected in manifest.get('additional_frozen_files', {}).items():
        if digest(root/relative) != expected:
            raise ValueError(f'Frozen file changed: {relative}')
    for relative, expected in manifest['source_sha256'].items():
        if digest(root/'source_snapshot/src'/relative) != expected:
            raise ValueError(f'Snapshot changed: {relative}')
    for item in manifest['input_sources']:
        if digest(item['path']) != item['sha256']:
            raise ValueError(f"Input changed: {item['path']}")
    for case in manifest['cases']:
        if digest(root/case['relative_directory']/'configs.yaml') != case['config_sha256']:
            raise ValueError(f"Case config changed: {case['relative_directory']}")
    if digest(root/'savings_improvement_analysis.py') != manifest['analysis_sha256']:
        raise ValueError('Frozen analysis changed')
    versions = {name: importlib.metadata.version(name) for name in manifest['packages']}
    if versions != manifest['packages'] or sys.version != manifest['python']:
        raise ValueError('Python/package versions changed; use a new experiment root')


def read_status(directory):
    path = directory/'status.json'
    return json.loads(path.read_text()) if path.exists() else {'status': 'pending'}


def run(root, workers, minimum_memory):
    # A flock is held until every child exits, preventing duplicate resume.
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((root/'manifest.json').read_text())
        validate(root, manifest)
        pending = []
        for case in manifest['cases']:
            directory = root/case['relative_directory']
            state = read_status(directory)
            if state['status'] == 'completed':
                continue
            if state['status'] == 'running' and state.get('pid'):
                try:
                    os.kill(state['pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f"Worker PID {state['pid']} may still be alive")
            if any(p.name not in ('configs.yaml', 'scope_audit.json') for p in directory.iterdir()):
                archive = directory.with_name(directory.name+'_attempt_'+
                                              datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                directory.rename(archive)
                directory.mkdir()
                for name in ('configs.yaml', 'scope_audit.json'):
                    shutil.copy2(archive/name, directory/name)
            pending.append(case)
        manifest.update(status='running', supervisor_pid=os.getpid())
        manifest.setdefault('launches', []).append({
            'at': now(), 'pid': os.getpid(), 'workers': workers,
            'min_available_memory_gb': minimum_memory,
        })
        active = {}
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True
            for proc, _, _ in active.values():
                proc.terminate()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        snapshot = root/'source_snapshot/src'
        env = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        last_completed = -1
        counts = Counter(read_status(root/c['relative_directory'])['status']
                         for c in manifest['cases'])
        while (pending and not stopping) or active:
            memory = available_memory_gb()
            if pending and not stopping and len(active) < workers and (
                    memory is not None and memory >= minimum_memory):
                case = pending.pop(0)
                directory = root/case['relative_directory']
                handle = (directory/'process.log').open('a')
                proc = subprocess.Popen([
                    sys.executable, str(snapshot/'scripts/control/full_control_matrix.py'),
                    '--worker-config', str(directory/'configs.yaml'), '--method', case['method'],
                ], cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
                active[proc.pid] = (proc, case, handle)
                print(f'{now()} START {proc.pid} {case["relative_directory"]}', flush=True)
            for pid, (proc, case, handle) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                handle.close()
                directory = root/case['relative_directory']
                state = read_status(directory)
                if code != 0 or state['status'] != 'completed':
                    state.update(status='cancelled' if stopping else 'failed',
                                 exit_code=code, finished_at=now())
                    write_json(directory/'status.json', state)
                print(f'{now()} EXIT {code} {case["relative_directory"]}', flush=True)
                del active[pid]
            states = [{**c, **read_status(root/c['relative_directory'])} for c in manifest['cases']]
            counts = Counter(s['status'] for s in states)
            manifest.update(updated_at=now(), counts=dict(counts), active_pids=list(active))
            write_json(root/'progress.json', {'updated_at': now(), 'cases': states})
            write_json(root/'manifest.json', manifest)
            if counts['completed'] != last_completed:
                with (root/'analysis.log').open('a') as handle:
                    code = subprocess.call([
                        sys.executable, str(root/'savings_improvement_analysis.py'), str(root),
                    ], cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
                manifest['analysis_exit_code'] = code
                last_completed = counts['completed']
            if active or (pending and not stopping):
                time.sleep(10)
        manifest.update(status=('cancelled' if stopping else
                                'completed_with_failures' if counts['failed'] else 'completed'),
                        finished_at=now(), supervisor_pid=None, active_pids=[])
        manifest['input_hashes_unchanged_at_end'] = all(
            digest(i['path']) == i['sha256'] for i in manifest['input_sources'])
        write_json(root/'manifest.json', manifest)
        return 0 if manifest['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--min-available-memory-gb', type=float, default=24)
    args = parser.parse_args()
    if args.workers < 1 or args.min_available_memory_gb < 0:
        parser.error('workers must be positive and memory threshold nonnegative')
    root = args.root.resolve()
    if not (root/'manifest.json').exists():
        if not args.source_root:
            parser.error('new root requires --source-root')
        prepare(args.source_root.resolve(), root)
    if args.prepare_only:
        validate(root, json.loads((root/'manifest.json').read_text()))
        print(f'PREPARED {root}', flush=True)
        return 0
    return run(root, args.workers, args.min_available_memory_gb)


if __name__ == '__main__':
    sys.exit(main())
