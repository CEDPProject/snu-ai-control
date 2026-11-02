"""Durable full-protocol experiment matrix, executed from an immutable snapshot."""

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime

from omegaconf import OmegaConf


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now().astimezone().isoformat()


def order_pending_cases(cases, method_order=None):
    """Queue order only: listed methods first in that order, ties keep manifest order."""
    if not method_order:
        return list(cases)
    method_order = [str(m) for m in method_order]
    known = {str(case['method']) for case in cases}
    unknown = sorted(set(method_order)-known)
    if unknown or len(set(method_order)) != len(method_order):
        raise ValueError(f'Invalid method_order {method_order}; unknown={unknown}')
    rank = {method: index for index, method in enumerate(method_order)}
    return sorted(cases, key=lambda case: rank.get(str(case['method']), len(rank)))


def available_memory_gb():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])/1024**2
    except OSError:
        pass
    return None


def mpc_fingerprint(cfg):
    text = json.dumps(OmegaConf.to_container(cfg.MPC, resolve=True),
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def scope_audit(cfg):
    """Precompute the exact original window cohort, without fitting any model."""
    import numpy as np
    from controllers.action_crossfit import align_boundaries_to_control_episodes
    from controllers.pipeline import _select_evaluation_windows, _training_control_support
    from controllers.total_power_evaluation import policy_window_transition_rows
    from data_provider.control_data import load_control_data

    data = load_control_data(cfg)
    train_end, test_start = data.split_indices(cfg.DATASET.split_rate)
    ev = cfg.MPC.evaluation
    alignment_fallback = None
    if ev.action_options.episode_group_split.enabled:
        try:
            train_end, test_start = align_boundaries_to_control_episodes(
                data, [train_end, test_start],
            )
        except ValueError as error:
            alignment_fallback = str(error)
    starts = _select_evaluation_windows(
        data, test_start, int(ev.closed_loop_steps), int(ev.stride), int(ev.max_windows),
        int(cfg.MPC.prediction_horizon), str(cfg.MPC.disturbance_forecast),
    )
    repeated = ev.action_options.repeated_joint_actions
    support, _, _ = _training_control_support(
        data, train_end, cfg.MPC.logical_rules, ev.restrict_to_training_control_support,
        repeated_action_filter_enabled=repeated.enabled,
        minimum_repeated_samples=repeated.minimum_training_samples,
        minimum_repeated_episodes=repeated.minimum_training_episodes,
    )
    report = ev.mpc_counterfactual_total_power_report
    report_rows = (policy_window_transition_rows(data, starts, int(ev.closed_loop_steps))
                   if str(report.evaluation_row_scope) == 'policy_windows'
                   else data.valid_transition_rows(np.arange(test_start, len(data.states)-1)))
    report_rows = report_rows[::int(report.row_stride_steps)]
    if int(report.maximum_evaluation_rows):
        report_rows = report_rows[:int(report.maximum_evaluation_rows)]
    return {
        'dataset_rows': len(data.states), 'train_end_exclusive': train_end,
        'test_start': test_start, 'split_alignment_fallback': alignment_fallback,
        'window_starts': starts, 'evaluated_windows': len(starts),
        'closed_loop_steps_per_window': int(ev.closed_loop_steps),
        'expected_policy_steps': len(starts)*int(ev.closed_loop_steps),
        'counterfactual_report_rows': len(report_rows) if report.enabled else 0,
        'train_supported_action_count': len(support),
        'mpc_config_sha256': mpc_fingerprint(cfg),
    }


def full_worker(config_path, method):
    """One entire original pipeline, including its post-MPC report-input sweep."""
    from controllers.full_control_backend import FullControlBackend
    from controllers.pipeline import run_mpc_pipeline
    from runners.mpc_runner import _run_counterfactual_input_ablation_if_enabled
    import torch

    cfg = OmegaConf.load(config_path)
    directory = Path(config_path).resolve().parent
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s',
        force=True, handlers=[logging.StreamHandler(),
                              logging.FileHandler(directory/'run.log')],
    )
    torch.set_num_threads(int(cfg.CONTROL_BENCHMARK.torch_threads))
    from controllers.guarded_milp import hard_timeout_seconds
    state = {'status': 'running', 'phase': 'original_full_pipeline',
             'method': method, 'pid': os.getpid(),
             'started_at': now(), 'mpc_config_sha256': mpc_fingerprint(cfg),
             'solver_hard_timeout_seconds': hard_timeout_seconds()}
    write_json(directory/'status.json', state)
    started = time.perf_counter()
    try:
        backend = FullControlBackend(cfg, method, directory)
        summary = run_mpc_pipeline(cfg, directory, control_backend=backend)
        audit_path = directory/'scope_audit.json'
        if audit_path.exists():
            import pandas as pd
            audit = json.loads(audit_path.read_text())
            trace = pd.read_csv(directory/'mpc_closed_loop_steps.csv')
            if (summary['evaluated_windows'] != audit['evaluated_windows']
                    or len(trace) != audit['expected_policy_steps']
                    or mpc_fingerprint(cfg) != audit['mpc_config_sha256']):
                raise AssertionError('Completed evaluation differs from predeclared original scope')
        state['phase'] = 'post_MPC_report_input_sweep'
        write_json(directory/'status.json', state)
        post = _run_counterfactual_input_ablation_if_enabled(cfg, directory)
        state.update({
            'status': 'completed', 'finished_at': now(),
            'wall_seconds': time.perf_counter()-started,
            'evaluated_windows': summary['evaluated_windows'],
            'closed_loop_steps_per_window': int(cfg.MPC.evaluation.closed_loop_steps),
            'post_evaluation_completed': post is not None,
        })
        write_json(directory/'status.json', state)
    except BaseException as error:
        state.update({'status': 'failed', 'finished_at': now(),
                      'wall_seconds': time.perf_counter()-started, 'error': str(error)})
        (directory/'error.txt').write_text(traceback.format_exc())
        write_json(directory/'status.json', state)
        raise


def run_full_matrix(cfg, *, existing_root=None, workers=None, prepare_only=False,
                    method_order=None, min_available_memory_gb=0.0):
    """Freeze source/config, queue every combination, never silently shrink scope.

    ``method_order`` and ``min_available_memory_gb`` change only when cases start,
    never the snapshot, configs, or evaluation scope of any case.
    """
    from data_provider.dataset_versions import resolve_dataset_csv_path

    spec = cfg.CONTROL_FULL_BENCHMARK
    count_workers = int(workers if workers is not None else spec.parallel_workers)
    if count_workers < 1:
        raise ValueError('parallel_workers must be positive')
    min_available_memory_gb = float(min_available_memory_gb or 0.0)
    if min_available_memory_gb < 0:
        raise ValueError('min_available_memory_gb must be non-negative')
    if existing_root:
        root = Path(existing_root).resolve()
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest.get('supervisor_pid') != os.getpid():
            pid = manifest.get('supervisor_pid')
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f'Matrix supervisor {pid} is still alive; refusing duplicate execution')
        for case in manifest['cases']:
            p = root/case['relative_directory']/'status.json'
            if p.exists():
                status = json.loads(p.read_text())
                if status['status'] == 'running':
                    try:
                        os.kill(status['pid'], 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError(f"Case process {status['pid']} is still running")
        # A failed or interrupted case is archived, never overwritten.
        for case in manifest['cases']:
            directory = root/case['relative_directory']
            state_path = directory/'status.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if state.get('status') != 'completed' and any(
                    p.name not in ('configs.yaml', 'scope_audit.json') for p in directory.iterdir()):
                archive = directory.with_name(directory.name+'_attempt_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                directory.rename(archive)
                directory.mkdir()
                shutil.copy2(archive/'configs.yaml', directory/'configs.yaml')
                if (archive/'scope_audit.json').exists():
                    shutil.copy2(archive/'scope_audit.json', directory/'scope_audit.json')
    else:
        stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
        root = (Path(spec.output_directory)/f'original_full_{stamp}').resolve()
        root.mkdir(parents=True, exist_ok=False)
        source = Path(__file__).resolve().parents[1]
        hashes = {}
        for path in source.rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.sh', '.yaml'):
                relative = path.relative_to(source)
                destination = root/'source_snapshot'/'src'/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                hashes[str(relative)] = digest(destination)
        OmegaConf.save(cfg, root/'matrix_config.yaml')
        cases = []
        original_hash = mpc_fingerprint(cfg)
        input_sources = {}
        scopes = []
        for group in spec.dataset_groups:
            for environment in spec.environments:
                for version in group.versions:
                    p = resolve_dataset_csv_path(cfg.DATASET.datadir, str(version), str(environment))
                    input_sources[str(p)] = {'path': str(p), 'sha256': digest(p)}
                version_list = [str(v) for v in group.versions]
                scope_cfg = OmegaConf.merge(cfg, {
                    'DATASET': {'environment_name': str(environment),
                                'data_version': version_list[0],
                                'data_versions': version_list if len(version_list) > 1 else None},
                })
                scope = scope_audit(scope_cfg)
                scopes.append({'group': str(group.name), 'environment': str(environment), **scope})
                for seed in spec.seeds:
                    for method in spec.methods:
                        relative = f'{group.name}/{environment}/seed_{seed}/{method}'
                        directory = root/relative
                        directory.mkdir(parents=True)
                        version_list = [str(v) for v in group.versions]
                        case_cfg = OmegaConf.merge(cfg, {
                            'DATASET': {'environment_name': str(environment),
                                        'data_version': version_list[0],
                                        'data_versions': version_list if len(version_list) > 1 else None},
                            'DEFAULT': {'seed': int(seed),
                                        'exp_name': f'original_full_{group.name}_{environment}_{method}'},
                            'RESULT': {'savedir': str(root/'post_evaluation')},
                            'CONTROL_BENCHMARK': {
                                'resume_from': None,
                                'planner': {'mode_matrix_strategy': str(spec.mode_matrix_strategy)},
                            },
                        })
                        if mpc_fingerprint(case_cfg) != original_hash:
                            raise AssertionError('Original MPC configuration was modified')
                        OmegaConf.save(case_cfg, directory/'configs.yaml')
                        write_json(directory/'scope_audit.json', scope)
                        cases.append({'group': str(group.name), 'environment': str(environment),
                                      'seed': int(seed), 'method': str(method),
                                      'relative_directory': relative})
        manifest = {
            'status': 'prepared', 'created_at': now(), 'cases': cases,
            'case_count': len(cases), 'source_sha256': hashes,
            'scopes': scopes,
            'expected_policy_steps_all_cases': sum(
                s['expected_policy_steps'] for s in scopes
            )*len(spec.seeds)*len(spec.methods),
            'expected_factual_report_decisions_all_cases': sum(
                s['counterfactual_report_rows'] for s in scopes
            )*len(spec.seeds)*len(spec.methods),
            'input_sources': list(input_sources.values()),
            'original_mpc_config_sha256': original_hash,
            'execution_source': 'immutable source_snapshot/src',
            'no_test_window_or_step_reduction': True,
            'original_pipeline_and_post_evaluation': True,
            'hardware_timing_scope': 'shared host; concurrent worker count recorded',
            'source_approximation_note': (
                'New model architectures are project implementations, not paper replications; '
                'DFL remains a teacher-cost proxy. MILP model2/4 strategy is recorded explicitly.'
            ),
        }
    # Resume always executes the existing snapshot and configs, not live source.
    for relative, expected in manifest['source_sha256'].items():
        if digest(root/'source_snapshot'/'src'/relative) != expected:
            raise ValueError(f'Corrupt source snapshot: {relative}')
    for source in manifest['input_sources']:
        if digest(source['path']) != source['sha256']:
            raise ValueError(f"Input CSV changed: {source['path']}")
    if prepare_only:
        write_json(root/'manifest.json', manifest)
        print(f'PREPARED_FULL_MATRIX_ROOT={root}', flush=True)
        return root
    pending = []
    for case in manifest['cases']:
        path = root/case['relative_directory']/'status.json'
        if not path.exists() or json.loads(path.read_text()).get('status') != 'completed':
            pending.append(case)
    pending = order_pending_cases(pending, method_order)
    launch = {'started_at': now(), 'supervisor_pid': os.getpid(),
              'parallel_workers': count_workers,
              'method_order': list(method_order) if method_order else None,
              'min_available_memory_gb': min_available_memory_gb,
              'pending_at_start': len(pending)}
    manifest.update({'status': 'running', 'supervisor_pid': os.getpid(),
                     'parallel_workers': count_workers, 'updated_at': now(),
                     'queue_method_order': launch['method_order'],
                     'min_available_memory_gb': min_available_memory_gb})
    manifest.setdefault('supervisor_launches', []).append(launch)
    write_json(root/'manifest.json', manifest)
    print(f'FULL_MATRIX_ROOT={root}', flush=True)
    print(f'CASES={manifest["case_count"]} WORKERS={count_workers} '
          f'METHOD_ORDER={launch["method_order"]} MIN_AVAILABLE_GB={min_available_memory_gb}',
          flush=True)
    snapshot = root/'source_snapshot'/'src'
    entry = snapshot/'scripts/control/full_control_matrix.py'
    active = {}
    previous_completed = -1
    while pending or active:
        while pending and len(active) < count_workers:
            if min_available_memory_gb and active:
                # Memory grows after start, so start at most one case per poll.
                available = available_memory_gb()
                if available is not None and available < min_available_memory_gb:
                    manifest['last_memory_deferral'] = {
                        'at': now(), 'available_gb': round(available, 2), 'active': len(active),
                    }
                    break
            case = pending.pop(0)
            directory = root/case['relative_directory']
            log = (directory/'process.log').open('a')
            env = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            process = subprocess.Popen(
                [sys.executable, str(entry), '--worker-config', str(directory/'configs.yaml'),
                 '--method', case['method']], stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            active[process.pid] = (process, case, log)
            print(f'START pid={process.pid} {case["relative_directory"]}', flush=True)
            if min_available_memory_gb:
                break
        for pid, (process, case, log) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close()
                directory = root/case['relative_directory']
                state_path = directory/'status.json'
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                if code != 0 or state.get('status') != 'completed':
                    state.update({'status': 'failed', 'exit_code': code,
                                  'finished_at': now(), 'pid': pid})
                    write_json(state_path, state)
                print(f'EXIT code={code} {case["relative_directory"]}', flush=True)
                del active[pid]
        states = []
        for case in manifest['cases']:
            path = root/case['relative_directory']/'status.json'
            state = json.loads(path.read_text()) if path.exists() else {'status': 'pending'}
            states.append({**case, **state})
        manifest.update({
            'updated_at': now(), 'active_pids': list(active),
            'completed_cases': sum(s['status'] == 'completed' for s in states),
            'failed_cases': sum(s['status'] == 'failed' for s in states),
            'pending_cases': sum(s['status'] == 'pending' for s in states),
        })
        write_json(root/'progress.json', {'updated_at': now(), 'cases': states})
        write_json(root/'manifest.json', manifest)
        if previous_completed != manifest['completed_cases']:
            from reporting.full_control_comparison import summarize_full_matrix
            try:
                summarize_full_matrix(root)
                previous_completed = manifest['completed_cases']
                manifest['comparison_status'] = 'up_to_date'
            except Exception:
                (root/'comparison_error.txt').write_text(traceback.format_exc())
                manifest['comparison_status'] = 'failed_retry_pending'
                # Evaluation workers continue even if a postprocessing table fails.
        if pending or active:
            time.sleep(10)
    manifest.update({'status': 'completed_with_failures' if manifest['failed_cases'] else 'completed',
                     'finished_at': now(), 'active_pids': []})
    manifest['input_csv_hashes_unchanged_at_end'] = all(
        digest(source['path']) == source['sha256'] for source in manifest['input_sources'])
    from reporting.full_control_comparison import summarize_full_matrix
    try:
        summarize_full_matrix(root)
        manifest['comparison_status'] = 'completed'
    except Exception:
        (root/'comparison_error.txt').write_text(traceback.format_exc())
        manifest['comparison_status'] = 'failed'
        manifest['status'] = 'evaluated_comparison_failed'
    write_json(root/'manifest.json', manifest)
    print(f'FINISHED {manifest["status"]}: {root}', flush=True)
    return root
