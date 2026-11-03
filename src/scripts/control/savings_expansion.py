"""m3 두 후보의 전체12셀 확대: 완료9조합 보존 복사, 신규27조합 실행."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import savings_improvement as base

CELLS = [(group, room) for group in ('v3', 'v4', 'v5', 'concat')
         for room in ('livingroom', 'room1', 'room2')]
ARMS = {name: base.ARMS[name] for name in
        ('m3_original', 'm3_b1', 'm3_b1_cem128x5')}
CPU_IDS = set(range(10))


def normalized_config(path):
    cfg = base.OmegaConf.load(path)
    cfg.RESULT.savedir = '<output>'
    cfg.CONTROL_BENCHMARK.resume_from = None
    return base.OmegaConf.to_container(cfg, resolve=True)


def reuse_case(root, case, previous_root, previous_case):
    """완료 산출물을 독립 복사한다. 다른 root에 쓰는 symlink는 사용하지 않는다."""
    old = previous_root/previous_case['relative_directory']
    target = root/case['relative_directory']
    if base.read_status(old)['status'] != 'completed':
        raise ValueError(f'Cannot reuse unfinished case: {old}')
    if base.digest(old/'configs.yaml') != previous_case['config_sha256']:
        raise ValueError(f'Reused config changed: {old}')
    if normalized_config(old/'configs.yaml') != normalized_config(target/'configs.yaml'):
        raise ValueError(f'Reused scientific config differs: {old}')
    if (old/'scope_audit.json').read_bytes() != (target/'scope_audit.json').read_bytes():
        raise ValueError(f'Reused evaluation scope differs: {old}')
    evidence = ('status.json', 'mpc_evaluation.json', 'mpc_closed_loop_steps.csv',
                'configs.yaml', 'scope_audit.json')
    hashes = {name: base.digest(old/name) for name in evidence}
    # Keep the previous output path inside completed configs as historical provenance.
    # Completed cases are never executed again by the coordinator.
    shutil.rmtree(target)
    shutil.copytree(old, target)
    if any(base.digest(target/name) != value for name, value in hashes.items()):
        raise ValueError(f'Reused artifact copy mismatch: {target}')
    case.update(config_sha256=hashes['configs.yaml'],
                mpc_config_sha256=previous_case['mpc_config_sha256'],
                reused_from=str(old), reused_artifact_sha256=hashes)


def prepare(previous_root, root):
    previous = json.loads((previous_root/'manifest.json').read_text())
    base.validate(previous_root, previous)
    if previous['status'] != 'completed':
        raise ValueError('Discovery run must be completed')
    manifest = base.prepare(
        Path(previous['source_root']), root, cells=CELLS, arms=ARMS,
        snapshot_src=previous_root/'source_snapshot/src')
    previous_cases = {(c['group'], c['environment'], c['arm']): c
                      for c in previous['cases']}
    reused = 0
    for case in manifest['cases']:
        key = (case['group'], case['environment'], case['arm'])
        if key in previous_cases:
            reuse_case(root, case, previous_root, previous_cases[key])
            reused += 1
    if reused != 9 or len(manifest['cases']) != 36:
        raise ValueError('Expected exactly9 reused and27 new cases')
    final = base.REPO/'src/reporting/savings_improvement_final_analysis.py'
    shutil.copy2(final, root/'savings_improvement_final_analysis.py')
    shutil.copy2(Path(__file__), root/'expansion_coordinator.py')
    shutil.copy2(Path(base.__file__), root/'scheduling_base.py')
    manifest.update(
        experiment_kind='m3_full_scope_expansion',
        discovery_root=str(previous_root), reused_case_count=reused,
        new_case_count=len(manifest['cases'])-reused,
        logical_cpu_ids=sorted(CPU_IDS), workers=6, nice=19,
        execution_source='discovery immutable science snapshot;9 completed cases copied;27 retrained',
        interpretation='all12 original cells, seed42, surrogate only; not independent deployment validation',
        aggregate_cohorts={'single_versions': ['v3', 'v4', 'v5'], 'concat': ['concat']},
        additional_frozen_files={name: base.digest(root/name) for name in (
            'savings_improvement_final_analysis.py', 'expansion_coordinator.py', 'scheduling_base.py')})
    base.write_json(root/'manifest.json', manifest)
    validate(root, manifest)
    return manifest


def validate(root, manifest):
    base.validate(root, manifest)
    if base.digest(Path(__file__)) != base.digest(root/'expansion_coordinator.py'):
        raise ValueError('Coordinator code changed; preserve this run and prepare a new root')
    if base.digest(Path(base.__file__)) != base.digest(root/'scheduling_base.py'):
        raise ValueError('Scheduling base changed; preserve this run and prepare a new root')
    for case in manifest['cases']:
        for name, expected in case.get('reused_artifact_sha256', {}).items():
            if base.digest(root/case['relative_directory']/name) != expected:
                raise ValueError(f'Reused artifact changed: {case["relative_directory"]}/{name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous-root', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--min-available-memory-gb', type=float, default=24)
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        parser.error('Run this coordinator inside tmux')
    if args.min_available_memory_gb < 0:
        parser.error('memory threshold must be nonnegative')
    # Affinity is a hard aggregate CPU placement cap inherited by every child/thread.
    if not CPU_IDS <= os.sched_getaffinity(0):
        parser.error('CPUs0-9 must be available')
    os.sched_setaffinity(0, CPU_IDS)
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        os.nice(19-os.getpriority(os.PRIO_PROCESS, 0))
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    root = args.root.resolve()
    if not (root/'manifest.json').exists():
        if not args.previous_root:
            parser.error('new root requires --previous-root')
        prepare(args.previous_root.resolve(), root)
    manifest = json.loads((root/'manifest.json').read_text())
    validate(root, manifest)
    if manifest.get('experiment_kind') != 'm3_full_scope_expansion':
        parser.error('root is not an m3 expansion')
    if args.prepare_only:
        print(f'PREPARED {root}:9 reused +27 new cases', flush=True)
        return 0
    result = base.run(root, workers=6, minimum_memory=args.min_available_memory_gb)
    if result == 0:
        validate(root, json.loads((root/'manifest.json').read_text()))
        env = dict(os.environ, PYTHONPATH=str(root/'source_snapshot/src'))
        with (root/'final_analysis.log').open('a') as handle:
            result = subprocess.call(
                [sys.executable, str(root/'savings_improvement_final_analysis.py'), str(root)],
                cwd=base.REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        manifest = json.loads((root/'manifest.json').read_text())
        manifest['final_analysis_exit_code'] = result
        base.write_json(root/'manifest.json', manifest)
    return result


if __name__ == '__main__':
    sys.exit(main())
