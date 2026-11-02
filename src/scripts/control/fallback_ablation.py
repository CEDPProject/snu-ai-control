"""원본 전체 비교와 분리된 fallback 완화 ablation 실행기.

원본 조합의 configs.yaml을 복사해 arm별 설정 하나만 바꾸고, 같은 worker
(`full_control_matrix.py --worker-config`)를 한 번에 하나씩 순차 실행한다.
원본 run root와 그 snapshot은 건드리지 않는다. baseline은 원본 결과를 그대로 쓴다.

arm
  b1_non_worsening : state_safety.out_of_band_policy=non_worsening
                     (모든 후보가 범위 밖 예측일 때만, 이전 action 유지보다 위반이 커지지 않는 후보 허용)
  b2_state_safety_off : state_safety.enabled=false (상한 참고용)
  b3_support15     : 후보 pool·global trust 최소 표본 30→15, episode 3→2

사용:
  python src/scripts/control/fallback_ablation.py --source-root <원본 run root> --root <ablation root> \
      [--min-available-memory-gb 14] [--prepare-only]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
ENTRY = REPO/'src/scripts/control/full_control_matrix.py'
METHOD = 'm3_physics_residual'
CELLS = [('v3', 'room2'), ('v4', 'livingroom'), ('v5', 'livingroom')]
ARMS = {
    'b1_non_worsening': {
        'MPC.evaluation.action_model_screen.state_safety.out_of_band_policy': 'non_worsening',
    },
    'b2_state_safety_off': {
        'MPC.evaluation.action_model_screen.state_safety.enabled': False,
    },
    'b3_support15': {
        'MPC.evaluation.action_options.repeated_joint_actions.minimum_training_samples': 15,
        'MPC.evaluation.action_options.repeated_joint_actions.minimum_training_episodes': 2,
        'MPC.evaluation.action_trust.global_support.minimum_samples': 15,
        'MPC.evaluation.action_trust.global_support.minimum_episodes': 2,
    },
}


def log(message):
    print(f'{datetime.now():%Y-%m-%d %H:%M:%S} {message}', flush=True)


def available_memory_gb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024**2
    return None


def prepare(source_root, root):
    cases = []
    for arm, overrides in ARMS.items():
        for group, environment in CELLS:
            source = source_root/group/environment/'seed_42'/METHOD/'configs.yaml'
            cfg = OmegaConf.load(source)
            for key, value in overrides.items():
                if OmegaConf.select(cfg, key, default='__missing__') == '__missing__' \
                        and not key.endswith('out_of_band_policy'):
                    raise KeyError(f'{key} not in {source}')
                OmegaConf.update(cfg, key, value, force_add=True)
            # Keep every output inside the ablation root; the post-MPC report sweep is
            # replaced by the separate fixed-state analysis script.
            cfg.RESULT.savedir = str(root/'post_evaluation')
            cfg.MPC.counterfactual_report_input_ablation.run_after_mpc = False
            directory = root/arm/group/environment/'seed_42'/METHOD
            directory.mkdir(parents=True, exist_ok=True)
            path = directory/'configs.yaml'
            if not path.exists():
                OmegaConf.save(cfg, path)
            cases.append({'arm': arm, 'group': group, 'environment': environment,
                          'directory': str(directory), 'source_config': str(source),
                          'overrides': overrides})
    (root/'ablation_manifest.json').write_text(json.dumps(
        {'source_root': str(source_root), 'method': METHOD, 'cases': cases,
         'baseline': 'original run result (not re-run)'}, ensure_ascii=False, indent=2))
    return cases


def status(directory):
    path = Path(directory)/'status.json'
    return json.loads(path.read_text()).get('status') if path.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--min-available-memory-gb', type=float, default=14.0)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source_root, root = Path(args.source_root).resolve(), Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases = prepare(source_root, root)
    log(f'prepared {len(cases)} cases under {root}')
    if args.prepare_only:
        return 0
    for case in cases:
        name = f"{case['arm']}/{case['group']}/{case['environment']}"
        if status(case['directory']) == 'completed':
            log(f'SKIP completed {name}')
            continue
        while (memory := available_memory_gb()) is not None and memory < args.min_available_memory_gb:
            log(f'WAIT {name}: MemAvailable {memory:.1f}GB < {args.min_available_memory_gb}GB')
            time.sleep(300)
        log(f'START {name} (MemAvailable {available_memory_gb():.1f}GB)')
        with open(Path(case['directory'])/'process.log', 'a') as handle:
            code = subprocess.call(
                ['nice', '-n', '19', sys.executable, str(ENTRY),
                 '--worker-config', str(Path(case['directory'])/'configs.yaml'), '--method', METHOD],
                cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                env={**__import__('os').environ, 'OMP_NUM_THREADS': '1',
                     'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        log(f'EXIT code={code} {name} status={status(case["directory"])}')
    log('ALL DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
