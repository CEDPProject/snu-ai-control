"""원본 전체 평가 matrix의 supervisor/worker 진입점."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omegaconf import OmegaConf
from runners.full_control_benchmark_runner import full_worker, run_full_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[2]/'configs/default_setting.yaml'))
    parser.add_argument('--resume')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker-config')
    parser.add_argument('--method')
    parser.add_argument('--method-order',
                        help='콤마 구분 method 실행 우선순위(큐 순서만 바꿈), 예: m1_affine_milp,m3_physics_residual')
    parser.add_argument('--min-available-memory-gb', type=float, default=0.0,
                        help='가용 메모리가 이 값 미만이면 새 worker 시작을 미룸(0=끔)')
    args = parser.parse_args()
    if args.worker_config:
        if not args.method:
            parser.error('--worker-config requires --method')
        return full_worker(args.worker_config, args.method)
    cfg = OmegaConf.load(
        Path(args.resume)/'matrix_config.yaml' if args.resume else args.config,
    )
    if not args.resume:
        from data_provider.dataset_versions import resolve_dataset_csv_path
        group = cfg.CONTROL_FULL_BENCHMARK.dataset_groups[0]
        environment = cfg.CONTROL_FULL_BENCHMARK.environments[0]
        first_csv = resolve_dataset_csv_path(cfg.DATASET.datadir, group.versions[0], environment)
        cfg.DATASET.datadir = str(first_csv.parent.parent)
    method_order = [m.strip() for m in args.method_order.split(',') if m.strip()] if args.method_order else None
    return run_full_matrix(cfg, existing_root=args.resume, workers=args.workers,
                           prepare_only=args.prepare_only, method_order=method_order,
                           min_available_memory_gb=args.min_available_memory_gb)


if __name__ == '__main__':
    main()
