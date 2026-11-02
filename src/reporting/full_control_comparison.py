"""원본 전체 평가 결과를 완료된 cohort만 대상으로 비교한다."""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pandas as pd


def summarize_full_matrix(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    rows, traces = [], {}
    for case in manifest['cases']:
        directory = root/case['relative_directory']
        state_path = directory/'status.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'status': 'pending'}
        item = {**case, 'status': state['status']}
        if state['status'] == 'completed':
            summary = json.loads((directory/'mpc_evaluation.json').read_text())
            cfg = OmegaConf.load(directory/'configs.yaml')
            trace = pd.read_csv(directory/'mpc_closed_loop_steps.csv')
            validation = summary['model_validation']['affine_controller_model']
            policy = summary['policy_evaluation']
            constraints = policy['constraint_violations']['mpc']
            penalties = {c.name: float(c.penalty) for c in cfg.MPC.state_constraints}
            soft_cost = sum(penalties[n]*c['violation_magnitude_minutes']
                            for n, c in constraints.items())
            # Reconstruct the exact one-hot L1 switch cost, including window bootstrap.
            from data_provider.control_data import load_control_data
            data = load_control_data(cfg)
            starts = json.loads((directory/'scope_audit.json').read_text())['window_starts']
            switch_count = 0
            for start, (_, group) in zip(starts, trace.groupby('window_timestamp', sort=False)):
                previous = data.controls[start-1] if start > 0 else data.encoder.encode_action()
                for _, record in group.iterrows():
                    z = data.encoder.encode_action({
                        name: record[f'action_{name}'] for name in data.encoder.groups
                    })
                    switch_count += float(np.abs(z-previous).sum()/2)
                    previous = z
            switching_cost = 2*float(cfg.MPC.objective.switch_penalty)*switch_count
            item.update({
                'evaluated_windows': summary['evaluated_windows'],
                'policy_steps': len(trace),
                'test_power_wape_percent': validation['one_step']['test']['power']['wape_percent'],
                'rollout_power_wape_percent': validation['recursive_recorded_control_rollout']['power']['wape_percent'],
                'surrogate_energy_wh': summary['optimized_energy_wh'],
                'reference_surrogate_energy_wh': summary['baseline_energy_wh'],
                'surrogate_energy_reduction_percent': summary['energy_reduction_percent'],
                'soft_constraint_cost': soft_cost,
                'switching_cost': switching_cost,
                'total_objective': summary['optimized_energy_wh']+soft_cost+switching_cost,
                'fallback_count': policy['solver']['fallback_count'],
                'deadline_miss_count': policy['solver']['deadline_miss_count'],
                'solve_gate_p95_seconds': float(trace.solve_seconds.quantile(.95)),
                'end_to_end_step_p95_seconds': float(trace.end_to_end_step_seconds.quantile(.95)),
                'model_gate_passed': summary['quality_assessment']['model_validation']['model_validation_passed'],
                'policy_gate_passed': summary['quality_assessment']['simulated_policy_acceptance']['passed'],
                'wall_seconds': state.get('wall_seconds'),
            })
            for name, values in constraints.items():
                item[f'violation_steps_{name}'] = values['violating_steps']
                item[f'violation_magnitude_minutes_{name}'] = values['violation_magnitude_minutes']
                item[f'maximum_violation_{name}'] = values['maximum_violation']
            traces[case['relative_directory']] = trace
        rows.append(item)
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'comparison.csv', index=False, encoding='utf-8-sig')
    aggregate = []
    for dataset, cohort in frame.groupby('group', sort=False):
        if not cohort.status.eq('completed').all():
            continue  # Never rank methods using different room/seed cohorts.
        for method, group in cohort.groupby('method', sort=False):
            trace = pd.concat([traces[r] for r in group.relative_directory], ignore_index=True)
            aggregate.append({
                'dataset_group': dataset, 'method': method, 'completed_cases': len(group),
                'policy_steps': int(group.policy_steps.sum()),
                'mean_test_power_wape_percent': float(group.test_power_wape_percent.mean()),
                'mean_rollout_power_wape_percent': float(group.rollout_power_wape_percent.mean()),
                'total_surrogate_energy_wh': float(group.surrogate_energy_wh.sum()),
                'total_soft_constraint_cost': float(group.soft_constraint_cost.sum()),
                'total_objective': float(group.total_objective.sum()),
                'fallback_count': int(group.fallback_count.sum()),
                'deadline_miss_count': int(group.deadline_miss_count.sum()),
                'pooled_solve_gate_p95_seconds': float(trace.solve_seconds.quantile(.95)),
                'pooled_end_to_end_step_p95_seconds': float(trace.end_to_end_step_seconds.quantile(.95)),
                'model_gate_passed_cases': int(group.model_gate_passed.sum()),
                'policy_gate_passed_cases': int(group.policy_gate_passed.sum()),
            })
    aggregate_frame = pd.DataFrame(aggregate) if aggregate else pd.DataFrame(columns=[
        'dataset_group', 'method', 'completed_cases', 'policy_steps',
        'mean_test_power_wape_percent', 'mean_rollout_power_wape_percent',
        'total_surrogate_energy_wh', 'total_soft_constraint_cost', 'total_objective',
        'fallback_count', 'deadline_miss_count', 'pooled_solve_gate_p95_seconds',
        'pooled_end_to_end_step_p95_seconds', 'model_gate_passed_cases',
        'policy_gate_passed_cases',
    ])
    aggregate_frame.to_csv(root/'aggregate_comparison.csv', index=False, encoding='utf-8-sig')
    lines = [
        '# 원본 전체 평가 비교 진행 상황', '',
        f'- 전체 {len(frame)}개 중 완료 {int(frame.status.eq("completed").sum())}개, '
        f'실패 {int(frame.status.eq("failed").sum())}개.',
        '- 전체 공간/seed/방법이 완료된 데이터 cohort만 집계한다. v3/v4/v5/concat은 따로 비교한다.',
        '- 에너지는 공통 독립 surrogate 결과이며 실제 절감이 아니다.',
        '- 시간은 공유 호스트에서 측정했다. solve+gate와 evaluator 갱신을 포함한 step 시간을 분리한다.',
        '', '| 데이터 | 방법 | test WAPE % | rollout WAPE % | 에너지 Wh | 총목적함수 | fallback | solve+gate p95 초 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for item in aggregate:
        lines.append(
            f'| {item["dataset_group"]} | {item["method"]} | '
            f'{item["mean_test_power_wape_percent"]:.3f} | {item["mean_rollout_power_wape_percent"]:.3f} | '
            f'{item["total_surrogate_energy_wh"]:.3f} | {item["total_objective"]:.3f} | '
            f'{item["fallback_count"]} | {item["pooled_solve_gate_p95_seconds"]:.3f} |'
        )
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    return frame
