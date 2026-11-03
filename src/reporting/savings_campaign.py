"""Campaign 진척/paired 절감/정확도 요약. 미완료 cohort는 전체 집계하지 않는다."""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pandas as pd


def case_tables(root, jobs):
    records, windows, states = [], {}, []
    for job in jobs:
        directory = root/job['relative_directory']
        state_file = directory/'campaign_status.json'
        if not state_file.exists() or json.loads(state_file.read_text())['status'] != 'completed':
            continue
        if job['kind'] != 'policy':
            continue
        cache = directory/'campaign_analysis_cache.json'
        key = (job['cohort'], job['group'], job['environment'], job['arm'])
        if cache.exists():
            saved = json.loads(cache.read_text())
            records.append(saved['record'])
            windows[key] = pd.DataFrame(saved['windows'])
            states.extend(saved['states'])
            continue
        evaluation = json.loads((directory/'mpc_evaluation.json').read_text())
        cfg = OmegaConf.load(directory/'configs.yaml')
        trace = pd.read_csv(directory/'mpc_closed_loop_steps.csv')
        solver = evaluation['policy_evaluation']['solver']
        identity = {k: job[k] for k in ('cohort', 'group', 'environment', 'arm', 'method')}
        frame = pd.DataFrame([{
            'timestamp': w['timestamp'],
            'energy_wh': w['policies']['mpc']['energy_wh'],
            'recorded_wh': w['policies']['recorded']['energy_wh'],
            'weighted_violation': sum(
                float(c.penalty)*w['policies']['mpc']['constraint_violations'][c.name][
                    'violation_magnitude_minutes'] for c in cfg.MPC.state_constraints)}
            for w in evaluation['windows']])
        windows[key] = frame
        energy, recorded = frame.energy_wh.sum(), frame.recorded_wh.sum()
        record = {
            **identity, 'windows': len(frame), 'steps': len(trace),
            'energy_wh': energy, 'recorded_wh': recorded,
            'saving_percent': 100*(1-energy/recorded) if recorded else None,
            'scenario_weighted_violation': frame.weighted_violation.sum(),
            'fallback_steps': solver['fallback_count'],
            'switching_steps': solver['control_switch_count'],
            'deadline_misses': solver['deadline_miss_count'],
            'training_support_ood_steps': solver['training_support_ood_count'],
            'end_to_end_step_p95_seconds': trace.end_to_end_step_seconds.quantile(.95),
            'model_gate': evaluation['quality_assessment']['model_validation']['model_validation_passed'],
            'policy_gate': evaluation['quality_assessment']['simulated_policy_acceptance']['passed'],
            'wall_seconds': json.loads(state_file.read_text())['wall_seconds'],
        }
        bands = json.loads((directory/'state_bands_audit.json').read_text())
        original = OmegaConf.load(job['source_config'])
        penalties = {c.name: float(c.penalty) for c in original.MPC.state_constraints}
        record['standard_weighted_violation'] = sum(
            penalties[name]*v['violation_magnitude_minutes']
            for name, v in bands['standard_bands'].items())
        switching = json.loads((directory/'switching_audit.json').read_text())
        record['switching_cost'] = switching['switching_cost']
        record['raw_categorical_changes'] = switching['raw_categorical_changes']
        record['raw_change_cap_exceedance_steps'] = switching['raw_change_cap_exceedance_steps']
        record['standard_objective'] = (energy+record['standard_weighted_violation']
                                        +record['switching_cost'])
        case_states = []
        for label, entries in bands.items():
            for name, metrics in entries.items():
                case_states.append({**identity, 'band_definition': label, 'state': name, **metrics})
        states.extend(case_states)
        records.append(record)
        cache.write_text(json.dumps({
            'record': record, 'windows': frame.to_dict('records'), 'states': case_states},
            ensure_ascii=False, indent=2, allow_nan=False))
    return pd.DataFrame(records), windows, pd.DataFrame(states)


def paired(a, b):
    if list(a.timestamp) != list(b.timestamp) or not np.allclose(a.recorded_wh, b.recorded_wh):
        raise ValueError('Paired campaign windows/recorded baseline differ')
    rng = np.random.default_rng(42)
    aa, bb = a.energy_wh.to_numpy(), b.energy_wh.to_numpy()
    indices = rng.integers(len(a), size=(2000, len(a)))
    with np.errstate(invalid='ignore', divide='ignore'):
        boot = 100*(bb[indices].sum(1)/aa[indices].sum(1)-1)
    finite = boot[np.isfinite(boot)]
    lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
    return {
        'windows': len(a),
        'energy_change_percent': 100*(bb.sum()/aa.sum()-1) if aa.sum() else None,
        'ci_low': lo, 'ci_high': hi,
        'claim': 'exploratory paired-window bootstrap,2000 draws,seed42,no multiplicity/serial correction',
    }


def write_campaign_report(root, manifest):
    root = Path(root)
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    cases, windows, state_rows = case_tables(root, manifest['jobs'])
    cases.to_csv(output/'policy_case_metrics.csv', index=False)
    state_rows.to_csv(output/'state_violations.csv', index=False)
    comparisons = []
    for (split, group, environment, arm), table in windows.items():
        reference = windows.get((split, group, environment, 'b1_reference'))
        if arm != 'b1_reference' and reference is not None:
            comparisons.append({'cohort': split, 'group': group, 'environment': environment,
                                'arm': arm, **paired(reference, table)})
    pd.DataFrame(comparisons).to_csv(output/'paired_vs_b1.csv', index=False)
    aggregates = []
    if len(cases):
        for (cohort, arm), subset in cases.groupby(['cohort', 'arm']):
            for label, groups, rooms in [
                ('single_versions', ['v3', 'v4', 'v5'], ['livingroom', 'room1', 'room2']),
                ('concat', ['concat'], ['livingroom', 'room1', 'room2']),
                ('livingroom_single_versions', ['v3', 'v4', 'v5'], ['livingroom']),
            ]:
                chosen = subset[subset.group.isin(groups) & subset.environment.isin(rooms)]
                if len(chosen) != len(groups)*len(rooms):
                    continue
                energy, baseline = chosen.energy_wh.sum(), chosen.recorded_wh.sum()
                aggregates.append({
                    'cohort': cohort, 'arm': arm, 'scope': label,
                    'cells': len(chosen), 'steps': chosen.steps.sum(),
                    'energy_wh': energy, 'recorded_wh': baseline,
                    'saving_percent': 100*(1-energy/baseline) if baseline else None,
                    'scenario_weighted_violation': chosen.scenario_weighted_violation.sum(),
                    'standard_weighted_violation': chosen.standard_weighted_violation.sum(),
                    'switching_cost': chosen.switching_cost.sum(),
                    'standard_objective': chosen.standard_objective.sum(),
                    'model_gate_passed_cases': int(chosen.model_gate.sum()),
                    'policy_gate_passed_cases': int(chosen.policy_gate.sum()),
                })
    pd.DataFrame(aggregates).to_csv(output/'policy_aggregate_metrics.csv', index=False)
    mape_file = output/'mape_metrics.csv'
    if mape_file.exists():
        metric_frame = pd.read_csv(mape_file)
        pooled = []
        keys = ['kind', 'cohort', 'arm', 'protocol', 'split', 'target', 'floor_w']
        for identity, subset in metric_frame.groupby(keys, dropna=False):
            for label, groups in [('single_versions', ['v3', 'v4', 'v5']),
                                  ('concat', ['concat'])]:
                chosen = subset[subset.group.isin(groups)]
                cells = set(zip(chosen.group, chosen.environment))
                expected = {(g, e) for g in groups for e in ('livingroom', 'room1', 'room2')}
                if cells != expected or len(chosen) != len(expected):
                    continue
                valid = int(chosen.mape_valid_samples.sum())
                total = int(chosen.sample_count.sum())
                denominator = chosen.absolute_actual_sum_w.sum()
                pooled.append({
                    **dict(zip(keys, identity)), 'scope': label, 'cells': len(chosen),
                    'sample_count': total, 'mape_valid_samples': valid,
                    'denominator_coverage_percent': 100*valid/total if total else 0.,
                    'pooled_mape_percent': (
                        100*chosen.absolute_percentage_error_sum.sum()/valid if valid else None),
                    'pooled_wape_percent': (
                        100*chosen.absolute_error_sum_w.sum()/denominator if denominator else None),
                    'cells_passing_mape_target': int((chosen.mape_target_pass == True).sum()),  # noqa: E712
                    'nonfinite_count': int(chosen.nonfinite_count.sum()),
                    'official_gate_pass_is_not_inferred_from_pooled_metric': True,
                })
        pd.DataFrame(pooled).to_csv(output/'mape_pooled_metrics.csv', index=False)
    lines = [
        '# 절감·MAPE campaign 진행 요약', '',
        f'- 상태: {manifest["status"]}; counts: {manifest.get("counts")}',
        f'- 선행 확대실험 대기: {manifest.get("waiting_for_predecessor", False)}',
        '- 실행 자원(CPU 집합·worker 수·nice)은 manifest.json에 기록된다.',
        '- MAPE108작업 + 공통report입력12작업 + 20조건×12셀×validation/test=480정책작업.',
        '- 정책작업에도 해당 체크포인트의 factual/recursive MAPE를 산출한다.',
        '- 완료된 원본 action trace가 있으면 S+A+D / S+A / A report-input 비교도 수행한다.',
        '- 공통report입력12작업에는 validation-ranked Top-K candidate-pool coverage 실험이 포함된다.',
        '- 단일버전·concat을 합산하지 않는다. livingroom은 사전 명시한 별도 범위다.',
        '- 서비스/전환/dwell 변경은 원본 조건과 다른 가정 기반 시뮬레이션이다.',
        '- validation은 early stopping/evaluator fitting에 쓰일 수 있어 독립 검증이 아니다.',
        '- test는 이미 관찰한 기간이므로 새 held-out 최종 실증으로 주장하지 않는다.',
        '- MAPE 목표 통과는 종합 모델/정책 gate 통과와 별개다.',
        '- MPC 변경 action에 대한 실제 전력 정답은 없으며 이를 factual MAPE로 채점하지 않는다.',
        '', '## 데이터·장비 정보가 필요한 미실행 항목', '',
    ]
    lines += [f'- {name}: {reason}' for name, reason in manifest['blocked_experiments'].items()]
    lines += ['', '## 완결 범위의 에너지 집계', '',
              '| 평가구간 | 범위 | 조건 | 절감률 % | 표준범위 가중위반 | 정책gate |',
              '|---|---|---|---:|---:|---:|']
    for item in aggregates:
        lines.append(f'| {item["cohort"]} | {item["scope"]} | {item["arm"]} | '
                     f'{item["saving_percent"]:.2f} | {item["standard_weighted_violation"]:.1f} | '
                     f'{item["policy_gate_passed_cases"]}/{item["cells"]} |')
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
