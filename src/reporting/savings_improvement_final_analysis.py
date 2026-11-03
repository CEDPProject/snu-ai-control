"""완료 cohort의 집계·층화 paired bootstrap·전환 비용을 계산한다.

사용:
  PYTHONPATH=<run>/source_snapshot/src python src/reporting/savings_improvement_final_analysis.py <run>
원본 결과를 읽기만 하고 새 분석 파일을 <run>/analysis/final/에 쓴다.
--cohort single_versions는 전체 실행이 진행 중이어도 완결된 단일버전 cohort만
검증해 analysis/interim_single_versions에 쓴다. 불완전 cohort는 거부한다.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf
import pandas as pd

CONTRASTS = [
    ('m3_original', 'm3_b1'), ('m3_b1', 'm3_b1_cem128x5'),
    ('m3_original', 'm3_b1_cem128x5'), ('m6_original', 'm6_b1'),
    ('m3_b1', 'm6_b1'), ('m3_b1_cem128x5', 'm6_b1'),
]


def stratified_paired(tables, samples=4000, seed=42):
    """tables: aligned [energy_a, energy_b, soft_a, soft_b, obj_a, obj_b]."""
    rng = np.random.default_rng(seed)
    observed = sum((t.sum(axis=0) for t in tables), np.zeros(6))
    bootstrap = np.zeros((samples, 6))
    for table in tables:
        if table.ndim != 2 or table.shape[1] != 6 or not len(table):
            raise ValueError('Each cell must have nonempty aligned N×6 values')
        idx = rng.integers(0, len(table), size=(samples, len(table)))
        bootstrap += table[idx].sum(axis=1)
    result = {'windows': sum(len(t) for t in tables)}
    for name, a, b, relative in [
        ('energy_b_vs_a_percent', 0, 1, True),
        ('soft_b_minus_a', 2, 3, False),
        ('objective_b_minus_a', 4, 5, False),
    ]:
        point = observed[b]-observed[a]
        draws = bootstrap[:, b]-bootstrap[:, a]
        if relative:
            if observed[a] <= 0 or (bootstrap[:, a] <= 0).any():
                raise ValueError('Energy denominators must be positive')
            point = 100*point/observed[a]
            draws = 100*draws/bootstrap[:, a]
        result[name] = float(point)
        result[name+'_ci_low'], result[name+'_ci_high'] = map(float, np.percentile(draws, [2.5, 97.5]))
    return result


def evaluation_cohorts(manifest):
    cells = sorted({(c['group'], c['environment']) for c in manifest['cases']})
    if manifest.get('experiment_kind') == 'm3_full_scope_expansion':
        expected = {(g, r, a) for g in ('v3', 'v4', 'v5', 'concat')
                    for r in ('livingroom', 'room1', 'room2')
                    for a in ('m3_original', 'm3_b1', 'm3_b1_cem128x5')}
        actual = {(c['group'], c['environment'], c['arm']) for c in manifest['cases']}
        if actual != expected or len(manifest['cases']) != 36:
            raise ValueError('Expansion requires all36 predeclared cases')
        return {'single_versions': [c for c in cells if c[0] != 'concat'],
                'concat': [c for c in cells if c[0] == 'concat']}
    if len(manifest['cases']) != 15:
        raise ValueError('Discovery requires the complete15-case cohort')
    return {'discovery': cells}


def analysis_scope(manifest, cohort=None):
    cohorts = evaluation_cohorts(manifest)
    if cohort is None:
        if manifest['status'] != 'completed':
            raise ValueError('Final analysis requires a completed cohort')
        return manifest['cases'], cohorts
    if cohort not in cohorts:
        raise ValueError(f'Unknown predeclared cohort: {cohort}')
    selected_cells = set(cohorts[cohort])
    cases = [c for c in manifest['cases'] if (c['group'], c['environment']) in selected_cells]
    return cases, {cohort: cohorts[cohort]}


def main(root, cohort=None):
    from data_provider.control_data import load_control_data

    manifest = json.loads((root/'manifest.json').read_text())
    cases, cohorts = analysis_scope(manifest, cohort)
    if cohort == 'single_versions':
        # 사전 정의된 전체 단일버전 cohort를 먼저 확보한 뒤 버전·재사용 여부를
        # 보조 분석한다. 성능을 보고 셀을 골라 합산하지 않는다.
        for version in ('v3', 'v4', 'v5'):
            cohorts[version] = sorted({(c['group'], c['environment']) for c in cases
                                       if c['group'] == version})
        for reused, label in ((True, 'reused_discovery_cells'), (False, 'new_single_version_cells')):
            cells = sorted({(c['group'], c['environment']) for c in cases
                            if bool(c.get('reused_from')) == reused})
            if cells:
                cohorts[label] = cells
    hashes = {p: hashlib.sha256((root/'source_snapshot/src'/p).read_bytes()).hexdigest()
              for p in manifest['source_sha256']}
    if hashes != manifest['source_sha256']:
        raise ValueError('Frozen source mismatch')
    for source in manifest['input_sources']:
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise ValueError(f"Input CSV mismatch: {source['path']}")
    data_by_cell, tables, records, states, traces = {}, {}, [], [], {}
    gate_counts = {'model': Counter(), 'policy': Counter()}
    for case in cases:
        d = root/case['relative_directory']
        status = json.loads((d/'status.json').read_text())
        if status['status'] != 'completed':
            raise ValueError(f'Incomplete case: {d}')
        if hashlib.sha256((d/'configs.yaml').read_bytes()).hexdigest() != case['config_sha256']:
            raise ValueError(f'Frozen config mismatch: {d}')
        cfg = OmegaConf.load(d/'configs.yaml')
        cell = (case['group'], case['environment'])
        if cell not in data_by_cell:
            data_by_cell[cell] = load_control_data(cfg)
        data = data_by_cell[cell]
        ev = json.loads((d/'mpc_evaluation.json').read_text())
        trace = pd.read_csv(d/'mpc_closed_loop_steps.csv')
        audit = json.loads((d/'scope_audit.json').read_text())
        if len(trace) != audit['expected_policy_steps'] or len(ev['windows']) != audit['evaluated_windows']:
            raise ValueError(f'Final evaluation scope mismatch: {d}')
        penalties = {c.name: float(c.penalty) for c in cfg.MPC.state_constraints}
        groups = list(trace.groupby('window_timestamp', sort=False))
        if len(groups) != len(ev['windows']):
            raise ValueError('Trace windows mismatch')
        window_records = []
        categorical_changes = 0.
        switched_steps = 0
        for start, window, (timestamp, group) in zip(audit['window_starts'], ev['windows'], groups):
            if pd.Timestamp(timestamp) != pd.Timestamp(window['timestamp']):
                raise ValueError('Window order mismatch')
            z = np.vstack([data.encoder.encode_action({
                name: row[f'action_{name}'] for name in data.encoder.groups
            }) for row in group.to_dict('records')])
            previous = data.controls[start-1] if start > 0 else data.encoder.encode_action()
            l1 = np.abs(z-np.vstack([previous, z[:-1]])).sum(axis=1)
            categorical_changes += l1.sum()/2
            switched_steps += int((l1 > 0).sum())
            switch_cost = float(cfg.MPC.objective.switch_penalty)*l1.sum()
            mpc, rec = window['policies']['mpc'], window['policies']['recorded']
            soft = sum(penalties[n]*v['violation_magnitude_minutes']
                       for n, v in mpc['constraint_violations'].items())
            window_records.append({
                'ts': window['timestamp'], 'base': rec['energy_wh'],
                'energy': mpc['energy_wh'], 'soft': soft, 'switch_cost': switch_cost,
                'objective': mpc['energy_wh']+soft+switch_cost,
            })
        table = pd.DataFrame(window_records)
        solver = ev['policy_evaluation']['solver']
        if switched_steps != solver['control_switch_count']:
            raise ValueError('Switching-step reconstruction mismatch')
        quality = ev['quality_assessment']
        for label, section in [('model', 'model_validation'), ('policy', 'simulated_policy_acceptance')]:
            for name, gate in quality[section]['gates'].items():
                if not gate['passed']:
                    gate_counts[label][name] += 1
        key = (*cell, case['arm'])
        tables[key], traces[key] = table, trace
        rec = {k: case[k] for k in ('group', 'environment', 'arm', 'method', 'seed')}
        rec.update({
            'reused_from': case.get('reused_from', ''),
            'windows': len(table), 'steps': len(trace), 'recorded_wh': table.base.sum(),
            'energy_wh': table.energy.sum(), 'reduction_percent': 100*(1-table.energy.sum()/table.base.sum()),
            'weighted_violation': table.soft.sum(), 'switching_cost': table.switch_cost.sum(),
            'objective': table.objective.sum(), 'switching_steps': switched_steps,
            'categorical_variable_changes': categorical_changes,
            'fallback_count': solver['fallback_count'],
            'fallback_percent': 100*solver['fallback_count']/len(trace),
            'deadline_miss': solver['deadline_miss_count'],
            'solve_gate_p95_seconds': trace.solve_seconds.quantile(.95),
            'end_to_end_p95_seconds': trace.end_to_end_step_seconds.quantile(.95),
            'wall_hours': status['wall_seconds']/3600,
            'model_gate': quality['model_validation']['model_validation_passed'],
            'policy_gate': quality['simulated_policy_acceptance']['passed'],
        })
        records.append(rec)
        for name, value in ev['policy_evaluation']['constraint_violations']['mpc'].items():
            recorded = ev['policy_evaluation']['constraint_violations']['recorded'][name]
            states.append({**{k: case[k] for k in ('group', 'environment', 'arm')}, 'state': name,
                           'magnitude_minutes': value['violation_magnitude_minutes'],
                           'maximum_violation': value['maximum_violation'],
                           'violating_steps': value['violating_steps'],
                           'recorded_magnitude_minutes': recorded['violation_magnitude_minutes'],
                           'recorded_maximum_violation': recorded['maximum_violation']})
    frame = pd.DataFrame(records)
    aggregate = []
    grouped = [(label, arm, frame[
        frame.apply(lambda r: (r['group'], r['environment']) in cells, axis=1)
        & (frame.arm == arm)])
        for label, cells in cohorts.items() for arm in frame.arm.unique()]
    for label, arm, subset in grouped:
        if len(subset) != len(cohorts[label]):
            raise ValueError('Aggregate requires every declared cell in its cohort')
        pooled_trace = pd.concat([traces[(r.group, r.environment, arm)]
                                  for r in subset.itertuples()], ignore_index=True)
        item = {'cohort': label, 'arm': arm, 'cells': len(subset)}
        for name in ('windows', 'steps', 'recorded_wh', 'energy_wh', 'weighted_violation',
                     'switching_cost', 'objective', 'switching_steps', 'categorical_variable_changes',
                     'fallback_count', 'deadline_miss', 'wall_hours'):
            item[name] = float(subset[name].sum())
        item.update(reduction_percent=100*(1-item['energy_wh']/item['recorded_wh']),
                    fallback_percent=100*item['fallback_count']/item['steps'],
                    model_gate_passed=int(subset.model_gate.sum()), policy_gate_passed=int(subset.policy_gate.sum()),
                    solve_gate_p95_seconds=pooled_trace.solve_seconds.quantile(.95),
                    end_to_end_p95_seconds=pooled_trace.end_to_end_step_seconds.quantile(.95))
        aggregate.append(item)
    contrasts = []
    arms = set(frame.arm)
    for label, cells in cohorts.items():
        for a, b in CONTRASTS:
            if a not in arms or b not in arms:
                continue
            arrays = []
            for group, env in cells:
                aa, bb = tables[(group, env, a)], tables[(group, env, b)]
                if list(aa.ts) != list(bb.ts) or not np.allclose(aa.base, bb.base):
                    raise ValueError('Paired timestamps or baseline mismatch')
                arrays.append(np.column_stack([aa.energy, bb.energy, aa.soft, bb.soft,
                                               aa.objective, bb.objective]))
            contrasts.append({'cohort': label, 'arm_a': a, 'arm_b': b,
                              'cells': len(cells), **stratified_paired(arrays)})
    out = root/'analysis'/('final' if cohort is None else f'interim_{cohort}')
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('case_metrics', frame), ('aggregate_metrics', pd.DataFrame(aggregate)),
                        ('state_violations', pd.DataFrame(states)),
                        ('stratified_paired_comparisons', pd.DataFrame(contrasts))]:
        value.to_csv(out/f'{name}.csv', index=False, encoding='utf-8-sig')
    (out/'gate_failure_counts.json').write_text(json.dumps(gate_counts, ensure_ascii=False, indent=2)+'\n')
    metadata = {
        'generated_at': datetime.now().astimezone().isoformat(),
        'run_status_at_analysis_start': manifest['status'],
        'selected_predeclared_cohort': cohort,
        'all_selected_cases_completed': True,
        'case_count': len(records), 'unique_cells': len(data_by_cell),
        'cohorts': {k: [list(c) for c in v] for k, v in cohorts.items()},
        'windows_per_cohort_per_arm': {
            label: int(sum(len(tables[(*cell, next(iter(arms)))]) for cell in cells))
            for label, cells in cohorts.items()},
        'policy_steps_all_cases': int(frame.steps.sum()), 'bootstrap_samples': 4000,
        'bootstrap_seed': 42, 'resampling': 'paired windows independently within each fixed cell; concat separate',
        'multiplicity_adjustment': False,
        'objective': 'energy_wh + sum(penalty*violation_magnitude_minutes) + switch_penalty*onehot_L1',
        'switching_steps': 'steps with any action change; not the number of individual variable changes',
        'limits': 'Only declared output cohorts included; all other cells excluded from this report; '
                  'single seed; no independent-state/deployment validation',
        'source_hashes_match': True, 'case_config_hashes_match': True, 'input_csv_hashes_match': True,
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out/'analysis_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(pd.DataFrame(aggregate).round(3).to_string(index=False))
    print(pd.DataFrame(contrasts).round(3).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--cohort', help='분석할 사전 정의 cohort; 생략하면 전체 완료 실행만 허용')
    args = parser.parse_args()
    main(args.root.resolve(), cohort=args.cohort)
