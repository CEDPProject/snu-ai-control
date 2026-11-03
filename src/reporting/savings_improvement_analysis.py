"""절감 개선 실험의 동일 셀·창 paired 비교. 완료 결과만 분석하며 원본과 합치지 않는다."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

CONTRASTS = [
    ('m3_original', 'm3_b1'),
    ('m3_b1', 'm3_b1_cem128x5'),
    ('m3_original', 'm3_b1_cem128x5'),
    ('m6_original', 'm6_b1'),
    ('m3_b1', 'm6_b1'),
    ('m3_b1_cem128x5', 'm6_b1'),
]


def paired(a, b, seed=42):
    if list(a.ts) != list(b.ts) or not np.allclose(a.base, b.base):
        raise ValueError('Paired window timestamps/baselines differ')
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(a), size=(4000, len(a)))
    aa, bb = a.energy.to_numpy(), b.energy.to_numpy()
    boot = 100*(bb[indices].sum(1)/aa[indices].sum(1)-1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        'windows': len(a), 'energy_a_wh': aa.sum(), 'energy_b_wh': bb.sum(),
        'energy_b_vs_a_percent': 100*(bb.sum()/aa.sum()-1),
        'ci_low': lo, 'ci_high': hi,
        'weighted_violation_a': a.soft.sum(), 'weighted_violation_b': b.soft.sum(),
        'weighted_violation_b_minus_a': b.soft.sum()-a.soft.sum(),
        'note': 'negative energy difference favors b; exploratory 95% paired bootstrap, no multiplicity correction',
    }


def analyze(root):
    manifest = json.loads((root/'manifest.json').read_text())
    records, windows = [], {}
    for case in manifest['cases']:
        d = root/case['relative_directory']
        status = json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else {}
        if status.get('status') != 'completed':
            continue
        ev = json.loads((d/'mpc_evaluation.json').read_text())
        cfg = OmegaConf.load(d/'configs.yaml')
        penalties = {c.name: float(c.penalty) for c in cfg.MPC.state_constraints}
        table = []
        for w in ev['windows']:
            mpc, recorded = w['policies']['mpc'], w['policies']['recorded']
            table.append({
                'ts': w['timestamp'], 'energy': mpc['energy_wh'], 'base': recorded['energy_wh'],
                'soft': sum(penalties[n]*v['violation_magnitude_minutes']
                            for n, v in mpc['constraint_violations'].items()),
            })
        frame = pd.DataFrame(table)
        key = (case['group'], case['environment'], case['arm'])
        windows[key] = frame
        solver = ev['policy_evaluation']['solver']
        quality = ev['quality_assessment']
        trace = pd.read_csv(d/'mpc_closed_loop_steps.csv')
        rec = {k: case[k] for k in ('arm', 'group', 'environment', 'method', 'seed')}
        rec.update({
            'windows': len(frame), 'steps': len(trace),
            'recorded_wh': frame.base.sum(), 'energy_wh': frame.energy.sum(),
            'reduction_percent': 100*(1-frame.energy.sum()/frame.base.sum()),
            'weighted_violation': frame.soft.sum(),
            'fallback_percent': 100*solver['fallback_count']/len(trace),
            'switches': solver['control_switch_count'],
            'deadline_miss': solver['deadline_miss_count'],
            'solve_gate_p95_seconds': trace.solve_seconds.quantile(.95),
            'end_to_end_p95_seconds': trace.end_to_end_step_seconds.quantile(.95),
            'wall_hours': status['wall_seconds']/3600,
            'model_gate': quality['model_validation']['model_validation_passed'],
            'policy_gate': quality['simulated_policy_acceptance']['passed'],
        })
        for name, value in ev['policy_evaluation']['constraint_violations']['mpc'].items():
            rec[f'violation::{name}'] = value['violation_magnitude_minutes']
        records.append(rec)
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(output/'case_metrics.csv', index=False, encoding='utf-8-sig')
    comparisons = []
    cells = sorted({(c['group'], c['environment']) for c in manifest['cases']})
    for a, b in CONTRASTS:
        for group, env in cells:
            if (group, env, a) in windows and (group, env, b) in windows:
                comparisons.append({
                    'group': group, 'environment': env, 'arm_a': a, 'arm_b': b,
                    **paired(windows[(group, env, a)], windows[(group, env, b)]),
                })
    pd.DataFrame(comparisons).to_csv(output/'paired_comparisons.csv', index=False, encoding='utf-8-sig')
    lines = [
        '# 절감 개선 실험 진행 결과', '',
        f'- 완료 {len(records)}/{len(manifest["cases"])}조합.',
        f'- 단일 seed·{len(cells)}셀·surrogate 탐색 결과. 원본 72조합과 별개.',
        '- concat은 개별 버전과 평가 데이터가 겹칠 수 있어 합산하지 않는다.',
        '- 기존 보고서 수치가 아니라 같은 서버에서 다시 실행한 원본 조건이 대조군이다.',
        '- 에너지 절감만으로 채택하지 않는다. 상태별 위반·전환·시간·gate를 함께 본다.',
        '- 쌍별 CI는 다중비교 미보정 탐색적 구간이며 다중 seed/독립 검증이 필요하다.', '',
        '| 셀 | 조건 | 절감률 % | 가중 위반 | fallback % | switching | wall h |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for r in records:
        lines.append(f'| {r["group"]}/{r["environment"]} | {r["arm"]} | '
                     f'{r["reduction_percent"]:.2f} | {r["weighted_violation"]:.1f} | '
                     f'{r["fallback_percent"]:.1f} | {r["switches"]} | {r["wall_hours"]:.2f} |')
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(f'Analyzed {len(records)} completed cases, {len(comparisons)} matched contrasts', flush=True)


if __name__ == '__main__':
    analyze(Path(sys.argv[1]).resolve())
