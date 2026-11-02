"""Full-guard MPC matrix over applied-action history and change constraints.

``L=0`` is intentionally not a controller intervention: it anchors the first
action to the current recorded action and supplies the factual report-model
baseline.  For ``L>0`` the controller can select every train-supported joint
action, but its physically applied first action is regularized against the
chronological actions from ``t-L`` through ``t-1``.  Closed-loop evaluation
uses recorded actions only to bootstrap a window and appends applied MPC
actions thereafter.

For every positive ``L``, the runner takes the product of categorical hard
distance ``K``, history group mode (equipment or one group per control
variable), and history-penalty coefficient. All scenarios use the same
Full-guard candidate screen and all chronological test transitions. Each saved
MPC trace is subsequently evaluated by each of the three frozen report-model
input bundles, without solving MILP again.
"""

from __future__ import annotations

from datetime import datetime
from itertools import product
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
from typing import Any

from omegaconf import OmegaConf
import pandas as pd

from controllers.pipeline import run_mpc_pipeline
from data_provider.dataset_versions import dataset_label_from_cfg, versions_from_cfg
from runners.counterfactual_report_input_ablation_runner import (
    run_counterfactual_report_input_ablation,
)
from utils.log import setup_default_logging


LOGGER = logging.getLogger('MPC.action_history_ablation')


def _plain(value: Any) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')


def _safe_name(value: str) -> str:
    result = re.sub(r'[^0-9A-Za-z._-]+', '_', str(value)).strip('._')
    if not result:
        raise ValueError('Experiment name must contain a filesystem-safe token')
    return result


def _result_directory(cfg) -> Path:
    timestamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    run_name = _safe_name(f'{cfg.DEFAULT.exp_name}_{timestamp}_{os.getpid()}')
    return (
        Path(cfg.RESULT.savedir)
        / dataset_label_from_cfg(cfg)
        / cfg.DATASET.environment_name
        / 'action_history_ablation'
        / run_name
    ).resolve()


def _configure_logging(cfg, result_dir: Path) -> Path:
    log_path = result_dir / 'logs' / 'action_history_ablation.log'
    level = logging.DEBUG if bool(cfg.get('DEBUG', False)) else logging.INFO
    setup_default_logging(default_level=level, log_path=str(log_path))
    logging.captureWarnings(True)
    return log_path


def _specification(cfg) -> dict[str, Any]:
    value = _plain(cfg.MPC.action_history_ablation)
    if not isinstance(value, dict):
        raise TypeError('MPC.action_history_ablation must be a mapping')
    return value


def _lookbacks(cfg) -> list[int]:
    raw = _specification(cfg).get('lookback_steps')
    if not isinstance(raw, list) or not raw:
        raise TypeError(
            'MPC.action_history_ablation.lookback_steps must be a nonempty list'
        )
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Every action-history lookback must be a nonnegative integer')
        values.append(int(value))
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError('Action-history lookbacks must be unique and ascending')
    return values


def _hard_distances(cfg) -> list[int]:
    """Return categorical hard-distance caps for L>0 scenarios."""
    raw = _specification(cfg).get('hard_distances', [len(cfg.MPC.control_variables)])
    if not isinstance(raw, list) or not raw:
        raise TypeError(
            'MPC.action_history_ablation.hard_distances must be a nonempty list'
        )
    maximum = len(cfg.MPC.control_variables)
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError('Every action-history hard distance must be an integer')
        if not 0 <= value <= maximum:
            raise ValueError(
                f'Action-history hard distance must be in [0, {maximum}], got {value}'
            )
        values.append(int(value))
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError('Action-history hard distances must be unique and ascending')
    return values


def _history_group_modes(cfg) -> list[str]:
    """Return explicit history-distance units used by the experiment matrix."""
    raw = _specification(cfg).get('history_group_modes', ['equipment'])
    if not isinstance(raw, list) or not raw:
        raise TypeError(
            'MPC.action_history_ablation.history_group_modes must be a nonempty list'
        )
    allowed = {'equipment', 'per_control_variable'}
    modes = [str(value) for value in raw]
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(
            'Unknown action-history group mode(s): '
            f'{unknown}; allowed={sorted(allowed)}'
        )
    if len(modes) != len(set(modes)):
        raise ValueError('Action-history group modes must be unique')
    return modes


def _history_penalties(cfg) -> list[float]:
    """Return nonnegative soft-regularization coefficients to compare."""
    raw = _specification(cfg).get(
        'history_penalties', [float(cfg.MPC.objective.action_history.penalty)]
    )
    if not isinstance(raw, list) or not raw:
        raise TypeError(
            'MPC.action_history_ablation.history_penalties must be a nonempty list'
        )
    penalties: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError('Every action-history penalty must be numeric')
        if float(value) < 0.0:
            raise ValueError('Every action-history penalty must be nonnegative')
        penalties.append(float(value))
    if penalties != sorted(penalties) or len(penalties) != len(set(penalties)):
        raise ValueError('Action-history penalties must be unique and ascending')
    return penalties


def _equipment_group_definitions(cfg):
    groups = cfg.MPC.objective.action_history.get('groups')
    if groups is None:
        raise ValueError(
            'history_group_modes includes equipment but '
            'MPC.objective.action_history.groups is null'
        )
    return OmegaConf.to_container(groups, resolve=True)


def _history_groups_for_mode(cfg, mode: str):
    if mode == 'per_control_variable':
        # MixedIntegerMPC interprets None as one history group per categorical
        # control variable, using the encoder's canonical feature groups.
        return None
    if mode == 'equipment':
        return _equipment_group_definitions(cfg)
    raise AssertionError(f'Unhandled history group mode: {mode!r}')


def _control_variable_names(cfg) -> list[str]:
    raw_variables = cfg.MPC.control_variables
    names: list[str] = []
    for raw in raw_variables:
        if OmegaConf.is_config(raw):
            raw = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(raw, dict) or not str(raw.get('name', '')).strip():
            raise TypeError('Every MPC.control_variables entry must define a name')
        names.append(str(raw['name']))
    return names


def _history_group_names(cfg, mode: str) -> list[str]:
    if mode == 'per_control_variable':
        return _control_variable_names(cfg)
    return [str(group['name']) for group in _equipment_group_definitions(cfg)]


def _normalized_recency_weights(cfg, lookback: int) -> list[float]:
    """Return lag-1..lag-L weights with the same total mass for all L>0.

    Holding the total mass fixed makes L a history-length sensitivity test,
    rather than silently imposing a much larger objective penalty at L=30
    than at L=1.  The output order is newest to oldest and matches the
    controller's ``action_history_recency_weights`` contract.
    """
    if lookback <= 0:
        return []
    spec = _specification(cfg)
    decay = float(spec.get('recency_weight_decay', 0.8))
    total = float(spec.get('recency_weight_total', 3.0))
    if not 0.0 < decay <= 1.0:
        raise ValueError('recency_weight_decay must be in (0, 1]')
    if total <= 0.0:
        raise ValueError('recency_weight_total must be positive')
    raw = [decay ** lag for lag in range(lookback)]
    scale = total / sum(raw)
    return [float(value * scale) for value in raw]


def _scaled_rolling_change_budget(cfg, lookback: int) -> int | None:
    """Scale a group-change rate, rather than reusing a five-minute count.

    The default reference is three equipment-group changes per five minutes.
    Therefore L=1,2,5,10,30 receive budgets 1,2,3,6,18 respectively.  A
    ceiling deliberately avoids turning a positive short horizon into a zero
    change budget merely because it was rounded down.
    """
    if lookback <= 0:
        return None
    spec = _specification(cfg)
    reference_steps = spec.get('rolling_change_budget_reference_steps', 5)
    reference_budget = spec.get('rolling_change_budget_reference_value', 3)
    if (
        isinstance(reference_steps, bool)
        or not isinstance(reference_steps, int)
        or reference_steps <= 0
    ):
        raise ValueError('rolling_change_budget_reference_steps must be positive')
    if reference_budget is None:
        # Per-group budgets can be intentionally used without a second
        # aggregate cap. This is useful when the experiment asks whether each
        # individual categorical variable should have its own allowance.
        return None
    if (
        isinstance(reference_budget, bool)
        or not isinstance(reference_budget, int)
        or reference_budget < 0
    ):
        raise ValueError('rolling_change_budget_reference_value must be nonnegative')
    return int(math.ceil(lookback * reference_budget / reference_steps))


def _scaled_per_group_rolling_change_budgets(
    cfg,
    lookback: int,
    mode: str,
) -> dict[str, int] | None:
    """Scale optional independent group budgets using the same L-rate rule."""
    if lookback <= 0:
        return None
    specification = _specification(cfg)
    raw = specification.get('per_group_rolling_change_budget_reference_values')
    if raw is None:
        return None
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(
            'per_group_rolling_change_budget_reference_values must be a mapping '
            'from group mode to a budget mapping or null'
        )
    reference_values = raw.get(mode)
    if reference_values is None:
        return None
    if not isinstance(reference_values, dict) or not reference_values:
        raise TypeError(
            f'Per-group reference budgets for {mode!r} must be a nonempty mapping'
        )
    names = set(_history_group_names(cfg, mode))
    unknown = sorted(set(reference_values) - names)
    if unknown:
        raise ValueError(
            f'Per-group reference budgets for {mode!r} contain unknown names: '
            f'{unknown}; expected subset of {sorted(names)}'
        )
    reference_steps = specification.get('rolling_change_budget_reference_steps', 5)
    if (
        isinstance(reference_steps, bool)
        or not isinstance(reference_steps, int)
        or reference_steps <= 0
    ):
        raise ValueError('rolling_change_budget_reference_steps must be positive')
    budgets: dict[str, int] = {}
    for name, value in reference_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f'Per-group reference budget for {name!r} must be a nonnegative integer'
            )
        budgets[str(name)] = int(math.ceil(lookback * value / reference_steps))
    return budgets


def _penalty_token(value: float) -> str:
    return format(float(value), '.12g').replace('-', 'm').replace('.', 'p')


def _scenario_descriptors(cfg) -> list[dict[str, Any]]:
    """Build a nonduplicated factual baseline plus the configured L×K matrix."""
    lookbacks = _lookbacks(cfg)
    descriptors: list[dict[str, Any]] = []
    if 0 in lookbacks:
        descriptors.append({
            'scenario_id': 'l00_factual',
            'lookback_l': 0,
            'kind': 'factual_baseline',
            'hard_distance_k': 0,
            'history_group_mode': 'factual',
            'history_group_names': [],
            'history_penalty': 0.0,
            'rolling_change_budget': None,
            'per_group_rolling_change_budgets': None,
        })
    for lookback, hard_distance, mode, penalty in product(
        (value for value in lookbacks if value > 0),
        _hard_distances(cfg),
        _history_group_modes(cfg),
        _history_penalties(cfg),
    ):
        descriptors.append({
            'scenario_id': (
                f'l{lookback:02d}_k{hard_distance}_{mode}_p{_penalty_token(penalty)}'
            ),
            'lookback_l': int(lookback),
            'kind': 'history_smoothed_mpc',
            'hard_distance_k': int(hard_distance),
            'history_group_mode': mode,
            'history_group_names': _history_group_names(cfg, mode),
            'history_penalty': float(penalty),
            'rolling_change_budget': _scaled_rolling_change_budget(cfg, lookback),
            'per_group_rolling_change_budgets': (
                _scaled_per_group_rolling_change_budgets(cfg, lookback, mode)
            ),
        })
    scenario_ids = [descriptor['scenario_id'] for descriptor in descriptors]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError('Action-history matrix generated duplicate scenario ids')
    return descriptors


def _scenario_cfg(
    cfg,
    lookback: int,
    *,
    hard_distance: int | None = None,
    history_group_mode: str | None = None,
    history_penalty: float | None = None,
    per_group_rolling_change_budgets: dict[str, int] | None = None,
):
    """Set an immutable Full-guard scenario for one explicit matrix point."""
    specification = _specification(cfg)
    families = specification.get('report_model_candidate_families') or []
    if not isinstance(families, list) or not families:
        raise TypeError('report_model_candidate_families must be a nonempty list')
    if specification.get('evaluation_row_scope') != 'all_chronological_test':
        raise ValueError(
            'action_history_ablation requires evaluation_row_scope='
            'all_chronological_test'
        )
    closed_loop_windows = specification.get('closed_loop_max_windows')
    if (
        isinstance(closed_loop_windows, bool)
        or not isinstance(closed_loop_windows, int)
        or closed_loop_windows < 0
    ):
        raise TypeError('closed_loop_max_windows must be a nonnegative integer')

    factual = lookback == 0
    if hard_distance is None:
        hard_distance = _hard_distances(cfg)[0]
    if history_group_mode is None:
        history_group_mode = _history_group_modes(cfg)[0]
    if history_penalty is None:
        history_penalty = _history_penalties(cfg)[0]
    maximum = len(cfg.MPC.control_variables)
    if not isinstance(hard_distance, int) or not 0 <= hard_distance <= maximum:
        raise ValueError(f'hard_distance must be an integer in [0, {maximum}]')
    if not factual and history_group_mode not in {
        'equipment',
        'per_control_variable',
    }:
        raise ValueError(f'Unknown history_group_mode: {history_group_mode!r}')
    if history_penalty < 0.0:
        raise ValueError('history_penalty must be nonnegative')
    reference = (
        'recorded_current_action' if factual else 'recorded_previous_action'
    )
    action_history = {
        'enabled': not factual,
        'lookback_steps': int(lookback),
        'recency_weights': _normalized_recency_weights(cfg, lookback),
        'penalty': 0.0 if factual else float(history_penalty),
        # The L=0 factual replay has no history constraints. Keeping this null
        # prevents an otherwise irrelevant equipment-group configuration from
        # rejecting a per-control-variable experiment.
        'groups': (
            None
            if factual
            else _history_groups_for_mode(cfg, history_group_mode)
        ),
        'rolling_change_budget': _scaled_rolling_change_budget(cfg, lookback),
        'per_group_rolling_change_budgets': (
            per_group_rolling_change_budgets
            if not factual
            else None
        ),
    }
    return OmegaConf.merge(
        cfg,
        {
            'DEFAULT': {'exp_name': f'{cfg.DEFAULT.exp_name}_history_l{lookback}'},
            'MPC': {
                'independent_evaluator': {'model_family': 'component_two_stage_hvac'},
                'objective': {
                    # L=0 is a factual replay. L>0 uses an explicit categorical
                    # hard-distance cap K in addition to history regularization.
                    'maximum_changed_control_variables_per_step': (
                        0 if factual else hard_distance
                    ),
                    'action_history': action_history,
                },
                'evaluation': {
                    'max_windows': int(closed_loop_windows),
                    'action_trust': {
                        'enabled': True,
                        'local_context_support': {
                            'enabled': True,
                            'minimum_validation_contexts': 30,
                        },
                    },
                    'action_model_screen': {'enabled': True},
                    'fixed_optimal_action_holdout': {
                        'total_power_report_model': {
                            'candidate_families': list(families),
                        },
                    },
                    'mpc_counterfactual_total_power_report': {
                        'enabled': True,
                        'first_action_reference': reference,
                        'evaluation_row_scope': 'all_chronological_test',
                        'row_stride_steps': 1,
                        'maximum_evaluation_rows': 0,
                    },
                },
                # The runner executes the three input bundles only after the
                # immutable action trace has been written.
                'counterfactual_report_input_ablation': {'run_after_mpc': False},
            },
        },
    )


def _history_summary_row(
    descriptor: dict[str, Any],
    mpc_dir: Path,
    summary,
) -> dict[str, Any]:
    evaluation = summary['mpc_counterfactual_total_power_report_evaluation']
    test = evaluation['test_evaluation']
    policy = summary.get('policy_evaluation') or {}
    solver = policy.get('solver') or {}
    return {
        'scenario_id': descriptor['scenario_id'],
        'lookback_l': int(descriptor['lookback_l']),
        'kind': descriptor['kind'],
        'hard_distance_k': descriptor['hard_distance_k'],
        'history_group_mode': descriptor['history_group_mode'],
        'history_group_names': json.dumps(
            descriptor['history_group_names'], ensure_ascii=False
        ),
        'history_penalty': descriptor['history_penalty'],
        'rolling_change_budget': descriptor['rolling_change_budget'],
        'per_group_rolling_change_budgets': json.dumps(
            descriptor['per_group_rolling_change_budgets'], ensure_ascii=False
        ),
        'status': 'OK',
        'mpc_result_directory': str(mpc_dir),
        'first_action_reference': evaluation['protocol']['first_action_reference'],
        'evaluated_transition_count': test['evaluated_transition_count'],
        'mean_changed_control_variables_from_reference': test[
            'changed_control_variables_from_reference'
        ]['mean'],
        'max_changed_control_variables_from_reference': test[
            'changed_control_variables_from_reference'
        ]['maximum'],
        'mpc_action_vs_recorded_actual_total_power_mape_percent': (
            evaluation['mape_cases']['mpc_action_vs_recorded_actual']['value']
        ),
        'report_to_report_total_power_mape_percent': evaluation['mape_cases'][
            'mpc_action_vs_recorded_action_report'
        ]['value'],
        'counterfactual_solver_fallback_count': test['solver_fallback_count'],
        'closed_loop_evaluated_windows': summary.get('evaluated_windows'),
        'closed_loop_solver_fallback_count': solver.get('fallback_count'),
        'closed_loop_control_switch_count': solver.get('control_switch_count'),
    }


def _input_rows(descriptor: dict[str, Any], sweep_dir: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(
        sweep_dir / 'counterfactual_report_input_ablation_summary.csv',
        encoding='utf-8-sig',
    )
    rows: list[dict[str, Any]] = []
    for source in frame.to_dict('records'):
        rows.append({
            'scenario_id': descriptor['scenario_id'],
            'lookback_l': int(descriptor['lookback_l']),
            'hard_distance_k': descriptor['hard_distance_k'],
            'history_group_mode': descriptor['history_group_mode'],
            'history_penalty': descriptor['history_penalty'],
            'case': str(source['case']),
            'description': str(source['description']),
            'total_report_to_report_mape_percent': source[
                'report_to_report_total_power_mape_percent'
            ],
            'vent_report_to_report_mape_percent': source[
                '환기유효전력(W)_report_to_report_mape_percent'
            ],
            'hvac_report_to_report_mape_percent': source[
                '냉난방유효전력(W)_report_to_report_mape_percent'
            ],
            'total_mpc_action_report_vs_recorded_actual_mape_percent': source[
                'mpc_action_report_vs_actual_total_power_mape_percent'
            ],
            'vent_mpc_action_report_vs_recorded_actual_mape_percent': source[
                '환기유효전력(W)_mpc_action_report_vs_actual_mape_percent'
            ],
            'hvac_mpc_action_report_vs_recorded_actual_mape_percent': source[
                '냉난방유효전력(W)_mpc_action_report_vs_actual_mape_percent'
            ],
            'counterfactual_report_input_ablation_directory': str(sweep_dir),
        })
    return rows


def _number(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return '—'
    return f'{float(value):.{digits}f}'


def _write_markdown(path: Path, history_rows, input_rows, manifest) -> None:
    lines = [
        '# Full guard action-history constraint matrix',
        '',
        '## 비교 규칙',
        '',
        '- L=0은 같은 시점의 기록 action을 고정한 factual 기준선이다. 제어 변경이나 절감 주장을 하지 않는다.',
        '- L>0은 각 시점의 `t-L, …, t-1` applied action 이력과 비교해 z_t의 급격한 변경을 억제한 Full-guard MPC다.',
        '- closed loop은 window 시작 전에는 기록 action으로 history를 채우고, 이후에는 이전에 실제 적용한 MPC action만 누적한다.',
        f"- 모든 L>0에서 recency weight 합은 {manifest['recency_weight_total']:.4f}로 고정했다. newest-to-oldest exponential decay={manifest['recency_weight_decay']:.4f}.",
        '- 모든 scenario는 동일하게 시간순 test 전체 first-action trace와 동일 Full-guard candidate screen을 사용한다.',
        '- report-model 세 입력 case는 저장된 같은 MPC trace를 재사용한다. 따라서 case 간 MILP action은 다르지 않다.',
        '',
        '## Matrix별 MPC trace',
        '',
        '| Scenario | L | K | history group | penalty | 성격 | first-action 기준 | 평가 행 | 평균 변경 변수 | 최대 변경 변수 | 실제값 대비 총전력 MAPE | action-effect 총전력 MAPE | counterfactual fallback |',
        '|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in history_rows:
        if row['status'] != 'OK':
            lines.append(
                f"| {row.get('scenario_id', '—')} | {row['lookback_l']} | — | — | — | — | — | — | — | — | 실패 | 실패 | — |"
            )
            continue
        lines.append(
            '| {scenario_id} | {lookback_l} | {hard_distance_k} | '
            '{history_group_mode} | {history_penalty} | {kind} | '
            '{first_action_reference} | {evaluated_transition_count} | {mean} | '
            '{maximum} | {actual}% | {effect}% | {fallback} |'.format(
                scenario_id=row['scenario_id'],
                lookback_l=row['lookback_l'],
                hard_distance_k=row['hard_distance_k'],
                history_group_mode=row['history_group_mode'],
                history_penalty=_number(row['history_penalty']),
                kind=row['kind'],
                first_action_reference=row['first_action_reference'],
                evaluated_transition_count=row['evaluated_transition_count'],
                mean=_number(row['mean_changed_control_variables_from_reference']),
                maximum=_number(row['max_changed_control_variables_from_reference'], 0),
                actual=_number(
                    row['mpc_action_vs_recorded_actual_total_power_mape_percent']
                ),
                effect=_number(row['report_to_report_total_power_mape_percent']),
                fallback=_number(row['counterfactual_solver_fallback_count'], 0),
            )
        )
    lines.extend([
        '',
        '## Report model 입력별 action-effect MAPE',
        '',
        '| Scenario | L | K | history group | penalty | Report model 입력 | 환기 MAPE | 냉난방 MAPE | 총전력 MAPE |',
        '|---|---:|---:|---|---:|---|---:|---:|---:|',
    ])
    for row in input_rows:
        lines.append(
            '| {scenario_id} | {lookback_l} | {hard_distance_k} | '
            '{history_group_mode} | {history_penalty} | {description} | '
            '{vent}% | {hvac}% | {total}% |'.format(
                scenario_id=row['scenario_id'],
                lookback_l=row['lookback_l'],
                hard_distance_k=row['hard_distance_k'],
                history_group_mode=row['history_group_mode'],
                history_penalty=_number(row['history_penalty']),
                description=row['description'],
                vent=_number(row['vent_report_to_report_mape_percent']),
                hvac=_number(row['hvac_report_to_report_mape_percent']),
                total=_number(row['total_report_to_report_mape_percent']),
            )
        )
    lines.extend([
        '',
        '## 해석 경계',
        '',
        '실제 장비에 MPC action을 적용한 관측 전력은 없으므로, L>0의 action-effect MAPE는 동일 report model 안에서 기록 action과 MPC action을 각각 넣어 비교한 반사실적 모델 추정치다. 실제 기록 전력 대비 MAPE는 action 변경 효과와 report-model 오차가 함께 들어간 diagnostic이며, 순수 전력 예측오차로 해석해서는 안 된다.',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_action_history_ablation(cfg):
    """Run the factual baseline and every configured history-control matrix point."""
    result_dir = _result_directory(cfg)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_logging(cfg, result_dir)
    lookbacks = _lookbacks(cfg)
    specification = _specification(cfg)
    descriptors = _scenario_descriptors(cfg)
    manifest = {
        'framework': 'full_guard_action_history_constraint_matrix',
        'lookback_steps': lookbacks,
        'hard_distances': _hard_distances(cfg),
        'history_group_modes': _history_group_modes(cfg),
        'history_penalties': _history_penalties(cfg),
        'l0_interpretation': 'factual recorded-current-action baseline',
        'l_positive_interpretation': (
            'Full-guard MPC with applied-action history: recorded bootstrap, '
            'then prior applied MPC actions in closed loop'
        ),
        'evaluation_row_scope': 'all_chronological_test',
        'recency_weight_decay': float(specification['recency_weight_decay']),
        'recency_weight_total': float(specification['recency_weight_total']),
        'rolling_budget_reference': {
            'steps': int(specification['rolling_change_budget_reference_steps']),
            # ``null`` deliberately disables the legacy aggregate equipment
            # budget when an experiment uses only independent per-variable
            # budgets.  Preserve that distinction in the manifest instead of
            # coercing it to an integer.
            'group_changes': (
                None
                if specification['rolling_change_budget_reference_value'] is None
                else int(specification['rolling_change_budget_reference_value'])
            ),
        },
        'scenarios': [
            {
                **descriptor,
                'action_history_enabled': bool(descriptor['lookback_l'] > 0),
                'first_action_reference': (
                    'recorded_current_action'
                    if descriptor['lookback_l'] == 0
                    else 'recorded_previous_action'
                ),
                'recency_weights_newest_to_oldest': _normalized_recency_weights(
                    cfg, descriptor['lookback_l']
                ),
            }
            for descriptor in descriptors
        ],
        'report_model_input_cases': [
            _plain(case) for case in cfg.MPC.report_model_input_ablation.cases
        ],
    }
    _write_json(result_dir / 'experiment_manifest.json', manifest)
    LOGGER.info('Action-history ablation started: log=%s', log_path)
    LOGGER.info(
        'Runtime: Python=%s | platform=%s | pid=%d | data=%s | versions=%s | '
        'environment=%s | scenarios=%d | lookbacks=%s | K=%s | groups=%s | '
        'penalties=%s',
        platform.python_version(), platform.platform(), os.getpid(),
        dataset_label_from_cfg(cfg), list(versions_from_cfg(cfg)),
        cfg.DATASET.environment_name, len(descriptors), lookbacks,
        manifest['hard_distances'], manifest['history_group_modes'],
        manifest['history_penalties'],
    )

    history_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    expected_cases = {
        str(case.name) for case in cfg.MPC.report_model_input_ablation.cases
    }
    for position, descriptor in enumerate(descriptors, start=1):
        lookback = int(descriptor['lookback_l'])
        scenario_dir = result_dir / descriptor['scenario_id']
        scenario_dir.mkdir(parents=True, exist_ok=True)
        mpc_dir = scenario_dir / 'mpc'
        LOGGER.info(
            'Scenario %d/%d started: id=%s | L=%d | K=%d | groups=%s | '
            'penalty=%s | artifacts=%s',
            position, len(descriptors), descriptor['scenario_id'], lookback,
            descriptor['hard_distance_k'], descriptor['history_group_mode'],
            descriptor['history_penalty'], scenario_dir,
        )
        try:
            scenario_cfg = _scenario_cfg(
                cfg,
                lookback,
                hard_distance=descriptor['hard_distance_k'],
                history_group_mode=descriptor['history_group_mode'],
                history_penalty=descriptor['history_penalty'],
                per_group_rolling_change_budgets=(
                    descriptor['per_group_rolling_change_budgets']
                ),
            )
            mpc_summary = run_mpc_pipeline(scenario_cfg, mpc_dir)
            history_row = _history_summary_row(descriptor, mpc_dir, mpc_summary)
            sweep_cfg = OmegaConf.merge(
                scenario_cfg,
                {'MPC': {'counterfactual_report_input_ablation': {
                    'reference_mpc_result_dir': str(mpc_dir),
                }}},
            )
            sweep_summary = run_counterfactual_report_input_ablation(sweep_cfg)
            sweep_dir = Path(sweep_summary['result_directory'])
            rows = _input_rows(descriptor, sweep_dir)
            actual_cases = {row['case'] for row in rows}
            if actual_cases != expected_cases:
                raise RuntimeError(
                    'Expected exactly the configured three report-input cases; '
                    f'expected={sorted(expected_cases)}, actual={sorted(actual_cases)}'
                )
            history_row['counterfactual_report_input_ablation_directory'] = str(
                sweep_dir
            )
            history_rows.append(history_row)
            input_rows.extend(rows)
            _write_json(scenario_dir / 'scenario_manifest.json', {
                **next(
                    item
                    for item in manifest['scenarios']
                    if item['scenario_id'] == descriptor['scenario_id']
                ),
                'mpc_result_directory': str(mpc_dir),
                'counterfactual_report_input_ablation_directory': str(sweep_dir),
                'evaluation_row_scope': 'all_chronological_test',
            })
            LOGGER.info(
                'Scenario %s complete: action-effect total MAPE=%s | '
                'mean changed variables=%s | report cases=%d',
                descriptor['scenario_id'],
                history_row['report_to_report_total_power_mape_percent'],
                history_row['mean_changed_control_variables_from_reference'],
                len(rows),
            )
        except Exception as error:
            LOGGER.exception('Scenario %s failed', descriptor['scenario_id'])
            failure = (
                f"{descriptor['scenario_id']}: {type(error).__name__}: {error}"
            )
            failures.append(failure)
            history_rows.append({
                'scenario_id': descriptor['scenario_id'],
                'lookback_l': int(lookback),
                'hard_distance_k': descriptor['hard_distance_k'],
                'history_group_mode': descriptor['history_group_mode'],
                'history_penalty': descriptor['history_penalty'],
                'status': 'FAIL',
                'mpc_result_directory': str(mpc_dir),
                'error': failure,
            })
        _write_csv(result_dir / 'lookback_summary.csv', history_rows)
        _write_csv(result_dir / 'lookback_input_report_summary.csv', input_rows)
        _write_markdown(
            result_dir / 'action_history_ablation_report.md',
            history_rows,
            input_rows,
            manifest,
        )

    summary = {
        'framework': 'full_guard_action_history_constraint_matrix',
        'result_directory': str(result_dir),
        'lookback_steps': lookbacks,
        'scenarios': descriptors,
        'history_rows': history_rows,
        'input_rows': input_rows,
        'failures': failures,
        'status': 'OK' if not failures else 'PARTIAL_FAILURE',
    }
    _write_json(result_dir / 'action_history_ablation.json', summary)
    if failures:
        raise RuntimeError(
            'Action-history ablation finished with failures; artifacts were '
            f'preserved at {result_dir}: ' + ' | '.join(failures)
        )
    LOGGER.info('Action-history ablation complete: result_dir=%s', result_dir)
    return summary
