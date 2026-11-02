"""Train and compare six control methods without changing the legacy MPC task."""

from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import platform
import shutil
import traceback
from time import perf_counter

import joblib
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from sklearn.metrics import r2_score
import torch

from controllers.action_crossfit import align_boundaries_to_control_episodes
from controllers.action_trust import ActionTrustEligibility
from controllers.control_benchmark import create_control_planner
from data_provider.control_data import load_control_data
from data_provider.dataset_versions import resolve_dataset_csv_path, versions_from_cfg
from dynamics.evaluator_factory import create_independent_evaluator
from models.control import ContextSpec, create_control_model, load_control_model
from models.control.base import categorical_changes, stage_cost


LOGGER = logging.getLogger('control_benchmark')


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def select_windows(data, start, length, count):
    """Outcome-blind deterministic windows with neither gaps nor segment joins."""
    valid = data.transition_mask(0, len(data.states)-1)
    possible = np.arange(start, len(data.states)-length)
    possible = np.asarray([i for i in possible if valid[i:i+length].all()], dtype=int)
    if not len(possible):
        raise ValueError('No contiguous test windows of the requested length')
    selected = []
    for target in np.linspace(0, len(possible)-1, min(count, len(possible)), dtype=int):
        candidate = int(possible[target])
        if all(abs(candidate-other) >= length for other in selected):
            selected.append(candidate)
    if not selected:
        raise ValueError('No nonoverlapping evaluation windows')
    return selected


def prediction_metrics(actual_x, predicted_x, actual_p, predicted_p, scale, constraints):
    actual_x, predicted_x, actual_p, predicted_p = map(
        np.asarray, (actual_x, predicted_x, actual_p, predicted_p))
    finite = bool(np.isfinite(predicted_x).all() and np.isfinite(predicted_p).all())
    if not finite:
        return {'finite': False, 'count': len(actual_x), 'power_wape_percent': None,
                'power_r2': None, 'state_mae': None, 'state_normalized_rmse': None}
    actual_bad = np.zeros(actual_x.shape, dtype=bool)
    predicted_bad = np.zeros_like(actual_bad)
    for j, c in enumerate(constraints):
        if c.get('lower') is not None:
            actual_bad[:, j] |= actual_x[:, j] < float(c['lower'])
            predicted_bad[:, j] |= predicted_x[:, j] < float(c['lower'])
        if c.get('upper') is not None:
            actual_bad[:, j] |= actual_x[:, j] > float(c['upper'])
            predicted_bad[:, j] |= predicted_x[:, j] > float(c['upper'])
    return {
        'finite': True, 'count': len(actual_x),
        'state_mae': np.abs(actual_x-predicted_x).mean(0).tolist(),
        'state_normalized_rmse': float(np.sqrt(np.mean(((actual_x-predicted_x)/scale)**2))),
        'power_mae_w': np.abs(actual_p-predicted_p).mean(0).tolist(),
        'power_wape_percent': float(100*np.abs(actual_p-predicted_p).sum()/max(np.abs(actual_p).sum(), 1e-6)),
        'power_r2': float(r2_score(actual_p.sum(1), predicted_p.sum(1))),
        'false_safe_state_fraction': float((actual_bad & ~predicted_bad).mean()),
    }


def _one_step(model, batch, scale, constraints):
    xs, ps = [], []
    for i in range(0, len(batch.rows), 4096):
        part = batch.take(slice(i, i+4096))
        x, p = model.predict(part.context, part.controls, part.disturbances)
        xs.append(x)
        ps.append(p)
    return prediction_metrics(batch.next_states, np.concatenate(xs),
                              batch.powers, np.concatenate(ps), scale, constraints)


def _recorded_rollouts(model, data, contexts, starts, horizon, scale, constraints):
    actual_x, predicted_x, actual_p, predicted_p = [], [], [], []
    for start in starts:
        ctx = contexts[start].copy()
        for t in range(horizon):
            row = start+t
            x, p = model.predict(ctx, data.controls[row], data.disturbances[row])
            actual_x.append(data.states[row+1])
            actual_p.append(data.powers[row])
            predicted_x.append(x[0])
            predicted_p.append(p[0])
            ctx = model.spec.advance(ctx, data.controls[row], x[0])
    return prediction_metrics(actual_x, predicted_x, actual_p, predicted_p, scale, constraints)


def _policy_metrics(traces, constraints, state_names):
    frame = pd.DataFrame(traces)
    violations, magnitudes = {}, {}
    any_bad = np.zeros(len(frame), dtype=bool)
    for j, item in enumerate(constraints):
        x = frame[f'next_state_{j}'].to_numpy()
        v = np.zeros(len(x))
        if item.get('lower') is not None:
            v += np.maximum(float(item['lower'])-x, 0)
        if item.get('upper') is not None:
            v += np.maximum(x-float(item['upper']), 0)
        violations[state_names[j]] = int((v > 1e-8).sum())
        magnitudes[state_names[j]] = float(v.sum())
        any_bad |= v > 1e-8
    return {
        'steps': len(frame),
        'energy_wh': float(frame.energy_wh.sum()),
        'soft_constraint_cost': float(frame.soft_constraint_cost.sum()),
        'switching_cost': float(frame.switching_cost.sum()),
        'total_objective': float(frame.total_cost.sum()),
        'state_violation_steps': violations, 'state_violation_magnitude_sum': magnitudes,
        'any_state_violation_fraction': float(any_bad.mean()),
        'changed_variable_count': int(frame.changed_variables.sum()),
        'fallback_steps': int(frame.fallback.sum()),
        'action_trust_rejection_steps': int((~frame.action_trusted).sum()),
        'change_cap_violation_steps': int((~frame.change_cap_satisfied).sum()),
        'solver_optimal_status_steps': int(frame.optimal.sum()),
        'median_decision_seconds': float(frame.decision_seconds.median()),
        'p95_decision_seconds': float(frame.decision_seconds.quantile(0.95)),
    }


def _evaluate_policy(model, planner, data, contexts, starts, steps, horizon,
                     evaluator, trust, actions, constraints, options, recorded=False):
    traces = []
    for window, start in enumerate(starts):
        ctx = contexts[start].copy()
        previous = ctx[2*model.spec.nx:2*model.spec.nx+model.spec.nz].copy()
        if hasattr(planner, 'warm'):
            planner.warm = None
        for t in range(steps):
            row = start+t
            state, disturbance = ctx[:model.spec.nx], data.disturbances[row]
            audit = trust.evaluate_candidates(state, disturbance)
            first = np.asarray([a.eligible for a in audit])
            first &= categorical_changes(actions, previous) <= int(options['maximum_changed_variables'])
            fallback, optimal, objective, mip_gap = False, False, None, None
            if recorded:
                chosen, status, duration = data.controls[row], 'recorded_action_reference', 0.0
            elif not first.any():
                chosen, status, duration, fallback = previous, 'hold_no_trusted_action', 0.0, True
            else:
                forecast = np.repeat(disturbance[None], horizon, axis=0)
                plan = planner.solve(ctx, forecast, previous, first)
                status, duration, objective, optimal = plan.status, plan.elapsed_seconds, plan.objective, plan.optimal
                mip_gap = plan.mip_gap
                if len(plan.controls):
                    chosen = plan.controls[0]
                    if not any(np.array_equal(chosen, a) for a in actions[first]):
                        raise AssertionError('Planner returned a masked or incompatible first action')
                else:
                    # A transparent hold fallback, not a claim of safety.
                    chosen, fallback = previous, True
            trusted = trust.evaluate(chosen, state, disturbance).eligible
            next_state, power = evaluator.step(state, chosen, disturbance)
            if not np.isfinite(next_state).all() or not np.isfinite(power).all():
                raise FloatingPointError('Independent evaluation plant became nonfinite')
            changes = float(categorical_changes(chosen, previous))
            cost = float(stage_cost(next_state, power, constraints))
            switching = 2*float(options['switch_penalty'])*changes
            energy = float(power.sum()/60)
            record = {
                'window': window, 'row': row, 'timestamp': str(data.timestamps.iloc[row]),
                'status': status, 'fallback': fallback, 'optimal': optimal,
                'action_trusted': trusted, 'eligible_first_action_count': int(first.sum()),
                'change_cap_satisfied': changes <= int(options['maximum_changed_variables']),
                'decision_seconds': duration, 'predicted_objective': objective,
                'mip_gap': mip_gap,
                'energy_wh': energy, 'soft_constraint_cost': cost-energy,
                'switching_cost': switching, 'total_cost': cost+switching,
                'changed_variables': changes,
                **{f'state_{j}': float(v) for j, v in enumerate(state)},
                **{f'next_state_{j}': float(v) for j, v in enumerate(next_state)},
                **{f'power_{j}_w': float(v) for j, v in enumerate(power)},
                **data.encoder.decode_action(chosen),
            }
            traces.append(record)
            ctx = model.spec.advance(ctx, chosen, next_state)
            previous = chosen.copy()
    return _policy_metrics(traces, constraints, data.state_names), traces


def _check_sources(cfg, data):
    sources = []
    for version in versions_from_cfg(cfg):
        path = resolve_dataset_csv_path(cfg.DATASET.datadir, version, cfg.DATASET.environment_name)
        raw = pd.read_csv(path)
        columns = list(data.state_names + data.disturbance_names + data.power_names)
        numeric = raw[columns].apply(pd.to_numeric, errors='coerce')
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(
                f'{path}: nonfinite numeric input. New benchmark refuses bidirectional '
                'interpolation across splits; prepare causal data explicitly first.')
        sources.append({'path': str(path), 'sha256': _hash_file(path), 'rows': len(raw)})
    return sources


def _write_report(root, rows, manifest):
    pd.DataFrame(rows).to_csv(root/'comparison.csv', index=False, encoding='utf-8-sig')
    _json(root/'comparison.json', {'manifest': manifest, 'results': rows})
    lines = ['# 제어 6종 비교 결과', '',
             '> 관측 로그 기반 모델 및 독립 surrogate 폐루프 평가다. 실제 장비 절감 실증이 아니다.', '',
             '| 공간 | 방법 | 상태 | test 전력 WAPE(%) | rollout 전력 WAPE(%) | 에너지(Wh) | 환경 위반 비율 | fallback | p95(s) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        def fmt(k):
            v = row.get(k)
            return f'{v:.4f}' if isinstance(v, (int, float)) else str(v if v is not None else '—')
        lines.append('| '+' | '.join([str(row['environment']), str(row['method']), str(row['status']),
                                     *[fmt(k) for k in ('test_power_wape_percent', 'rollout_power_wape_percent',
                                                       'energy_wh', 'any_state_violation_fraction',
                                                       'fallback_steps', 'p95_decision_seconds')]])+' |')
    lines += ['', '## 해석 제한', '',
              '- 동일 시간 분할·창·외란·support·변경 cap·물리 단위 목적함수를 사용한다.',
              '- baseline은 기존 MILP, 모드/ensemble MILP는 train-derived 상태 box 및 설정에 따른 nominal linearization 또는 exact disjunction을 사용한다.',
              '- 신경망 방법은 CEM 근사 탐색이며 전역 최적성을 보장하지 않는다.',
              '- DFL은 train-prefix teacher의 유한 후보 smoothed decision loss 구현이다. 원 논문의 differentiable MIP 재현이 아니다.',
              '- physics residual은 구조화 grey-box이며 PCNN의 전체 보증을 주장하지 않는다.',
              '- 별도 evaluator도 같은 관측 데이터로 식별하므로 미관측 action의 실제 정답이 아니다.',
              '- 모델/정책 gate 실패를 숨기지 않으며 모든 순위는 탐색적이다. 한 seed·제한된 창의 결과를 일반화하지 않는다.', '']
    (root/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def run_control_benchmark(cfg):
    config = _plain(cfg.CONTROL_BENCHMARK)
    if (int(config['evaluation_windows']) < 1 or int(config['closed_loop_steps']) < 1
            or int(config['model_parameters']['epochs']) < 1):
        raise ValueError('Positive evaluation windows, closed-loop steps and training epochs required')
    if not 1 <= int(cfg.MPC.control_horizon) <= int(cfg.MPC.prediction_horizon):
        raise ValueError('Require 1 <= control_horizon <= prediction_horizon')
    if not 0 <= float(config['planner']['risk_weight']) <= 1:
        raise ValueError('risk_weight must be in [0,1]')
    if not config['selected_methods'] or any(m not in config['methods'] for m in config['selected_methods']):
        raise ValueError('selected_methods must be nonempty registered method names')
    torch.set_num_threads(int(config['torch_threads']))
    seeds = [int(v) for v in config['seeds']]
    if not seeds or not config['environments']:
        raise ValueError('At least one seed and environment are required')
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
    root = (Path(config['output_directory'])/f'{cfg.DEFAULT.exp_name}_{stamp}').resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for source in [*source_root.rglob('*.py'), *source_root.rglob('*.sh'), *source_root.glob('configs/*.yaml')]:
        relative = source.relative_to(source_root)
        dest = root/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        source_hashes[str(relative)] = _hash_file(source)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(root/'run.log')])
    OmegaConf.save(cfg, root/'config.yaml')
    manifest = {'status': 'running', 'result_directory': str(root), 'python': platform.python_version(),
                'torch': torch.__version__, 'numpy': np.__version__,
                'methods': config['selected_methods'], 'seeds': seeds,
                'source_sha256': source_hashes,
                'real_equipment_closed_loop': False,
                'decision_focused_implementation': 'finite-candidate smoothed teacher cost; not exact MIP differentiation'}
    _json(root/'manifest.json', manifest)
    results, failed = [], []
    for environment in config['environments']:
        ecfg = OmegaConf.merge(cfg, {'DATASET': {'environment_name': environment}})
        data = load_control_data(ecfg)
        sources = _check_sources(ecfg, data)
        train_end, test_start = data.split_indices(ecfg.DATASET.split_rate)
        if config['align_episode_boundaries']:
            train_end, test_start = align_boundaries_to_control_episodes(data, [train_end, test_start])
        spec = ContextSpec(data, config['memory_decay'])
        contexts = spec.contexts(data)
        train = spec.batch(data, contexts, 0, train_end)
        validation = spec.batch(data, contexts, train_end, test_start)
        test = spec.batch(data, contexts, test_start, len(data.states))
        max_train = int(config['maximum_training_samples'])
        if max_train > 0 and len(train.rows) > max_train:
            train = train.take(np.linspace(0, len(train.rows)-1, max_train, dtype=int))
        constraints = _plain(ecfg.MPC.state_constraints)
        logical_rules = ecfg.MPC.logical_rules
        options = {**config['planner'], 'prediction_horizon': int(ecfg.MPC.prediction_horizon),
                   'control_horizon': int(ecfg.MPC.control_horizon)}
        horizon, steps = options['prediction_horizon'], int(config['closed_loop_steps'])
        starts = select_windows(data, test_start, max(horizon, steps), int(config['evaluation_windows']))
        scale = np.maximum(train.context[:, :spec.nx].std(0), 1e-3)
        global_config = dict(config['action_trust'])
        global_config['local_context_support'] = {'enabled': False}
        global_trust = ActionTrustEligibility.build(
            data, train.rows, global_config, logical_rules=logical_rules)
        actions = np.asarray([d.encoded_control for d in global_trust.evaluate_candidates(
            train.context[0, :spec.nx], train.disturbances[0]) if d.eligible], dtype=float)
        if not len(actions):
            raise ValueError('No train-supported logical actions; adjust train-only thresholds')
        trust = ActionTrustEligibility.build(data, train.rows, config['action_trust'],
                                            logical_rules=logical_rules, candidate_controls=actions)
        shared = root/environment
        shared.mkdir()
        shared_manifest = {
            'sources': sources, 'train_end_exclusive': train_end, 'test_start': test_start,
            'train_count': len(train.rows), 'validation_count': len(validation.rows),
            'test_count': len(test.rows), 'window_starts': starts, 'window_timestamps': [
                str(data.timestamps.iloc[i]) for i in starts],
            'prediction_horizon': horizon, 'control_horizon': options['control_horizon'],
            'closed_loop_steps': steps, 'action_count': len(actions),
            'actual_evaluation_steps_per_method': len(starts)*steps,
            'disturbance_forecast': 'persistence; evaluation plant receives recorded exogenous disturbances',
            'state_scale': scale.tolist(), 'context_is_strictly_causal': True,
            'preprocessing': data.control_preprocessing,
        }
        _json(shared/'data_manifest.json', shared_manifest)
        _json(shared/'action_trust.json', trust.to_metadata())
        _json(shared/'actions.json', [data.encoder.decode_action(a) for a in actions])
        np.savez_compressed(shared/'split_rows.npz', train=train.rows, validation=validation.rows, test=test.rows,
                            window_starts=np.asarray(starts), action_vectors=actions)
        LOGGER.info('Environment=%s train=%d val=%d test=%d actions=%d windows=%s',
                    environment, len(train.rows), len(validation.rows), len(test.rows), len(actions), starts)
        for seed in seeds:
            seed_dir = shared/f'seed_{seed}'
            seed_dir.mkdir()
            evaluator = create_independent_evaluator(ecfg.MPC.independent_evaluator, data, train_end,
                                                     random_state=seed+10000, training_rows=train.rows)
            joblib.dump(evaluator, seed_dir/'independent_evaluator.joblib')
            _json(seed_dir/'evaluator_metadata.json', evaluator.to_metadata())
            ex, ep = evaluator.predict_state(test.context[:, :spec.nx], test.controls, test.disturbances), (
                evaluator.predict_power(test.context[:, :spec.nx], test.controls, test.disturbances))
            evaluator_metrics = prediction_metrics(test.next_states, ex, test.powers, ep, scale, constraints)
            _json(seed_dir/'evaluator_test_metrics.json', evaluator_metrics)
            for method in config['selected_methods']:
                settings = config['methods'][method]
                directory = seed_dir/method
                directory.mkdir()
                LOGGER.info('START environment=%s seed=%d method=%s', environment, seed, method)
                begin = perf_counter()
                try:
                    parameters = {**config['model_parameters'], **settings.get('parameters', {})}
                    model = create_control_model(settings['model'], spec, parameters, seed)
                    if settings['model'] == 'decision_focused':
                        model.decision_actions, model.decision_constraints = actions, constraints
                    resume = config.get('resume_from')
                    checkpoint = (Path(resume)/environment/f'seed_{seed}'/method/'model') if resume else None
                    loaded_checkpoint = False
                    original_training_seconds = None
                    if resume and checkpoint.is_dir():
                        old = Path(resume)/environment
                        previous_manifest = json.loads((old/'data_manifest.json').read_text())
                        if previous_manifest['sources'] != sources:
                            raise ValueError('Checkpoint dataset fingerprint differs')
                        if (previous_manifest['train_end_exclusive'] != train_end
                                or previous_manifest['test_start'] != test_start):
                            raise ValueError('Checkpoint chronological boundaries differ')
                        model = load_control_model(checkpoint)
                        if model.family != settings['model'] or not np.array_equal(model.fit_rows, train.rows):
                            raise ValueError('Checkpoint model family or training rows differ')
                        for name in ('state_names', 'power_names', 'disturbance_names', 'feature_names', 'decay'):
                            if getattr(model.spec, name) != getattr(spec, name):
                                raise ValueError(f'Checkpoint input contract differs: {name}')
                        loaded_checkpoint = True
                        original_evaluation = checkpoint.parent/'evaluation.json'
                        if original_evaluation.is_file():
                            original_training_seconds = json.loads(original_evaluation.read_text()).get('training_seconds')
                    else:
                        if resume and not config.get('resume_missing_models', False):
                            raise FileNotFoundError(f'Missing checkpoint: {checkpoint}')
                        model.fit(train, validation)
                    fit_or_load_seconds = perf_counter()-begin
                    fit_seconds = original_training_seconds if loaded_checkpoint else fit_or_load_seconds
                    model.save(directory/'model')
                    # Reload before inference: the very artifact users swap is tested.
                    model = load_control_model(directory/'model')
                    validation_metrics = _one_step(model, validation, scale, constraints)
                    test_metrics = _one_step(model, test, scale, constraints)
                    rollout_metrics = _recorded_rollouts(model, data, contexts, starts, horizon, scale, constraints)
                    _json(directory/'prediction_metrics.json', {'validation': validation_metrics,
                                                               'test': test_metrics, 'recorded_action_rollout': rollout_metrics})
                    planner = create_control_planner(settings['planner'], model, actions, constraints,
                                                     logical_rules, {**options, 'seed': seed}, train.context)
                    policy, trace = _evaluate_policy(model, planner, data, contexts, starts, steps, horizon,
                                                     evaluator, trust, actions, constraints, options)
                    pd.DataFrame(trace).to_csv(directory/'policy_trace.csv', index=False, encoding='utf-8-sig')
                    reference, reference_trace = _evaluate_policy(
                        model, planner, data, contexts, starts, steps, horizon,
                        evaluator, trust, actions, constraints, options, recorded=True)
                    pd.DataFrame(reference_trace).to_csv(directory/'recorded_reference_trace.csv', index=False, encoding='utf-8-sig')
                    model_passed = (validation_metrics['finite'] and rollout_metrics['finite']
                                    and validation_metrics['power_r2'] >= float(ecfg.MPC.affine_models.minimum_test_power_r2)
                                    and rollout_metrics['power_wape_percent'] <= float(ecfg.MPC.affine_models.maximum_rollout_power_wape_percent)
                                    and 100*validation_metrics['false_safe_state_fraction'] <= float(ecfg.MPC.affine_models.maximum_constraint_false_safe_rate_percent))
                    policy_passed = (model_passed
                                     and policy['any_state_violation_fraction'] <= float(config['maximum_policy_violation_fraction'])
                                     and not (policy['fallback_steps'] or policy['action_trust_rejection_steps']
                                              or policy['change_cap_violation_steps']))
                    savings = 100*(reference['energy_wh']-policy['energy_wh'])/max(reference['energy_wh'], 1e-6)
                    evaluation = {
                        'method': method, 'model_family': model.family, 'planner_family': settings['planner'],
                        'training_seconds': fit_seconds, 'policy': policy, 'recorded_reference': reference,
                        'fit_or_load_seconds': fit_or_load_seconds,
                        'loaded_from_checkpoint': str(checkpoint) if loaded_checkpoint else None,
                        'surrogate_energy_savings_percent': savings,
                        'quality': {'model_validation_passed': bool(model_passed),
                                    'exploratory_policy_gate_passed': bool(policy_passed),
                                    'not_a_real_equipment_safety_certificate': True},
                        'actual_closed_loop_mape': {'available': False, 'reason': 'No equipment commands executed'},
                        'scope': 'independent observational surrogate closed-loop only',
                        'mode_matrix_strategy': options.get('mode_matrix_strategy') if settings['planner'] in (
                            'joint_milp', 'scenario_milp') else None,
                        'optimal_flag_scope': 'MILP subproblem; nominal_linearization is not a global model optimum',
                    }
                    _json(directory/'evaluation.json', evaluation)
                    row = {'environment': environment, 'seed': seed, 'method': method, 'status': 'completed',
                           'model_family': model.family, 'planner_family': settings['planner'],
                           'mode_matrix_strategy': options.get('mode_matrix_strategy') if settings['planner'] in (
                               'joint_milp', 'scenario_milp') else None,
                           'training_seconds': fit_seconds,
                           'fit_or_load_seconds': fit_or_load_seconds,
                           'loaded_from_checkpoint': loaded_checkpoint,
                           'test_power_wape_percent': test_metrics['power_wape_percent'],
                           'test_state_normalized_rmse': test_metrics['state_normalized_rmse'],
                           'rollout_power_wape_percent': rollout_metrics['power_wape_percent'],
                           'surrogate_energy_savings_percent': savings,
                           'model_validation_passed': bool(model_passed),
                           'exploratory_policy_gate_passed': bool(policy_passed), **policy}
                    LOGGER.info('DONE %s/%s energy=%.3fWh WAPE=%s fallback=%d p95=%.3fs',
                                environment, method, policy['energy_wh'], test_metrics['power_wape_percent'],
                                policy['fallback_steps'], policy['p95_decision_seconds'])
                except Exception as error:
                    LOGGER.exception('FAILED %s/%s', environment, method)
                    (directory/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                    row = {'environment': environment, 'seed': seed, 'method': method,
                           'status': 'failed', 'error': str(error)}
                    failed.append(row)
                results.append(row)
                _write_report(root, results, manifest)
    manifest['status'] = 'completed_with_failures' if failed else 'completed'
    manifest['completed_method_runs'] = sum(r['status'] == 'completed' for r in results)
    manifest['failed_method_runs'] = len(failed)
    manifest['source_changed_during_run'] = [
        path for path, digest in source_hashes.items() if _hash_file(source_root/path) != digest]
    _json(root/'manifest.json', manifest)
    _write_report(root, results, manifest)
    LOGGER.info('RESULT_DIRECTORY=%s', root)
    if failed:
        raise RuntimeError(f'{len(failed)} method runs failed; inspect {root}')
    return {'result_directory': str(root), 'results': results}
