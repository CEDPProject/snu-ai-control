"""절감 ablation + m1–m6 MAPE campaign. 실행 자원(CPU 집합·worker 수)은 manifest에 기록한다."""

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from runners.full_control_benchmark_runner import (
    available_memory_gb, digest, mpc_fingerprint, now, scope_audit, write_json,
)

REPO = Path(__file__).resolve().parents[3]


def mape_target(cfg):
    """설정에 MAPE 목표(%)가 있으면 읽고, 없으면 목표 판정을 생략한다."""
    value = OmegaConf.select(cfg, 'MPC.component_kpi_gate.target_percent', default=None)
    return None if value is None else float(value)
CELLS = [(g, e) for g in ('v3', 'v4', 'v5', 'concat')
         for e in ('livingroom', 'room1', 'room2')]
METHODS = ['m1_affine_milp', 'm2_mode_history_milp', 'm3_physics_residual',
           'm4_scenario_ensemble', 'm5_neural_cem', 'm6_decision_focused']
M3 = METHODS[2]
P = 'CONTROL_BENCHMARK.planner.'
FAMILY = f'CONTROL_BENCHMARK.methods.{M3}.planner'
SAFETY = 'MPC.evaluation.action_model_screen.state_safety.out_of_band_policy'
SMART = {P+'explicit_warm_candidate': True, P+'cem_block_steps': 2,
         P+'stratified_first_actions': True}
ARMS = {
    'b1_reference': {},
    'energy3': {P+'energy_weight': 3.},
    'energy10': {P+'energy_weight': 10.},
    'quality_budget': {P+'quality_budget': True},
    'physical_transition': {P+'physical_transition_count': True},
    'assumed_dwell3': {P+'assumed_minimum_onoff_steps': 3},
    'assumed_dwell5': {P+'assumed_minimum_onoff_steps': 5},
    'explicit_warm': {P+'explicit_warm_candidate': True},
    'block2': {P+'cem_block_steps': 2},
    'stratified_first': {P+'stratified_first_actions': True},
    'smart_cem': SMART,
    'energy3_physical_smart': {
        **SMART, P+'energy_weight': 3., P+'physical_transition_count': True},
    'quality_physical_smart': {
        **SMART, P+'quality_budget': True, P+'physical_transition_count': True},
    'vent_energy3': {P+'power_weights': [3., 1.]},
    'separate_power_heads': {
        'CONTROL_BENCHMARK.model_parameters.separate_power_heads': True},
    'memory095': {'CONTROL_BENCHMARK.memory_decay': .95},
    'rollout5': {
        'CONTROL_BENCHMARK.model_parameters.rollout_training_steps': 5,
        'CONTROL_BENCHMARK.model_parameters.rollout_training_weight': .1},
    'vent_model_smart': {
        **SMART, 'CONTROL_BENCHMARK.model_parameters.separate_power_heads': True,
        'CONTROL_BENCHMARK.memory_decay': .95,
        'CONTROL_BENCHMARK.model_parameters.rollout_training_steps': 5,
        'CONTROL_BENCHMARK.model_parameters.rollout_training_weight': .1},
    'service_envelope': {'CAMPAIGN.service_envelope': True},
    'service_quality_smart': {
        **SMART, P+'quality_budget': True, 'CAMPAIGN.service_envelope': True},
}
CURRENT_ARMS = ['m3_original', 'm3_b1', 'm3_b1_cem128x5']
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                              'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                              'VECLIB_MAXIMUM_THREADS')}
PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'torch', 'pandas', 'omegaconf')


def runtime():
    return {'python': sys.version,
            'packages': {name: importlib.metadata.version(name) for name in PACKAGES}}


def configure_policy(cfg, arm, split):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.update(cfg, SAFETY, 'non_worsening', force_add=True)
    # Reference uses exactly the unchanged original CEM. The other arms are opt-in.
    if arm != 'b1_reference':
        OmegaConf.update(cfg, FAMILY, 'energy_categorical_cem', force_add=True)
    for key, value in ARMS[arm].items():
        OmegaConf.update(cfg, key, value, force_add=True)
    cfg.MPC.counterfactual_report_input_ablation.run_after_mpc = False
    cfg.CONTROL_BENCHMARK.resume_from = None
    OmegaConf.update(cfg, 'MPC.evaluation.policy_evaluation_split', split, force_add=True)
    OmegaConf.update(cfg, 'CAMPAIGN.capture_next_states', True, force_add=True)
    if OmegaConf.select(cfg, 'CAMPAIGN.service_envelope', default=False):
        # Explicit hypothetical sensitivity, NOT a recommendation or approved comfort standard.
        for c in cfg.MPC.state_constraints:
            if c.name == '실내온도(℃)':
                c.lower, c.upper = 19., 27.
            elif c.name == '실내습도(%)':
                c.lower, c.upper = 25., 75.
    return cfg


def prepare(root, source, predecessor):
    if root.exists():
        raise ValueError('New campaign root must not exist')
    origin = json.loads((source/'manifest.json').read_text())
    previous = json.loads((predecessor/'manifest.json').read_text())
    root.mkdir(parents=True)
    hashes = {}
    for path in (REPO/'src').rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.sh', '.yaml'):
            relative = path.relative_to(REPO/'src')
            target = root/'source_snapshot/src'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(relative)] = digest(target)
    inputs = origin['input_sources']
    for item in inputs:
        if digest(item['path']) != item['sha256']:
            raise ValueError(f'Input mismatch: {item["path"]}')
    jobs, cache = [], {}

    def add_job(kind, cohort, group, environment, arm, method, source_cfg, **extra):
        directory = root/kind/cohort/arm/group/environment/'seed_42'/method
        directory.mkdir(parents=True)
        cfg = OmegaConf.load(source_cfg)
        cfg.RESULT.savedir = str(root/'post_evaluation')
        if kind == 'policy':
            cfg = configure_policy(cfg, arm, cohort)
        elif kind == 'report_input':
            cfg.RESULT.savedir = str(directory/'report_input_outputs')
        OmegaConf.save(cfg, directory/'configs.yaml')
        job = {'kind': kind, 'cohort': cohort, 'group': group,
               'environment': environment, 'arm': arm, 'method': method, 'seed': 42,
               'relative_directory': str(directory.relative_to(root)),
               'config_sha256': digest(directory/'configs.yaml'),
               'source_config': str(source_cfg), 'source_config_sha256': digest(source_cfg),
               **extra}
        if kind == 'policy':
            key = (cohort, group, environment)
            if key not in cache:
                cache[key] = scope_audit(cfg)
            audit = {**cache[key], 'mpc_config_sha256': mpc_fingerprint(cfg)}
            if cohort == 'test':
                old_audit = json.loads((source_cfg.parent/'scope_audit.json').read_text())
                for field in ('train_end_exclusive', 'test_start', 'window_starts',
                              'expected_policy_steps', 'counterfactual_report_rows'):
                    if audit[field] != old_audit[field]:
                        raise ValueError(f'Original test scope changed: {group}/{environment}/{field}')
            elif any(start < audit['train_end_exclusive'] or
                     start+int(cfg.MPC.evaluation.closed_loop_steps) >= audit['test_start']
                     for start in audit['window_starts']):
                raise ValueError('Validation windows cross split boundaries')
            write_json(directory/'scope_audit.json', audit)
            job['expected_policy_steps'] = audit['expected_policy_steps']
        jobs.append(job)

    # All methods, including failed/incomplete MILP policies: prediction needs no MILP.
    for group, environment in CELLS:
        for method in METHODS:
            parent = source/group/environment/'seed_42'/method
            add_job('mape', 'original', group, environment, method, method,
                    parent/'configs.yaml', checkpoint_source=str(parent/'model'),
                    artifact_source=str(parent), reuse_checkpoint=False)
    prior_cases = {(c['group'], c['environment'], c['arm']): c for c in previous['cases']}
    for group, environment in CELLS:
        for arm in CURRENT_ARMS:
            case = prior_cases[group, environment, arm]
            parent = predecessor/case['relative_directory']
            add_job('mape', 'current', group, environment, arm, M3,
                    parent/'configs.yaml', checkpoint_source=str(parent/'model'),
                    artifact_source=str(parent))
    for group, environment in CELLS:
        parent = source/group/environment/'seed_42'/M3
        add_job('report_input', 'shared_report_model', group, environment,
                'shared_report_inputs', 'shared_report_instrument', parent/'configs.yaml')
    # Every predeclared arm goes through validation, then test; no test-based pruning.
    for split in ('validation', 'test'):
        for group, environment in CELLS:
            for arm in ARMS:
                parent = source/group/environment/'seed_42'/M3
                add_job('policy', split, group, environment, arm, M3,
                        parent/'configs.yaml')
    manifest = {
        'status': 'prepared', 'created_at': now(), 'source_root': str(source),
        'repository_root': str(REPO),
        'predecessor_root': str(predecessor), 'source_sha256': hashes,
        'input_sources': inputs, 'jobs': jobs, 'case_count': len(jobs),
        'workers': 8, 'logical_cpu_ids': list(range(10)), 'nice': 19,
        'phase_order': ['mape', 'report_input', 'policy/validation', 'policy/test'],
        'counts': {'pending': len(jobs)}, 'active_pids': [],
        'arms': ARMS, 'seed': 42,
        'runtime': runtime(),
        'minimum_available_memory_gb': 24, 'minimum_free_disk_gb': 32,
        'interpretation': (
            'Predeclared exploratory surrogate ablations; test periods already observed. '
            'Original m1–m6 prediction models refit in this run; current3 arms reuse verified checkpoints. '
            'Validation can be used for early stopping and evaluator fitting. '
            'Concat is not pooled with single versions. No real equipment claim.'),
        'blocked_experiments': {
            'verified_occupancy_schedule': 'No measured occupancy/schedule channel supplied.',
            'occupancy_conditioned_emissions': 'Current dynamics has no occupancy intervention input.',
            'hardware_minimum_on_off': 'No manufacturer dwell limits or initial lockout history.',
            'new_period_livingroom_validation': 'No newly supplied untouched period/device.',
        },
        'hypothetical_scenarios': {
            'service_envelope': 'Temperature19–27C/humidity25–75%; air-quality bounds unchanged. '
                                'Not user-approved operating limits; simulation only.',
            'physical_transition': 'Ignore ancillary changes involving OFF, keep raw switch cost. '
                                   'A weaker nominal categorical cap, not verified hardware semantics.',
            'assumed_dwell3_and5': '3/5-minute minimum power on/off, lockout-free window bootstrap; '
                                  'simulation sensitivity, not verified manufacturer limits.',
        },
    }
    write_json(root/'manifest.json', manifest)
    return manifest


def validate(root, manifest):
    if manifest['runtime'] != runtime():
        raise ValueError('Runtime changed; use a new root')
    for relative, expected in manifest['source_sha256'].items():
        if digest(root/'source_snapshot/src'/relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')
    for item in manifest['input_sources']:
        if digest(item['path']) != item['sha256']:
            raise ValueError(f'Input changed: {item["path"]}')
    for job in manifest['jobs']:
        if digest(root/job['relative_directory']/'configs.yaml') != job['config_sha256']:
            raise ValueError(f'Frozen config changed: {job["relative_directory"]}')
        if digest(job['source_config']) != job['source_config_sha256']:
            raise ValueError('Source config changed')


def split_data(cfg):
    from controllers.action_crossfit import align_boundaries_to_control_episodes
    from data_provider.control_data import load_control_data
    data = load_control_data(cfg)
    train_end, test_start = data.split_indices(cfg.DATASET.split_rate)
    if cfg.MPC.evaluation.action_options.episode_group_split.enabled:
        try:
            train_end, test_start = align_boundaries_to_control_episodes(data, [train_end, test_start])
        except ValueError:
            pass  # Same explicit nominal fallback as the original pipeline.
    return data, train_end, test_start


def model_for_mape(cfg, job, directory, data, train_end, test_start):
    from controllers.full_control_backend import FullControlBackend
    from dynamics.affine import fit_affine_models
    from models.control import ContextSpec, load_control_model
    spec = ContextSpec(data, cfg.CONTROL_BENCHMARK.memory_decay)
    contexts = spec.contexts(data)
    train = spec.batch(data, contexts, 0, train_end)
    source = Path(job['checkpoint_source'])
    evidence = {'checkpoint_source': str(source), 'mode': 'refit_full_original_train'}
    if job.get('reuse_checkpoint', True) and (source/'model.joblib').exists():
        # Only locally trusted project artifacts are accepted.
        expected_hashes = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
        destination = directory/'model'
        if source.resolve() != destination.resolve():
            shutil.copytree(source, destination)
        model = load_control_model(destination)
        if not np.array_equal(model.fit_rows, train.rows):
            raise ValueError('Checkpoint training rows differ from chronological full train')
        if (tuple(model.spec.feature_names) != tuple(spec.feature_names)
                or tuple(model.spec.state_names) != tuple(spec.state_names)
                or tuple(model.spec.power_names) != tuple(spec.power_names)
                or tuple(model.spec.disturbance_names) != tuple(spec.disturbance_names)
                or model.spec.decay != spec.decay):
            raise ValueError('Checkpoint feature/context contract differs')
        expected_family = cfg.CONTROL_BENCHMARK.methods[job['method']].model
        if model.family != expected_family or model.seed != int(cfg.DEFAULT.seed):
            raise ValueError('Checkpoint family or seed differs')
        if expected_hashes != {p.name: digest(p) for p in source.iterdir() if p.is_file()}:
            raise ValueError('Checkpoint source changed while copying')
        evidence.update(mode='copied_verified_checkpoint', checkpoint_sha256=expected_hashes)
    else:
        sm, pm = fit_affine_models(
            *data.model_arrays(0, train_end-1),
            state_names=data.state_names, control_feature_names=data.encoder.feature_names,
            disturbance_names=data.disturbance_names, power_names=data.power_names,
            state_alpha=cfg.MPC.affine_models.state_ridge_alpha,
            power_alpha=cfg.MPC.affine_models.power_ridge_alpha)
        backend = FullControlBackend(cfg, job['method'], directory)
        backend.attach(data, train_end, test_start, sm, pm)
        model = backend.model
    write_json(directory/'checkpoint_provenance.json', evidence)
    return model, spec, contexts


def artifact_mape(cfg, source):
    """Report factual 값과 기록 action이 정확히 일치한 경우만 정답과 비교한다. 반사실 오차는 정확도로 쓰지 않는다."""
    from controllers.control_mape_audit import power_metric_rows
    from data_provider.control_data import _category_token
    file = source/'mpc_counterfactual_total_power_report_predictions.csv'
    result = []
    if file.exists():
        frame = pd.read_csv(file)
        names = list(cfg.MPC.power_variables)
        columns_a = ['recorded_actual_'+n for n in names]
        columns_p = ['recorded_action_report_predicted_'+n for n in names]
        columns_m = ['mpc_action_report_predicted_'+n for n in names]
        if set(columns_a+columns_p+columns_m) <= set(frame):
            actual = frame[columns_a].to_numpy(float)
            factual = frame[columns_p].to_numpy(float)
            exact = np.ones(len(frame), dtype=bool)
            controls = [str(c.name) for c in cfg.MPC.control_variables]
            for name in controls:
                recorded = frame['recorded_control_'+name].map(_category_token).to_numpy()
                selected = frame['mpc_control_'+name].map(_category_token).to_numpy()
                exact &= (recorded == selected) & (recorded != '')
            for floor in (.1, 5.):
                result += power_metric_rows(
                    actual, factual, names, protocol='report_factual_recorded_action',
                    split='artifact_declared_scope', floor_w=floor, target_percent=mape_target(cfg))
                rows = power_metric_rows(
                    actual[exact], frame.loc[exact, columns_m].to_numpy(float),
                    names, protocol='report_MPC_exact_recorded_action_matches_only',
                    split='artifact_declared_scope', floor_w=floor, target_percent=mape_target(cfg))
                for row in rows:
                    row['exact_action_match_count'] = int(exact.sum())
                    row['action_match_coverage_percent'] = 100*float(exact.mean()) if len(exact) else 0.
                result += rows
    gates = {}
    for name in ('fixed_action_candidate_pool_evaluation.json',
                 'mpc_counterfactual_total_power_report_evaluation.json',
                 'action_selection_gate.json', 'kpi_coverage_summary.json',
                 'model_validation.json', 'component_kpi_gate.json', 'vent_isolated_kpi.json'):
        path = source/name
        gates[name] = json.loads(path.read_text()) if path.exists() else {
            'available': False, 'reason': 'source artifact not generated; not assumed pass'}
    gates['_source_artifact_sha256'] = {
        p.name: digest(p) for p in [file, *[source/name for name in gates]]
        if p.is_file()}
    return result, gates


def run_mape(cfg, job, directory):
    from controllers.control_mape_audit import power_metric_rows
    from controllers.pipeline import _select_evaluation_windows
    data, train_end, test_start = split_data(cfg)
    model, spec, contexts = model_for_mape(cfg, job, directory, data, train_end, test_start)
    rows, arrays = [], {}
    for split, begin, end in (('validation', train_end, test_start),
                              ('test', test_start, len(data.states))):
        batch = spec.batch(data, contexts, begin, end)
        xs, powers = [], []
        for i in range(0, len(batch.rows), 4096):
            part = batch.take(slice(i, i+4096))
            x, p = model.predict(part.context, part.controls, part.disturbances)
            xs.append(x)
            powers.append(p)
        predicted_x, predicted = np.concatenate(xs), np.concatenate(powers)
        for floor in (.1, 5.):
            rows += power_metric_rows(
                batch.powers, predicted, data.power_names,
                protocol='controller_factual_one_step', split=split, floor_w=floor,
                target_percent=mape_target(cfg))
        arrays[split+'_rows'] = batch.rows
        arrays[split+'_actual_power'] = batch.powers
        arrays[split+'_predicted_power'] = predicted
        arrays[split+'_actual_state'] = batch.next_states
        arrays[split+'_predicted_state'] = predicted_x
        starts = _select_evaluation_windows(
            data, begin, 60, 60, 0, 60, 'persistence', end_exclusive=end)
        # One vectorized model call per lead; reset to factual context per window.
        ctx = contexts[starts].copy()
        all_actual, all_pred, all_rows = [], [], []
        for lead in range(1, 61):
            indices = np.asarray(starts)+lead-1
            x, p = model.predict(ctx, data.controls[indices], data.disturbances[indices])
            all_actual.append(data.powers[indices])
            all_pred.append(p)
            all_rows.append(indices)
            if lead in (1, 10, 30, 60):
                for floor in (.1, 5.):
                    rows += power_metric_rows(
                        data.powers[indices], p, data.power_names,
                        protocol=f'controller_recorded_recursive_lead_{lead}',
                        split=split, floor_w=floor, target_percent=mape_target(cfg))
            ctx = spec.advance(ctx, data.controls[indices], x)
        actual, predicted = np.concatenate(all_actual), np.concatenate(all_pred)
        for floor in (.1, 5.):
            rows += power_metric_rows(
                actual, predicted, data.power_names,
                protocol='controller_recorded_recursive_all60', split=split, floor_w=floor,
                target_percent=mape_target(cfg))
        arrays[split+'_rollout_rows_lead_major'] = np.stack(all_rows)
        arrays[split+'_rollout_actual_power'] = actual
        arrays[split+'_rollout_predicted_power'] = predicted
        arrays[split+'_window_starts'] = np.asarray(starts)
    artifact_rows, gates = artifact_mape(cfg, Path(job['artifact_source']))
    rows += artifact_rows
    np.savez_compressed(directory/'predictions.npz', **arrays)
    pd.DataFrame(rows).to_csv(directory/'mape_metrics.csv', index=False)
    write_json(directory/'source_formal_gates.json', gates)
    reference = Path(job['artifact_source'])
    if ((reference/'mpc_evaluation.json').exists()
            and (reference/'mpc_counterfactual_total_power_report_predictions.csv').exists()):
        from runners.counterfactual_report_input_ablation_runner import run_counterfactual_report_input_ablation
        report_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        report_cfg.RESULT.savedir = str(directory/'counterfactual_report_input_outputs')
        report_cfg.MPC.counterfactual_report_input_ablation.reference_mpc_result_dir = str(reference)
        result = run_counterfactual_report_input_ablation(report_cfg)
        write_json(directory/'counterfactual_report_input_run.json', result)
        counterfactual_input_status = 'completed_with_frozen_reference_actions_no_MILP'
    else:
        counterfactual_input_status = 'not_available_source_policy_has_no_final_action_artifacts'
    write_json(directory/'mape_protocol.json', {
        'train_end_exclusive': train_end, 'test_start': test_start,
        'factual_power_truth': 'measured recorded power with recorded controls',
        'recursive_power_truth': 'recorded controls/disturbances; predicted state feedback',
        'MAPE': '100*mean(abs(actual-predicted)/abs(actual)) over abs(actual)>floor',
        'floors_w': [.1, 5.], 'target_percent': mape_target(cfg), 'minimum_samples': 30,
        'selection': 'none; all methods/settings included regardless of error or gates',
        'counterfactual_truth_available': False,
        'counterfactual_report_input_ablation': counterfactual_input_status,
        'counterfactual_report_values_are_not_measured_accuracy': True,
        'model_quality_gates_are_separate': True,
    })


def policy_supplement(cfg, directory, standard_constraints):
    """변경 서비스 범위의 유리함과 원래 범위의 상태 위반을 함께 남긴다."""
    from controllers.energy_cem import violation_magnitudes
    frame = pd.read_csv(directory/'mpc_closed_loop_steps.csv')
    constraints = OmegaConf.to_container(cfg.MPC.state_constraints, resolve=True)
    standard = OmegaConf.to_container(standard_constraints, resolve=True)
    if [c['name'] for c in standard] != [c['name'] for c in constraints]:
        raise ValueError('Source and scenario state names differ')
    states = frame[['next_state_'+c['name'] for c in standard]].to_numpy(float)
    results = {}
    for label, bands in [('scenario_bands', constraints), ('standard_bands', standard)]:
        magnitudes = violation_magnitudes(states, bands)
        results[label] = {
            c['name']: {'violation_magnitude_minutes': float(magnitudes[:, j].sum()),
                        'maximum_violation': float(magnitudes[:, j].max()),
                        'violating_steps': int((magnitudes[:, j] > 0).sum())}
            for j, c in enumerate(bands)}
    write_json(directory/'state_bands_audit.json', results)
    from controllers.energy_cem import physical_changes
    data, _, _ = split_data(cfg)
    audit_path = directory/'scope_audit.json'
    groups = list(frame.groupby('window_timestamp', sort=False))
    if audit_path.exists():
        starts = json.loads(audit_path.read_text())['window_starts']
    else:
        starts = []
        for timestamp, _ in groups:
            matches = np.flatnonzero(pd.to_datetime(data.timestamps).to_numpy()
                                     == np.datetime64(pd.Timestamp(timestamp)))
            if len(matches) != 1:
                raise ValueError('Ambiguous window bootstrap without scope audit')
            starts.append(int(matches[0]))
    if len(starts) != len(groups):
        raise ValueError('Window count differs from scope audit')
    switching = []
    raw_violations = 0
    for start, (timestamp, group) in zip(starts, groups):
        if pd.Timestamp(timestamp) != pd.Timestamp(data.timestamps.iloc[start]):
            raise ValueError('Window bootstrap timestamp differs')
        z = data.encoder.transform(pd.DataFrame({
            name: group['action_'+name].to_numpy() for name in data.encoder.groups}))
        prior = data.controls[start-1] if start else data.encoder.encode_action()
        previous = np.vstack([prior, z[:-1]])
        raw = np.abs(z-previous).sum(1)/2
        effective = physical_changes(data.encoder, z, previous)
        cap = int(cfg.MPC.objective.maximum_changed_control_variables_per_step)
        raw_violations += int((raw > cap).sum())
        switching.append({
            'timestamp': str(timestamp), 'raw_categorical_changes': float(raw.sum()),
            'off_semantic_changes': float(effective.sum()),
            'switching_steps': int((raw > 0).sum()),
            'switching_cost': float(2*cfg.MPC.objective.switch_penalty*raw.sum())})
    evaluation = json.loads((directory/'mpc_evaluation.json').read_text())
    if sum(r['switching_steps'] for r in switching) != evaluation[
            'policy_evaluation']['solver']['control_switch_count']:
        raise ValueError('Applied switching reconstruction differs from pipeline')
    write_json(directory/'switching_audit.json', {
        'windows': switching, 'raw_change_cap_exceedance_steps': raw_violations,
        'physical_semantics_are_hypothetical': True,
        'switching_cost': sum(r['switching_cost'] for r in switching),
        'raw_categorical_changes': sum(r['raw_categorical_changes'] for r in switching)})
    artifact_rows, gates = artifact_mape(cfg, directory)
    pd.DataFrame(artifact_rows).to_csv(directory/'report_mape_audit.csv', index=False)
    write_json(directory/'report_formal_gates_audit.json', gates)


def worker(root, index):
    manifest = json.loads((root/'manifest.json').read_text())
    job = manifest['jobs'][index]
    directory = root/job['relative_directory']
    cfg = OmegaConf.load(directory/'configs.yaml')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    state = {'status': 'running', 'pid': os.getpid(), 'started_at': now(),
             'kind': job['kind'], 'phase': job['cohort']}
    write_json(directory/'campaign_status.json', state)
    try:
        if digest(directory/'configs.yaml') != job['config_sha256']:
            raise ValueError('Frozen job config changed')
        if job['kind'] == 'mape':
            run_mape(cfg, job, directory)
        elif job['kind'] == 'report_input':
            from runners.report_model_input_ablation_runner import run_report_model_input_ablation
            result = run_report_model_input_ablation(cfg)
            write_json(directory/'report_input_run.json', result)
            write_json(directory/'report_input_artifacts.json', {
                'summaries': [
                    str(p.relative_to(directory))
                    for p in (directory/'report_input_outputs').rglob('*summary.csv')],
                'claim': 'shared report instrument; not six different control policies'})
        else:
            from runners.full_control_benchmark_runner import full_worker
            full_worker(str(directory/'configs.yaml'), job['method'])
            standard = OmegaConf.load(job['source_config']).MPC.state_constraints
            policy_supplement(cfg, directory, standard)
            run_mape(cfg, {**job, 'checkpoint_source': str(directory/'model'),
                           'artifact_source': str(directory)}, directory)
            compress_diagnostics(directory)
        state.update(status='completed', finished_at=now(), wall_seconds=time.monotonic()-started)
        write_json(directory/'campaign_status.json', state)
        return 0
    except BaseException as error:
        state.update(status='failed', finished_at=now(), error=str(error),
                     wall_seconds=time.monotonic()-started)
        write_json(directory/'campaign_status.json', state)
        (directory/'campaign_error.txt').write_text(traceback.format_exc())
        raise


def compress_diagnostics(directory):
    """새 campaign의 자체 산출물만 gzip으로 무손실 보존한다."""
    evidence = {}
    for name in ('mpc_action_candidate_diagnostics.csv',):
        source = directory/name
        if not source.exists():
            continue
        target = source.with_suffix(source.suffix+'.gz')
        if target.exists():
            raise ValueError('Compression destination already exists')
        original_hash, original_size = digest(source), source.stat().st_size
        temporary = target.with_suffix(target.suffix+'.tmp')
        with source.open('rb') as inp, gzip.open(temporary, 'wb', compresslevel=3) as out:
            shutil.copyfileobj(inp, out, length=1024*1024)
        recovered = hashlib.sha256()
        with gzip.open(temporary, 'rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                recovered.update(block)
        if recovered.hexdigest() != original_hash or digest(source) != original_hash:
            raise ValueError('Lossless compression verification failed; raw artifact preserved')
        temporary.replace(target)
        evidence[name] = {
            'raw_sha256': original_hash, 'raw_bytes': original_size,
            'gzip_sha256': digest(target), 'gzip_bytes': target.stat().st_size,
            'recovered_sha256_verified': True}
        source.unlink()
    write_json(directory/'lossless_compression.json', evidence)


def status(root, job):
    file = root/job['relative_directory']/(
        'campaign_status.json' if 'kind' in job else 'status.json')
    return json.loads(file.read_text()) if file.exists() else {'status': 'pending'}


def live_predecessor_pids(predecessor):
    manifest = json.loads((predecessor/'manifest.json').read_text())
    pids = set(manifest.get('active_pids', []))
    # Includes reparented/adopted workers and individual statuses.
    for case in manifest['cases']:
        state = status(predecessor, case)
        if state.get('status') == 'running' and state.get('pid'):
            pids.add(state['pid'])
    alive = []
    for pid in pids:
        try:
            proc = Path(f'/proc/{pid}')
            command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if str(predecessor) in command and (proc/'stat').read_text().split()[2] != 'Z':
                alive.append(pid)
        except FileNotFoundError:
            pass
    return alive, manifest.get('status')


def aggregate(root, manifest):
    metrics, policies, report_inputs = [], [], []
    for job in manifest['jobs']:
        directory = root/job['relative_directory']
        if status(root, job)['status'] != 'completed':
            continue
        identity = {k: job[k] for k in ('kind', 'cohort', 'group', 'environment', 'arm', 'method')}
        for name in ('mape_metrics.csv', 'report_mape_audit.csv'):
            if (directory/name).exists():
                try:
                    frame = pd.read_csv(directory/name)
                except pd.errors.EmptyDataError:
                    continue
                for k, value in identity.items():
                    frame[k] = value
                metrics.append(frame)
        if job['kind'] == 'policy':
            evaluation = json.loads((directory/'mpc_evaluation.json').read_text())
            policies.append({**identity, 'evaluation': evaluation,
                             'relative_directory': job['relative_directory']})
        index_file = directory/'report_input_artifacts.json'
        if index_file.exists():
            for relative in json.loads(index_file.read_text())['summaries']:
                frame = pd.read_csv(directory/relative)
                for k, value in identity.items():
                    frame[k] = value
                frame['artifact_name'] = Path(relative).name
                frame['claim_boundary'] = 'shared_report_instrument_prediction_accuracy'
                report_inputs.append(frame)
        cf = directory/'counterfactual_report_input_run.json'
        if cf.exists():
            result = json.loads(cf.read_text())
            path = Path(result['result_directory'])/'counterfactual_report_input_ablation_summary.csv'
            frame = pd.read_csv(path)
            for k, value in identity.items():
                frame[k] = value
            frame['artifact_name'] = path.name
            frame['claim_boundary'] = (
                'frozen_action_report_sensitivity;changed_actions_have_no_measured_truth')
            report_inputs.append(frame)
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    if metrics:
        combined = pd.concat(metrics, ignore_index=True)
        temporary = output/'mape_metrics.csv.tmp'
        combined.to_csv(temporary, index=False)
        temporary.replace(output/'mape_metrics.csv')
    write_json(output/'policy_results.json', policies)
    if report_inputs:
        pd.concat(report_inputs, ignore_index=True).to_csv(output/'report_input_metrics.csv', index=False)
    from reporting.savings_campaign import write_campaign_report
    write_campaign_report(root, manifest)


def supervise(root):
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((root/'manifest.json').read_text())
        validate(root, manifest)
        predecessor = Path(manifest['predecessor_root'])
        active, pending = {}, []
        for i, job in enumerate(manifest['jobs']):
            state = status(root, job)
            if state['status'] == 'completed':
                continue
            if state['status'] == 'running' and state.get('pid') and Path(f'/proc/{state["pid"]}').exists():
                raise RuntimeError('An existing campaign worker is alive; refusing duplicate supervisor')
            directory = root/job['relative_directory']
            if any(p.name not in ('configs.yaml', 'scope_audit.json') for p in directory.iterdir()):
                archive = directory.with_name(directory.name+'_attempt_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                directory.rename(archive)
                directory.mkdir()
                for name in ('configs.yaml', 'scope_audit.json'):
                    if (archive/name).exists():
                        shutil.copy2(archive/name, directory/name)
            pending.append(i)
        stopping = False

        def stop(signum, frame):
            nonlocal stopping
            stopping = True
            for proc, _ in active.values():
                proc.terminate()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        manifest.update(status='running', supervisor_pid=os.getpid(), started_at=now())
        env = dict(os.environ, **THREAD_ENV, PYTHONPATH=str(root/'source_snapshot/src'),
                   PYTHONUNBUFFERED='1', MPC_SOLVER_HARD_TIMEOUT_SECONDS='900')
        last_completed = -1
        while pending or active:
            old_pids, old_status = live_predecessor_pids(predecessor)
            # Existing matrix may still start its final pending case. Wait for terminal
            # status instead of racing it for a slot; preserve its fixed worker setup.
            predecessor_finished = old_status in ('completed', 'completed_with_failures', 'cancelled')
            manifest['predecessor_active_pids'] = old_pids
            manifest['predecessor_status'] = old_status
            manifest['waiting_for_predecessor'] = not predecessor_finished or bool(old_pids)
            manifest['available_memory_gb'] = available_memory_gb()
            manifest['free_disk_gb'] = shutil.disk_usage(root).free/1024**3
            manifest['waiting_for_resources'] = (
                (manifest['available_memory_gb'] or 0) < manifest['minimum_available_memory_gb']
                or manifest['free_disk_gb'] < manifest['minimum_free_disk_gb'])
            if (pending and not stopping and not manifest['waiting_for_predecessor']
                    and len(active) < 8 and not manifest['waiting_for_resources']):
                # Do not start test until all validation jobs have finished.
                index = pending[0]
                next_job = manifest['jobs'][index]
                phase = (next_job['kind'], next_job['cohort'])
                active_phases = {(manifest['jobs'][i]['kind'], manifest['jobs'][i]['cohort'])
                                 for i in active}
                if not active_phases or active_phases == {phase}:
                    pending.pop(0)
                    directory = root/next_job['relative_directory']
                    handle = (directory/'process.log').open('a')
                    proc = subprocess.Popen([
                        sys.executable, str(root/'source_snapshot/src/scripts/control/savings_campaign.py'),
                        '--root', str(root), '--worker-index', str(index)],
                        cwd=manifest['repository_root'], env=env,
                        stdout=handle, stderr=subprocess.STDOUT)
                    active[index] = (proc, handle)
                    print(now(), 'START', index, proc.pid, next_job['relative_directory'], flush=True)
            for index, (proc, handle) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                handle.close()
                job = manifest['jobs'][index]
                current = status(root, job)
                if code or current['status'] != 'completed':
                    write_json(root/job['relative_directory']/'campaign_status.json', {
                        **current, 'status': 'failed', 'exit_code': code, 'finished_at': now()})
                print(now(), 'EXIT', index, code, flush=True)
                del active[index]
            counts = Counter(status(root, job)['status'] for job in manifest['jobs'])
            manifest.update(updated_at=now(), counts=dict(counts),
                            active_pids=[p.pid for p, _ in active.values()],
                            pending_jobs=len(pending))
            write_json(root/'manifest.json', manifest)
            if counts['completed'] != last_completed:
                aggregate(root, manifest)
                last_completed = counts['completed']
            if stopping and not active:
                break
            time.sleep(5)
        validate(root, manifest)
        counts = Counter(status(root, job)['status'] for job in manifest['jobs'])
        manifest.update(status='cancelled' if stopping else
                        'completed_with_failures' if counts['failed'] else 'completed',
                        finished_at=now(), counts=dict(counts), supervisor_pid=None, active_pids=[])
        write_json(root/'manifest.json', manifest)
        aggregate(root, manifest)
        return 0 if manifest['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--predecessor-root', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker-index', type=int)
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        parser.error('All campaign runs, including preparation/workers, require tmux')
    if not set(range(10)) <= os.sched_getaffinity(0):
        parser.error('CPUs0–9 are unavailable')
    os.sched_setaffinity(0, set(range(10)))
    os.nice(19-os.getpriority(os.PRIO_PROCESS, 0))
    os.environ.update(THREAD_ENV)
    root = args.root.resolve()
    if args.worker_index is not None:
        return worker(root, args.worker_index)
    if not (root/'manifest.json').exists():
        if not args.source_root or not args.predecessor_root:
            parser.error('New campaign requires source and predecessor roots')
        prepare(root, args.source_root.resolve(), args.predecessor_root.resolve())
    if args.prepare_only:
        return 0
    return supervise(root)


if __name__ == '__main__':
    sys.exit(main())
