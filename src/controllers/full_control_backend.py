"""Adapt independently trained models to the unchanged full MPC evaluation."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from controllers.control_benchmark import create_control_planner
from controllers.evaluation import AffinePlantAdapter
from controllers.mixed_integer_mpc import MPCSolution
from dynamics.evaluator_base import TransitionPredictions
from models.control import ContextSpec, create_control_model, load_control_model


class ModelProjection:
    """Legacy state/power API; online predictions use the applied causal context."""

    def __init__(self, backend, component, affine_reference):
        self.backend, self.component = backend, component
        self.affine_reference = affine_reference
        self.state_names = backend.data.state_names
        self.power_names = backend.data.power_names
        self.control_feature_names = backend.data.encoder.feature_names
        self.disturbance_names = backend.data.disturbance_names

    def _predict(self, states, controls, disturbances):
        states, controls, disturbances = map(
            np.atleast_2d, (states, controls, disturbances),
        )
        count = max(len(states), len(controls), len(disturbances))
        states = np.broadcast_to(states, (count, self.backend.spec.nx))
        if self.backend.online_context is None:
            raise RuntimeError('History model prediction requires explicit causal context')
        ctx = np.repeat(self.backend.online_context[None], count, axis=0)
        ctx[:, :self.backend.spec.nx] = states
        return self.backend.model.predict(
            ctx, np.broadcast_to(controls, (count, self.backend.spec.nz)),
            np.broadcast_to(disturbances, (count, self.backend.spec.nd)),
        )

    def predict(self, states, controls, disturbances):
        return self._predict(states, controls, disturbances)[0]

    def predict_components(self, states, controls, disturbances):
        return self._predict(states, controls, disturbances)[1]

    def predict_components_for_rows(self, data, rows):
        if data is not self.backend.data:
            raise ValueError('Calibration rows must refer to the attached dataset')
        return self.backend.model.predict(
            self.backend.contexts[rows], data.controls[rows], data.disturbances[rows],
        )[1]

    def to_dict(self):
        return {
            'model_type': self.backend.model.family,
            'component': self.component,
            'checkpoint': 'model/model.joblib',
            'is_affine_matrix_serialization': False,
        }


class HistoryPlant:
    """Factual batch and independent recursive sessions with causal bootstrap."""

    required_history_steps = 0

    def __init__(self, backend):
        self.backend = backend
        self.state_names = backend.data.state_names
        self.power_names = backend.data.power_names
        self.rows_by_timestamp = {}
        for row, timestamp in enumerate(backend.data.timestamps):
            self.rows_by_timestamp.setdefault(pd.Timestamp(timestamp).value, []).append(row)

    def predict_transitions(self, data, start, end):
        rows = data.valid_transition_rows(np.arange(start, end, dtype=int))
        x, p = self.backend.model.predict(
            self.backend.contexts[rows], data.controls[rows], data.disturbances[rows],
        )
        return TransitionPredictions(data.states[rows+1], x, data.powers[rows], p, rows)

    def start_rollout(self, history=None):
        del history
        plant = self

        class Session:
            context = None

            def step(self, state, control, disturbance, *, timestamp=None):
                if self.context is None:
                    if timestamp is None:
                        raise ValueError('History rollout must bootstrap at an explicit timestamp')
                    candidates = plant.rows_by_timestamp.get(pd.Timestamp(timestamp).value, [])
                    candidates = [r for r in candidates if np.array_equal(
                        plant.backend.data.states[r], np.asarray(state),
                    )]
                    if len(candidates) != 1:
                        raise ValueError('Ambiguous/missing causal rollout bootstrap row')
                    self.context = plant.backend.contexts[candidates[0]].copy()
                x, p = plant.backend.model.predict(self.context, control, disturbance)
                self.context = plant.backend.spec.advance(self.context, control, x[0])
                return x[0], p[0]

        return Session()


class FullControlBackend:
    """Only the model/planner changes; split, screen, metrics and gates are original."""

    def __init__(self, cfg, method, result_dir):
        self.cfg, self.method, self.result_dir = cfg, method, Path(result_dir)
        self.online_context = None

    def attach(self, data, train_end, test_start, state_model, power_model):
        self.data = data
        self.spec = ContextSpec(data, self.cfg.CONTROL_BENCHMARK.memory_decay)
        self.contexts = self.spec.contexts(data)
        self.train = self.spec.batch(data, self.contexts, 0, train_end)
        validation = self.spec.batch(data, self.contexts, train_end, test_start)
        settings = OmegaConf.to_container(
            self.cfg.CONTROL_BENCHMARK.methods[self.method], resolve=True,
        )
        self.settings = settings
        parameters = {
            **OmegaConf.to_container(self.cfg.CONTROL_BENCHMARK.model_parameters, resolve=True),
            **settings.get('parameters', {}),
        }
        self.model = create_control_model(settings['model'], self.spec, parameters,
                                          int(self.cfg.DEFAULT.seed))
        started = perf_counter()
        if self.model.family == 'affine':
            # Reuse the exact original fit, including separate configured alphas.
            self.model.state_model, self.model.power_model = state_model, power_model
            self.model.fit_rows = self.train.rows.copy()
        else:
            if self.model.family == 'decision_focused':
                from controllers.pipeline import _training_control_support
                ev = self.cfg.MPC.evaluation
                repeated = ev.action_options.repeated_joint_actions
                actions, _, _ = _training_control_support(
                    data, train_end, self.cfg.MPC.logical_rules,
                    ev.restrict_to_training_control_support,
                    repeated_action_filter_enabled=repeated.enabled,
                    minimum_repeated_samples=repeated.minimum_training_samples,
                    minimum_repeated_episodes=repeated.minimum_training_episodes,
                )
                self.model.decision_actions = actions
                self.model.decision_constraints = OmegaConf.to_container(
                    self.cfg.MPC.state_constraints, resolve=True,
                )
            self.model.fit(self.train, validation)
        self.fit_seconds = perf_counter()-started
        self.model.save(self.result_dir/'model')
        self.model = load_control_model(self.result_dir/'model')
        if not np.array_equal(self.model.fit_rows, self.train.rows):
            raise AssertionError('Reloaded checkpoint training rows changed')
        if self.model.family == 'affine':
            return (self.model.state_model, self.model.power_model,
                    AffinePlantAdapter(self.model.state_model, self.model.power_model))
        self.state_projection = ModelProjection(self, 'state', state_model)
        self.power_projection = ModelProjection(self, 'power', power_model)
        return self.state_projection, self.power_projection, HistoryPlant(self)

    def wrap_controller(self, original, actions):
        self.actions = np.asarray(actions)
        if self.model.family == 'affine':
            return original
        if original.action_history_enabled:
            raise ValueError('New planners do not implement legacy history budgets; refusing silent omission')
        options = {
            **OmegaConf.to_container(self.cfg.CONTROL_BENCHMARK.planner, resolve=True),
            'prediction_horizon': original.prediction_horizon,
            'control_horizon': original.control_horizon,
            'switch_penalty': float(self.cfg.MPC.objective.switch_penalty),
            'maximum_changed_variables': original.maximum_changed_control_variables_per_step,
            'time_limit_seconds': float(self.cfg.MPC.solver.time_limit_seconds),
            'mip_relative_gap': float(self.cfg.MPC.solver.mip_relative_gap),
            'seed': int(self.cfg.DEFAULT.seed),
        }
        planner = create_control_planner(
            self.settings['planner'], self.model, self.actions,
            OmegaConf.to_container(self.cfg.MPC.state_constraints, resolve=True),
            self.cfg.MPC.logical_rules, options, self.train.context,
        )
        return ControllerAdapter(self, original, planner)

    def metadata(self):
        return {
            'method': self.method, 'model_family': self.model.family,
            'planner_family': self.settings['planner'],
            'training_transition_count': len(self.train.rows),
            'fit_seconds_excluding_original_affine_reference_fit': self.fit_seconds,
            'checkpoint_reloaded_before_evaluation': True,
            'evaluation_pipeline': 'controllers.pipeline.run_mpc_pipeline',
            'original_mpc_configuration_preserved': True,
            'effective_planner_parameters': {
                **OmegaConf.to_container(self.cfg.CONTROL_BENCHMARK.planner, resolve=True),
                'prediction_horizon': int(self.cfg.MPC.prediction_horizon),
                'control_horizon': int(self.cfg.MPC.control_horizon),
                'time_limit_seconds': float(self.cfg.MPC.solver.time_limit_seconds),
                'mip_relative_gap': float(self.cfg.MPC.solver.mip_relative_gap),
                'switch_penalty': float(self.cfg.MPC.objective.switch_penalty),
                'maximum_changed_variables': self.cfg.MPC.objective.maximum_changed_control_variables_per_step,
            },
            'ignored_pilot_fields': (
                'CONTROL_BENCHMARK evaluation_windows, closed_loop_steps, action_trust, '
                'maximum_training_samples, resume_from; common planner values come from MPC'
            ),
            'approximation_disclosure': {
                'mode_matrix_strategy': str(self.cfg.CONTROL_BENCHMARK.planner.mode_matrix_strategy),
                'physics_residual_is_not_full_PCNN': True,
                'decision_focused_is_proxy_not_exact_MIP_differentiation': True,
                'CEM_has_no_global_optimality_certificate': True,
            },
        }

    def window_completed(self, number, total, result, records):
        from runners.full_control_benchmark_runner import write_json
        write_json(self.result_dir/'policy_progress.json', {
            'completed_windows': number, 'total_windows': total,
            'last_window_timestamp': result['timestamp'],
            'last_window_energy_wh': result['policies']['mpc']['energy_wh'],
        })
        path = self.result_dir/'mpc_closed_loop_steps.partial.csv'
        pd.DataFrame([{'window_timestamp': result['timestamp'], **r} for r in records]).to_csv(
            path, index=False, mode='w' if number == 1 else 'a', header=number == 1,
        )


class ControllerAdapter:
    """Expose the existing solve API and explicitly observe actually applied actions."""

    def __init__(self, backend, original, planner):
        self.backend, self.original, self.planner = backend, original, planner
        self.state_model = backend.state_projection
        self.power_model = backend.power_projection

    def __getattr__(self, name):
        return getattr(self.original, name)

    def begin_window(self, data, start):
        self.prepare_factual(data, start)

    def prepare_factual(self, data, row):
        if data is not self.backend.data:
            raise ValueError('Controller bootstrap dataset differs')
        self.backend.online_context = self.backend.contexts[row].copy()
        if hasattr(self.planner, 'warm'):
            self.planner.warm = None

    def observe_step(self, state, control, next_state):
        ctx = self.backend.online_context
        if not np.allclose(ctx[:self.backend.spec.nx], state):
            raise AssertionError('Controller context is not the applied plant state')
        self.backend.online_context = self.backend.spec.advance(ctx, control, next_state)

    def solve(self, initial_state, disturbances, previous_control,
              first_step_forbidden_control_vectors=None, **kwargs):
        if kwargs.get('action_history') is not None:
            raise ValueError('Legacy history budgets are unsupported for this planner')
        ctx = self.backend.online_context
        if ctx is None or not np.allclose(ctx[:self.backend.spec.nx], initial_state):
            raise AssertionError('Explicit history bootstrap/observation is required before solve')
        forbidden = np.asarray(
            first_step_forbidden_control_vectors if first_step_forbidden_control_vectors is not None
            else np.empty((0, self.backend.spec.nz)),
        )
        allowed = np.asarray([
            not any(np.array_equal(a, f) for f in forbidden) for a in self.backend.actions
        ])
        plan = self.planner.solve(ctx, disturbances, previous_control, allowed)
        with (self.backend.result_dir/'planner_solves.jsonl').open('a') as stream:
            stream.write(json.dumps({
                'status': plan.status, 'elapsed_seconds': plan.elapsed_seconds,
                'MILP_subproblem_optimal_status': plan.optimal, 'mip_gap': plan.mip_gap,
                'has_incumbent': bool(len(plan.controls)),
            }, allow_nan=False)+'\n')
        spec = self.backend.spec
        if not len(plan.controls):
            return MPCSolution(False, 1, plan.status, float('inf'), (),
                               np.empty((0, spec.nz)), np.empty((0, spec.nx)),
                               np.empty((0, spec.np)), np.empty((0, spec.nx)),
                               np.empty((0, spec.nx)))
        controls = np.stack([plan.controls[min(t, len(plan.controls)-1)]
                             for t in range(len(disturbances))])
        states, powers = [], []
        rollout_ctx = ctx.copy()
        for z, d in zip(controls, disturbances):
            x, p = self.backend.model.predict(rollout_ctx, z, d)
            states.append(x[0])
            powers.append(p[0])
            rollout_ctx = spec.advance(rollout_ctx, z, x[0])
        states = np.asarray(states)
        lower, upper = np.zeros_like(states), np.zeros_like(states)
        for j, constraint in enumerate(self.backend.cfg.MPC.state_constraints):
            if constraint.get('lower') is not None:
                lower[:, j] = np.maximum(float(constraint.lower)-states[:, j], 0)
            if constraint.get('upper') is not None:
                upper[:, j] = np.maximum(states[:, j]-float(constraint.upper), 0)
        return MPCSolution(
            bool(plan.optimal), 0 if plan.optimal else 1, plan.status,
            float(plan.objective), tuple(spec.encoder.decode_action(z) for z in plan.controls),
            plan.controls, states, np.asarray(powers), lower, upper,
        )
