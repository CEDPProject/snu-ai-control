"""Contracts for the optional HiGHS wall-clock watchdog."""

import os
import time
import unittest
from unittest import mock

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import controllers.guarded_milp as guarded


def _problem():
    c = np.array([-1.0, -2.0])
    constraints = LinearConstraint(np.array([[1.0, 1.0]]), -np.inf, 1.0)
    return dict(c=c, integrality=np.ones(2), bounds=Bounds(0, 1), constraints=constraints,
                options={'time_limit': 10.0, 'presolve': True})


def _hanging_milp(*args, **kwargs):
    time.sleep(30)


class GuardedMILPTest(unittest.TestCase):
    def test_disabled_by_default_and_identical_to_milp(self):
        with mock.patch.dict(os.environ, {guarded.ENV_VARIABLE: ''}):
            self.assertIsNone(guarded.hard_timeout_seconds())
            result = guarded.guarded_milp(**_problem())
        reference = milp(**_problem())
        np.testing.assert_allclose(result.x, reference.x)
        self.assertEqual(result.status, reference.status)

    def test_enabled_watchdog_returns_child_result_unchanged(self):
        with mock.patch.dict(os.environ, {guarded.ENV_VARIABLE: '60'}):
            result = guarded.guarded_milp(**_problem())
        np.testing.assert_allclose(result.x, milp(**_problem()).x)
        self.assertTrue(result.success)

    def test_hung_solve_is_killed_and_reported_as_no_incumbent(self):
        with mock.patch.object(guarded, 'milp', _hanging_milp):
            started = time.perf_counter()
            result = guarded.guarded_milp(**_problem(), timeout_seconds=1.0)
        self.assertLess(time.perf_counter()-started, 10.0)
        self.assertIsNone(result.x)
        self.assertFalse(result.success)
        self.assertEqual(result.status, guarded.HARD_TIMEOUT_STATUS)
        self.assertIn('hard_timeout_killed', result.message)

    def test_invalid_environment_value_is_rejected(self):
        with mock.patch.dict(os.environ, {guarded.ENV_VARIABLE: '-5'}):
            with self.assertRaises(ValueError):
                guarded.hard_timeout_seconds()

    def test_mpc_turns_hard_timeout_into_hold_previous_fallback(self):
        import controllers.mixed_integer_mpc as mpc_module
        from controllers.evaluation import solve_first_action_with_gate
        from data_provider.control_data import ControlEncoder, ControlVariable
        from dynamics.affine import AffinePowerModel, AffineStateModel

        encoder = ControlEncoder([ControlVariable('switch', ('OFF', 'ON'), 'OFF')])
        controller = mpc_module.MixedIntegerMPC(
            state_model=AffineStateModel(
                A=np.array([[1.0]]), B=np.zeros((1, 2)), E=np.zeros((1, 0)), c=np.zeros(1),
                state_names=('state',), control_feature_names=encoder.feature_names,
                disturbance_names=()),
            power_model=AffinePowerModel(
                F=np.zeros((1, 1)), G=np.array([[0.0, -9.0]]), H=np.zeros((1, 0)),
                b=np.array([10.0]), power_names=('power_w',), state_names=('state',),
                control_feature_names=encoder.feature_names, disturbance_names=()),
            encoder=encoder,
            state_constraints=[{'name': 'state', 'lower': None, 'upper': None, 'penalty': 1.0}],
            logical_rules=[], prediction_horizon=1, control_horizon=1, switch_penalty=0.0)
        off = encoder.encode_action({'switch': 'OFF'})
        timeout = guarded._failure('hard_timeout_killed_after_1.0s')
        with mock.patch.object(mpc_module, 'guarded_milp', return_value=timeout):
            result = solve_first_action_with_gate(
                controller, initial_state=np.array([0.0]), disturbances=np.empty((1, 0)),
                previous_control=off, evaluator=object(), action_gate=None)
        self.assertFalse(result.has_incumbent)
        self.assertTrue(result.used_fallback)
        np.testing.assert_array_equal(result.applied_control, off)
        self.assertEqual(result.solution.status, guarded.HARD_TIMEOUT_STATUS)

if __name__ == '__main__':
    unittest.main()
