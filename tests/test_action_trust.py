"""Focused contracts for train-only MPC action trust eligibility."""

import unittest

import numpy as np
import pandas as pd

from controllers.action_trust import (
    ActionTrustConfig,
    ActionTrustEligibility,
    canonical_control_episode_ids,
)
from data_provider.control_data import ControlData, ControlEncoder, ControlVariable


class ActionTrustEligibilityTest(unittest.TestCase):
    @staticmethod
    def _data():
        encoder = ControlEncoder(
            [
                ControlVariable('power', ('OFF', 'ON'), 'OFF'),
                ControlVariable('mode', ('neutral', 'low', 'high'), 'neutral'),
            ]
        )
        # OFF has deliberately non-canonical raw sub-settings.  The canonical
        # OFF/neutral action has six train rows in five distinct episodes.
        action_rows = [
            {'power': 'OFF', 'mode': 'high'},
            {'power': 'OFF', 'mode': 'low'},
            {'power': 'ON', 'mode': 'low'},
            {'power': 'OFF', 'mode': 'high'},
            {'power': 'ON', 'mode': 'high'},
            {'power': 'OFF', 'mode': 'low'},
            {'power': 'ON', 'mode': 'low'},
            {'power': 'OFF', 'mode': 'high'},
            {'power': 'ON', 'mode': 'high'},
            {'power': 'OFF', 'mode': 'low'},
            # These are held out of training.  They must not make ON/low
            # eligible even though it becomes very frequent in the full data.
            *({'power': 'ON', 'mode': 'low'} for _ in range(10)),
        ]
        controls = np.vstack([encoder.encode_action(action) for action in action_rows])
        state_values = np.asarray(
            [0.0, 0.1, 5.0, -0.1, 6.0, 0.2, 5.1, -0.2, 6.1, 0.05]
            + [5.2] * 10,
            dtype=float,
        )[:, None]
        disturbance_values = np.zeros((len(action_rows), 1), dtype=float)
        return ControlData(
            timestamps=pd.Series(
                pd.date_range('2026-01-01', periods=len(action_rows), freq='min')
            ),
            states=state_values,
            controls=controls,
            disturbances=disturbance_values,
            powers=np.ones((len(action_rows), 2), dtype=float),
            state_names=('indoor_state',),
            disturbance_names=('outdoor_state',),
            power_names=('vent_power', 'hvac_power'),
            encoder=encoder,
        )

    @staticmethod
    def _rules():
        return (
            {
                'when': {'variable': 'power', 'value': 'OFF'},
                'require': {'mode': 'neutral'},
            },
            {
                'when': {'variable': 'power', 'value': 'ON'},
                'forbid': {'mode': ['neutral']},
            },
        )

    @staticmethod
    def _config():
        return {
            'enabled': True,
            'global_support': {
                'minimum_samples': 6,
                'minimum_episodes': 3,
            },
            'local_context_support': {
                'enabled': True,
                'minimum_samples_within_radius': 3,
                'maximum_standardized_rms_distance': 0.20,
                'neighbor_count': 3,
                'maximum_mean_k_distance': 0.20,
            },
        }

    def test_canonical_action_uses_train_only_global_and_local_support(self):
        data = self._data()
        encoder = data.encoder
        raw_off_high = encoder.encode_action({'power': 'OFF', 'mode': 'high'})
        on_low = encoder.encode_action({'power': 'ON', 'mode': 'low'})
        trust = ActionTrustEligibility.build(
            data,
            training_rows=np.arange(10),
            config=self._config(),
            logical_rules=self._rules(),
            candidate_controls=np.vstack((raw_off_high, on_low)),
        )
        chronological_trust = ActionTrustEligibility.build_from_chronological_train_end(
            data,
            train_end_exclusive=10,
            config=self._config(),
            logical_rules=self._rules(),
            candidate_controls=np.vstack((raw_off_high, on_low)),
        )
        self.assertEqual(
            chronological_trust.to_metadata()['training_transition_count'],
            trust.to_metadata()['training_transition_count'],
        )

        # The canonical OFF representation keeps the two adjacent raw OFF
        # records in one episode instead of incorrectly counting two episodes.
        canonical_episodes = canonical_control_episode_ids(data, self._rules())
        self.assertEqual(int(canonical_episodes[0]), int(canonical_episodes[1]))
        self.assertEqual(len(trust.candidate_controls), 2)

        nearby = trust.evaluate(raw_off_high, state=np.array([0.0]), disturbance=np.array([0.0]))
        self.assertTrue(nearby.eligible)
        self.assertEqual(nearby.action, {'power': 'OFF', 'mode': 'neutral'})
        self.assertIn('canonicalized_inactive_equipment_subsettings', nearby.notes)
        self.assertEqual(nearby.global_support['training_sample_count'], 6)
        self.assertGreaterEqual(nearby.local_support['samples_within_radius'], 3)

        far_away = trust.evaluate(raw_off_high, state=np.array([100.0]), disturbance=np.array([0.0]))
        self.assertFalse(far_away.eligible)
        self.assertIn('insufficient_local_context_samples', far_away.rejection_reasons)
        self.assertIn(
            'mean_k_nearest_context_distance_above_threshold',
            far_away.rejection_reasons,
        )

        unsupported = trust.evaluate(on_low, state=np.array([5.0]), disturbance=np.array([0.0]))
        self.assertFalse(unsupported.eligible)
        self.assertEqual(unsupported.global_support['training_sample_count'], 2)
        self.assertEqual(unsupported.global_support['training_episode_count'], 2)
        self.assertIn('insufficient_global_training_samples', unsupported.rejection_reasons)
        self.assertIn('insufficient_global_training_episodes', unsupported.rejection_reasons)
        self.assertNotIn('not_observed_in_training', unsupported.rejection_reasons)

        trusted, decisions = trust.eligible_control_vectors(
            state=np.array([0.0]),
            disturbance=np.array([0.0]),
        )
        self.assertEqual(len(trusted), 1)
        self.assertEqual(len(decisions), 2)
        self.assertEqual(encoder.decode_action(trusted[0]), {'power': 'OFF', 'mode': 'neutral'})

    def test_invalid_config_and_outside_candidate_have_clear_reasons(self):
        with self.assertRaisesRegex(ValueError, 'maximum_context_distance'):
            ActionTrustConfig.from_mapping(
                {
                    'local_context_support': {
                        'enabled': True,
                        'minimum_samples_within_radius': 1,
                    }
                }
            )

        data = self._data()
        raw_off_high = data.encoder.encode_action({'power': 'OFF', 'mode': 'high'})
        trust = ActionTrustEligibility.build(
            data,
            training_rows=np.arange(10),
            config=self._config(),
            logical_rules=self._rules(),
            candidate_controls=np.vstack((raw_off_high,)),
        )
        on_high = data.encoder.encode_action({'power': 'ON', 'mode': 'high'})
        decision = trust.evaluate(on_high, state=np.array([6.0]), disturbance=np.array([0.0]))
        self.assertFalse(decision.eligible)
        self.assertIn('outside_configured_candidate_pool', decision.rejection_reasons)

    def test_canonical_episode_never_crosses_source_segment_boundary(self):
        encoder = ControlEncoder(
            [ControlVariable('switch', ('OFF', 'ON'), 'OFF')]
        )
        controls = np.vstack(
            [encoder.encode_action({'switch': 'OFF'}) for _ in range(4)]
        )
        data = ControlData(
            # The timestamps intentionally are consecutive at the boundary.
            timestamps=pd.Series(
                pd.date_range('2026-01-01', periods=4, freq='min')
            ),
            states=np.zeros((4, 1)),
            controls=controls,
            disturbances=np.zeros((4, 1)),
            powers=np.ones((4, 1)),
            state_names=('state',),
            disturbance_names=('disturbance',),
            power_names=('power',),
            encoder=encoder,
            segment_ids=np.array([0, 0, 1, 1]),
        )

        episode_ids = canonical_control_episode_ids(data)

        self.assertEqual(episode_ids.tolist(), [0, 0, 1, 1])


if __name__ == '__main__':
    unittest.main()
