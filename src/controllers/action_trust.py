"""Train-only action trust gates for offline MPC.

The MPC controller can only optimize the affine surrogate that it was given.
This module therefore answers a deliberately narrower question before an
action is offered to the MILP:

    "Was this *exact canonical joint action* observed often enough in training,
    and under a state/disturbance context close enough to the current one?"

It contains no power, next-state, validation, or test targets.  Consequently
it is safe to build from the chronological training transition rows and to use
online at every receding-horizon decision step.  A caller may supply a smaller
``candidate_controls`` pool (for example the repeated-action pool already used
by the MPC); the trust gate then adds local context eligibility on top of it.

The module intentionally does *not* make a model-quality or safety decision.
Those require the independent evaluator and are kept separate so that an
action audit can state exactly why an action was rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from omegaconf import OmegaConf

from data_provider.control_data import ControlData, ControlEncoder, _category_token


def _plain(value):
    """Convert OmegaConf/dict input to ordinary Python containers."""
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError('Action trust configuration must be a mapping or OmegaConf')


def _action_key(control: np.ndarray) -> tuple[int, ...]:
    """Return the exact binary-one-hot joint-action key used across MPC code."""
    array = np.asarray(control, dtype=float).reshape(-1)
    rounded = np.rint(array).astype(int)
    if not np.allclose(array, rounded):
        raise ValueError('Control vector must contain binary one-hot values')
    return tuple(int(value) for value in rounded.tolist())


def _validate_one_hot_control(encoder: ControlEncoder, control) -> np.ndarray:
    """Validate and normalize one exact encoded joint action."""
    vector = np.asarray(control, dtype=float).reshape(-1)
    if vector.shape != (encoder.n_features,):
        raise ValueError(
            'Control vector shape must be '
            f'({encoder.n_features},), got {vector.shape}'
        )
    rounded = np.rint(vector).astype(int)
    if not np.allclose(vector, rounded) or not np.isin(rounded, (0, 1)).all():
        raise ValueError('Control vector must contain only binary one-hot values')
    for variable_name, group in encoder.groups.items():
        if int(rounded[list(group)].sum()) != 1:
            raise ValueError(
                f'Control vector must select exactly one category for '
                f'{variable_name!r}'
            )
    return rounded.astype(float)


def _normalized_logical_rules(logical_rules: Iterable[Mapping] = ()):
    """Return validated logical-rule dictionaries without mutating cfg values."""
    normalized = []
    for raw_rule in logical_rules or ():
        rule = _plain(raw_rule)
        if 'when' not in rule:
            raise ValueError("Every logical action rule requires a 'when' mapping")
        when = _plain(rule['when'])
        if 'variable' not in when or 'value' not in when:
            raise ValueError("Logical rule 'when' requires variable and value")
        require = {
            str(variable): _category_token(value)
            for variable, value in _plain(rule.get('require', {})).items()
        }
        forbid = {
            str(variable): tuple(_category_token(value) for value in values)
            for variable, values in _plain(rule.get('forbid', {})).items()
        }
        normalized.append(
            {
                'when': {
                    'variable': str(when['variable']),
                    'value': _category_token(when['value']),
                },
                'require': require,
                'forbid': forbid,
            }
        )
    return tuple(normalized)


def canonicalize_action(
    encoder: ControlEncoder,
    action: Mapping[str, object],
    logical_rules: Iterable[Mapping] = (),
) -> Dict[str, str]:
    """Canonicalize a categorical action using deterministic ``require`` rules.

    It is useful even when :func:`load_control_data` already canonicalized the
    recorded controls: callers can present an action generated outside that
    loader and receive the same inactive-equipment representative.  ``forbid``
    rules remain hard validation rules because they cannot be converted to a
    unique representative without an additional policy decision.
    """
    # ``encode_action`` validates every supplied category and supplies defaults
    # for omitted fields.  Decoding immediately normalizes numeric categories
    # (e.g., YAML 22 and CSV "22") to the encoder's stable string tokens.
    normalized = encoder.decode_action(encoder.encode_action(dict(action)))
    rules = _normalized_logical_rules(logical_rules)
    for rule in rules:
        encoder.index(rule['when']['variable'], rule['when']['value'])
        for variable, value in rule['require'].items():
            encoder.index(variable, value)
        for variable, values in rule['forbid'].items():
            for value in values:
                encoder.index(variable, value)

    # A required category can itself trigger a later rule.  Iterate to a fixed
    # point and fail loudly on a malformed cyclic rule set rather than silently
    # emitting an ambiguous action.
    maximum_passes = max(1, len(rules) * max(1, len(encoder.variables)) + 1)
    for _ in range(maximum_passes):
        changed = False
        for rule in rules:
            when = rule['when']
            if normalized[when['variable']] != when['value']:
                continue
            for variable, required_value in rule['require'].items():
                if normalized[variable] != required_value:
                    normalized[variable] = required_value
                    changed = True
        if not changed:
            break
    else:
        raise ValueError('Logical require rules do not converge to a canonical action')

    for rule in rules:
        when = rule['when']
        if normalized[when['variable']] != when['value']:
            continue
        for variable, forbidden_values in rule['forbid'].items():
            if normalized[variable] in forbidden_values:
                raise ValueError(
                    'Action violates logical rule: '
                    f"when {when['variable']}={when['value']!r}, "
                    f"{variable} must not be {normalized[variable]!r}"
                )
    return dict(normalized)


def canonicalize_control_vector(
    encoder: ControlEncoder,
    control,
    logical_rules: Iterable[Mapping] = (),
) -> np.ndarray:
    """Return the canonical binary control vector for an exact joint action."""
    vector = _validate_one_hot_control(encoder, control)
    canonical = canonicalize_action(
        encoder,
        encoder.decode_action(vector),
        logical_rules,
    )
    return encoder.encode_action(canonical)


def canonical_control_matrix(
    data: ControlData,
    logical_rules: Iterable[Mapping] = (),
) -> np.ndarray:
    """Canonicalize all recorded control rows without changing ``ControlData``."""
    # The data may contain tens of thousands of rows but only a small number
    # of categorical combinations.  Cache by the exact one-hot key so action
    # trust construction remains cheap even for long chronological histories.
    canonical_by_raw_key: Dict[tuple[int, ...], np.ndarray] = {}
    canonical_rows = []
    for control in np.asarray(data.controls, dtype=float):
        key = _action_key(control)
        if key not in canonical_by_raw_key:
            canonical_by_raw_key[key] = canonicalize_control_vector(
                data.encoder,
                control,
                logical_rules,
            )
        canonical_rows.append(canonical_by_raw_key[key])
    return np.asarray(canonical_rows, dtype=float)


def canonical_control_episode_ids(
    data: ControlData,
    logical_rules: Iterable[Mapping] = (),
    *,
    canonical_controls: np.ndarray | None = None,
) -> np.ndarray:
    """Episode labels for contiguous canonical joint-action runs.

    A raw inactive-equipment sub-setting must not split an otherwise identical
    canonical OFF action into artificial independent episodes.  The definition
    remains compatible with ``control_episode_ids`` for already canonical data.
    """
    controls = (
        canonical_control_matrix(data, logical_rules)
        if canonical_controls is None
        else np.asarray(canonical_controls, dtype=float)
    )
    if controls.shape != (len(data.controls), data.encoder.n_features):
        raise ValueError(
            'canonical_controls must have shape '
            f'({len(data.controls)}, {data.encoder.n_features})'
        )
    episode_ids = np.zeros(len(controls), dtype=int)
    for position in range(1, len(controls)):
        adjacent = (
            data.timestamps.iloc[position] - data.timestamps.iloc[position - 1]
        ).total_seconds() == 60.0
        # A concatenated dataset may contain source segments whose local
        # timestamps happen to be consecutive.  They still do not form one
        # physical operating episode: no transition may be inferred across a
        # segment boundary, consistent with ``ControlData.valid_transition_rows``.
        if data.segment_ids is not None:
            adjacent = bool(
                adjacent
                and data.segment_ids[position] == data.segment_ids[position - 1]
            )
        same_action = np.array_equal(controls[position], controls[position - 1])
        episode_ids[position] = episode_ids[position - 1] + int(
            not (adjacent and same_action)
        )
    return episode_ids


def _context_vector(state, disturbance, *, state_width: int, disturbance_width: int):
    state = np.asarray(state, dtype=float).reshape(-1)
    disturbance = np.asarray(disturbance, dtype=float).reshape(-1)
    if state.shape != (state_width,):
        raise ValueError(f'State must have shape ({state_width},), got {state.shape}')
    if disturbance.shape != (disturbance_width,):
        raise ValueError(
            'Disturbance must have shape '
            f'({disturbance_width},), got {disturbance.shape}'
        )
    context = np.concatenate((state, disturbance))
    if not np.isfinite(context).all():
        raise ValueError('State and disturbance values must be finite')
    return context


@dataclass(frozen=True)
class ActionTrustConfig:
    """Thresholds for action eligibility, expressed in training-only units.

    ``maximum_context_distance`` and ``maximum_mean_k_distance`` are
    standardized RMS distances over ``[state, disturbance]``.  Their values
    must be calibrated without test targets (normally on a frozen validation
    protocol) and then supplied as a config mapping.  A disabled local gate
    deliberately preserves existing repeated-action-only behavior.
    """

    enabled: bool = True
    minimum_training_samples: int = 1
    minimum_training_episodes: int = 1
    local_support_enabled: bool = False
    minimum_local_samples: int = 1
    maximum_context_distance: float | None = None
    neighbor_count: int = 1
    maximum_mean_k_distance: float | None = None

    @classmethod
    def from_mapping(cls, config=None) -> 'ActionTrustConfig':
        """Parse a small, explicit config schema with backwards-safe aliases.

        Preferred schema::

            enabled: true
            global_support:
              minimum_samples: 30
              minimum_episodes: 3
            local_context_support:
              enabled: true
              minimum_samples_within_radius: 5
              maximum_standardized_rms_distance: 1.50
              neighbor_count: 5
              maximum_mean_k_distance: 1.20

        The top-level aliases mirror the existing ``repeated_joint_actions``
        setting and make standalone use convenient.
        """
        specification = _plain(config)
        global_support = _plain(specification.get('global_support', {}))
        local_support = _plain(
            specification.get(
                'local_context_support',
                specification.get('local_support', {}),
            )
        )

        def choose(mapping, keys, default):
            for key in keys:
                if key in mapping:
                    return mapping[key]
            return default

        result = cls(
            enabled=bool(specification.get('enabled', True)),
            minimum_training_samples=int(
                choose(
                    global_support,
                    ('minimum_samples', 'minimum_training_samples'),
                    choose(
                        specification,
                        ('minimum_training_samples', 'minimum_global_samples'),
                        1,
                    ),
                )
            ),
            minimum_training_episodes=int(
                choose(
                    global_support,
                    ('minimum_episodes', 'minimum_training_episodes'),
                    choose(
                        specification,
                        ('minimum_training_episodes', 'minimum_global_episodes'),
                        1,
                    ),
                )
            ),
            local_support_enabled=bool(
                choose(
                    local_support,
                    ('enabled',),
                    specification.get('local_support_enabled', False),
                )
            ),
            minimum_local_samples=int(
                choose(
                    local_support,
                    ('minimum_samples_within_radius', 'minimum_local_samples'),
                    specification.get('minimum_local_samples', 1),
                )
            ),
            maximum_context_distance=(
                None
                if choose(
                    local_support,
                    (
                        'maximum_standardized_rms_distance',
                        'maximum_context_distance',
                    ),
                    specification.get('maximum_context_distance'),
                )
                is None
                else float(
                    choose(
                        local_support,
                        (
                            'maximum_standardized_rms_distance',
                            'maximum_context_distance',
                        ),
                        specification.get('maximum_context_distance'),
                    )
                )
            ),
            neighbor_count=int(
                choose(
                    local_support,
                    ('neighbor_count', 'k_neighbors'),
                    specification.get('neighbor_count', 1),
                )
            ),
            maximum_mean_k_distance=(
                None
                if choose(
                    local_support,
                    ('maximum_mean_k_distance',),
                    specification.get('maximum_mean_k_distance'),
                )
                is None
                else float(
                    choose(
                        local_support,
                        ('maximum_mean_k_distance',),
                        specification.get('maximum_mean_k_distance'),
                    )
                )
            ),
        )
        result._validate()
        return result

    def _validate(self) -> None:
        if self.minimum_training_samples <= 0:
            raise ValueError('minimum_training_samples must be positive')
        if self.minimum_training_episodes <= 0:
            raise ValueError('minimum_training_episodes must be positive')
        if self.neighbor_count <= 0:
            raise ValueError('neighbor_count must be positive')
        if self.local_support_enabled:
            if self.minimum_local_samples <= 0:
                raise ValueError('minimum_local_samples must be positive')
            if (
                self.maximum_context_distance is None
                or not np.isfinite(self.maximum_context_distance)
                or self.maximum_context_distance < 0.0
            ):
                raise ValueError(
                    'Enabled local context support requires a finite, non-negative '
                    'maximum_context_distance calibrated before test use'
                )
        if (
            self.maximum_mean_k_distance is not None
            and (
                not np.isfinite(self.maximum_mean_k_distance)
                or self.maximum_mean_k_distance < 0.0
            )
        ):
            raise ValueError(
                'maximum_mean_k_distance must be finite and non-negative'
            )

    def to_metadata(self) -> Dict[str, object]:
        return {
            'enabled': bool(self.enabled),
            'global_support': {
                'minimum_samples': int(self.minimum_training_samples),
                'minimum_episodes': int(self.minimum_training_episodes),
            },
            'local_context_support': {
                'enabled': bool(self.local_support_enabled),
                'minimum_samples_within_radius': int(self.minimum_local_samples),
                'maximum_standardized_rms_distance': (
                    None
                    if self.maximum_context_distance is None
                    else float(self.maximum_context_distance)
                ),
                'neighbor_count': int(self.neighbor_count),
                'maximum_mean_k_distance': (
                    None
                    if self.maximum_mean_k_distance is None
                    else float(self.maximum_mean_k_distance)
                ),
            },
            'threshold_selection_contract': (
                'Threshold values are supplied by configuration and must be '
                'calibrated without test power/state targets.'
            ),
        }


@dataclass(frozen=True)
class ActionTrustDecision:
    """Eligibility result and audit-ready reasons for one action at one context."""

    encoded_control: tuple[int, ...] | None
    action: Dict[str, str] | None
    eligible: bool
    rejection_reasons: tuple[str, ...]
    notes: tuple[str, ...]
    global_support: Dict[str, object]
    local_support: Dict[str, object]

    def to_metadata(self) -> Dict[str, object]:
        return {
            'encoded_control': (
                None if self.encoded_control is None else list(self.encoded_control)
            ),
            'action': None if self.action is None else dict(self.action),
            'eligible': bool(self.eligible),
            'rejection_reasons': list(self.rejection_reasons),
            'notes': list(self.notes),
            'global_support': dict(self.global_support),
            'local_support': dict(self.local_support),
        }


class ActionTrustEligibility:
    """Immutable train-only action-support index usable at each MPC step.

    Typical integration::

        trust = ActionTrustEligibility.build(
            data,
            training_rows=np.arange(train_end),
            config=cfg.MPC.evaluation.action_trust,
            logical_rules=cfg.MPC.logical_rules,
            candidate_controls=observed_support,
        )
        trusted, decisions = trust.eligible_control_vectors(x_t, d_t)

    ``trusted`` may be translated to MILP no-good cuts by the caller.  The
    returned decisions are designed to be persisted in a per-step audit CSV.
    """

    def __init__(
        self,
        *,
        data: ControlData,
        training_rows: np.ndarray,
        config: ActionTrustConfig,
        logical_rules: Iterable[Mapping],
        canonical_controls: np.ndarray,
        canonical_episode_ids: np.ndarray,
        candidate_controls: np.ndarray,
    ):
        self.data = data
        self.training_rows = np.asarray(training_rows, dtype=int)
        self.config = config
        self.logical_rules = tuple(_normalized_logical_rules(logical_rules))
        self._canonical_controls = np.asarray(canonical_controls, dtype=float)
        self._canonical_episode_ids = np.asarray(canonical_episode_ids, dtype=int)
        self.candidate_controls = np.asarray(candidate_controls, dtype=float)
        self._candidate_keys = {
            _action_key(vector) for vector in self.candidate_controls
        }

        raw_contexts = np.concatenate(
            (
                data.states[self.training_rows],
                data.disturbances[self.training_rows],
            ),
            axis=1,
        )
        if not np.isfinite(raw_contexts).all():
            raise ValueError('Training state and disturbance contexts must be finite')
        self._context_mean = raw_contexts.mean(axis=0)
        self._context_scale = raw_contexts.std(axis=0)
        self._context_scale = np.where(
            self._context_scale > 1e-12,
            self._context_scale,
            1.0,
        )
        self._training_contexts = (
            raw_contexts - self._context_mean
        ) / self._context_scale

        self._support_by_key: Dict[tuple[int, ...], Dict[str, object]] = {}
        for position, row in enumerate(self.training_rows.tolist()):
            key = _action_key(self._canonical_controls[row])
            support = self._support_by_key.setdefault(
                key,
                {
                    'row_indices': [],
                    'episode_ids': set(),
                    'context_positions': [],
                },
            )
            support['row_indices'].append(int(row))
            support['episode_ids'].add(int(self._canonical_episode_ids[row]))
            support['context_positions'].append(int(position))

    @classmethod
    def build(
        cls,
        data: ControlData,
        training_rows,
        config=None,
        *,
        logical_rules: Iterable[Mapping] = (),
        candidate_controls: np.ndarray | None = None,
    ) -> 'ActionTrustEligibility':
        """Index valid chronological training transition rows only.

        ``training_rows`` are source-row indices, not positions in a shuffled
        array.  Invalid one-minute transitions and cross-segment transitions
        are removed via :meth:`ControlData.valid_transition_rows` before any
        support statistic is computed.
        """
        threshold_config = (
            config
            if isinstance(config, ActionTrustConfig)
            else ActionTrustConfig.from_mapping(config)
        )
        rows = data.valid_transition_rows(training_rows)
        if not len(rows):
            raise ValueError('Action trust requires at least one training transition')
        rules = _normalized_logical_rules(logical_rules)
        canonical_controls = canonical_control_matrix(data, rules)
        canonical_episodes = canonical_control_episode_ids(
            data,
            rules,
            canonical_controls=canonical_controls,
        )

        if candidate_controls is None:
            logical_vectors = data.encoder.valid_control_vectors(rules)
            candidates = np.asarray(
                [
                    canonicalize_control_vector(data.encoder, vector, rules)
                    for vector in logical_vectors
                ],
                dtype=float,
            )
        else:
            raw_candidates = np.asarray(candidate_controls, dtype=float)
            if raw_candidates.ndim == 1:
                raw_candidates = raw_candidates.reshape(1, -1)
            if raw_candidates.ndim != 2 or raw_candidates.shape[1] != data.encoder.n_features:
                raise ValueError(
                    'candidate_controls must have shape '
                    f'(n_actions, {data.encoder.n_features})'
                )
            candidates = np.asarray(
                [
                    canonicalize_control_vector(data.encoder, vector, rules)
                    for vector in raw_candidates
                ],
                dtype=float,
            )

        # Stable de-duplication matters when off-state sub-settings collapse to
        # one canonical joint action.
        unique = []
        seen = set()
        for vector in candidates:
            key = _action_key(vector)
            if key not in seen:
                unique.append(vector)
                seen.add(key)
        candidates = (
            np.asarray(unique, dtype=float)
            if unique
            else np.empty((0, data.encoder.n_features), dtype=float)
        )
        if not len(candidates):
            raise ValueError('Action trust candidate pool is empty after canonicalization')

        return cls(
            data=data,
            training_rows=rows,
            config=threshold_config,
            logical_rules=rules,
            canonical_controls=canonical_controls,
            canonical_episode_ids=canonical_episodes,
            candidate_controls=candidates,
        )

    @classmethod
    def build_from_chronological_train_end(
        cls,
        data: ControlData,
        train_end_exclusive: int,
        config=None,
        *,
        logical_rules: Iterable[Mapping] = (),
        candidate_controls: np.ndarray | None = None,
    ) -> 'ActionTrustEligibility':
        """Build from the prefix ``[0, train_end_exclusive)`` only.

        This convenience constructor is the preferred integration point for
        the ordinary chronological MPC split.  It makes the no-future-row
        contract explicit instead of requiring a caller to construct an index
        array correctly.  Episode-grouped protocols can use :meth:`build`
        with their explicit train episode rows.
        """
        train_end = int(train_end_exclusive)
        if not 1 <= train_end <= len(data.states) - 1:
            raise ValueError(
                'train_end_exclusive must leave at least one later observation '
                'for a valid transition'
            )
        return cls.build(
            data,
            training_rows=np.arange(train_end, dtype=int),
            config=config,
            logical_rules=logical_rules,
            candidate_controls=candidate_controls,
        )

    @property
    def context_feature_count(self) -> int:
        return int(len(self.data.state_names) + len(self.data.disturbance_names))

    def _global_metadata(self, key: tuple[int, ...]) -> Dict[str, object]:
        support = self._support_by_key.get(key)
        samples = 0 if support is None else len(support['row_indices'])
        episodes = 0 if support is None else len(support['episode_ids'])
        return {
            'training_sample_count': int(samples),
            'training_episode_count': int(episodes),
            'minimum_training_samples': int(self.config.minimum_training_samples),
            'minimum_training_episodes': int(self.config.minimum_training_episodes),
        }

    def _local_metadata(
        self,
        key: tuple[int, ...],
        standardized_context: np.ndarray,
    ) -> Dict[str, object]:
        support = self._support_by_key.get(key)
        if support is None:
            return {
                'reference_training_sample_count': 0,
                'samples_within_radius': 0,
                'nearest_context_distance': None,
                'mean_k_nearest_context_distance': None,
                'effective_neighbor_count': 0,
                'minimum_samples_within_radius': int(
                    self.config.minimum_local_samples
                ),
                'maximum_standardized_rms_distance': (
                    self.config.maximum_context_distance
                ),
                'neighbor_count': int(self.config.neighbor_count),
                'maximum_mean_k_distance': self.config.maximum_mean_k_distance,
            }
        positions = np.asarray(support['context_positions'], dtype=int)
        references = self._training_contexts[positions]
        width = max(1, references.shape[1])
        distances = np.sqrt(
            np.square(references - standardized_context[None, :]).sum(axis=1)
            / float(width)
        )
        effective_neighbors = min(int(self.config.neighbor_count), len(distances))
        nearest = float(distances.min()) if len(distances) else None
        if effective_neighbors:
            nearest_k = np.partition(distances, effective_neighbors - 1)[
                :effective_neighbors
            ]
            mean_k = float(nearest_k.mean())
        else:
            mean_k = None
        radius = self.config.maximum_context_distance
        within_radius = (
            int((distances <= float(radius) + 1e-12).sum())
            if radius is not None
            else int(len(distances))
        )
        return {
            'reference_training_sample_count': int(len(distances)),
            'samples_within_radius': int(within_radius),
            'nearest_context_distance': nearest,
            'mean_k_nearest_context_distance': mean_k,
            'effective_neighbor_count': int(effective_neighbors),
            'minimum_samples_within_radius': int(self.config.minimum_local_samples),
            'maximum_standardized_rms_distance': radius,
            'neighbor_count': int(self.config.neighbor_count),
            'maximum_mean_k_distance': self.config.maximum_mean_k_distance,
        }

    def evaluate(self, control, state, disturbance) -> ActionTrustDecision:
        """Assess one proposed exact joint action for the current MPC context."""
        try:
            original = _validate_one_hot_control(self.data.encoder, control)
            canonical = canonicalize_control_vector(
                self.data.encoder,
                original,
                self.logical_rules,
            )
            key = _action_key(canonical)
            action = self.data.encoder.decode_action(canonical)
            notes = (
                ('canonicalized_inactive_equipment_subsettings',)
                if not np.array_equal(original, canonical)
                else ()
            )
        except (KeyError, TypeError, ValueError) as error:
            return ActionTrustDecision(
                encoded_control=None,
                action=None,
                eligible=False,
                rejection_reasons=('invalid_control_vector',),
                notes=(f'{type(error).__name__}: {error}',),
                global_support={},
                local_support={},
            )

        try:
            raw_context = _context_vector(
                state,
                disturbance,
                state_width=len(self.data.state_names),
                disturbance_width=len(self.data.disturbance_names),
            )
        except ValueError as error:
            return ActionTrustDecision(
                encoded_control=key,
                action=action,
                eligible=False,
                rejection_reasons=('invalid_state_or_disturbance_context',),
                notes=notes + (f'{type(error).__name__}: {error}',),
                global_support=self._global_metadata(key),
                local_support={},
            )
        standardized = (raw_context - self._context_mean) / self._context_scale
        global_support = self._global_metadata(key)
        local_support = self._local_metadata(key, standardized)
        reasons = []
        if key not in self._candidate_keys:
            reasons.append('outside_configured_candidate_pool')

        if self.config.enabled:
            sample_count = int(global_support['training_sample_count'])
            episode_count = int(global_support['training_episode_count'])
            if sample_count == 0:
                reasons.append('not_observed_in_training')
            if sample_count < self.config.minimum_training_samples:
                reasons.append('insufficient_global_training_samples')
            if episode_count < self.config.minimum_training_episodes:
                reasons.append('insufficient_global_training_episodes')

            if self.config.local_support_enabled:
                if int(local_support['reference_training_sample_count']) == 0:
                    reasons.append('no_local_training_context')
                nearest = local_support['nearest_context_distance']
                maximum_distance = self.config.maximum_context_distance
                if (
                    maximum_distance is not None
                    and (nearest is None or float(nearest) > maximum_distance + 1e-12)
                ):
                    reasons.append('nearest_context_distance_above_threshold')
                if (
                    int(local_support['samples_within_radius'])
                    < self.config.minimum_local_samples
                ):
                    reasons.append('insufficient_local_context_samples')
                maximum_mean = self.config.maximum_mean_k_distance
                mean_k = local_support['mean_k_nearest_context_distance']
                if (
                    maximum_mean is not None
                    and (mean_k is None or float(mean_k) > maximum_mean + 1e-12)
                ):
                    reasons.append('mean_k_nearest_context_distance_above_threshold')

        return ActionTrustDecision(
            encoded_control=key,
            action=action,
            eligible=not reasons,
            rejection_reasons=tuple(reasons),
            notes=notes,
            global_support=global_support,
            local_support=local_support,
        )

    def evaluate_candidates(self, state, disturbance) -> list[ActionTrustDecision]:
        """Audit every configured candidate action for one current context."""
        return [
            self.evaluate(vector, state, disturbance)
            for vector in self.candidate_controls
        ]

    def eligible_control_vectors(
        self,
        state,
        disturbance,
    ) -> tuple[np.ndarray, list[ActionTrustDecision]]:
        """Return currently trusted action vectors and all audit decisions."""
        decisions = self.evaluate_candidates(state, disturbance)
        eligible = [
            np.asarray(decision.encoded_control, dtype=float)
            for decision in decisions
            if decision.eligible and decision.encoded_control is not None
        ]
        vectors = (
            np.asarray(eligible, dtype=float)
            if eligible
            else np.empty((0, self.data.encoder.n_features), dtype=float)
        )
        return vectors, decisions

    def to_metadata(self) -> Dict[str, object]:
        """Return train-only audit metadata safe to save with MPC outputs."""
        actions = []
        for vector in self.candidate_controls:
            key = _action_key(vector)
            actions.append(
                {
                    'encoded_control': list(key),
                    'action': self.data.encoder.decode_action(vector),
                    'global_support': self._global_metadata(key),
                }
            )
        return {
            'protocol': {
                'name': 'train_only_exact_action_local_context_trust_gate',
                'training_rows_are_valid_one_minute_transitions_only': True,
                'uses_power_targets': False,
                'uses_next_state_targets': False,
                'training_rows_input_contract': (
                    'Caller must supply train-only source rows; '
                    'build_from_chronological_train_end enforces the ordinary '
                    'chronological-prefix case.'
                ),
                'context_features': [
                    *self.data.state_names,
                    *self.data.disturbance_names,
                ],
                'distance': 'standardized RMS Euclidean distance',
                'episode_definition': (
                    'maximal contiguous one-minute run of one canonical exact '
                    'joint action'
                ),
            },
            'thresholds': self.config.to_metadata(),
            'training_transition_count': int(len(self.training_rows)),
            'candidate_action_count': int(len(self.candidate_controls)),
            'actions': actions,
        }


__all__ = [
    'ActionTrustConfig',
    'ActionTrustDecision',
    'ActionTrustEligibility',
    'canonicalize_action',
    'canonicalize_control_vector',
    'canonical_control_matrix',
    'canonical_control_episode_ids',
]
