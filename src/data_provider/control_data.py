"""Physical-unit data contract used only by the MPC framework."""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def _category_token(value) -> str:
    """Return one stable token for YAML, string, integer, and float categories."""
    if pd.isna(value):
        return ''
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


@dataclass(frozen=True)
class ControlVariable:
    name: str
    categories: tuple[str, ...]
    default: str


class ControlEncoder:
    """One-hot encoder whose category order is owned by the MPC configuration."""

    def __init__(self, variables: Sequence[ControlVariable]):
        self.variables = tuple(variables)
        self._indices: Dict[tuple[str, str], int] = {}
        self.groups: Dict[str, tuple[int, ...]] = {}
        feature_names: List[str] = []
        offset = 0
        for variable in self.variables:
            group = []
            for category in variable.categories:
                self._indices[(variable.name, category)] = offset
                group.append(offset)
                feature_names.append(f'{variable.name}={category}')
                offset += 1
            self.groups[variable.name] = tuple(group)
        self.feature_names = tuple(feature_names)
        self.n_features = offset

    @classmethod
    def from_config(cls, entries: Iterable[Mapping]) -> 'ControlEncoder':
        variables = []
        for raw_entry in entries:
            entry = OmegaConf.to_container(raw_entry, resolve=True)
            categories = tuple(_category_token(v) for v in entry['categories'])
            default = _category_token(entry['default'])
            if len(categories) != len(set(categories)):
                raise ValueError(f"Duplicate categories for {entry['name']}: {categories}")
            if default not in categories:
                raise ValueError(
                    f"Default {default!r} is not a category of {entry['name']!r}"
                )
            variables.append(ControlVariable(entry['name'], categories, default))
        return cls(variables)

    def index(self, variable: str, category) -> int:
        key = (variable, _category_token(category))
        if key not in self._indices:
            raise KeyError(f'Unknown MPC control category: {key}')
        return self._indices[key]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = np.zeros((len(frame), self.n_features), dtype=float)
        for variable in self.variables:
            values = frame[variable.name].map(_category_token)
            unknown = sorted(set(values) - set(variable.categories))
            if unknown:
                raise ValueError(
                    f"{variable.name!r} contains categories absent from MPC config: {unknown}"
                )
            for category in variable.categories:
                encoded[:, self.index(variable.name, category)] = (
                    values == category
                ).to_numpy(dtype=float)
        return encoded

    def encode_action(self, action: Mapping[str, object] | None = None) -> np.ndarray:
        action = action or {}
        vector = np.zeros(self.n_features, dtype=float)
        for variable in self.variables:
            value = action.get(variable.name, variable.default)
            vector[self.index(variable.name, value)] = 1.0
        return vector

    def decode_action(self, vector: np.ndarray) -> Dict[str, str]:
        vector = np.asarray(vector)
        decoded = {}
        for variable in self.variables:
            group = self.groups[variable.name]
            selected = group[int(np.argmax(vector[list(group)]))]
            decoded[variable.name] = self.feature_names[selected].split('=', 1)[1]
        return decoded

    def valid_control_vectors(self, logical_rules=()) -> np.ndarray:
        """Enumerate categorical actions that satisfy all configured rules."""
        rules = [
            OmegaConf.to_container(rule, resolve=True)
            if not isinstance(rule, dict)
            else dict(rule)
            for rule in logical_rules
        ]
        vectors = []
        category_sets = [variable.categories for variable in self.variables]
        for categories in product(*category_sets):
            action = {
                variable.name: category
                for variable, category in zip(self.variables, categories)
            }
            valid = True
            for rule in rules:
                when = rule['when']
                if action[when['variable']] != _category_token(when['value']):
                    continue
                for variable, value in rule.get('require', {}).items():
                    if action[variable] != _category_token(value):
                        valid = False
                        break
                if not valid:
                    break
                for variable, values in rule.get('forbid', {}).items():
                    forbidden = {_category_token(value) for value in values}
                    if action[variable] in forbidden:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                vectors.append(self.encode_action(action))
        if not vectors:
            raise ValueError('MPC logical rules leave no valid control combination')
        return np.asarray(vectors, dtype=float)


@dataclass
class ControlData:
    timestamps: pd.Series
    states: np.ndarray
    controls: np.ndarray
    disturbances: np.ndarray
    powers: np.ndarray
    state_names: tuple[str, ...]
    disturbance_names: tuple[str, ...]
    power_names: tuple[str, ...]
    encoder: ControlEncoder
    control_preprocessing: Dict[str, object] = field(default_factory=dict)

    def split_indices(self, split_rate: Sequence[float]) -> tuple[int, int]:
        if len(split_rate) != 3 or not np.isclose(sum(split_rate), 1.0):
            raise ValueError('DATASET.split_rate must contain three values summing to 1')
        train_end = int(len(self.states) * split_rate[0])
        test_start = int(len(self.states) * (split_rate[0] + split_rate[1]))
        return train_end, test_start

    def transition_mask(self, start: int, end: int) -> np.ndarray:
        """Mask transitions that are exactly one minute apart."""
        start = max(0, int(start))
        end = min(int(end), len(self.timestamps) - 1)
        delta = (
            self.timestamps.iloc[start + 1 : end + 1].reset_index(drop=True)
            - self.timestamps.iloc[start:end].reset_index(drop=True)
        )
        return delta.eq(pd.Timedelta(minutes=1)).to_numpy()

    def model_arrays(self, start: int, end: int):
        """Return x(t), z(t), d(t), x(t+1), and p(t) for valid transitions."""
        mask = self.transition_mask(start, end)
        x = self.states[start:end][mask]
        z = self.controls[start:end][mask]
        d = self.disturbances[start:end][mask]
        x_next = self.states[start + 1 : end + 1][mask]
        power = self.powers[start:end][mask]
        return x, z, d, x_next, power

    def valid_transition_rows(self, row_indices) -> np.ndarray:
        """Return unique source rows whose next observation is one minute later.

        Evaluation-only grouped holdouts are represented by explicit transition
        rows rather than by reordering the physical time series.  Keeping the
        original row numbers makes split membership auditable and prevents a
        synthetic transition from being created between unrelated episodes.
        """
        rows = np.asarray(row_indices, dtype=int).reshape(-1)
        if not len(rows):
            return np.empty(0, dtype=int)
        if (rows < 0).any() or (rows >= len(self.states) - 1).any():
            raise IndexError('Transition row indices must be in [0, n_rows - 1)')
        rows = np.unique(rows)
        delta = (
            self.timestamps.iloc[rows + 1].reset_index(drop=True)
            - self.timestamps.iloc[rows].reset_index(drop=True)
        )
        return rows[delta.eq(pd.Timedelta(minutes=1)).to_numpy()]

    def model_arrays_for_rows(self, row_indices):
        """Return model arrays for explicit, leakage-auditable transition rows."""
        rows = self.valid_transition_rows(row_indices)
        if not len(rows):
            raise ValueError('No valid one-minute transitions in requested rows')
        return (
            self.states[rows],
            self.controls[rows],
            self.disturbances[rows],
            self.states[rows + 1],
            self.powers[rows],
        )


def _resolve_csv_path(cfg) -> Path:
    base = Path(cfg.DATASET.datadir).expanduser()
    src_dir = Path(__file__).resolve().parents[1]
    candidates = [base, Path.cwd() / base, src_dir / base]
    for candidate in candidates:
        csv_path = (
            candidate.resolve()
            / cfg.DATASET.data_version
            / f'{cfg.DATASET.environment_name}.csv'
        )
        if csv_path.is_file():
            return csv_path
    searched = ', '.join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f'MPC data directory was not found. Searched: {searched}')


def _fill_numeric(frame: pd.DataFrame, names: Sequence[str]) -> None:
    for name in names:
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    frame[list(names)] = (
        frame[list(names)].interpolate(limit_direction='both').ffill().bfill()
    )
    if frame[list(names)].isna().any().any():
        missing = frame[list(names)].columns[frame[list(names)].isna().any()].tolist()
        raise ValueError(f'Numeric MPC columns contain unresolved missing values: {missing}')


def consolidate_control_categories(
    frame: pd.DataFrame,
    encoder: ControlEncoder,
    config,
) -> Dict[str, object]:
    """Map fine-grained recorded categories to configured representatives."""
    enabled = bool(config.get('enabled', False)) if config is not None else False
    if not enabled:
        return {
            'enabled': False,
            'mappings': {},
            'replacement_count_by_variable': {},
            'total_replacement_count': 0,
        }

    raw_mappings = config.get('mappings', {})
    if OmegaConf.is_config(raw_mappings):
        raw_mappings = OmegaConf.to_container(raw_mappings, resolve=True)
    raw_mappings = dict(raw_mappings or {})
    if not raw_mappings:
        raise ValueError(
            'Category consolidation is enabled but no mappings are configured'
        )

    known_variables = {variable.name for variable in encoder.variables}
    unknown_variables = sorted(set(raw_mappings) - known_variables)
    if unknown_variables:
        raise ValueError(
            'Category consolidation references unknown control variables: '
            f'{unknown_variables}'
        )

    normalized: Dict[str, Dict[str, str]] = {}
    for variable_name, raw_mapping in raw_mappings.items():
        if not isinstance(raw_mapping, Mapping):
            raise ValueError(
                f'Category consolidation for {variable_name!r} must be a mapping'
            )
        mapping = {
            _category_token(source): _category_token(target)
            for source, target in raw_mapping.items()
        }
        for source, target in mapping.items():
            encoder.index(variable_name, source)
            encoder.index(variable_name, target)

        resolved = {}
        for source in mapping:
            target = mapping[source]
            visited = {source}
            while target in mapping and mapping[target] != target:
                if target in visited:
                    raise ValueError(
                        f'Category consolidation cycle for {variable_name!r}: '
                        f'{mapping}'
                    )
                visited.add(target)
                target = mapping[target]
            resolved[source] = target
        normalized[variable_name] = resolved

    replacement_count_by_variable = {}
    for variable_name, mapping in normalized.items():
        original = frame[variable_name].map(_category_token)
        variable_count = 0
        for source, target in mapping.items():
            if source == target:
                continue
            selected = original.eq(source)
            count = int(selected.sum())
            frame.loc[selected, variable_name] = target
            variable_count += count
        replacement_count_by_variable[variable_name] = variable_count

    return {
        'enabled': True,
        'mappings': normalized,
        'replacement_count_by_variable': replacement_count_by_variable,
        'total_replacement_count': int(
            sum(replacement_count_by_variable.values())
        ),
    }


def _canonicalize_controls(frame: pd.DataFrame, encoder: ControlEncoder, rules) -> None:
    """Apply the same inactive-equipment semantics used by the MILP."""
    for raw_rule in rules:
        rule = OmegaConf.to_container(raw_rule, resolve=True)
        when = rule["when"]
        when_value = _category_token(when["value"])
        encoder.index(when["variable"], when_value)
        active = frame[when["variable"]].map(_category_token).eq(when_value)
        for variable, value in rule.get("require", {}).items():
            required_value = _category_token(value)
            encoder.index(variable, required_value)
            frame.loc[active, variable] = required_value
        for variable, values in rule.get("forbid", {}).items():
            forbidden = {_category_token(value) for value in values}
            invalid = active & frame[variable].map(_category_token).isin(forbidden)
            if invalid.any():
                raise ValueError(
                    f"Observed controls violate logical rule {rule}: "
                    f"{int(invalid.sum())} rows"
                )


def load_control_data(cfg) -> ControlData:
    """Load chronological, unscaled data for system identification and MPC."""
    encoder = ControlEncoder.from_config(cfg.MPC.control_variables)
    state_names = tuple(item['name'] for item in cfg.MPC.state_constraints)
    disturbance_names = tuple(cfg.MPC.disturbance_variables)
    power_names = tuple(cfg.MPC.power_variables)
    required = {
        '데이터 시간',
        *state_names,
        *disturbance_names,
        *power_names,
        *(variable.name for variable in encoder.variables),
    }

    csv_path = _resolve_csv_path(cfg)
    frame = pd.read_csv(csv_path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'MPC data is missing required columns: {missing}')

    frame['데이터 시간'] = pd.to_datetime(frame['데이터 시간'], errors='raise')
    frame = (
        frame.sort_values('데이터 시간')
        .drop_duplicates('데이터 시간', keep='last')
        .reset_index(drop=True)
    )

    start_date = cfg.DATASET.get("start_date")
    end_date = cfg.DATASET.get("end_date")
    if start_date is not None:
        frame = frame[frame["데이터 시간"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame["데이터 시간"] <= pd.Timestamp(end_date)]
    frame = frame.reset_index(drop=True)

    for variable in encoder.variables:
        frame[variable.name] = frame[variable.name].map(_category_token)
        frame.loc[frame[variable.name] == '', variable.name] = variable.default

    action_options = cfg.MPC.evaluation.get('action_options', {})
    category_config = action_options.get('category_consolidation', {})
    category_consolidation = consolidate_control_categories(
        frame,
        encoder,
        category_config,
    )
    _canonicalize_controls(frame, encoder, cfg.MPC.logical_rules)

    numeric_names = tuple(dict.fromkeys(state_names + disturbance_names + power_names))
    _fill_numeric(frame, numeric_names)

    if len(frame) < int(cfg.MPC.prediction_horizon) + 2:
        raise ValueError('Not enough rows for the configured MPC prediction horizon')

    return ControlData(
        timestamps=frame['데이터 시간'].copy(),
        states=frame[list(state_names)].to_numpy(dtype=float),
        controls=encoder.transform(frame),
        disturbances=frame[list(disturbance_names)].to_numpy(dtype=float),
        powers=frame[list(power_names)].to_numpy(dtype=float),
        state_names=state_names,
        disturbance_names=disturbance_names,
        power_names=power_names,
        encoder=encoder,
        control_preprocessing={
            'category_consolidation': category_consolidation,
        },
    )
