"""Explicit registry for six independently trained control-model families."""

MODEL_FAMILIES = (
    'affine', 'mode_history', 'physics_residual',
    'scenario_ensemble', 'neural_ensemble', 'decision_focused', 'neural_history',
)


def create_control_model(family, spec, parameters=None, seed=42):
    from .affine import AffineControlModel, ModeHistoryModel
    from .ensemble import EnsembleControlModel
    from .neural import NeuralHistoryModel, PhysicsResidualModel
    from .decision_focused import DecisionFocusedModel
    constructors = {
        'affine': AffineControlModel,
        'mode_history': ModeHistoryModel,
        'physics_residual': PhysicsResidualModel,
        'neural_history': NeuralHistoryModel,
        'decision_focused': DecisionFocusedModel,
    }
    if family == 'scenario_ensemble':
        return EnsembleControlModel(spec, parameters, seed, 'mode_history')
    if family == 'neural_ensemble':
        return EnsembleControlModel(spec, parameters, seed, 'neural_history')
    if family not in constructors:
        raise ValueError(f'Unknown control model {family!r}: {MODEL_FAMILIES}')
    return constructors[family](spec, parameters, seed)
