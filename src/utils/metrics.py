import numpy as np


def RSE(pred, true, eps=1e-8):
    numerator = np.sqrt(np.sum((true - pred) ** 2))
    denominator = np.sqrt(np.sum((true - true.mean()) ** 2))
    return numerator / denominator if denominator > eps else None


def CORR(pred, true, eps=1e-8):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    valid = d > eps
    return (u[valid] / d[valid]).mean(-1) if np.any(valid) else None


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true, threshold=0.1):
    """Conventional MAPE, excluding only near-zero actual values."""
    true_flat = true.flatten()
    pred_flat = pred.flatten()
    mask = np.abs(true_flat) > threshold
    if mask.sum() == 0:
        return None
    return np.mean(np.abs((pred_flat[mask] - true_flat[mask]) / true_flat[mask])) * 100


def MSPE(pred, true, threshold=0.1):
    true_flat = true.flatten()
    pred_flat = pred.flatten()
    mask = np.abs(true_flat) > threshold
    if mask.sum() == 0:
        return None
    return np.mean(np.square((pred_flat[mask] - true_flat[mask]) / true_flat[mask])) * 100


def WAPE(pred, true, eps=1e-8):
    """Aggregate percentage error, robust to individual near-zero samples."""
    denominator = np.sum(np.abs(true))
    if denominator <= eps:
        return None
    return np.sum(np.abs(pred - true)) / denominator * 100


def SMAPE(pred, true, eps=1e-8):
    """Symmetric percentage error bounded between 0 and 200 percent."""
    denominator = np.abs(pred) + np.abs(true)
    mask = denominator > eps
    if mask.sum() == 0:
        return None
    return np.mean(2.0 * np.abs(pred[mask] - true[mask]) / denominator[mask]) * 100


def NMAE(pred, true):
    mae = MAE(pred, true)
    true_flat = true.flatten()
    range_val = true_flat.max() - true_flat.min()
    return (mae / range_val) * 100 if range_val > 1e-8 else None


def Kweather_metric(pred, true, eps=1e-8):
    """Return 1 - MAE / mean(actual power), as a fraction and percentage."""
    mae = MAE(pred, true)
    mean_true = float(np.mean(true))
    if abs(mean_true) <= eps:
        return None, None
    accuracy_fraction = 1.0 - (mae / mean_true)
    return accuracy_fraction, accuracy_fraction * 100.0


def _rounded(value):
    return round(float(value), 4) if value is not None else None


def cal_metric(pred, true):
    accuracy_fraction, accuracy_percent = Kweather_metric(pred, true)
    return {
        # Preserve the historical keys used by existing report consumers.
        'prediction_accuracy': _rounded(accuracy_fraction),
        'prediction_accuracy*100': _rounded(accuracy_percent),
        'mse': _rounded(MSE(pred, true)),
        'mae': _rounded(MAE(pred, true)),
        'rmse': _rounded(RMSE(pred, true)),
        'mape': _rounded(MAPE(pred, true)),
        'mspe': _rounded(MSPE(pred, true)),
        'wape': _rounded(WAPE(pred, true)),
        'smape': _rounded(SMAPE(pred, true)),
        'rse': _rounded(RSE(pred, true)),
        'nmae': _rounded(NMAE(pred, true)),
    }
