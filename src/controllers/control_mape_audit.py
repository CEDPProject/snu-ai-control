"""전력 MAPE 감사. factual 정답, 분모 제외율, rollout을 명시적으로 구분한다."""

import numpy as np


def power_metric_rows(actual, predicted, names, *, protocol, split, floor_w,
                      target_percent=None, minimum_samples=30):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.ndim != 2 or actual.shape != predicted.shape or len(names) != actual.shape[1]:
        raise ValueError('Power arrays/names must have matching two-dimensional shapes')
    if not np.isfinite(floor_w) or floor_w < 0:
        raise ValueError('floor_w must be finite and nonnegative')
    rows = []
    for j, name in enumerate([*names, '총전력']):
        a = actual[:, j] if j < len(names) else actual.sum(1)
        p = predicted[:, j] if j < len(names) else predicted.sum(1)
        finite = np.isfinite(a) & np.isfinite(p)
        mask = finite & (np.abs(a) > floor_w)
        count = int(mask.sum())
        error = np.abs(a[finite]-p[finite])
        denominator = float(np.abs(a[finite]).sum())
        value = float(100*np.mean(np.abs(a[mask]-p[mask])/np.abs(a[mask]))) if count else None
        mae = float(error.mean()) if len(error) else None
        sse = float(np.square(a[finite]-p[finite]).sum())
        sst = float(np.square(a[finite]-a[finite].mean()).sum()) if finite.any() else 0
        valid = bool(finite.all() and count >= minimum_samples)
        rows.append({
            'protocol': protocol, 'split': split, 'target': name,
            'floor_w': float(floor_w),
            'target_percent': None if target_percent is None else float(target_percent),
            'sample_count': len(a), 'finite_count': int(finite.sum()),
            'nonfinite_count': int((~finite).sum()), 'mape_valid_samples': count,
            'denominator_coverage_percent': 100*count/len(a) if len(a) else 0.,
            'mape_percent': value,
            'wape_percent': 100*float(error.sum())/denominator if denominator > 1e-12 else None,
            'mae_w': mae,
            'rmse_w': float(np.sqrt(sse/len(error))) if len(error) else None,
            'r2': 1-sse/sst if sst > 1e-12 else None,
            'absolute_error_sum_w': float(error.sum()),
            'absolute_actual_sum_w': denominator,
            'absolute_percentage_error_sum': float(
                np.sum(np.abs(a[mask]-p[mask])/np.abs(a[mask]))) if count else 0.,
            'minimum_samples': minimum_samples,
            'metric_reportable': valid,
            'mape_target_pass': None if target_percent is None else bool(
                valid and value is not None and value <= target_percent),
            'claim_boundary': 'prediction_accuracy_not_energy_savings_or_counterfactual_truth',
        })
    return rows
