import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true, threshold=0.1):
    true_flat = true.flatten()
    pred_flat = pred.flatten()
    mask = np.abs(true_flat) > threshold
    if mask.sum() > 0:
        return np.mean(np.abs((pred_flat[mask] - true_flat[mask]) / true_flat[mask])) * 100
    else:
        return None

def MSPE(pred, true, threshold=0.1):
    true_flat = true.flatten()
    pred_flat = pred.flatten()
    mask = np.abs(true_flat) > threshold
    if mask.sum() > 0:
        return np.mean(np.square((pred_flat[mask] - true_flat[mask]) / true_flat[mask])) * 100
    else:
        return None

def NMAE(pred, true):
    mae = MAE(pred, true)
    true_flat = true.flatten()
    range_val = true_flat.max() - true_flat.min()
    if range_val > 1e-8:
        return (mae / range_val) * 100
    else:
        return None

def Kweather_metric(pred, true, eps=1e-8):
    """
    전력량예측정확도(슬라이드): 1 - MAE / (평균 실제사용전력량)
    반환: (정확도-분수값[0~1], 정확도-백분율[%])
    """
    mae = MAE(pred, true)
    mean_true = float(np.mean(true))
    # # 분모가 0 또는 매우 작으면 |true|의 평균으로 보정
    # if abs(mean_true) <= eps:
    #     mean_true = float(np.mean(np.abs(true)))
    # if abs(mean_true) <= eps:
    #     return None, None
    acc_frac = 1.0 - (mae / mean_true)
    acc_pct = acc_frac * 100.0
    return acc_frac, acc_pct

def cal_metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    nmae = NMAE(pred, true)

    # 추가: 슬라이드 지표
    acc_img_frac, acc_img_pct = Kweather_metric(pred, true)

    metric_dict = {
        'prediction_accuracy': round(float(acc_img_frac), 4) if acc_img_frac is not None else None,  # 원본 유지
        'prediction_accuracy*100': round(float(acc_img_pct), 4) if acc_img_pct  is not None else None,  # 원본 유지
        'mse': round(float(mse), 4),
        'mae': round(float(mae), 4),
        'rmse': round(float(rmse), 4),
        'mape': round(float(mape), 4) if mape is not None else None,
        'mspe': round(float(mspe), 4) if mspe is not None else None,
        'rse': round(float(rse), 4),
        'nmae': round(float(nmae), 4) if nmae is not None else None,
    }
    return metric_dict