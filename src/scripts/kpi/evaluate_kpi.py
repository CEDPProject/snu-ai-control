"""KPI 평가 프로그램: 고정된 forecast 모델로 시험 구간을 추론하고 표준화 MSE를 산출한다.

학습은 하지 않는다. 공간(environment)마다 ``<model-dir>/<env>/configs.yaml``(학습 시 저장된 설정)과
``best_model.pt``(학습된 가중치)를 읽고, 학습 파이프라인과 같은 데이터 로더로 시험 구간의
모든 예측 구간을 추론한다. MSE는 학습 구간의 평균·표준편차로 표준화한 값에서 계산하며,
3개 공간의 모든 예측값을 합친 통합 MSE를 KPI 목표와 비교한다.

사용 예 (저장소 루트에서):
    python src/scripts/kpi/evaluate_kpi.py \
        --data-dir /path/to/data/version4_260805_processed \
        --model-dir kpi_release/models --out kpi_result
"""

import argparse
import hashlib
import json
import logging
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from omegaconf import OmegaConf

from data_provider import create_dataloader
from models import create_model
from utils.tools import update_information

ENV_LABELS = {'livingroom': '거실', 'room1': '방1', 'room2': '방2'}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1 for ch in str(text))


def pad(text, width: int, align: str = 'left') -> str:
    text = str(text)
    space = ' ' * max(0, width - display_width(text))
    return text + space if align == 'left' else space + text


def print_row(cells, widths, aligns):
    print('  ' + '  '.join(pad(c, w, a) for c, w, a in zip(cells, widths, aligns)))


def load_split(cfg, data_dir: Path):
    """Build the loaders exactly as the forecast runner does, with the dataset path overridden."""
    dataset = cfg.DATASET
    return create_dataloader(
        datadir=str(data_dir.parent),
        scaler=dataset.scaler,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        data_version=data_dir.name,
        data_versions=None,
        merge_strategy=dataset.get('merge_strategy', 'segmented_concat'),
        prevent_cross_segment_windows=dataset.get('prevent_cross_segment_windows', True),
        segment_split=dataset.get('segment_split', 'global'),
        split_rate=dataset.split_rate,
        seq_len=dataset.seq_len,
        stride_len=dataset.stride_len,
        pred_len=dataset.pred_len,
        model_name=cfg.MODEL.modelname,
        environment_name=dataset.environment_name,
        target_features=dataset.target_features,
        input_features=dataset.input_features,
        ord_categorical_feature=dataset.ord_categorical_feature,
        categorical_feature=dataset.categorical_features,
        cat_encode_method=dataset.cat_encode_method,
        numerical_feature=dataset.numerical_features,
        del_feature=dataset.ignore_feature,
        split_date=dataset.split_date,
        end_date=dataset.end_date,
        start_date=dataset.start_date,
    )


def period(timestamps) -> str:
    ts = np.asarray(timestamps).astype('datetime64[m]')
    fmt = lambda t: str(t).replace('T', ' ')
    return f'{fmt(ts[0])} ~ {fmt(ts[-1])}'


@torch.no_grad()
def predict(model, loader, device):
    outputs, targets = [], []
    for item in loader:
        inputs = item['input']
        input_cat = inputs['cat'].to(device) if torch.is_tensor(inputs['cat']) else None
        input_num = inputs['num'].to(device) if torch.is_tensor(inputs['num']) else None
        out = model(input_cat=input_cat, input_num=input_num, input_ts=None)
        outputs.append(out.float().cpu().numpy())
        targets.append(item['target']['target'].float().numpy())
    return np.concatenate(outputs), np.concatenate(targets)


def evaluate_env(env: str, model_dir: Path, data_dir: Path, device) -> dict:
    env_dir = model_dir / env
    cfg = OmegaConf.load(env_dir / 'configs.yaml')
    checkpoint = env_dir / 'best_model.pt'

    trn, val, tst, metadata, scaler_dict = load_split(cfg, data_dir)
    update_information(model_name=cfg.MODEL.modelname, cfg=cfg, information_dict=metadata)
    model = create_model(modelname=cfg.MODEL.modelname, params=cfg.MODELSETTING)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.to(device).eval()

    out_norm, tgt_norm = predict(model, tst, device)
    target_scaler = scaler_dict['target_scaler']
    shape = out_norm.shape
    out_w = target_scaler.inverse_transform(out_norm.reshape(-1, shape[-1])).reshape(shape)
    tgt_w = target_scaler.inverse_transform(tgt_norm.reshape(-1, shape[-1])).reshape(shape)

    dataset = tst.dataset
    all_ts = np.asarray(dataset.timestamps)
    starts = np.asarray(dataset.start_indices)
    first_target_time = all_ts[starts + dataset.seq_len]

    return {
        'env': env,
        'model': cfg.MODEL.modelname,
        'target': list(cfg.DATASET.target_features),
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256(checkpoint),
        'rows': {name: int(len(d.dataset.data)) for name, d in (('train', trn), ('val', val), ('test', tst))},
        'test_period': period(all_ts),
        'train_period': period(trn.dataset.timestamps),
        'windows': int(shape[0]),
        'points': int(out_norm.size),
        'sum_sq_err_norm': float(((out_norm - tgt_norm) ** 2).sum()),
        'mse_norm': float(((out_norm - tgt_norm) ** 2).mean()),
        'mse_w2': float(((out_w - tgt_w) ** 2).mean()),
        'target_mean_w': float(target_scaler.mean_[0]),
        'target_std_w': float(target_scaler.scale_[0]),
        'seq_len': int(dataset.seq_len),
        'pred_len': int(dataset.pred_len),
        '_arrays': {'outputs_w': out_w, 'targets_w': tgt_w, 'first_target_time': first_target_time},
    }


def plot_results(results, pooled, kpi, out_dir: Path, font_path, day: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates
    from matplotlib import font_manager

    if font_path and Path(font_path).is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    ink, ink2, grid, surface = '#0b0b0b', '#52514e', '#e4e3de', '#fcfcfb'
    actual_c, pred_c = '#2a78d6', '#eb6834'  # categorical slots 1-2 of the validated palette

    # 1) 실측 vs 60분 앞 예측 (대표 하루, 공간별)
    fig, axes = plt.subplots(len(results), 1, figsize=(11, 3.1 * len(results)), dpi=150, facecolor=surface)
    axes = np.atleast_1d(axes)
    day_start = np.datetime64(day)
    day_end = day_start + np.timedelta64(1, 'D')
    for ax, res in zip(axes, results):
        arr = res['_arrays']
        horizon = res['pred_len']
        # 각 예측 구간의 마지막 시점 = 예측 시작 시각에서 60분 앞 예측
        target_time = arr['first_target_time'] + np.timedelta64(horizon - 1, 'm')
        keep = (target_time >= day_start) & (target_time < day_end)
        t = target_time[keep].astype('datetime64[m]').astype(object)
        ax.set_facecolor(surface)
        ax.plot(t, arr['targets_w'][keep, -1, 0], color=actual_c, lw=1.6, label='실측')
        ax.plot(t, arr['outputs_w'][keep, -1, 0], color=pred_c, lw=1.6, label='60분 앞 예측')
        ax.set_title(f"{ENV_LABELS.get(res['env'], res['env'])}  (시험 구간 전체 MSE {res['mse_norm']:.4f})",
                     loc='left', fontsize=11, color=ink)
        ax.set_ylabel('환기 유효전력 (W)', color=ink2, fontsize=9)
        ax.grid(True, color=grid, lw=0.6)
        ax.tick_params(colors=ink2, labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.legend(loc='upper right', frameon=False, fontsize=9, labelcolor=ink)
    axes[-1].set_xlabel(f'시각 ({day})', color=ink2, fontsize=9)
    fig.suptitle('환기 유효전력 실측값과 60분 앞 예측값 (케이웨더 v4 시험 구간 중 하루)',
                 x=0.01, ha='left', fontsize=12, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / 'pred_vs_actual.png', facecolor=surface)
    plt.close(fig)

    # 2) 공간별·통합 MSE와 목표선
    labels = [ENV_LABELS.get(r['env'], r['env']) for r in results] + ['3개 공간 통합']
    values = [r['mse_norm'] for r in results] + [pooled['mse_norm']]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    bars = ax.bar(labels, values, width=0.55, color=actual_c, edgecolor=surface, linewidth=2)
    bars[-1].set_hatch('///')
    bars[-1].set_facecolor('#8fb5e6')
    bars[-1].set_edgecolor(actual_c)
    y_max = max(max(values), kpi) * 1.2
    gap = y_max * 0.02
    labels_above = kpi - max(values) > y_max * 0.12  # 목표선이 막대보다 충분히 위에 있으면 값을 막대 위에
    for i, (bar, value) in enumerate(zip(bars, values)):
        pooled_bar = i == len(values) - 1
        x = bar.get_x() + bar.get_width() / 2
        if labels_above:
            ax.text(x, value + gap, f'{value:.4f}', ha='center', va='bottom', fontsize=9.5, color=ink, fontweight='bold')
        else:
            ax.text(x, value - gap, f'{value:.4f}', ha='center', va='top', fontsize=9.5,
                    color=ink if pooled_bar else surface, fontweight='bold')
    ax.axhline(kpi, color=ink2, lw=1.2, ls=(0, (5, 3)))
    if labels_above:
        ax.text(-0.45, kpi - gap, f'목표 {kpi:.3f} 이하', ha='left', va='top', fontsize=9, color=ink2)
    else:
        ax.text(-0.45, kpi + gap / 2, f'목표 {kpi:.3f} 이하', ha='left', va='bottom', fontsize=9, color=ink2)
    ax.set_ylim(0, y_max)
    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.set_ylabel('MSE (학습 구간 기준 표준화)', color=ink2, fontsize=9)
    ax.set_title('공간별·통합 MSE (케이웨더 v4 시험 구간)', loc='left', fontsize=11, color=ink)
    ax.grid(True, axis='y', color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=ink2, labelsize=9)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(grid)
    fig.tight_layout()
    fig.savefig(out_dir / 'mse_summary.png', facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Forecast KPI evaluation (inference only)')
    parser.add_argument('--data-dir', required=True, type=Path,
                        help='데이터 버전 폴더 (예: .../kweather/version4_260805_processed)')
    parser.add_argument('--model-dir', required=True, type=Path,
                        help='<env>/best_model.pt 와 <env>/configs.yaml 이 있는 폴더')
    parser.add_argument('--envs', nargs='+', default=['livingroom', 'room1', 'room2'])
    parser.add_argument('--out', type=Path, default=Path('kpi_result'))
    parser.add_argument('--kpi', type=float, default=0.233, help='통합 MSE 목표(이하이면 PASS)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--plot-day', default='2026-07-20', help='비교 그래프에 그릴 날짜 (YYYY-MM-DD)')
    parser.add_argument('--font', type=Path, default=None,
                        help='그래프용 한글 폰트 (기본: <data-dir>/../fonts/NanumGothic-Regular.ttf)')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    data_dir = args.data_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[KPI 평가] {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  데이터   : {data_dir}")
    print(f"  모델     : {args.model_dir}")
    print(f"  장치     : {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == 'cuda' else ''))
    print(f"  목표     : 3개 공간 통합 MSE(표준화) <= {args.kpi:.3f}")

    results = [evaluate_env(env, args.model_dir, data_dir, device) for env in args.envs]
    first = results[0]

    print(f"\n[데이터 분할] 시간순 8:1:1, 입력 {first['seq_len']}분 → 예측 {first['pred_len']}분")
    widths, aligns = [10, 8, 8, 7, 7, 35, 9], ['left'] + ['right'] * 4 + ['left', 'right']
    print_row(['공간', '전체행', '학습행', '검증행', '시험행', '시험 구간', '예측구간'], widths, aligns)
    for r in results:
        rows = r['rows']
        print_row([ENV_LABELS.get(r['env'], r['env']), f"{sum(rows.values()):,}", f"{rows['train']:,}",
                   f"{rows['val']:,}", f"{rows['test']:,}", r['test_period'], f"{r['windows']:,}"], widths, aligns)

    pooled_points = sum(r['points'] for r in results)
    pooled = {
        'windows': sum(r['windows'] for r in results),
        'points': pooled_points,
        'mse_norm': sum(r['sum_sq_err_norm'] for r in results) / pooled_points,
    }
    passed = pooled['mse_norm'] <= args.kpi

    print("\n[추론 결과]")
    widths, aligns = [13, 9, 10, 12, 9, 8, 8], ['left'] + ['right'] * 6
    print_row(['공간', '예측구간', '예측값 수', 'MSE(표준화)', 'MSE(W²)', 'μ(W)', 'σ(W)'], widths, aligns)
    for r in results:
        print_row([ENV_LABELS.get(r['env'], r['env']), f"{r['windows']:,}", f"{r['points']:,}", f"{r['mse_norm']:.4f}",
                   f"{r['mse_w2']:.2f}", f"{r['target_mean_w']:.2f}", f"{r['target_std_w']:.2f}"], widths, aligns)
    print_row(['3개 공간 통합', f"{pooled['windows']:,}", f"{pooled['points']:,}", f"{pooled['mse_norm']:.4f}", '', '', ''],
              widths, aligns)
    print(f"\n[판정] 통합 MSE {pooled['mse_norm']:.4f} {'<=' if passed else '>'} {args.kpi:.3f} → {'PASS' if passed else 'FAIL'}")

    for r in results:
        arrays = r.pop('_arrays')
        np.savez_compressed(args.out / f"predictions_{r['env']}.npz",
                            outputs_w=arrays['outputs_w'], targets_w=arrays['targets_w'],
                            first_target_time=arrays['first_target_time'].astype('datetime64[m]').astype(str))
        r['_arrays'] = arrays
    report = {
        'executed_at': datetime.now().isoformat(timespec='seconds'),
        'data_dir': str(data_dir),
        'model_dir': str(args.model_dir),
        'device': str(device),
        'kpi_target_mse': args.kpi,
        'metric': 'MSE on targets standardized with the training split mean/std of each space',
        'spaces': {r['env']: {k: v for k, v in r.items() if not k.startswith('_') and k != 'sum_sq_err_norm'}
                   for r in results},
        'pooled': pooled,
        'result': 'PASS' if passed else 'FAIL',
    }
    (args.out / 'kpi_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    if not args.no_plot:
        font = args.font or (data_dir.parent / 'fonts' / 'NanumGothic-Regular.ttf')
        plot_results(results, pooled, args.kpi, args.out, font, args.plot_day)
    print(f"결과 파일: {args.out / 'kpi_result.json'}" + ('' if args.no_plot else
          f", {args.out / 'pred_vs_actual.png'}, {args.out / 'mse_summary.png'}"))


if __name__ == '__main__':
    main()
