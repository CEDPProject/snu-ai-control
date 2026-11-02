"""예측 성능 시험 프로그램: 시험 데이터로 예측 모델을 학습한 뒤 시험 구간을 평가해 공간별 MSE를 산출한다.

공간(environment)마다 시간순 8:1:1(학습/검증/시험)로 나누고, 학습·검증 구간으로 모델을 학습한 다음
(학습 파이프라인 ``src/main.py --task forecast`` 그대로 실행), 시험 구간의 모든 예측 구간(과거 120분 입력 →
향후 60분 예측)을 추론해 학습 구간 기준 표준화 MSE를 계산한다. 대표값은 공간별 MSE의 산술평균(공간 평균)이고
공간별 값은 세부 측정값으로 함께 출력한다. 프로그램은 측정값만 출력하고,
목표값과의 비교·적합 판정은 하지 않는다(시험기관이 판정).

학습 배치는 무작위 순서로 공급하고(``DATASET.shuffle=true``), 계속 증가하기만 하는 누적전력량 열
(``IGNORE_FEATURES``)은 시험 구간 값이 학습 범위를 벗어나므로 입력에서 뺀다. ``--seeds``로 시드를 여러 개
주면 공간마다 시드별로 학습·평가해 시드별 MSE와 평균·표준편차를 출력한다. 같은 시드·같은 PC에서는 같은 MSE가 나온다.

사용 예 (저장소 루트에서):
    python src/scripts/kpi/run_kpi.py \
        --data-dir /path/to/data/version4_260805_processed \
        --envs livingroom room1 room2 --out kpi_release/test_run --seeds 40 41 42 43 44
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

KPI_DIR = Path(__file__).resolve().parent
SRC_DIR = KPI_DIR.parents[1]
REPO_DIR = SRC_DIR.parent
for path in (SRC_DIR, KPI_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_kpi import ENV_LABELS, evaluate_env, load_split, period, print_row  # noqa: E402

TARGET = '환기유효전력(W)'
IGNORE_FEATURES = ['환기누적전력량(Wh)', '냉난방누적전력량(Wh)']


def build_opts(env: str, data_dir: Path, runs_dir: Path, exp_name: str, seed: int) -> list:
    return [
        f'DEFAULT.exp_name={exp_name}',
        f'DEFAULT.seed={seed}',
        f'DATASET.environment_name={env}',
        f"DATASET.target_features=['{TARGET}']",
        f'DATASET.ignore_feature={IGNORE_FEATURES}',
        'DATASET.shuffle=true',
        f'DATASET.datadir={data_dir.parent}',
        f'DATASET.data_version={data_dir.name}',
        'DATASET.segment_split=global',
        'DATASET.seq_len=120',
        'DATASET.pred_len=60',
        'DATASET.stride_len=1',
        'DATASET.split_rate=[0.8,0.1,0.1]',
        f'RESULT.savedir={runs_dir}',
        'TRAIN.del_pt_file=false',
    ]


def build_cfg(model: str, opts: list):
    """Assemble the same config as src/arguments.py does for --task forecast."""
    cfg = OmegaConf.load(SRC_DIR / 'configs' / 'default_setting.yaml')
    cfg = OmegaConf.merge(cfg, {'TASK': {'name': 'forecast'}, 'MODEL': {'modelname': model}})
    cfg = OmegaConf.merge(cfg, {'MODELSETTING': OmegaConf.load(SRC_DIR / 'configs' / 'model_setting.yaml')[model]})
    for opt in opts:
        key, value = opt.split('=', 1)
        OmegaConf.update(cfg, key, yaml.safe_load(value), merge=True)
    return cfg


def parse_curve(log_text: str):
    train = [float(v) for v in re.findall(r'Epoch \d+/\d+ .*?epoch_mean=([0-9.]+)', log_text)]
    val = [float(v) for v in re.findall(r'VALID summary \| loss_normalized=([0-9.]+)', log_text)]
    return train, val


def train_env(env, model, data_dir, out_dir, exp_name, seed):
    runs_dir = (out_dir / 'runs').resolve()
    run_name = f'{exp_name}_seed{seed}'
    opts = build_opts(env, data_dir, runs_dir, run_name, seed)
    log_path = out_dir / 'logs' / f'train_{env}_seed{seed}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        proc = subprocess.run(
            [sys.executable, str(SRC_DIR / 'main.py'), '--task', 'forecast', '--model_name', model, '--opts', *opts],
            cwd=REPO_DIR, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise RuntimeError(f'{env} 시드 {seed} 학습 실패 (로그: {log_path})')

    run_root = runs_dir / data_dir.name / env / model / run_name
    run_dir = max(run_root.glob('version*'), key=lambda p: p.stat().st_mtime)
    model_dir = seed_model_dir(out_dir, seed) / env
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in ('best_model.pt', 'configs.yaml'):
        shutil.copy2(run_dir / name, model_dir / name)
    best = json.loads((run_dir / 'best_results.json').read_text())
    latest = json.loads((run_dir / 'latest_results.json').read_text())
    train_curve, val_curve = parse_curve(log_path.read_text(encoding='utf-8', errors='ignore'))
    return {
        'run_dir': str(run_dir),
        'log': str(log_path),
        'elapsed_s': round(elapsed, 1),
        'epochs': int(latest['latest_epoch']),
        'best_epoch': int(best['best_epoch']),
        'best_val_loss': float(best['best_loss']),
        'train_curve': train_curve,
        'val_curve': val_curve,
    }


def seed_model_dir(out_dir: Path, seed: int) -> Path:
    return out_dir / 'models' / f'seed{seed}'


def compact_json(obj) -> str:
    """indent=2 JSON에서 가장 안쪽의 작은 객체·배열만 한 줄로 접는다 (사람이 읽기 쉬운 결과 파일)."""
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    pattern = re.compile(r'([\[{])\n\s+([^\[\]{}]*?)\n\s*([\]}])')
    return pattern.sub(lambda m: m.group(1) + re.sub(r',\n\s+', ', ', m.group(2)) + m.group(3), text)


def _korean_font(font_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    plt.rcdefaults()
    if font_path and Path(font_path).is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    return plt


PASTEL = {'blue': '#7FAFD9', 'coral': '#F29B9B', 'ink': '#3A3A3A', 'muted': '#7A7A7A', 'grid': '#EEEEEE',
          'axis': '#CFCFCF', 'mark': '#B9B9B9'}


def _pastel_axes(ax):
    ax.grid(True, color=PASTEL['grid'], lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(PASTEL['axis'])
    ax.tick_params(colors=PASTEL['muted'], labelsize=8, length=3)
    ax.xaxis.label.set_color(PASTEL['muted'])
    ax.yaxis.label.set_color(PASTEL['muted'])


def plot(results, trainings, out_dir: Path, font_path, day: str, note: str = ''):
    import matplotlib.dates as mdates

    plt = _korean_font(font_path)
    day_start = np.datetime64(day)
    day_end = day_start + np.timedelta64(1, 'D')
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.8 * len(results)), dpi=150, squeeze=False)
    for ax, res in zip(axes[:, 0], results):
        arr = res['_arrays']
        # 각 예측 구간의 마지막 시점 = 60분 앞 예측
        target_time = arr['first_target_time'] + np.timedelta64(res['pred_len'] - 1, 'm')
        keep = (target_time >= day_start) & (target_time < day_end)
        t = target_time[keep].astype('datetime64[m]').astype(object)
        ax.plot(t, arr['targets_w'][keep, -1, 0], color=PASTEL['blue'], lw=2.0, label='실측')
        ax.plot(t, arr['outputs_w'][keep, -1, 0], color=PASTEL['coral'], lw=1.8, label='예측 (60분 후)')
        ax.set_title(f"{ENV_LABELS.get(res['env'], res['env'])}   MSE {res['mse_norm']:.4f}{note}", loc='left',
                     fontsize=10, color=PASTEL['ink'])
        ax.set_ylabel('환기 유효전력 (W)', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        _pastel_axes(ax)
        ax.legend(fontsize=8, frameon=False, loc='upper left', labelcolor=PASTEL['ink'])
    axes[-1, 0].set_xlabel(f'시각 ({day})', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / 'pred_vs_actual.png', facecolor='white')
    plt.close(fig)
    plot_learning_curves(trainings, out_dir, font_path, note)


def plot_learning_curves(trainings, out_dir: Path, font_path, note: str = ''):
    plt = _korean_font(font_path)
    fig, axes = plt.subplots(1, len(trainings), figsize=(4.2 * len(trainings), 3.3), dpi=150, squeeze=False)
    for ax, (env, tr) in zip(axes[0], trainings.items()):
        epochs = np.arange(1, len(tr['train_curve']) + 1)
        ax.plot(epochs, tr['train_curve'], '-o', color=PASTEL['blue'], lw=2, ms=4, mec='white', mew=0.8, label='학습')
        ax.plot(epochs[:len(tr['val_curve'])], tr['val_curve'], '-o', color=PASTEL['coral'], lw=2, ms=4, mec='white',
                mew=0.8, label='검증')
        ax.axvline(tr['best_epoch'], color=PASTEL['mark'], ls='--', lw=1, label=f"선택 epoch ({tr['best_epoch']})")
        ax.set_title(ENV_LABELS.get(env, env) + note, loc='left', fontsize=10, color=PASTEL['ink'])
        ax.set_xticks([1] + list(range(5, len(epochs) + 1, 5)))
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Loss (MSE)', fontsize=9)
        _pastel_axes(ax)
        ax.legend(fontsize=8, frameon=False, labelcolor=PASTEL['ink'])
    fig.tight_layout()
    fig.savefig(out_dir / 'learning_curve.png', facecolor='white')
    plt.close(fig)


def plot_seed(seeds, requested=None):
    """그래프에 쓸 대표 시드: 지정값, 없으면 42(포함 시), 그 외에는 첫 시드."""
    if requested is not None:
        return requested
    return 42 if 42 in seeds else seeds[0]


def main():
    parser = argparse.ArgumentParser(description='Forecast performance test: train on the data, then evaluate')
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--envs', nargs='+', default=['livingroom'])
    parser.add_argument('--out', type=Path, default=Path('kpi_release/test_run'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                        help='학습 난수 시드 목록 (가중치 초기값·배치 순서). 공간마다 시드별로 학습·평가')
    parser.add_argument('--model', default='DLinear', help='학습할 예측 모델 (configs/model_setting.yaml 키)')
    parser.add_argument('--exp-name', default='kpi_test')
    parser.add_argument('--plot-day', default='2026-07-20')
    parser.add_argument('--plot-seed', type=int, default=None, help='그래프에 쓸 시드 (기본: 42가 있으면 42)')
    parser.add_argument('--font', type=Path, default=None)
    parser.add_argument('--replot', action='store_true',
                        help='학습 없이 저장된 모델로 다시 평가해 그래프 2종을 다시 그림')
    parser.add_argument('--replot-curves', action='store_true',
                        help='학습·평가 없이 <out>/logs 와 kpi_result.json 으로 학습 곡선만 다시 그림')
    args = parser.parse_args()

    if args.replot or args.replot_curves:
        report = json.loads((args.out / 'kpi_result.json').read_text(encoding='utf-8'))
        seed = plot_seed(report['training_setup']['seeds'], args.plot_seed)
        font = args.font or (args.data_dir.resolve().parent / 'fonts' / 'NanumGothic-Regular.ttf')
        trainings = {}
        for env, info in report['spaces'].items():
            log = (args.out / 'logs' / f'train_{env}_seed{seed}.log').read_text(encoding='utf-8', errors='ignore')
            train, val = parse_curve(log)
            trainings[env] = {'train_curve': train, 'val_curve': val, 'best_epoch': info['seeds'][str(seed)]['best_epoch']}
        if args.replot_curves:
            plot_learning_curves(trainings, args.out, font, f'  (시드 {seed})')
            print(f"다시 그림: {args.out / 'learning_curve.png'}")
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = [evaluate_env(env, seed_model_dir(args.out, seed), args.data_dir.resolve(), device)
                   for env in report['spaces']]
        plot(results, trainings, args.out, font, args.plot_day, f'  (시드 {seed})')
        print('다시 그림:', ', '.join(f"{r['env']} MSE {r['mse_norm']:.4f}" for r in results))
        return

    data_dir = args.data_dir.resolve()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = list(dict.fromkeys(args.seeds))
    labels = {env: ENV_LABELS.get(env, env) for env in args.envs}

    print(f"[예측 성능 시험] {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  데이터   : {data_dir}")
    print(f"  공간     : {', '.join(labels.values())}")
    print(f"  장치     : {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == 'cuda' else ''))
    print(f"  시드     : {', '.join(map(str, seeds))} (공간마다 시드별로 학습·평가, {len(args.envs) * len(seeds)}회)")

    print("\n[1/3 데이터 분할] 시간순 8:1:1 (학습/검증/시험), 입력 120분 → 예측 60분")
    print(f"  입력 제외 열: {', '.join(IGNORE_FEATURES)} (계속 증가하는 누적값)")
    widths, aligns = [8, 8, 8, 7, 7, 35, 9], ['left'] + ['right'] * 4 + ['left', 'right']
    print_row(['공간', '전체행', '학습행', '검증행', '시험행', '시험 구간', '예측구간'], widths, aligns)
    for env in args.envs:
        cfg = build_cfg(args.model, build_opts(env, data_dir, out_dir / 'runs', args.exp_name, seeds[0]))
        trn, val, tst, _, _ = load_split(cfg, data_dir)
        rows = [len(d.dataset.data) for d in (trn, val, tst)]
        print_row([labels[env], f"{sum(rows):,}", f"{rows[0]:,}", f"{rows[1]:,}", f"{rows[2]:,}",
                   period(tst.dataset.timestamps), f"{len(tst.dataset):,}"], widths, aligns)

    print("\n[2/3 학습] 학습 구간으로 학습(배치 순서 무작위), 검증 손실이 가장 낮은 epoch의 모델을 저장")
    trainings = {}  # (env, seed) -> 학습 정보
    for env in args.envs:
        for seed in seeds:
            print(f"  {labels[env]} 시드 {seed:>3}: {datetime.now():%H:%M:%S} 시작 → ", end='', flush=True)
            tr = train_env(env, args.model, data_dir, out_dir, args.exp_name, seed)
            trainings[env, seed] = tr
            print(f"{datetime.now():%H:%M:%S} 완료 (소요 {tr['elapsed_s'] / 60:.1f}분) | "
                  f"최적 epoch {tr['best_epoch']}/{tr['epochs']}, 검증 손실 {tr['best_val_loss']:.4f}", flush=True)

    print("\n[3/3 평가] 시험 구간 전체 예측 구간 추론, MSE 계산")
    results = {(env, seed): evaluate_env(env, seed_model_dir(out_dir, seed), data_dir, device)
               for env in args.envs for seed in seeds}
    widths, aligns = [8, 6, 9, 10, 12, 9], ['left'] + ['right'] * 5
    print_row(['공간', '시드', '예측구간', '예측값 수', 'MSE(표준화)', 'MSE(W²)'], widths, aligns)
    for (env, seed), r in results.items():
        print_row([labels[env], str(seed), f"{r['windows']:,}", f"{r['points']:,}", f"{r['mse_norm']:.4f}",
                   f"{r['mse_w2']:.2f}"], widths, aligns)

    summary = {}
    for env in args.envs:
        values = np.array([results[env, seed]['mse_norm'] for seed in seeds])
        summary[env] = {'mean': float(values.mean()), 'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        'min': float(values.min()), 'max': float(values.max())}
    # 대표값: 공간별 MSE(시드 평균)의 산술평균 = 시드별 '공간 평균'의 평균
    by_seed = {seed: float(np.mean([results[env, seed]['mse_norm'] for env in args.envs])) for seed in seeds}
    seed_avgs = np.array(list(by_seed.values()))
    overall = {'mean': float(np.mean([summary[env]['mean'] for env in args.envs])),
               'std': float(seed_avgs.std(ddof=1)) if len(seed_avgs) > 1 else 0.0,
               'min': float(seed_avgs.min()), 'max': float(seed_avgs.max())}
    scope = f"{len(args.envs)}개 공간 평균"

    print(f"\n[측정 결과] MSE(표준화), 시드 {len(seeds)}개 평균. 대표값은 {scope}, 공간별 값은 세부 측정값")
    widths, aligns = [14, 10, 10, 9, 9, 8, 8], ['left'] + ['right'] * 6
    print_row(['구분', '평균 MSE', '표준편차', '최소', '최대', 'μ(W)', 'σ(W)'], widths, aligns)
    print_row([scope, f"{overall['mean']:.4f}", f"{overall['std']:.4f}", f"{overall['min']:.4f}", f"{overall['max']:.4f}",
               '', ''], widths, aligns)
    for env in args.envs:
        sm, r = summary[env], results[env, seeds[0]]
        print_row([f"  {labels[env]} (세부)", f"{sm['mean']:.4f}", f"{sm['std']:.4f}", f"{sm['min']:.4f}",
                   f"{sm['max']:.4f}", f"{r['target_mean_w']:.2f}", f"{r['target_std_w']:.2f}"], widths, aligns)
    details = ', '.join(f"{labels[env]} {summary[env]['mean']:.4f}" for env in args.envs)
    print(f"\n[대표값] {scope} MSE(표준화): {overall['mean']:.4f}  (세부 측정값: {details})")

    report = {
        'executed_at': datetime.now().isoformat(timespec='seconds'),
        'data_dir': str(data_dir),
        'device': str(device),
        'metric': 'MSE on targets standardized with the training split mean/std',
        'training_setup': {'seeds': seeds, 'shuffle_train_batches': True, 'excluded_input_columns': IGNORE_FEATURES},
        'overall': {'definition': 'mean of the per-space MSEs (each averaged over seeds)', 'spaces': list(args.envs),
                    'mse_norm_mean': overall['mean'], 'mse_norm_std': overall['std'],
                    'mse_norm_min': overall['min'], 'mse_norm_max': overall['max'],
                    'by_seed': {str(seed): v for seed, v in by_seed.items()}},
        'spaces': {},
    }
    skip = ('sum_sq_err_norm', 'checkpoint_sha256', 'checkpoint', 'mse_norm', 'mse_w2')
    for env in args.envs:
        first = results[env, seeds[0]]
        report['spaces'][env] = {
            **{k: v for k, v in first.items() if not k.startswith('_') and k not in skip},
            'mse_norm_mean': summary[env]['mean'],
            'mse_norm_std': summary[env]['std'],
            'mse_norm_min': summary[env]['min'],
            'mse_norm_max': summary[env]['max'],
            'seeds': {str(seed): {'mse_norm': results[env, seed]['mse_norm'],
                                  'mse_w2': round(results[env, seed]['mse_w2'], 4),
                                  'best_epoch': trainings[env, seed]['best_epoch'],
                                  'epochs': trainings[env, seed]['epochs'],
                                  'train_s': trainings[env, seed]['elapsed_s']} for seed in seeds},
        }
    (out_dir / 'kpi_result.json').write_text(compact_json(report) + '\n', encoding='utf-8')

    seed = plot_seed(seeds, args.plot_seed)
    plot([results[env, seed] for env in args.envs], {env: trainings[env, seed] for env in args.envs}, out_dir,
         args.font or (data_dir.parent / 'fonts' / 'NanumGothic-Regular.ttf'), args.plot_day, f'  (시드 {seed})')
    print(f"결과 파일: {out_dir / 'kpi_result.json'}, {out_dir / 'pred_vs_actual.png'}, {out_dir / 'learning_curve.png'}")


if __name__ == '__main__':
    main()
