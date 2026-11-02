import argparse
from pathlib import Path

import yaml
from omegaconf import OmegaConf


SRC_DIR = Path(__file__).resolve().parent


def parser():
    cli = argparse.ArgumentParser(description='snu-ai-control')
    cli.add_argument(
        '--task',
        choices=(
            'forecast',
            'mpc',
            'evaluator_benchmark',
            'action_stratified_evaluation',
            'report_model_input_ablation',
            'counterfactual_report_input_ablation',
            'action_distance_ablation',
            'action_history_ablation',
            'control_benchmark',
        ),
        default='forecast',
        help=(
            'Run forecasting, the full MPC pipeline, the evaluator/action-scope '
            'benchmark, the supplementary action-stratified row evaluation, the '
            'observed-action report-model input ablation, the counterfactual '
            'report-model input ablation, or the recorded-action anchored '
            'Full-guard distance sweep, or the applied-action-history '
            'constraint matrix (lookback, hard distance, grouping, penalty). '
            'Forecasting and MPC share data, not execution.'
        ),
    )
    cli.add_argument('--model_name', type=str, help='Forecast model name')
    cli.add_argument(
        '--default_cfg',
        type=str,
        default=str(SRC_DIR / 'configs' / 'default_setting.yaml'),
        help='Common configuration file',
    )
    cli.add_argument(
        '--model_cfg',
        type=str,
        default=str(SRC_DIR / 'configs' / 'model_setting.yaml'),
        help='Forecast model configuration file',
    )
    cli.add_argument(
        '--opts',
        help='Modify config options using KEY=VALUE entries',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = cli.parse_args()

    cfg = OmegaConf.load(args.default_cfg)
    cfg = OmegaConf.merge(cfg, {'TASK': {'name': args.task}})

    if args.task == 'forecast':
        if not args.model_name:
            cli.error('--model_name is required when --task=forecast')
        cfg = OmegaConf.merge(cfg, {'MODEL': {'modelname': args.model_name}})
        model_cfg = OmegaConf.load(args.model_cfg)
        if args.model_name not in model_cfg:
            cli.error(f"Model '{args.model_name}' not found in {args.model_cfg}")
        cfg = OmegaConf.merge(
            cfg,
            {'MODELSETTING': OmegaConf.create(model_cfg[args.model_name])},
        )

    if args.opts:
        for opt in args.opts:
            if '=' not in opt:
                continue
            key, value = opt.split('=', 1)
            try:
                parsed_value = yaml.safe_load(value)
                OmegaConf.update(cfg, key, parsed_value, merge=True)
            except Exception as exc:
                print(f"Warning: failed to parse '{value}' for '{key}': {exc}")
                OmegaConf.update(cfg, key, value, merge=True)

    return cfg
