from arguments import parser


def main(cfg):
    task = cfg.TASK.name
    if task == 'forecast':
        from runners.forecast_runner import run_forecast

        return run_forecast(cfg)
    if task == 'mpc':
        from runners.mpc_runner import run_mpc

        return run_mpc(cfg)
    if task == 'evaluator_benchmark':
        from runners.evaluator_benchmark_runner import run_evaluator_benchmark

        return run_evaluator_benchmark(cfg)
    raise ValueError(f"Unsupported task: {task!r}")


if __name__ == '__main__':
    main(parser())
