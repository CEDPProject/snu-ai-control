from arguments import parser


def main(cfg):
    task = cfg.TASK.name
    if task == 'control_benchmark':
        from runners.control_benchmark_runner import run_control_benchmark

        return run_control_benchmark(cfg)
    if task == 'forecast':
        from runners.forecast_runner import run_forecast

        return run_forecast(cfg)
    if task == 'mpc':
        from runners.mpc_runner import run_mpc

        return run_mpc(cfg)
    if task == 'evaluator_benchmark':
        from runners.evaluator_benchmark_runner import run_evaluator_benchmark

        return run_evaluator_benchmark(cfg)
    if task == 'action_stratified_evaluation':
        from runners.action_stratified_evaluation_runner import (
            run_action_stratified_evaluation,
        )

        return run_action_stratified_evaluation(cfg)
    if task == 'report_model_input_ablation':
        from runners.report_model_input_ablation_runner import (
            run_report_model_input_ablation,
        )

        return run_report_model_input_ablation(cfg)
    if task == 'counterfactual_report_input_ablation':
        from runners.counterfactual_report_input_ablation_runner import (
            run_counterfactual_report_input_ablation,
        )

        return run_counterfactual_report_input_ablation(cfg)
    if task == 'action_distance_ablation':
        from runners.action_distance_ablation_runner import (
            run_action_distance_ablation,
        )

        return run_action_distance_ablation(cfg)
    if task == 'action_history_ablation':
        from runners.action_history_ablation_runner import (
            run_action_history_ablation,
        )

        return run_action_history_ablation(cfg)
    raise ValueError(f"Unsupported task: {task!r}")


if __name__ == '__main__':
    main(parser())
