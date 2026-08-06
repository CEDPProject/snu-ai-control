import json
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import wandb
from accelerate import Accelerator

from utils.metrics import cal_metric
from utils.tools import EarlyStopping
from utils.utils import Float32Encoder


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.val = float(value)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


def _current_learning_rates(optimizer):
    return [float(group['lr']) for group in optimizer.param_groups]


def training_dl(
    model, trn_dataloader, val_dataloader, criterion, optimizer,
    accelerator: Accelerator, savedir: str, epochs: int, eval_epochs: int,
    log_epochs: int, log_eval_iter: int, use_wandb: bool, wandb_iter: int,
    ckp_metric: str, model_name: str, early_stopping_metric: str,
    early_stopping_count: int, early_stopping_min_delta: float,
    checkpoint_min_delta: float, lr_scheduler, lr_scheduler_monitor: str,
    model_config: dict, scaler_dict: dict = None,
):
    logger = logging.getLogger('TRAINING')
    if epochs < 1:
        raise ValueError('epochs must be at least 1')
    if eval_epochs < 1 or eval_epochs > epochs:
        raise ValueError('eval_epochs must be in [1, epochs]')

    model.train()
    started_at = time.time()
    early_stopping = EarlyStopping(
        patience=early_stopping_count,
        min_delta=early_stopping_min_delta,
    )
    best_score = np.inf
    best_state = None
    last_eval_metrics = None
    global_step = 0

    for epoch in range(epochs):
        # Each meter covers one epoch. Previously this was a whole-run
        # cumulative average mislabeled as an epoch average.
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_loss = AverageMeter()
        end_time = time.time()

        for idx, item in enumerate(trn_dataloader):
            data_time.update(time.time() - end_time)
            optimizer.zero_grad()
            inputs = item['input']
            target = item['target']['target']
            outputs = model(
                input_cat=inputs['cat'],
                input_num=inputs['num'],
                input_ts=inputs['ts'],
            )
            loss = accelerator.gather(criterion(outputs, target)).mean()
            accelerator.backward(loss)
            optimizer.step()

            train_loss.update(loss.item(), n=target.size(0))
            batch_time.update(time.time() - end_time)
            global_step += 1
            if use_wandb and wandb_iter > 0 and global_step % wandb_iter == 0:
                wandb.log({
                    'lr': optimizer.param_groups[0]['lr'],
                    'train_loss_epoch_mean': train_loss.avg,
                }, step=global_step)
            end_time = time.time()

        if (epoch + 1) % log_epochs == 0:
            logger.info(
                'Epoch %d/%d | batches=%d/%d | '
                'train_loss_normalized(last_batch=%.6f, epoch_mean=%.6f) | '
                'lr=%.3e | batch_time_s(last=%.3f, mean=%.3f) | '
                'data_time_s(last=%.3f, mean=%.3f)',
                epoch + 1, epochs, idx + 1, len(trn_dataloader),
                train_loss.val, train_loss.avg,
                optimizer.param_groups[0]['lr'],
                batch_time.val, batch_time.avg, data_time.val, data_time.avg,
            )

        if (epoch + 1) % eval_epochs != 0:
            continue

        eval_metrics = test_dl(
            model=model, dataloader=val_dataloader, criterion=criterion,
            accelerator=accelerator, log_interval=log_eval_iter,
            savedir=savedir, model_config=model_config, model_name=model_name,
            name='VALID', return_output=False, scaler_dict=scaler_dict,
        )
        last_eval_metrics = eval_metrics
        model.train()
        eval_results = {f'eval_{key}': value for key, value in eval_metrics.items()}
        if use_wandb:
            wandb.log(eval_results, step=global_step)

        checkpoint_value = eval_metrics.get(ckp_metric)
        if checkpoint_value is None:
            raise KeyError(f'Checkpoint metric {ckp_metric!r} is unavailable')
        if best_score - float(checkpoint_value) > checkpoint_min_delta:
            previous_best = best_score
            best_score = float(checkpoint_value)
            best_state = {
                'best_epoch': epoch + 1,
                'best_step': global_step,
                f'best_{ckp_metric}': best_score,
                **eval_results,
            }
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                with open(os.path.join(savedir, 'best_results.json'), 'w') as stream:
                    json.dump(best_state, stream, indent='\t', cls=Float32Encoder)
                torch.save(model.state_dict(), os.path.join(savedir, 'best_model.pt'))
            logger.info(
                'Checkpoint saved | metric=%s | previous=%s | current=%.8f | '
                'epoch=%d | global_step=%d',
                ckp_metric,
                'inf' if not np.isfinite(previous_best) else f'{previous_best:.8f}',
                best_score, epoch + 1, global_step,
            )

        if lr_scheduler is not None:
            scheduler_value = eval_metrics.get(lr_scheduler_monitor)
            if scheduler_value is None:
                raise KeyError(
                    f'LR scheduler monitor {lr_scheduler_monitor!r} is unavailable'
                )
            old_lrs = _current_learning_rates(optimizer)
            lr_scheduler.step(float(scheduler_value))
            new_lrs = _current_learning_rates(optimizer)
            if new_lrs != old_lrs:
                logger.info(
                    'Learning rate reduced on validation plateau | monitor=%s | '
                    'value=%.8f | old=%s | new=%s',
                    lr_scheduler_monitor, float(scheduler_value), old_lrs, new_lrs,
                )

        early_value = eval_metrics.get(early_stopping_metric)
        if early_value is None:
            raise KeyError(
                f'Early-stopping metric {early_stopping_metric!r} is unavailable'
            )
        early_improved = early_stopping(float(early_value))
        if not early_improved:
            logger.info(
                'Early-stopping monitor unchanged | metric=%s | value=%.8f | '
                'counter=%d/%d | min_delta=%g',
                early_stopping_metric, float(early_value), early_stopping.counter,
                early_stopping.patience, early_stopping.min_delta,
            )
        if early_stopping.early_stop:
            logger.info(
                'Early stopping triggered | epoch=%d | no_improvement_count=%d',
                epoch + 1, early_stopping.counter,
            )
            break

    if best_state is None or last_eval_metrics is None:
        raise RuntimeError('Training completed without a validation checkpoint')

    logger.info(
        'Best checkpoint | metric=%s | value=%.8f | epoch=%d | global_step=%d',
        ckp_metric, best_state[f'best_{ckp_metric}'],
        best_state['best_epoch'], best_state['best_step'],
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(model.state_dict(), os.path.join(savedir, 'latest_model.pt'))
        latest_state = {
            'latest_epoch': epoch + 1,
            'latest_step': global_step,
            f'latest_{ckp_metric}': last_eval_metrics[ckp_metric],
            **{f'eval_{key}': value for key, value in last_eval_metrics.items()},
        }
        with open(os.path.join(savedir, 'latest_results.json'), 'w') as stream:
            json.dump(latest_state, stream, indent='\t', cls=Float32Encoder)
        logger.info(
            'Latest checkpoint saved | epoch=%d | global_step=%d | %s=%.8f',
            epoch + 1, global_step, ckp_metric,
            float(last_eval_metrics[ckp_metric]),
        )

    elapsed = time.time() - started_at
    logger.info('Training complete | elapsed_seconds=%.3f', elapsed)
    return elapsed


def _format_metric(value):
    return 'N/A' if value is None else f'{float(value):.4f}'


def test_dl(
    model, dataloader, criterion, accelerator: Accelerator, log_interval: int,
    savedir: str, model_config: dict, model_name: str, name: str = 'TEST',
    dataname: str = None, wandb_config: dict = None, return_output: bool = False,
    plot_result: bool = False, training_time=None, scaler_dict=None,
) -> dict:
    logger = logging.getLogger(name)
    logger.info('Start %s evaluation', name)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    total_targets = []
    total_outputs = []
    end_time = time.time()

    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            data_time.update(time.time() - end_time)
            inputs = item['input']
            target = item['target']['target']
            outputs = model(
                input_cat=inputs['cat'],
                input_num=inputs['num'],
                input_ts=inputs['ts'],
            )
            loss = accelerator.gather(criterion(outputs, target)).mean()
            outputs, target = accelerator.gather_for_metrics(
                (outputs.contiguous(), target.contiguous())
            )
            losses.update(loss.item(), n=target.size(0))
            total_outputs.append(outputs.detach().cpu().numpy())
            total_targets.append(target.detach().cpu().numpy())
            batch_time.update(time.time() - end_time)

            if log_interval > 0 and (idx + 1) % log_interval == 0:
                logger.info(
                    '%s progress | batches=%d/%d | '
                    'loss_normalized(last_batch=%.6f, running_mean=%.6f) | '
                    'batch_time_s(last=%.3f, mean=%.3f)',
                    name, idx + 1, len(dataloader), losses.val, losses.avg,
                    batch_time.val, batch_time.avg,
                )
            end_time = time.time()

    total_outputs = np.concatenate(total_outputs, axis=0)
    total_targets = np.concatenate(total_targets, axis=0)
    if scaler_dict is not None and 'target_scaler' in scaler_dict:
        target_scaler = scaler_dict['target_scaler']
        shape = total_outputs.shape
        total_outputs_denorm = target_scaler.inverse_transform(
            total_outputs.reshape(-1, shape[-1])
        ).reshape(shape)
        total_targets_denorm = target_scaler.inverse_transform(
            total_targets.reshape(-1, shape[-1])
        ).reshape(shape)
        pred_for_metric = total_outputs_denorm
        true_for_metric = total_targets_denorm
        metric_scale = 'original_target_scale'
    else:
        logger.warning('No target scaler found; metrics use normalized values')
        pred_for_metric = total_outputs
        true_for_metric = total_targets
        metric_scale = 'normalized_scale'

    metrics = cal_metric(pred=pred_for_metric, true=true_for_metric)
    results = OrderedDict([('loss', losses.avg), *metrics.items()])
    logger.info(
        '%s summary | loss_normalized=%.6f | metric_scale=%s | '
        'power_accuracy_pct=%s | mse=%s | mae=%s | rmse=%s | '
        'mape_pct(actual_abs_gt_0.1)=%s | wape_pct=%s | smape_pct=%s | '
        'rse=%s | nmae_pct=%s',
        name, losses.avg, metric_scale,
        _format_metric(metrics['prediction_accuracy*100']),
        _format_metric(metrics['mse']), _format_metric(metrics['mae']),
        _format_metric(metrics['rmse']), _format_metric(metrics['mape']),
        _format_metric(metrics['wape']), _format_metric(metrics['smape']),
        _format_metric(metrics['rse']), _format_metric(metrics['nmae']),
    )
    logger.info('End %s evaluation', name)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and return_output:
        if scaler_dict is not None and 'target_scaler' in scaler_dict:
            np.save(os.path.join(savedir, f'{name}_total_outputs.npy'), total_outputs_denorm)
            np.save(os.path.join(savedir, f'{name}_total_targets.npy'), total_targets_denorm)
        else:
            np.save(os.path.join(savedir, f'{name}_total_outputs.npy'), total_outputs)
            np.save(os.path.join(savedir, f'{name}_total_targets.npy'), total_targets)
    return results
