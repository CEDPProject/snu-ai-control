import logging
import wandb
import time
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from accelerate import Accelerator
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import cal_metric
from utils.utils import Float32Encoder

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def training_dl(
    model, trn_dataloader, val_dataloader, criterion, optimizer, accelerator: Accelerator, 
    savedir: str, epochs: int, eval_epochs: int, log_epochs: int, log_eval_iter: int, 
    use_wandb: bool, wandb_iter: int, ckp_metric: str, model_name: str, 
    early_stopping_metric: str, early_stopping_count: int, lradj: int, 
    learning_rate: int, model_config: dict, scaler_dict: dict = None):

    logger = logging.getLogger('TRAINING') 

    # avereage meter
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    # set mode
    model.train()
    start_training_time = time.time()
    end_time = time.time()
    
    early_stopping = EarlyStopping(patience=early_stopping_count)
        
    # init best score and step
    best_score = np.inf
    wandb_iteration = 0
    
    for epoch in range(epochs):
        epoch_time = time.time()
        
        for idx, item in enumerate(trn_dataloader):
            
            data_time_m.update(time.time() - end_time)
            optimizer.zero_grad()
            
            # input
            input_cat = item['input']['cat']
            input_num = item['input']['num']
            input_ts = item['input']['ts']
            target = item['target']['target']
            target_ts = item['target']['ts']
            
            outputs = model(input_cat = input_cat, 
                            input_num = input_num, 
                            input_ts  = input_ts)
            
            loss = criterion(outputs, target)
            loss = accelerator.gather(loss).mean()
            outputs, target = accelerator.gather_for_metrics((outputs.contiguous(), target.contiguous()))
            
            accelerator.backward(loss)
            
            # loss update
            optimizer.step()
            optimizer.zero_grad()
            
            losses_m.update(loss.item(), n = target.size(0))
            
            # batch time
            batch_time_m.update(time.time() - end_time)
            wandb_iteration += 1
            
            if use_wandb and (wandb_iteration+1) % wandb_iter:
                train_results = OrderedDict([
                    ('lr',optimizer.param_groups[0]['lr']),
                    ('train_loss',losses_m.avg)
                ])
                wandb.log(train_results, step=idx+1)
        
        if (epoch + 1) % log_epochs == 0:
            logger.info("[🚀 TRAINING Status]")
            logger.info(
                'EPOCH {:>3d}/{} | TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                'LR: {lr:.3e} '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    (epoch + 1), epochs,
                    (idx + 1),
                    len(trn_dataloader),
                    loss=losses_m,
                    lr=optimizer.param_groups[0]['lr'],
                    batch_time=batch_time_m,
                    rate=input_num.size(0) / batch_time_m.val,
                    rate_avg=input_num.size(0) / batch_time_m.avg,
                    data_time=data_time_m
                )
            )
            
        if (epoch + 1) % eval_epochs == 0:
            eval_metrics = test_dl(
                accelerator   = accelerator,
                model         = model, 
                dataloader    = val_dataloader, 
                criterion     = criterion,
                name          = 'VALID',
                log_interval  = log_eval_iter,
                savedir       = savedir,
                model_name    = model_name,
                model_config  = model_config,
                return_output = False,
                scaler_dict   = scaler_dict
                )
            
            model.train()
            
            # eval results
            eval_results = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])
            
            # wandb
            if use_wandb:
                wandb.log(eval_results, step=idx+1)
                
            # check_point
            if best_score > eval_metrics[ckp_metric]:
                # save results
                state = {'best_epoch':epoch ,
                            'best_step':idx+1, 
                            f'best_{ckp_metric}':eval_metrics[ckp_metric]}
                
                print('Save best model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
                    to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))
    
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    state.update(eval_results)
                    json.dump(state, open(os.path.join(savedir, f'best_results.json'),'w'), indent='\t', cls=Float32Encoder)
                    # save model
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                
                logger.info('Best {0} {1:6.6f} to {2:6.6f}'.format(ckp_metric.upper(), best_score, eval_metrics[ckp_metric]))
                print("✅ Saved best model")
                
                best_score = eval_metrics[ckp_metric]
                
            early_stopping(eval_metrics[early_stopping_metric])
            if early_stopping.early_stop:
                logger.info("⏳ Early stopping triggered")
                break
            
            adjust_learning_rate(optimizer, epoch + 1, lradj, learning_rate)
            
        end_time = time.time()
            
    # logging best score and step
    logger.info('Best Metric: {0:6.6f} (step {1:})'.format(state[f'best_{ckp_metric}'], state['best_step']))
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # save latest model
        torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))
        logger.info('Save latest model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
            to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))

        # save latest results
        state = {'best_epoch':epoch ,'best_step':idx+1, f'latest_{ckp_metric}':eval_metrics[ckp_metric]}
        state.update(eval_results)
        json.dump(state, open(os.path.join(savedir, f'latest_results.json'),'w'), indent='\t', cls=Float32Encoder)
        
    # 전체 학습 시간 측정 종료
    total_training_time = time.time() - start_training_time
    logger.info("🎉 Training complete for all datasets")    
    return total_training_time


def test_dl(model, dataloader, criterion, accelerator: Accelerator, log_interval: int, 
            savedir: str, model_config: dict, model_name: str, name: str = 'TEST', dataname: str = None, 
            wandb_config: dict = None, return_output: bool = False, plot_result:bool = False, 
            training_time = None, scaler_dict = None) -> dict:
    
    logger = logging.getLogger(name) 
    
    logger.info(f'[🔍 Start {name} Evaluation]')
    
    # avereage meter
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    # targets and outputs
    total_targets = []
    total_outputs = []
    
    end_time = time.time()
    wandb_iteration = 0
    
    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            data_time_m.update(time.time() - end_time)
            
            # input
            input_cat = item['input']['cat']
            input_num = item['input']['num']
            input_ts = item['input']['ts']
            target = item['target']['target']
            target_ts = item['target']['ts']
            
            outputs = model(input_cat = input_cat, 
                            input_num = input_num, 
                            input_ts  = input_ts)
            
            loss = criterion(outputs, target)
            loss = accelerator.gather(loss).mean()
            outputs, target = accelerator.gather_for_metrics((outputs.contiguous(), target.contiguous()))
            
            losses_m.update(loss.item(), n=target.size(0))
                        
            outputs = outputs.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            total_outputs.append(outputs)
            total_targets.append(target)
            
            # batch time        
            batch_time_m.update(time.time() - end_time)
            
            if (idx+1) % log_interval == 0:
                logger.info('{name} [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1), 
                            len(dataloader),
                            name       = name,
                            loss       = losses_m, 
                            batch_time = batch_time_m,
                            rate       = input_num.size(0) / batch_time_m.val,
                            rate_avg   = input_num.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))
                
            end_time = time.time()
    
    total_outputs = np.concatenate(total_outputs, axis=0)
    total_targets = np.concatenate(total_targets, axis=0)
    
    # Denormalization
    if scaler_dict is not None and 'target_scaler' in scaler_dict:
        target_scaler = scaler_dict['target_scaler']
        
        # shape: (n_samples, pred_len, n_target) -> (n_samples * pred_len, n_target)
        original_shape = total_outputs.shape
        
        # Denormalize outputs
        total_outputs_denorm = target_scaler.inverse_transform(
            total_outputs.reshape(-1, original_shape[-1])
        ).reshape(original_shape)
        
        # Denormalize targets
        total_targets_denorm = target_scaler.inverse_transform(
            total_targets.reshape(-1, original_shape[-1])
        ).reshape(original_shape)
        
        # Use denormalized values for metric calculation
        pred_for_metric = total_outputs_denorm
        true_for_metric = total_targets_denorm
    else:
        logger.warning('⚠️  No target_scaler found. Computing metrics on normalized values.')
        pred_for_metric = total_outputs
        true_for_metric = total_targets
    
    results = OrderedDict([('loss',losses_m.avg)])
    
    # Calculate metrics on denormalized values
    metrics = cal_metric(pred=pred_for_metric, true=true_for_metric)
    
    results.update([(f'{k}', m) for k, m in metrics.items()])
    
    # logging metrics
    logger.info(
        "{}: ".format(name) +
        " ".join(
            "{}: {:.4f}".format(k.upper(), v) if isinstance(v, (int, float)) and v is not None 
            else "{}: {}".format(k.upper(), v if v is not None else "N/A")
            for k, v in metrics.items()
        )
    )
    
    # return results
    results.update([(f'{k}', m) for k, m in metrics.items()])
    logger.info('[End {}]'.format(name))
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if return_output:
            # Save denormalized values if available
            if scaler_dict is not None and 'target_scaler' in scaler_dict:
                np.save(os.path.join(savedir, f'{name}_total_outputs.npy'), total_outputs_denorm)
                np.save(os.path.join(savedir, f'{name}_total_targets.npy'), total_targets_denorm)
            else:
                np.save(os.path.join(savedir, f'{name}_total_outputs.npy'), total_outputs)
                np.save(os.path.join(savedir, f'{name}_total_targets.npy'), total_targets)
            
    return results