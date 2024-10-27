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

_logger = logging.getLogger('train')

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
        
def training(
    model, trainloader, validloader, criterion, optimizer, accelerator: Accelerator, 
    epochs: int, eval_epochs: int, log_epochs: int, log_eval_iter: int, wandb_iter: int,
    use_wandb: bool, ckp_metric: str, savedir: str, model_name: str, 
    pre_training:int, early_stopping_count: int, classification: bool, return_output: bool):

    # avereage meter
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    # set mode
    model.train()
    optimizer.zero_grad()
    end_time = time.time()

    # targets and outputs
    total_targets = []
    total_outputs = []
    
    early_stopping = EarlyStopping(patience=early_stopping_count)
        
    # init best score and step
    best_score = np.inf
    wandb_iteration = 0
    for epoch in range(epochs):
        epoch_time = time.time()
        for idx, item in enumerate(trainloader):
            data_time_m.update(time.time() - end_time)
            
            # input
            tabular_cat, tabular_con = item['inputs']['tabular']['cat'].type(torch.float), item['inputs']['tabular']['con'].type(torch.float)
            tsa = item['inputs']['tsa'].type(torch.float)
            tsb = item['inputs']['tsb'].type(torch.float)
            target = item['targets'].type(torch.float)
                        
            # output
            if model_name == 'TemporalFusionTransformer':
                TFT_input = {'s_cat': tabular_cat.int(),
                            's_cont': tabular_con,
                            'k_cont': tsa,
                            'o_cont': tsb,
                            'targets': target}
                outputs = model(TFT_input)
            else:
                outputs = model(x_cat = tabular_cat, x_cons = tabular_con, x_tsa = tsa, x_tsb = tsb)
            
            # final loss
            if pre_training:
                target = torch.concat([tabular_cat, tabular_con, 
                            tsa.reshape(tabular_cat.size(0),-1), 
                            tsb.reshape(tabular_cat.size(0),-1)], axis=1)
                
            if classification:
                roi_dict = get_roi_dict()
                target = classification_label_generator(target.detach().cpu().numpy(), roi_dict)
                target = torch.tensor(np.where(target == 1, 0, 1)).type(torch.float).cuda()

                # target = torch.tensor(classification_label_generator(target.detach().cpu().numpy(), roi_dict)).type(torch.float).cuda()
                
                outputs = outputs.squeeze(1)

            loss = criterion(outputs, target)
            accelerator.backward(loss)
            
            # loss update
            optimizer.step()
            optimizer.zero_grad()
            losses_m.update(loss.item(), n = target.size(0))
            
            # batch time
            batch_time_m.update(time.time() - end_time)
            wandb_iteration += 1
            
            outputs = outputs.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            total_outputs.append(outputs)
            total_targets.append(target)
            
            if use_wandb and (wandb_iteration+1) % wandb_iter:
                train_results = OrderedDict([
                    ('lr',optimizer.param_groups[0]['lr']),
                    ('train_loss',losses_m.avg)
                ])
                wandb.log(train_results, step=idx+1)
        
        if (epoch+1) % log_epochs == 0:
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (idx+1), 
                        len(trainloader), 
                        loss       = losses_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = tabular_cat.size(0) / batch_time_m.val,
                        rate_avg   = tabular_cat.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
                
        if (epoch+1) % eval_epochs == 0:
            eval_metrics = test(
                model          = model, 
                dataloader     = validloader, 
                criterion      = criterion,
                name           = 'VALID',
                log_eval_iter  = log_eval_iter,
                pre_training   = pre_training,
                return_output  = False,
                savedir        = savedir,
                model_name     = model_name,
                accelerator    = accelerator,
                classification = classification
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
                
                _logger.info('Best {0} {1:6.6f} to {2:6.6f}'.format(ckp_metric.upper(), best_score, eval_metrics[ckp_metric]))

                best_score = eval_metrics[ckp_metric]
                
            early_stopping(eval_metrics['loss'])
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        end_time = time.time()
            
    # logging best score and step
    _logger.info('Best Metric: {0:6.6f} (step {1:})\n'.format(state[f'best_{ckp_metric}'], state['best_step']))

    total_outputs = np.concatenate(total_outputs, axis=0)
    total_targets = np.concatenate(total_targets, axis=0)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # save latest model
        torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))
        print('Save latest model complete, epoch: {0:}: Best metric has changed from {1:.5f} \
            to {2:.5f}'.format(epoch, best_score, eval_metrics[ckp_metric]))

        # save latest results
        state = {'best_epoch':epoch ,'best_step':idx+1, f'latest_{ckp_metric}':eval_metrics[ckp_metric]}
        state.update(eval_results)
        json.dump(state, open(os.path.join(savedir, f'latest_results.json'),'w'), indent='\t', cls=Float32Encoder)
        
        if return_output:
            np.save(os.path.join(savedir, f'Train_total_outputs.npy'), total_outputs)
            np.save(os.path.join(savedir, f'Train_total_targets.npy'), total_targets)

def test(model, dataloader, criterion, accelerator: Accelerator, pre_training: int, log_eval_iter: int, 
        savedir: str, model_name: str, name: str = 'TEST', return_output: bool = False, 
        classification: bool = False):
    
    _logger.info('\n[Start {}]'.format(name))
    
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
            tabular_cat, tabular_con = item['inputs']['tabular']['cat'].type(torch.float), item['inputs']['tabular']['con'].type(torch.float)
            tsa = item['inputs']['tsa'].type(torch.float)
            tsb = item['inputs']['tsb'].type(torch.float)
            target = item['targets'].type(torch.float)
            
            # output
            if model_name == 'TemporalFusionTransformer':
                TFT_input = {'s_cat': tabular_cat.int(),
                            's_cont': tabular_con,
                            'k_cont': tsa,
                            'o_cont': tsb,
                            'targets': target}
                outputs = model(TFT_input)
                
            else:
                outputs = model(x_cat = tabular_cat, x_cons = tabular_con, x_tsa = tsa, x_tsb = tsb)

            # final loss
            if pre_training:
                target = torch.concat([tabular_cat, tabular_con, 
                            tsa.reshape(tabular_cat.size(0),-1), 
                            tsb.reshape(tabular_cat.size(0),-1)], axis=1)
                
            if classification:
                roi_dict = get_roi_dict()
                target = classification_label_generator(target.detach().cpu().numpy(), roi_dict)
                target = torch.tensor(np.where(target == 1, 0, 1)).type(torch.float).cuda()

                # target = torch.tensor(classification_label_generator(target.detach().cpu().numpy(), roi_dict)).type(torch.float).cuda()
                outputs = outputs.squeeze(1)
                
            # final loss
            loss = criterion(outputs, target)

            loss = accelerator.gather(loss)
            loss = torch.mean(loss)
            
            outputs, target = accelerator.gather_for_metrics((outputs.contiguous(), target.contiguous()))
            losses_m.update(loss.item(), n=target.size(0))
                        
            outputs = outputs.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            total_outputs.append(outputs)
            total_targets.append(target)
            
            # batch time        
            batch_time_m.update(time.time() - end_time)
            
            if (idx+1) % log_eval_iter == 0:
                _logger.info('{name} [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1), 
                            len(dataloader),
                            name       = name,
                            loss       = losses_m, 
                            batch_time = batch_time_m,
                            rate       = tabular_cat.size(0) / batch_time_m.val,
                            rate_avg   = tabular_cat.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))
                
            end_time = time.time()
    
    total_outputs = np.concatenate(total_outputs, axis=0)
    total_targets = np.concatenate(total_targets, axis=0)
    print(f'{name} shape:', total_outputs.shape, total_targets.shape)
    
    results = OrderedDict([('loss',losses_m.avg)])
    
    if classification:
        total_outputs = torch.Tensor(total_outputs).unsqueeze(1)
        probs = torch.sigmoid(total_outputs)
        binary_outputs = (probs >= 0.5).float()
        binary_outputs = np.array(binary_outputs.squeeze())
        metrics = cal_classification_metric(pred=binary_outputs, true=total_targets)
    else:
        metrics = cal_metric(pred = total_outputs, true = total_targets)
        result_generator_SFR = ResultGeneratorSFR(total_outputs.reshape(total_outputs.shape[0], -1), 
                                                total_targets.reshape(total_targets.shape[0], -1), 
                                                './output/SFR')
        LG_result_dict = result_generator_SFR.calc_acc_sfr()
        results.update([(f'{k}', m) for k, m in LG_result_dict.items()])
    
    # logging metrics
    _logger.info("\n{}: ".format(name) + " ".join('{}: {}'.format(k.upper(), m) for k, m in metrics.items()) + "\n")
    
    # return results
    results.update([(f'{k}', m) for k, m in metrics.items()])
    _logger.info('[End {}]\n'.format(name))
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if return_output:
            np.save(os.path.join(savedir, f'{name}_total_outputs.npy'), total_outputs)
            np.save(os.path.join(savedir, f'{name}_total_targets.npy'), total_targets)
    return results