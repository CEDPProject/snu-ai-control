import numpy as np
import os
import wandb
import json
import shutil
import logging
from glob import glob
import torch
from omegaconf import OmegaConf

from exp_builder import training_dl, test_dl
from data_provider import create_dataloader
from losses import create_criterion
from optimizers import create_optimizer
from models import create_model
from utils.log import setup_main_logging
from utils.utils import version_build, load_resume_model, Float32Encoder
from utils.tools import update_information
from arguments import parser

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

def main(cfg):    
    # set seed
    set_seed(cfg.DEFAULT.seed)
    
    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )
    
    # make save directory
    savedir = os.path.join(cfg.RESULT.savedir,
                            cfg.DATASET.data_version,
                            cfg.DATASET.environment_name, 
                            cfg.MODEL.modelname, 
                            cfg.DEFAULT.exp_name)
    
    loaddir, savedir = version_build(accelerator = accelerator,
                                    logdir = savedir, 
                                    resume = cfg.TRAIN.resume)
    
    # logging
    logger = setup_main_logging(
        accelerator, 
        savedir, 
        debug=cfg.get('DEBUG', False)
    )
    
    # device
    logger.info('Device: {}'.format(accelerator.device))
    
    # load and define dataloader
    trn_dataloader, val_dataloader, tst_dataloader, metadata, scaler_dict = create_dataloader(
                            datadir                  = cfg.DATASET.datadir,
                            scaler                   = cfg.DATASET.scaler,
                            batch_size               = cfg.DATASET.batch_size,
                            shuffle                  = cfg.DATASET.shuffle,
                            num_workers              = cfg.DATASET.num_workers,
                            pin_memory               = cfg.DATASET.pin_memory,
                            drop_last                = cfg.DATASET.drop_last,
                            data_version             = cfg.DATASET.data_version,
                            split_rate               = cfg.DATASET.split_rate,
                            seq_len                  = cfg.DATASET.seq_len,
                            stride_len               = cfg.DATASET.stride_len,
                            pred_len                 = cfg.DATASET.pred_len,
                            model_name               = cfg.MODEL.modelname,
                            environment_name         = cfg.DATASET.environment_name,
                            target_features          = cfg.DATASET.target_features,
                            input_features           = cfg.DATASET.input_features,
                            ord_categorical_feature  = cfg.DATASET.ord_categorical_feature,
                            categorical_feature      = cfg.DATASET.categorical_features,
                            cat_encode_method        = cfg.DATASET.cat_encode_method,
                            numerical_feature        = cfg.DATASET.numerical_features,
                            del_feature              = cfg.DATASET.ignore_feature,
                            split_date               = cfg.DATASET.split_date,
                            end_date                 = cfg.DATASET.end_date,
                            start_date               = cfg.DATASET.start_date,
                            )
    
    update_information(model_name        = cfg.MODEL.modelname, 
                        cfg              = cfg, 
                        information_dict = metadata)
    
    # save configs
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        # print(OmegaConf.to_yaml(cfg))
        
    # MPC 실행 및 평가
    if cfg.MPC.use_mpc:
        # build Model
        model = create_model(
            modelname    = 'MPCController',
            params       = cfg.MODELSETTING
            )
        
        # 시스템 동역학 학습
        model.learn_dynamics(trn_dataloader)
        
        # 시뮬레이션
        mpc_results = model.simulate(tst_dataloader, use_mpc=True)
        baseline_results = model.simulate(tst_dataloader, use_mpc=False)
        
        # 평가
        model.evaluate(mpc_results, baseline_results, tst_dataloader)
        model.print_report()
    
    # build Model
    model = create_model(
        modelname    = cfg.MODEL.modelname,
        params       = cfg.MODELSETTING
        )
    
    logger.info('Model name: {}'.format(cfg.MODEL.modelname))
    
    # load weights
    if cfg.TRAIN.resume:
        load_resume_model(model=model, savedir=savedir, resume_num=cfg.TRAIN.resume_number)

    logger.info('# of learnable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))
    
    # set training
    criterion = create_criterion(loss_name=cfg.LOSS.loss_name)
    optimizer = create_optimizer(model     =  model, 
                                opt_name   =  cfg.OPTIMIZER.opt_name, 
                                lr         =  cfg.OPTIMIZER.lr, 
                                params     =  cfg.OPTIMIZER.params)
    
    model, optimizer, trn_dataloader, val_dataloader, tst_dataloader = accelerator.prepare(
        model, optimizer, trn_dataloader, val_dataloader, tst_dataloader
    )
    
    training_time = training_dl(
    model                 = model, 
    trn_dataloader        = trn_dataloader, 
    val_dataloader        = val_dataloader, 
    criterion             = criterion, 
    optimizer             = optimizer,
    accelerator           = accelerator, 
    savedir               = savedir,
    epochs                = cfg.TRAIN.epoch,
    eval_epochs           = cfg.TRAIN.eval_epochs, 
    log_epochs            = cfg.TRAIN.log_epochs,
    log_eval_iter         = cfg.TRAIN.log_eval_iter,
    use_wandb             = cfg.TRAIN.wandb.use, 
    wandb_iter            = cfg.TRAIN.wandb.wandb_iter,
    ckp_metric            = cfg.TRAIN.ckp_metric,
    model_name            = cfg.MODEL.modelname,
    early_stopping_metric = cfg.TRAIN.early_stopping_metric,
    early_stopping_count  = cfg.TRAIN.early_stopping_count,
    lradj                 = cfg.TRAIN.lradj,
    learning_rate         = cfg.OPTIMIZER.lr,
    model_config          = cfg.MODELSETTING,
    scaler_dict           = scaler_dict
    )

    # load best checkpoint weights
    model.load_state_dict(torch.load(os.path.join(savedir, 'best_model.pt')))
    
    # test results
    test_metrics = test_dl(
    model         = model, 
    dataloader    = tst_dataloader,     
    criterion     = criterion, 
    training_time = training_time,
    accelerator   = accelerator,
    log_interval  = cfg.TRAIN.log_eval_iter, 
    savedir       = savedir,
    model_config  = cfg.MODELSETTING,
    wandb_config  = cfg.TRAIN.wandb,
    model_name    = cfg.MODEL.modelname,
    name          = 'TEST',
    return_output = cfg.TRAIN.return_output,
    plot_result   = cfg.TRAIN.plot_result,
    scaler_dict   = scaler_dict
    )
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        test_metrics = {str(key): value for key, value in test_metrics.items()}
        logger.info('test_metrics: {}'.format(test_metrics))
        json.dump(test_metrics, open(os.path.join(savedir, 'test_results.json'),'w'), indent='\t', cls=Float32Encoder)

        if cfg.TRAIN.del_pt_file:
            pt_files = glob(os.path.join(savedir, "*.pt"))
            for pt_file in pt_files:
                try:
                    os.remove(pt_file)
                except Exception as e:
                    logger.warning(f"Failed to delete {pt_file}: {e}")
    
    print('🎉 Done 🎉 \n')
    
if __name__=='__main__':
    cfg = parser()
    main(cfg)
    