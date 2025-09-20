import numpy as np
import os
import wandb
import json
import shutil
import logging
from glob import glob
import torch
from omegaconf import OmegaConf

from exp_builder import training, test
from data_provider import create_dataloader
from losses import create_criterion
from optimizers import create_optimizer
from models import create_model
from utils.log import setup_default_logging
from utils.utils import version_build
from utils.tools import update_information
from arguments import parser

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

_logger = get_logger('train')

def main(cfg):

    # set logger
    setup_default_logging()
    
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
                            cfg.MODEL.modelname, 
                            cfg.DEFAULT.exp_name)
    
    loaddir, savedir = version_build(accelerator = accelerator,
                                    logdir = savedir, 
                                    resume = cfg.TRAIN.resume)
    
    # set device
    if accelerator.is_local_main_process:
        _logger.info('Device: {}'.format(accelerator.device))
    
    # load and define dataloader
    data_loader, metadata, scaler_dict = create_dataloader(
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
                            pred_len                 = cfg.DATASET.pred_len,
                            model_name               = cfg.MODEL.modelname,
                            environment_name         = cfg.DATASET.environment_name,
                            ord_categorical_feature  = cfg.DATASET.ord_categorical_feature,
                            categorical_feature      = cfg.DATASET.categorical_feature,
                            numerical_feature        = cfg.DATASET.numerical_feature,
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
        
    # build Model
    model = create_model(
        modelname    = cfg.MODEL.modelname,
        params       = cfg.MODELSETTING
        )
    
    # MPC 실행 및 평가
    if cfg.MPC.use_mpc:
        # 시스템 동역학 학습
        model.learn_dynamics(data_loader['train'])
        
        # 시뮬레이션
        mpc_results = model.simulate(data_loader['test'], use_mpc=True)
        baseline_results = model.simulate(data_loader['test'], use_mpc=False)
        
        # 평가
        model.evaluate(mpc_results, baseline_results, data_loader['test'])
        model.print_report()

if __name__=='__main__':
    cfg = parser()
    main(cfg)
    