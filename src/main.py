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
from data_provider import LG_create_dataloader
from losses import create_criterion
from optimizers import create_optimizer
from models import create_model
from utils.log import setup_default_logging
from utils.utils import version_build
from arguments import parser

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

_logger = get_logger('train')

def main(cfg):

    # set logger
    logger = setup_default_logging()

    # set seed
    set_seed(cfg.DEFAULT.seed)
    
    # set accelrator
    accelerator = Accelerator()

    # make save directory
    savedir = os.path.join(cfg.RESULT.savedir, cfg.MODEL.modelname, cfg.DEFAULT.exp_name)
    loaddir, savedir = version_build(accelerator = accelerator,
                                    logdir = savedir, 
                                    resume = cfg.TRAIN.resume)
    
    # set device
    accelerator.print('Device: {}'.format(accelerator.device), main_process_only=False)
    
    # load and define dataloader
    info_dict, trn_dataloader, val_dataloader, tst_dataloader = LG_create_dataloader(
                                    datadir                  = cfg.DATASET.datadir,
                                    train_only               = cfg.DATASET.train_only, 
                                    scale                    = cfg.DATASET.scale,
                                    batch_size               = cfg.DATASET.batch_size,
                                    train_test_predefined    = cfg.DATASET.train_test_predefined,
                                    shuffle                  = cfg.DATASET.shuffle,
                                    num_workers              = cfg.DATASET.num_workers,
                                    pre_training             = cfg.TRAIN.pre_training,
                                    data_num                 = cfg.TRAIN.data_num,
                                    pin_memory               = cfg.DATASET.pin_memory,
                                    drop_last                = cfg.DATASET.drop_last,
                                    split_rate               = cfg.DATASET.split_rate,
                                    random_seed              = cfg.DEFAULT.seed
                                    )
    
    update_information(model_name       = cfg.MODEL.modelname, 
                       cfg              = cfg, 
                       information_dict = information_dict)

    # save configs
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        print(OmegaConf.to_yaml(cfg))

    # build Model
    model = create_model(
        modelname    = cfg.MODEL.modelname,
        params       = cfg.MODELSETTING
        )
    
    # load weights
    if cfg.TRAIN.resume is not None:
        load_resume_model(model=model, 
                          loaddir=loaddir, 
                          resume=cfg.TRAIN.resume, 
                          pre_training=cfg.TRAIN.pre_training,
                          fine_tuning_method=cfg.TRAIN.fine_tuning_method)

    if cfg.MODEL.modelname != 'TemporalFusionTransformer':
        _logger.info('# of learnable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))
    
    # set training
    criterion = create_criterion(loss_name='BCE') if cfg.DATASET.classification else create_criterion(loss_name=cfg.LOSS.loss_name)

    optimizer = create_optimizer(model=model, opt_name=cfg.OPTIMIZER.opt_name, lr=cfg.OPTIMIZER.lr, params=cfg.OPTIMIZER.params)

    model, optimizer, trn_dataloader, val_dataloader, tst_dataloader = accelerator.prepare(
        model, optimizer, trn_dataloader, val_dataloader, tst_dataloader
    )
    
    # wandb
    if cfg.TRAIN.wandb.use:
        # initialize wandb
        wandb.init(name    = cfg.DEFAULT.exp_name, 
                   group   = cfg.TRAIN.wandb.exp_name,
                   project = cfg.TRAIN.wandb.project_name, 
                   entity  = cfg.TRAIN.wandb.entity, 
                   config  = OmegaConf.to_container(cfg))

    # fitting model
    training(
    model                = model, 
    trainloader          = trn_dataloader, 
    validloader          = val_dataloader, 
    criterion            = criterion, 
    optimizer            = optimizer,
    accelerator          = accelerator, 
    epochs               = cfg.TRAIN.epoch,
    eval_epochs          = cfg.TRAIN.eval_epochs, 
    log_epochs           = cfg.TRAIN.log_epochs,
    log_eval_iter        = cfg.TRAIN.log_eval_iter,
    use_wandb            = cfg.TRAIN.wandb.use, 
    wandb_iter           = cfg.TRAIN.wandb.wandb_iter, 
    ckp_metric           = cfg.TRAIN.ckp_metric, 
    savedir              = savedir,
    model_name           = cfg.MODEL.modelname,
    early_stopping_count = cfg.TRAIN.early_stopping_count,
    pre_training         = cfg.TRAIN.pre_training,
    classification       = cfg.DATASET.classification,
    return_output        = False
    )
    
    # load best checkpoint weights
    model.load_state_dict(torch.load(os.path.join(savedir, 'best_model.pt')))

    val_metrics = test(
    model          = model, 
    dataloader     = val_dataloader, 
    criterion      = criterion, 
    log_eval_iter  = cfg.TRAIN.log_eval_iter,
    name           = 'VALID',
    pre_training   = cfg.TRAIN.pre_training,
    savedir        = savedir,
    accelerator    = accelerator,
    model_name     = cfg.MODEL.modelname,
    classification = cfg.DATASET.classification, 
    return_output  = False
    )
    
    # test results
    test_metrics = test(
    model          = model, 
    dataloader     = tst_dataloader, 
    criterion      = criterion, 
    log_eval_iter  = cfg.TRAIN.log_eval_iter,
    name           = 'TEST',
    pre_training   = cfg.TRAIN.pre_training,
    savedir        = savedir,
    accelerator    = accelerator,
    model_name     = cfg.MODEL.modelname,
    classification = cfg.DATASET.classification, 
    return_output  = False
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _logger.info('test_metrics: {}'.format(test_metrics))
        json.dump(test_metrics, open(os.path.join(savedir, 
                                f'test_results.json'),'w'), indent='\t', cls=Float32Encoder)

if __name__=='__main__':
    cfg = parser()
    main(cfg)
