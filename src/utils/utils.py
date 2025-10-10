import numpy as np
import torch
from tqdm import tqdm
import time
import logging
import os
import glob
import json
import shutil
import random
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def version_build(accelerator, logdir: str, resume: int=None) -> str:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
    if resume is None:
        version = len(os.listdir(logdir))
        load_dir = os.path.join(logdir, f'version{version}')
        save_dir = os.path.join(logdir, f'version{version}')
    else:
        version = resume
        load_dir = os.path.join(logdir, f'version{version}')
        save_dir = os.path.join(logdir, f'version{len(os.listdir(logdir))}')
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(save_dir)
    return load_dir, save_dir


def load_resume_model(model, loaddir: str, resume: int, pre_training: bool, fine_tuning_method: str):
    new_weights = torch.load(os.path.join(loaddir,'best_model.pt'))
    
    if not pre_training:
        # List of weights to be removed
        weights_to_remove = [
            'TFTpart2.recon_layer.weight',
            'TFTpart2.recon_layer.bias',
            'TFTpart2.mlp.0.weight',
            'TFTpart2.mlp.0.bias'
        ]
        # Remove the specified weights
        for weight in weights_to_remove:
            if weight in new_weights:
                del new_weights[weight]
                
    # load weights
    model.load_state_dict(new_weights, strict=False)

    if not pre_training:
        if fine_tuning_method == 'full':
            for param in model.parameters():
                param.requires_grad = True
            print(f'Fine tuning method: Full')
        elif fine_tuning_method == 'linear_probing':
            train_params = ['final_layers']
            for name, param in model.named_parameters():
                if any(x in name for x in train_params):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f'Fine tuning method: Linear probing')
    print('load model from (version {})'.format(resume))
    
class Float32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)
