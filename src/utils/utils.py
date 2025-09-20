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