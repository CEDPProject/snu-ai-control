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

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def cal_metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    
    metric_dict = {'MAE':mae,
                   'MSE':mse,
                   'RMSE':rmse,
                   'MAPE':mape,
                   'MSPE':mspe,
                   'RSE':rse,
                   }
    
    print('mse: {:.4f}\t mae: {:.4f}\t mape: {:.4f}'.format(mse, mae, mape))
    return metric_dict

def cal_classification_metric(pred, true):
    accuracy = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    recall = recall_score(true, pred)
    precision = precision_score(true, pred)
    metric_dict = {
        'accuracy': accuracy,
        'f1' : f1,
        'recall' : recall,
        'precision' : precision
    }
    return metric_dict
    

def set_seed(random_seed: int = 42):
    """
    set deterministic seed

    Parameters
    --------
    random_seed : int(default=72)
        random seed number

    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # CUDA Randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def seed_worker_1(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_worker_2():
    g = torch.Generator()
    g.manual_seed(0)
    return g

def train_tool(model, optimizer, train_loader, criterion, device, train_epochs = 10):
    
    time_now = time.time()
    train_steps = len(train_loader)
    model.train()
    total_loss = 0
    
    
    for epoch in range(train_epochs):
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        
        for i, input in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()
    
            tabular_cat, tabular_con = input['inputs']['tabular']['cat'].to(device), input['inputs']['tabular']['con'].to(device)
            tsa = input['inputs']['tsa'].to(device)
            tsb = input['inputs']['tsb'].to(device)
            target = input['targets'].to(device)
            
            output = model(x_cat = tabular_cat, x_cons = tabular_con,
                        x_tsa = tsa, x_tsb = tsb)
            
            loss = criterion(output, target)
            train_loss.append(loss.item())
            
            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            
            loss.backward()
            optimizer.step()
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
            epoch + 1, train_steps, train_loss))
        
    return total_loss / len(train_loader)

def test_tool(model, test_loader, criterion, device):    
    preds = []
    trues = []

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            tabular_cat, tabular_con = input['inputs']['tabular']['cat'].to(device), input['inputs']['tabular']['con'].to(device)
            tsa = input['inputs']['tsa'].to(device)
            tsb = input['inputs']['tsb'].to(device)
            target = input['targets'].to(device)
                        
            output = model(x_cat = tabular_cat, x_cons = tabular_con,
                        x_tsa = tsa, x_tsb = tsb)
            loss = criterion(output, target)
            
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            
            pred = output
            true = target
            
            preds.append(pred)
            trues.append(true)
            
    preds = np.array(preds)
    trues = np.array(trues)
    
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}, mape:{}'.format(mse, mae, mape))
    LG_accuracy = (1.0 - mape) * 100
    print('Final accuracy: {}'.format(LG_accuracy))
    
    # np.save('metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    # np.save('pred.npy', preds)
    # np.save('true.npy', trues)
    
    return preds, trues, LG_accuracy


def log_setting(logdir: str,
                log_name: str,
                formatter: str = '%(asctime)s|%(name)s|%(levelname)s:%(message)s'):
    """
    logger setting

    Parameters
    ----------
    log_name : str
        logger's name
    formatter : str(default=%(asctime)s|%(name)s|%(levelname)s:%(message)s)
        logging formatter

    Return
    ------
    logger

    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(logdir, f'{log_name}.log'))
    formatter = logging.Formatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

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
        
def make_save(accelerator, savedir: str, resume: bool = False) -> str:
    # resume
    if resume:
        assert os.path.isdir(savedir), f'{savedir} does not exist'
        # check version
        version = len([f for f in glob.glob(os.path.join(savedir, '*')) if os.path.isdir(f)])
        # init version
        if version == 0:
            # check saved files
            files = [f for f in glob.glob(os.path.join(savedir, '*')) if os.path.isfile(f)]
            # make version0
            version0_dir = os.path.join(savedir, f'train{version}')
            
            # move saved files into version0
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.makedirs(version0_dir)
                for f in files:
                    shutil.move(f, f.replace(savedir, version0_dir))
            version += 1
        
        savedir = os.path.join(savedir, f'train{version}')

    # make save directory
    # assert not os.path.isdir(savedir), f'{savedir} already exists'
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(savedir, exist_ok=True)
    print("make save directory {}".format(savedir))

    return savedir

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