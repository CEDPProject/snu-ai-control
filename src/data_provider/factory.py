import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_provider.load_dataset import load_dataset
from data_provider.build_dataset import LG_BuildDataset
from utils.utils import seed_worker_1, seed_worker_2
from omegaconf import OmegaConf
import warnings
import pdb

warnings.filterwarnings('ignore')

def LG_create_dataloader(
                datadir: str,
                train_only: bool, 
                train_test_predefined: bool,
                scale: str,
                batch_size: int,
                shuffle: bool,
                num_workers: int,
                pin_memory: bool,
                drop_last: bool,
                pre_training: bool,
                data_num: int,
                split_rate: list = [0.7, 0.1, 0.2]
                ):

    assert scale in (None, 'minmax', 'minmax_square', 'standard')
    assert category_name in ('solar', 'windw')
    assert structure in ('independence', 'merge')

    information_dict, trn_list, val_list, tst_list = load_dataset(
                                                        datadir                = datadir,
                                                        train_only             = train_only,
                                                        train_test_predefined  = train_test_predefined,
                                                        scale                  = scale,
                                                        pre_training           = pre_training,
                                                        data_num               = data_num,
                                                        split_rate             = split_rate
                                                        )

    # build dataset
    trn_dataset = LG_BuildDataset(data_list = trn_list)
    val_dataset = LG_BuildDataset(data_list = val_list)
    tst_dataset = LG_BuildDataset(data_list = tst_list)
    
    # Randomness adjust
    g = seed_worker_2()
    
    # torch dataloader
    trn_dataloader = torch.utils.data.DataLoader(trn_dataset,
                                                batch_size     =  batch_size,
                                                shuffle        =  shuffle,
                                                num_workers    =  num_workers,
                                                pin_memory     =  pin_memory,
                                                worker_init_fn =  seed_worker_1,
                                                generator      =  g,
                                                drop_last      =  drop_last)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size     =  batch_size,
                                                shuffle        =  shuffle,
                                                num_workers    =  num_workers,
                                                pin_memory     =  pin_memory,
                                                worker_init_fn =  seed_worker_1,
                                                generator      =  g,
                                                drop_last      =  drop_last)

    tst_dataloader = torch.utils.data.DataLoader(tst_dataset,
                                                batch_size     =  batch_size,
                                                shuffle        =  False,
                                                num_workers    =  num_workers,
                                                pin_memory     =  pin_memory,
                                                worker_init_fn =  seed_worker_1,
                                                generator      =  g,
                                                drop_last      =  False)

    # information dict update
    information_dict.update({'tsa_n_series': next(iter(trn_dataset))['inputs']['tsa'].size(-1)})
    information_dict.update({'tsb_n_series': next(iter(trn_dataset))['inputs']['tsb'].size(-1)})

    information_dict.update({'tsa_n_step': next(iter(trn_dataset))['inputs']['tsa'].size(0)})
    information_dict.update({'tsb_n_step': next(iter(trn_dataset))['inputs']['tsb'].size(0)})
    
    information_dict.update({'target_num': next(iter(trn_dataloader))['targets'].size(-1)})

    
    return information_dict, trn_dataloader, val_dataloader, tst_dataloader
