import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            
def update_information(model_name, cfg, information_dict):
    if model_name == 'TabTransformer':
        cfg.MODELSETTING.pre_training = cfg.TRAIN.pre_training
        cfg.MODELSETTING.classification = cfg.DATASET.classification
        cfg.MODELSETTING.tab_model_args.cat_field_dims = information_dict['cat_field_dims']
        cfg.MODELSETTING.tab_model_args.cons_dims = information_dict['num_continuous']

        cfg.MODELSETTING.tsa_model_args.enc_in = information_dict['tsa_n_series']
        cfg.MODELSETTING.tsa_model_args.dec_in = information_dict['tsa_n_series']
        cfg.MODELSETTING.tsa_model_args.c_out = information_dict['tsa_n_series']
        cfg.MODELSETTING.tsa_model_args.n_series = information_dict['tsa_n_series']
        
        cfg.MODELSETTING.tsb_model_args.enc_in = information_dict['tsb_n_series']
        cfg.MODELSETTING.tsb_model_args.dec_in = information_dict['tsb_n_series']
        cfg.MODELSETTING.tsb_model_args.c_out = information_dict['tsb_n_series']
        cfg.MODELSETTING.tsb_model_args.n_series = information_dict['tsb_n_series']

    elif model_name == 'FTTransformer':
        cfg.MODELSETTING.pre_training = cfg.TRAIN.pre_training
        cfg.MODELSETTING.classification = cfg.DATASET.classification
        cfg.MODELSETTING.tab_model_args.d_numerical = information_dict['num_continuous']
        cfg.MODELSETTING.tab_model_args.categories = information_dict['cat_field_dims']
        
        cfg.MODELSETTING.tsa_model_args.enc_in = information_dict['tsa_n_series']
        cfg.MODELSETTING.tsa_model_args.dec_in = information_dict['tsa_n_series']
        cfg.MODELSETTING.tsa_model_args.c_out = information_dict['tsa_n_series']
        cfg.MODELSETTING.tsa_model_args.n_series = information_dict['tsa_n_series']
        
        cfg.MODELSETTING.tsb_model_args.enc_in = information_dict['tsb_n_series']
        cfg.MODELSETTING.tsb_model_args.dec_in = information_dict['tsb_n_series']
        cfg.MODELSETTING.tsb_model_args.c_out = information_dict['tsb_n_series']
        cfg.MODELSETTING.tsb_model_args.n_series = information_dict['tsb_n_series']

    elif model_name == 'GatedTabTransformer':
        cfg.MODELSETTING.pre_training = cfg.TRAIN.pre_training
        cfg.MODELSETTING.classification = cfg.DATASET.classification
        cfg.MODELSETTING.tab_model_args.cat_field_dims = information_dict['cat_field_dims']
        cfg.MODELSETTING.tab_model_args.cons_dims = information_dict['num_continuous']
        
        cfg.MODELSETTING.tsa_model_args.enc_in = information_dict['tsa_n_series']
        cfg.MODELSETTING.tsa_model_args.dec_in = information_dict['tsa_n_series']
        cfg.MODELSETTING.tsa_model_args.c_out = information_dict['tsa_n_series']
        cfg.MODELSETTING.tsa_model_args.n_series = information_dict['tsa_n_series']
        
        cfg.MODELSETTING.tsb_model_args.enc_in = information_dict['tsb_n_series']
        cfg.MODELSETTING.tsb_model_args.dec_in = information_dict['tsb_n_series']
        cfg.MODELSETTING.tsb_model_args.c_out = information_dict['tsb_n_series']
        cfg.MODELSETTING.tsb_model_args.n_series = information_dict['tsb_n_series']
        
    elif model_name == 'TemporalFusionTransformer':
        cfg.MODELSETTING.pre_training = cfg.TRAIN.pre_training
        cfg.MODELSETTING.classification = cfg.DATASET.classification
        cfg.MODELSETTING.static_categorical_inp_lens = information_dict['cat_field_dims']
        cfg.MODELSETTING.categories = information_dict['cat_field_dims']
        
        cfg.MODELSETTING.temporal_tsb_continuous_inp_size = information_dict['tsb_n_series']
        cfg.MODELSETTING.tsb_times = information_dict['tsb_n_step']
        
        cfg.MODELSETTING.temporal_tsa_continuous_inp_size = information_dict['tsa_n_series']
        cfg.MODELSETTING.tsa_times = information_dict['tsa_n_step']
        
        cfg.MODELSETTING.temporal_target_size = information_dict['target_num']
        cfg.MODELSETTING.static_continuous_inp_size = information_dict['num_continuous']
        
        cfg.MODELSETTING.num_static_vars = information_dict['num_continuous'] + information_dict['num_category'] 
        cfg.MODELSETTING.num_future_vars = information_dict['tsa_n_series'] + len(cfg.MODELSETTING.temporal_tsa_categorical_inp_lens)
        cfg.MODELSETTING.num_historic_vars = information_dict['tsb_n_series'] + len(cfg.MODELSETTING.temporal_tsb_categorical_inp_lens)
        cfg.MODELSETTING.n_times = information_dict['tsa_n_step'] + information_dict['tsb_n_step'] 