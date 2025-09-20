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
    """
    cfg 딕셔너리에 metadata를 METADATA 섹션으로 추가
    """
    # MODEL 섹션 업데이트
    cfg['MODEL']['modelname'] = model_name
    
    # METADATA 섹션에 전체 metadata 저장
    cfg['MODELSETTING']['METADATA'] = information_dict
    
    return cfg