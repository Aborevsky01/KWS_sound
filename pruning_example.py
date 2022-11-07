from utils import prune_conv_and_gru
from seminar import TaskConfig
import torch
import torch.nn as nn

config = TaskConfig(hidden_size=32, kernel_size=(7, 20), stride=(5, 12), gru_num_layers=1)
pr_model = CRNN(config, mode='light', qat=True).to(config.device)


opt = torch.optim.Adam(
    pr_model.parameters(),
    lr=0.003,
    weight_decay=config.weight_decay
)

scheduler_one = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6, eta_min=0.0008)

prun_mode_conv = {
    'dark kd'   : {'T' : 10, 'alpha' : 0.35}, 
    'attention' : {'T' : 10, 'alpha' : 0.25},
    'pruning' : {
        'amount' : 1,
        'runs'   : 2,
        'epochs' : 3
    }
}

_, updated_conv = prune_conv_and_gru(pr_model, 'conv', opt, scheduler_one, base_config, 
                                 exp_logger, prun_mode_conv, base_model_qat)

config = TaskConfig(hidden_size=32, cnn_out_channels=6, kernel_size=(7, 20), stride=(5, 12), gru_num_layers=1)
pr_model_gru = CRNN(config, mode='light', qat=True).to(config.device)
pr_model_gru.conv.weight = nn.Parameter(pr_model.conv.weight[updated_conv['conv']['mask'] == 1]
                                .reshape(config.cnn_out_channels, 1, *config.kernel_size))

opt = torch.optim.Adam(
    pr_model.parameters(),
    lr=0.003,
    weight_decay=config.weight_decay
)
scheduler_one = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6, eta_min=0.0008)

prun_mode_gru = {
    'dark kd'   : {'T' : 10, 'alpha' : 0.35}, 
    'attention' : {'T' : 10, 'alpha' : 0.25},
    'pruning' : {
        'amount' : 24,
        'runs'   : 2,
        'epochs' : 3
    }
}


_, updated_gru = prune_conv_and_gru(pr_model_gru, 'gru', opt, base_config, exp_logger, prun_mode_gru, base_model_qat)
