import torch
import numpy as np
import random

from pathlib import Path
from collections import OrderedDict
import json

from torch.nn.utils import prune
import copy
import time


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def set_random_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def create_log(log_dict):
    open(log_dict['name'] + '.log', 'w+')
    log_config = read_json(log_dict['json'])
    log_config['handlers']['info_file_handler']['filename'] = log_dict['name'] + '.log'
    logging.config.dictConfig(log_config)
    return logging.getLogger(log_dict['name'])
    

def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)
    
    
def prune_conv_gru(model, module, opt, scheduler, config, logger, mode,  *args, **kwargs):
    if module == 'conv':
        updated = {
            'conv' : {
                'module': model.conv,
                'name'  : 'weight',
                'mask'  : torch.ones(model.conv.weight.shape).to(TaskConfig.device),
                'amount': mode['pruning']['amount']
            }
        }
    elif module == 'gru':
        updated = {
            'gru_one' : {
            'module': model.gru,
            'name'  : 'weight_ih_l0',
            'mask'  : torch.ones(model.gru.weight_ih_l0.shape).to(TaskConfig.device),
            'amount': mode['pruning']['amount']
        },
            'gru_two' : {
            'module': model.gru,
            'name'  : 'weight_hh_l0',
            'mask'  : torch.ones(model.gru.weight_hh_l0.shape).to(TaskConfig.device),
            'amount': mode['pruning']['amount']
        }
    }
    logger.info('Pruning starts for {0}'.format(module))
    start = time.time()
    for run in range(mode['pruning']['runs']):
        for epoch in range(mode['pruning']['epochs']):
            train_epoch(model, opt, train_loader,
                    melspec_train, config.device, scheduler, mode, *args, **kwargs)
        
        for (key, value) in updated.items():
            prune.ln_structured(value['module'], value['name'], amount=value['amount'], n=1, dim=0)
            value['mask'] = dict(value['module'].named_buffers())[value['name'] + '_mask']
    logger.info('Pruning has been accomplished for {0} minutes'.format(np.round((time.time() - start) / 60, 2)))
    return model, updated