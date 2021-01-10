import argsparse, math , csv , os, runpy, time
import numpy as np
from shutil import copyfile
from datetime import datetime
from warmup_scheduler import GradualWarmupScheduler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from pytorch_lightning.loggers import CSVLogger

from src.data_loader import CSVDataset, Alphabet, AudioDataLoader
from src.decoder import GreedyDecoder, levenshtein, f_merge_repeated
from src.utils import clear_checkpoints, get_last_checkpoints, get_lr, print_config_module
from src.utils.eval_utils import print_samples_steps, eval_batch
from src.utils.log_utils import print_loss_steps, save_summary, save_checkpoint

def train():
    torch.manual_seed(config_module.get('random_seed',0 ))
    np.random.seed(config_module.get('random_seed', 0))

    base_config = config_module.get('base_config', None)
    audio_config = config_module.get('audio_config', None)
    decoder_config = config_module.get('decoder_config', None)

    # =============== setup hyper parameters ===============

    ##### setup for continue_learnning #####
    if(continue_learning):
        base_config['load_model'] = get_last_checkpoints(base_config['log_dir'])
        base_config['load_optim'] = True
        base_config['load_only_lr'] = False

    ##### hyper parameters #####
    batch_size = base_config['batch_size']
    max_epochs = base_config.get('max_epochs', 200)
    ctc_loss_weight = base_config.get('ctc_loss_weight', 1)
    ct_loss_left_weight = base_config.get('ct_loss_left_weight', 0)
    ct_loss_right_weight = base_config.get('ct_loss_right_weight', 0)
    n_left_context_heads = len(ct_loss_left_weight)
    n_right_context_heads = len(ct_loss_right_weight)
    eval_ct_steps = base_config.get('eval_steps') * base_config.get('eval_ct_steps', 0)
    num_data_loader_workers = base_config.get('num_data_loader_workers', 0)
    device = base_config.get('device', 'cuda')

    print("data size", len(train_set))
    print("batch_size", batch_size)
    print("max_epochs", max_epochs)
    print('ctc_loss_weight', ctc_loss_weight)
    print('cctc_loss_weight', ct_loss_left_weight, ct_loss_right_weight)
    print("feature:", audio_config['features_type'])
    print("num_audio_features:", audio_config['num_audio_features'])
    print("alphabet_count", alphabet.size())
    print("num_data_loader_workers", num_data_loader_workers)
    print("mixed_precision", base_config.get('mixed_precision', False))
    print("log_dir", base_config['log_dir'])
    print('num_pool_workers', base_config.get('num_pool_workers', 8) )
    print("train_version", train_version)
    print()


    # ============ Initialize model, loss, optimizer ============
    model = base_config['model'](audio_config['num_audio_features'], alphabet.size(), 
                n_left_context_heads=n_left_context_heads, n_right_context_heads=n_right_context_heads,
                blank_index=blank_index)
    
    # model.to(device)
    ## print('model', model)
    # ctc_criterion = nn.CTCLoss(blank=blank_index, reduction='mean', zero_infinity=True)
    # ct_criterion = CTLoss(blank_index=blank_index, version='numpy')
    # total_steps = len(train_loader) * max_epochs
    # epoch = 0
    # step = 0

    model.configure_optimizers(optimizer)
    optimizer = base_config['optimizer'](model.parameters(), **base_config.get('optimizer_params', {}))
    optimizer_wrapper = base_config.get('optimizer_wrapper', None)
    if(optimizer_wrapper != None):
        optimizer = optimizer_wrapper(optimizer, **base_config.get('optimizer_wrapper_params', {}))
    scheduler = base_config.get('scheduler', None)
    
    if(scheduler != None): 
        #if('max_decay_steps' not in base_config['scheduler_params']):
        #    max_decay_steps = total_steps - step
        #    base_config['scheduler_params']['max_decay_steps'] = max_decay_steps
        lr_scheduler = scheduler(optimizer, **base_config.get('scheduler_params', {}))
        warmup_params = base_config.get('warmup_params', None)
        if(warmup_params != None):
            scheduler = GradualWarmupScheduler(optimizer, **warmup_params, after_scheduler=lr_scheduler)
        else:
            scheduler = lr_scheduler 

    
    if(base_config.get('load_model', "") != "" and base_config.get('load_model', "") != None):
        print('load model: ', base_config.get('load_model'), '\n')
        checkpoint = torch.load(base_config.get('load_model'))

        pretrained_dict = checkpoint['model_state_dict'].copy()
        if(train_version == 5 and not continue_learning):
            for lat_name in pretrained_dict.keys():
                if( 'output' in lat_name ):
                    del checkpoint['model_state_dict'][lat_name]

        pretrained_dict = checkpoint['model_state_dict'] 
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        unmet_params = {k: v for k, v in model_dict.items() if k not in pretrained_dict or pretrained_dict[k].shape != model_dict[k].shape}
        pretrained_dict.update(unmet_params)

        pretrained_dict = { k: v for k, v in pretrained_dict.items() if k in model_dict }

        # 3. load the new state dict
        model.load_state_dict(pretrained_dict)
        if(len(unmet_params) > 0):
            print("New Initialization")
            for k in unmet_params:
                print(k)
        
        if(base_config.get("load_optim", True)):
            opt_pre_state = checkpoint['optimizer_state_dict']
            current_opt_state = optimizer.state_dict()
            if(base_config.get("load_only_lr", False)):
                current_opt_state['param_groups'][0].update({'lr' : opt_pre_state['param_groups'][0]['lr']})
                optimizer.load_state_dict(current_opt_state)
            else:
                try:
                    optimizer.load_state_dict(opt_pre_state)
                except:
                    current_opt_state['param_groups'][0].update({'lr' : opt_pre_state['param_groups'][0]['lr']})
                    optimizer.load_state_dict(current_opt_state)
                
            if(checkpoint.get('scheduler_state_dict', None) != None and \
                    base_config.get("load_scheduler", True)):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        step = checkpoint['step']
        epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
    else:
        print('training from scratch')
    print()
    print('optimizer', optimizer)
    print('scheduler', scheduler)


    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    model = LitAutoEncoder()
    trainer = pl.Trainer(max_epochs=1, gpus=8, precision=16)
    trainer.fit(model, DataLoader(train), DataLoader(val))



if __name__ == "__main__":
    # try:
    #     mp.set_start_method('fork')
    # except RuntimeError:
    #     pass 
    parser = argparse.ArgumentParser(description='Wav2Letter')

    parser.add_argument('--config', type=str, default="config.py", metavar='N',
                        help='path to config.py')

    parser.add_argument('--batch_size', type=int, default=0, metavar='N',
                        help='input batch size for training (default: 0)')
    parser.add_argument('--max_epochs', type=int, default=200, metavar='N',
                        help='total max_epochs (default: 100)')
    
    parser.add_argument('--mode', type=str, default='train_eval', metavar='N',
                        help='chhose between train, train_eval and eval')
    parser.add_argument('--continue_learning', action='store_true', default=False)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--parallel",action='store_true', default=False)

    args = parser.parse_args()
    config_module = runpy.run_path(args.config)
    train(config_module, continue_learning=args.continue_learning, config_path=args.config, args=args)
