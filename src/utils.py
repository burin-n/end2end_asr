import sys
import os


def clear_checkpoints(path, num_to_keep=5, ascending=True):
    checkpoints = [] 
    for file in os.listdir(path):
        if(file[-3:] == '.pt' or file[-4:] == '.pth'):
            if('save_epoch' in path):
                skip = False
                for ep in ['130', '200', '250', '300']:
                    if ep in file:
                        skip=True
                if(skip):
                    continue
            checkpoints.append(os.path.join(path, file))
    # sorted by iteration
    if(ascending):
        func = lambda x: float('.'.join(x.split('/')[-1].split('-')[-1].split('.')[:-1]))
    else:
        func = lambda x: -float('.'.join(x.split('/')[-1].split('-')[-1].split('.')[:-1]))

    checkpoints = sorted(checkpoints, key=func)
    for ckpt in checkpoints[:-num_to_keep]:
        os.remove(ckpt)
        # print(ckpt)
        

def get_last_checkpoints(path, reverse=False):
    checkpoints = [] 
    for file in os.listdir(path):
        if(file[-3:] == '.pt' or file[-4:] == '.pth'):
            checkpoints.append(os.path.join(path, file))

    if(len(checkpoints) == 0): return None

    # sorted by iteration
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-1])) 
        
    if(len(checkpoints) == 0): return "" 
    elif(reverse): return checkpoints[0]
    else: return checkpoints[-1]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def print_pretty(d, lvl=0, indent=0, file=sys.stdout):
    for key, value in d.items():
        print(' ' * lvl * indent + str(key), file=file, end=": ")
        if isinstance(value, dict):
            print(file=file)
            print_pretty(value, lvl+1, indent, file)
        else:
            #print(str(value), file=file)
            print(' ' * (lvl+1) * indent + str(value), file=file)

def print_config_module(config_module, indent, file=sys.stdout):
    for key in ['base_config', 'audio_config', 'decoder_config', 'train_set', 'eval_set', 'test_set']:
        print(f"{key}:", file=file)
        value = config_module.get(key)
        if isinstance(value, dict):
            print_pretty(value, lvl=1, indent=indent, file=file)
        elif isinstance(value, list):
            for v in value:
                print(' ' * indent + str(v), file=file)



if __name__ == '__main__':
    clear_checkpoints(sys.argv[1], 10, ascending=True)
