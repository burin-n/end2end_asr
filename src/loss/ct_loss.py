import torch
import torch.multiprocessing as mp
from itertools import repeat
import time
import numpy as np

class CTLoss():
    def __init__(self, blank_index, version='numpy'):
        self.blank_index = blank_index
        self.version = version

    def __call__(self, log_prob_mid, log_probs_left, log_probs_right, input_lengths):
        left_labs, right_labs = self.get_ct_label(log_prob_mid, input_lengths, blank_index=self.blank_index,\
            lhead=len(log_probs_left), rhead=len(log_probs_right), version=self.version)
        return self.compute_ct_loss(log_probs_left, log_probs_right, left_labs, right_labs, input_lengths) 


    def get_ct_label(self, log_prob_mid, input_lengths, blank_index=0, lhead=1, rhead=1, version='numpy'):
        device=log_prob_mid.device
        log_prob_mid_argmax = log_prob_mid.argmax(dim=2).permute(1, 0).cpu()
        for i in range(log_prob_mid_argmax.size(0)):
            log_prob_mid_argmax[i, input_lengths[i]:] = blank_index
        
        p = self.get_p_fast(log_prob_mid_argmax) # M's index list
        M = self.merge_repeated_batch(log_prob_mid_argmax) # character list

        if version == 'tensor':
            _get_ct_label = self._get_ct_label_fast
        elif version == 'numpy':
            _get_ct_label = self._get_ct_label_fast_numpy
        else:
            raise NotImplementedError() 
        
        left_labs = []; omega_st = p
        for i in range(lhead):
            c, omega_st = _get_ct_label(omega_st, M, blank_index=blank_index)
            left_labs.append(c.to(device))
            
        right_labs = []; omega_st = p
        for i in range(rhead):
            c, omega_st = _get_ct_label(omega_st, M, blank_index=blank_index, right=True)
            right_labs.append(c.to(device))
        
        return left_labs, right_labs


    def compute_ct_loss(self, log_probs_left, log_probs_right, left_labs, right_labs, input_lengths):
        # NLLLoss take argument in shape (batch, class, time_step)
        criterion = torch.nn.NLLLoss(reduction='none')
        select = torch.full(log_probs_left[0].size()[:-1], 1., device=log_probs_left[0].device).transpose(1,0)
        for b in range(input_lengths.size(0)):
            select[b, input_lengths[b]:] = 0.
        left_loss = [criterion(log_probs_left[i].permute(1,2,0), left_labs[i]) for i in range(len(left_labs))]
        right_loss = [criterion(log_probs_right[i].permute(1,2,0), right_labs[i]) for i in range(len(right_labs))]
        left_loss = [(select * left_loss[i]).sum(dim=1) / input_lengths.float() for i in range(len(left_loss))]
        right_loss = [(select * right_loss[i]).sum(dim=1) / input_lengths.float() for i in range(len(right_loss))]
        
        left_loss = [left_loss[i].mean() for i in range(len(left_loss))]
        right_loss = [right_loss[i].mean() for i in range(len(right_loss))]

        return torch.stack(left_loss), torch.stack(right_loss)


    def get_p_fast(self, pi):
        if(torch.is_tensor(pi)):
            pi = pi.numpy()
        idx = np.full(pi.shape, 0, dtype=np.long)
        step = 1
        start = 1
        end = pi.shape[1]

        temp = idx[:, 0].copy()
        for t in range(start, end, step):
            change = pi[:, t] != pi[:, t-step]
            temp[change] += 1
            idx[:, t] = temp

        return idx


    def merge_repeated_batch(self, x):
        if(torch.is_tensor(x)):    
            x = x.numpy()
        merged = np.full(x.shape, 0, dtype=np.long)
        cur_position = np.full((x.shape[0],), 0, dtype=np.long)
        
        merged[:, 0] = x[:, 0].copy()
        for t in range(1, merged.shape[1]):
            change = x[:, t] != x[:, t-1]
            cur_position[change] += 1
            merged[np.arange(merged.shape[0]), cur_position] = x[:, t]
        return merged


    def _get_ct_label_fast(self, omega_st, M, blank_index=0, right=False):
        if(not torch.is_tensor(omega_st)):
            omega_st = torch.from_numpy(omega_st)
        M = torch.from_numpy(M)
        omega_n = omega_st.clone() - 1    
        char = torch.full(omega_st.shape, blank_index, dtype=torch.long)

        N, T = omega_st.shape
        if(right):
            step = -1
        else:
            step = 1

        for t in range(T):
            step_vector = torch.full((N,), step, dtype=torch.long)

            look_ahead_index = omega_st[:, t] - step_vector
            in_range = (look_ahead_index >= 0) & (look_ahead_index < T)
            candidate = torch.full(step_vector.shape, blank_index, dtype=torch.long) 
            candidate[in_range] = M[in_range, look_ahead_index[in_range]]
            ### skip blank alphabets
            step_vector[candidate == blank_index] += step

            ### new omega
            omega_n[:, t] = omega_st[:, t] - step_vector 
            in_range = (omega_n[:, t] >= 0) & (omega_n[:, t] < T)
            candidate = torch.full(step_vector.shape, blank_index, dtype=torch.long)
            candidate[in_range] = M[in_range, omega_n[in_range, t]]
            char[:, t] = candidate

        return char, omega_n


    def _get_ct_label_fast_numpy(self, omega_st, M, blank_index=0, right=False):
        omega_n = omega_st.copy() - 1    
        char = np.full(omega_st.shape, blank_index, dtype=np.long)

        N, T = omega_st.shape
        if(right):
            step = -1
        else:
            step = 1

        for t in range(T):
            look_ahead_index = omega_st[:, t] - step
            in_range = (look_ahead_index >= 0) & (look_ahead_index < T)

            candidate = np.full(look_ahead_index.shape, blank_index, dtype=np.long) 
            candidate[in_range] = M[in_range, look_ahead_index[in_range]]

            ### skip blank alphabets
            is_blank = in_range & (candidate == blank_index)
            look_ahead_index[is_blank] -= step

            in_range = (look_ahead_index >= 0) & (look_ahead_index < T) 
            candidate[in_range] =  M[in_range, look_ahead_index[in_range]]

            omega_n[:, t] = look_ahead_index
            char[:, t] = candidate 

        return torch.from_numpy(char), omega_n


if __name__ == '__main__':
    from tqdm import tqdm
    n = 10

    print('generating data...')
    # pre-random
    W = [torch.randn(2000, 64, 94, requires_grad=True) for i in range(n)]
    X = [torch.randn(2000, 64, 94, requires_grad=True) for i in range(n)]
    Y = [torch.randn(2000, 64, 94, requires_grad=True) for i in range(n)]
    Z = [torch.full((64, ), 2000, dtype=torch.int) for i in range(n)]
    blank_index = 0

    print('start!')
    start = time.time()
    A = []
    ct_criterion = CTLoss(blank_index, 'tensor')
    for w, x, y, z in tqdm(zip(W, X, Y, Z)):
        A.append( ct_criterion(w, [x,x], [y,y], z) )
    print('tensor version:', round((time.time() - start) / n, 5))

    start = time.time()
    B = []
    ct_criterion = CTLoss(blank_index, 'numpy')
    for w, x, y, z in tqdm(zip(W, X, Y, Z)):
        B.append( ct_criterion(w, [x,x], [y,y], z) )
    print('numpy version:', round((time.time() - start) / n, 5))
    
    print('are they the same?')
    for a, b in zip(A,B):
        #print( (a-b < 1e-7).all() )
        for a_v, b_v in zip(a,b):
            print(a_v - b_v < 1e-7) 