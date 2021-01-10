import tensorflow as tf
import time
import numpy as np

class CCTCLoss():
    def __init__(self, blank_index=0,
                ct_loss_left_weight=0,
                ct_loss_right_weight=0,
                ctc_loss_weight=1, 
                logits_time_major=False,
                version='numpy', reduction='mean', zero_infinity=True):

        self.blank_index = blank_index
        self.ct_loss_left_weight = ct_loss_left_weight
        self.ct_loss_right_weight = ct_loss_right_weight
        self.ctc_loss_weight = ctc_loss_weight
        self.logits_time_major = logits_time_major

        self.ct_criterion = CTLoss(blank_index, version)
        self.n_left_context_heads = len(ct_loss_left_weight)
        self.n_right_context_heads = len(ct_loss_right_weight)


    def __call__(self, logits_mid, logits_left, logits_right, input_lengths, labels, label_lengths):
        # logits_mid (batch, time_step, num_classes)
        # logits_left [(batch, time_step, num_classes)] list of left order
        # logits_right [(batch, time_step, num_classes)] list of right order
        # input_lengths (batch)


        # CTC loss
        ctc_loss = tf.nn.ctc_loss(
            labels, logits, label_length, logit_length, logits_time_major=False, blank_index=self.blank_index
        )
        # CT loss
        if( ((self.ct_loss_left_weight > 0) & (self.ct_loss_right_weight > 0)).any() ):
            ct_loss_left, ct_loss_right = self.ct_criterion(logits_mid, \
                logits_left, logits_right, input_lengths)
        else:
            ct_loss_left = tf.zeros(self.ct_loss_left_weight.shape)
            ct_loss_right = tf.zeros(self.ct_loss_right_weight.shape)

        # CCTC loss
        cctc_loss = (self.ctc_loss_weight * ctc_loss) + (self.ct_loss_left_weight * ct_loss_left).sum() \
                + (self.ct_loss_right_weight * ct_loss_right).sum()
        return cctc_loss



class CTLoss():
    def __init__(self, blank_index, version='numpy'):
        self.blank_index = blank_index
        self.version = version


    def __call__(self, logits_mid, logits_left, logits_right, input_lengths):
        '''
            logits_mid (batch x times x nclass)
            logits_left (batch x times x nclass)
            logits_right (batch x times x nclass)
        '''
        left_labs, right_labs = self.get_ct_label(logits_mid, input_lengths, blank_index=self.blank_index,\
            lhead=len(logits_left), rhead=len(logits_right), version=self.version)
        return self.compute_ct_loss(logits_left, logits_right, left_labs, right_labs, input_lengths) 


    def get_ct_label(self, logits_mid, input_lengths, blank_index=0, lhead=1, rhead=1, version='numpy'):
        logits_mid_argmax = tf.argmax(logits_mid, axis=-1).numpy()
        print(logits_mid_argmax.shape) 
        for i in range(logits_mid_argmax.shape[0]):
            logits_mid_argmax[i, input_lengths[i]:] = blank_index
        
        p = self.get_p_fast(logits_mid_argmax) # M's index list
        M = self.merge_repeated_batch(logits_mid_argmax) # character list

        if version == 'tensor':
            raise NotImplementedError("Tensor version is deprecated, Please use numpy version")
        elif version == 'numpy':
            _get_ct_label = self._get_ct_label_fast_numpy
        else:
            raise NotImplementedError() 
        
        left_labs = []; omega_st = p
        for i in range(lhead):
            c, omega_st = _get_ct_label(omega_st, M, blank_index=blank_index)
            left_labs.append(c)
            
        right_labs = []; omega_st = p
        for i in range(rhead):
            c, omega_st = _get_ct_label(omega_st, M, blank_index=blank_index, right=True)
            right_labs.append(c)
        
        return left_labs, right_labs


    def compute_ct_loss(self, logits_left, logits_right, left_labs, right_labs, input_lengths):
        #select = tf.fill(logits_left[0].shape[:-1], 1.)
        select = np.ones(logits_left[0].shape[:-1])
        for b in range(input_lengths.shape[0]):
            select[b, input_lengths[b]:] = 0.
        
        criterion = tf.nn.sparse_softmax_cross_entropy_with_logits

        left_loss = [criterion(left_labs[i], logits_left[i]) for i in range(len(left_labs))]
        right_loss = [criterion(right_labs[i], logits_right[i]) for i in range(len(right_labs))]

        left_loss = [tf.reduce_sum(select * left_loss[i],axis=1) / tf.cast(input_lengths, tf.dtypes.float32) for i in range(len(left_loss))]
        right_loss = [tf.reduce_sum(select * right_loss[i],axis=1) / tf.cast(input_lengths, tf.dtypes.float32) for i in range(len(right_loss))]
        
        left_loss = [tf.reduce_mean(left_loss[i]) for i in range(len(left_loss))]
        right_loss = [tf.reduce_mean(right_loss[i]) for i in range(len(right_loss))]

        return tf.stack(left_loss), tf.stack(right_loss)


    def get_p_fast(self, pi):
        if(tf.is_tensor(pi)):
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
        if(tf.is_tensor(x)):    
            x = x.numpy()
        merged = np.full(x.shape, 0, dtype=np.long)
        cur_position = np.full((x.shape[0],), 0, dtype=np.long)
        
        merged[:, 0] = x[:, 0].copy()
        for t in range(1, merged.shape[1]):
            change = x[:, t] != x[:, t-1]
            cur_position[change] += 1
            merged[np.arange(merged.shape[0]), cur_position] = x[:, t]
        return merged


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

        return tf.convert_to_tensor(char), omega_n


if __name__ == '__main__':
    from tqdm import tqdm

    n = 10
    n_class = 94
    batch_size=64
    time_size=2000

    print('generating data...')
    # pre-random
    W = [tf.random.normal([batch_size, time_size, n_class]) for i in range(n)]
    X = [tf.random.normal([batch_size, time_size, n_class]) for i in range(n)]
    Y = [tf.random.normal([batch_size, time_size, n_class]) for i in range(n)]
    Z = [tf.fill((batch_size, ), time_size) for i in range(n)]
    blank_index = n_class-1

    print('start!')
    start = time.time()
    A = []
    ct_criterion = CTLoss(blank_index, 'numpy')
    for w, x, y, z in tqdm(zip(W, X, Y, Z)):
        A.append( ct_criterion(w, [x,x], [y,y], z) )
    print('tensor version:', round((time.time() - start) / n, 5))

    start = time.time()
    B = []
    ct_criterion = CTLoss(blank_index, 'tensor')
    for w, x, y, z in tqdm(zip(W, X, Y, Z)):
        B.append( ct_criterion(w, [x,x], [y,y], z) )
    print('numpy version:', round((time.time() - start) / n, 5))
    
    print('are they the same?')
    for a, b in zip(A,B):
        #print( (a-b < 1e-7).all() )
        for a_v, b_v in zip(a,b):
            print(a_v - b_v < 1e-7) 
