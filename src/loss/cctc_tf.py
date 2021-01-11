import tensorflow as tf
from itertools import repeat
import time
import numpy as np
import tensorflow.keras.backend as K


class CCTCLoss():
    def __init__(self, blank_index=0,
                ct_loss_left_weight=[0],
                ct_loss_right_weight=[0],
                ctc_loss_weight=1, 
                logits_time_major=False,
                from_logits=False,
                version='numpy',
                ctc_version='tf.nn'):

        self.blank_index = blank_index
        self.ct_loss_left_weight = ct_loss_left_weight
        self.ct_loss_right_weight = ct_loss_right_weight
        self.ctc_loss_weight = ctc_loss_weight
        self.logits_time_major = logits_time_major
        self.version = version
        self.from_logits = from_logits
       
        if ctc_version == 'tf.nn':
            self.ctc_criterion = self.ctc_criterion_nn
        elif ctc_version == 'keras.backend':
            self.ctc_criterion = self.ctc_criterion_backend
        else:
            raise NotImplementedError() 
         

        self.ct_criterion = CTLoss(blank_index, from_logits, version)
        self.n_left_context_heads = len(ct_loss_left_weight)
        self.n_right_context_heads = len(ct_loss_right_weight)
               

    def __call__(self, output_mid, output_left, output_right, input_length, labels, label_length):
        # output_mid (batch, time_step, num_classes)
        # output_left [(batch, time_step, num_classes)] list of left order
        # output_right [(batch, time_step, num_classes)] list of right order
        # input_lengths (batch)

        # CTC loss
        ctc_loss = self.ctc_criterion(labels, output_mid, label_length, input_length)
        
        # CT loss
        if( ((self.ct_loss_left_weight > 0) & (self.ct_loss_right_weight > 0)).any() ):
            ct_loss_left, ct_loss_right = self.ct_criterion(output_mid, \
                output_left, output_right, input_length)
        else:
            ct_loss_left = tf.zeros(self.ct_loss_left_weight.shape)
            ct_loss_right = tf.zeros(self.ct_loss_right_weight.shape)

        # CCTC loss
        ct_loss_left_reduced = tf.reduce_sum(self.ct_loss_left_weight * ct_loss_left)
        ct_loss_right_reduced = tf.reduce_sum(self.ct_loss_right_weight * ct_loss_right)
        cctc_loss = (self.ctc_loss_weight * ctc_loss) + ct_loss_left_reduced + ct_loss_right_reduced
        return cctc_loss, ctc_loss, ct_loss_left_reduced, ct_loss_right_reduced


    # Compute CTC loss using tf.nn.ctc_loss
    def ctc_criterion_nn(self, labels, output_mid, label_length, input_length):
        if not self.from_logits:
            raise AssertionError("tf.nn.ctc_loss takes logits as input")

        return tf.nn.ctc_loss(labels, output_mid, label_length, input_length, 
                    logits_time_major=self.logits_time_major, blank_index=self.blank_index
                )
        
    # Compute CTC loss using tf.keras.backend.ctc_batch_cost
    def ctc_criterion_backend(self, labels, output_mid, label_length, input_length):
        # This assume blank_index=n_class-1
        if not self.blank_index == output_mid.shape[-1]-1 :
            raise AssertionError("keras.backend.ctc requires blank_index = nclass-1")
        if self.from_logits:
            output_mid = tf.nn.softmax(output_mid)
        if tf.is_tensor(input_length):
            input_length = tf.reshape(input_length, (-1,1))
        else:
            input_length = input_length.reshape(-1,1)
        if tf.is_tensor(label_length):
            label_length = tf.reshape(label_length, (-1,1))
        else:
            label_length = label_length.reshape(-1,1)


        return K.ctc_batch_cost(labels, output_mid, input_length, label_length)



class CTLoss():
    def __init__(self, blank_index, from_logits=False, version='numpy'):
        self.blank_index = blank_index
        self.version = version
        self.from_logits = from_logits


    def __call__(self, output_mid, output_left, output_right, input_lengths):
        '''
            output_mid (batch x times x nclass)
            output_left [(batch x times x nclass)] size of left
            output_right [(batch x times x nclass)] size of right
        '''
        left_labs, right_labs = self.get_ct_label(output_mid, input_lengths, blank_index=self.blank_index,\
            lhead=len(output_left), rhead=len(output_right), version=self.version)
        return self.compute_ct_loss(output_left, output_right, left_labs, right_labs, input_lengths) 


    def get_ct_label(self, output_mid, input_lengths, blank_index=0, lhead=1, rhead=1, version='numpy'):
        output_mid_argmax = tf.argmax(output_mid, axis=-1).numpy()
        for i in range(output_mid_argmax.shape[0]):
            # print('inp', input_lengths[i])
            output_mid_argmax[i, input_lengths[i]:] = blank_index
        
        p = self.get_p_fast(output_mid_argmax) # M's index list
        M = self.merge_repeated_batch(output_mid_argmax) # character list

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


    def compute_ct_loss(self, output_left, output_right, left_labs, right_labs, input_lengths):
        #select = tf.fill(output_left[0].shape[:-1], 1.)
        select = np.ones(output_left[0].shape[:-1])
        for b in range(input_lengths.shape[0]):
            select[b, input_lengths[b]:] = 0.
        
        #criterion = tf.nn.sparse_softmax_cross_entropy_with_logits
        criterion = K.sparse_categorical_crossentropy

        left_loss = [criterion(left_labs[i], output_left[i], from_logits=self.from_logits) for i in range(len(left_labs))]
        right_loss = [criterion(right_labs[i], output_right[i], from_logits=self.from_logits) for i in range(len(right_labs))]

        left_loss = [tf.reduce_sum(select * left_loss[i],axis=1) / tf.cast(input_lengths, tf.dtypes.float32) for i in range(len(left_loss))]
        right_loss = [tf.reduce_sum(select * right_loss[i],axis=1) / tf.cast(input_lengths, tf.dtypes.float32) for i in range(len(right_loss))]
        
        # left_loss = [tf.reduce_mean(left_loss[i]) for i in range(len(left_loss))]
        # right_loss = [tf.reduce_mean(right_loss[i]) for i in range(len(right_loss))]

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
    lab_len=30
    from_logits = True

    print('generating data...')
    # pre-random
    W = [tf.random.normal([batch_size, time_size, n_class]) for i in range(n)]
    X = [tf.random.normal([batch_size, time_size, n_class]) for i in range(n)]
    Y = [tf.random.normal([batch_size, time_size, n_class]) for i in range(n)]

    # tf.nn.ctc_loss
    Z = [np.ones((batch_size), dtype=np.int32)*time_size for i in range(n)]
    labels = np.random.randint(n_class-1, size=batch_size*lab_len, dtype=np.int32).reshape(batch_size, -1)
    labels_length = np.ones(batch_size, dtype=np.int32)*lab_len
    blank_index=0

    blank_index = n_class-1

    print('start!')
    start = time.time()
    A = []  
    ct_criterion = CTLoss(blank_index, 'numpy')
    for w, x, y, z in tqdm(zip(W, X, Y, Z)):
        # print(z)
        A.append( ct_criterion(w, [x,x], [y,y], z ))
    print('ct loss {:.5f} seconds'.format((time.time() - start) / n))


    B = []
    start = time.time()
    cctc_criterion = CCTCLoss(blank_index, 
                ct_loss_left_weight=np.array([0.1]),
                ct_loss_right_weight=np.array([0.1]),
                ctc_loss_weight=1, 
                logits_time_major=False,
                version='numpy',
                from_logits=from_logits,
                ctc_version='tf.nn')
    for w, x, y, z in tqdm(zip(W, X, Y, Z)):
        B.append( cctc_criterion(w, [x,x], [y,y], z, labels, labels_length) )
    print('tf.nn version: {:.5f} seconds'.format((time.time() - start) / n))


    C = []
    start = time.time()
    cctc_criterion = CCTCLoss(blank_index, 
                ct_loss_left_weight=np.array([0.1]),
                ct_loss_right_weight=np.array([0.1]),
                ctc_loss_weight=1, 
                logits_time_major=False,
                from_logits=from_logits,
                version='numpy',
                ctc_version='keras.backend')
    for w, x, y, z in tqdm(zip(W, X, Y, Z)):
        C.append( cctc_criterion(w, [x,x], [y,y], z, labels, labels_length) )
    print('keras backend version: {:.5f} second'.format((time.time() - start) / n))


    # start = time.time()
    # B = []
    # ct_criterion = CTLoss(blank_index, 'tensor')
    # for w, x, y, z in tqdm(zip(W, X, Y, Z)):
    #     B.append( ct_criterion(w, [x,x], [y,y], z) )
    # print('tensor version:', round((time.time() - start) / n, 5))
    
    # print('are they the same?')
    # for a, b in zip(A,B):
    #     #print( (a-b < 1e-7).all() )
    #     for a_v, b_v in zip(a,b):
    #         print(a_v - b_v < 1e-7) 
