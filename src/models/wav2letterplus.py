from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.loss import ct_loss
from src.eval_utils import eval_batch

# have no projection layers, context heads are used for training embedding layer only
# middle uses features from backbone only 

class Wav2LetterPlusSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, 
                    dropout=0, repeat=1, padding='same', activation=True, batch_norm=True):
        super(Wav2LetterPlusSubBlock, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.conv1d = []
        self.batch_norm = []
        
        pad_num = (dilation * (kernel_size - 1))//2

        for layer_idx in range(repeat):
            self.conv1d.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, \
                    bias=False, padding=pad_num)
            )
            if(batch_norm):
                self.batch_norm.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))

            in_channels = out_channels
                
        self.conv1d = nn.ModuleList(self.conv1d)
        
        if(batch_norm): self.batch_norm = nn.ModuleList(self.batch_norm)
        else: self.batch_norm = [None] * repeat

        if(dropout > 0): self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(repeat)])
        else: self.dropout = [None] * repeat

        # for layer in self.conv1d:
            # torch.nn.init.xavier_normal_(layer.weight)
            # torch.nn.init.kaiming_normal_(layer.weight)



    def padding_same(self, L_in, kernel_size, stride, dilation):
        return ((L_in - 1) * stride + dilation * (kernel_size - 1) + 1 - L_in + 1)//2


    def forward(self, x, training=True):

        #to_pad = self.padding_same(x.size(2), self.kernel_size, self.stride, self.dilation)
        
        for conv1d, batch_norm, dropout in zip(self.conv1d, self.batch_norm, self.dropout):
            
            #if(self.padding == 'same'):
            #    x = nn.functional.pad(x, (to_pad, to_pad), mode='constant')

            x = conv1d(x)

            if(batch_norm != None):
                x = batch_norm(x)

            if(self.activation):
                # clipped ReLU [0,20]
                x = torch.clamp(x, 0.0, 20.0)

            if(training and dropout != None):
                x = dropout(x)

        return x


class Wav2LetterPlus(pl.LightningModule):

    def __init__(self, num_features, num_classes, base_config, blank_index=0):
        super(Wav2LetterPlus, self).__init__()
        self.layers = nn.ModuleList ([
            Wav2LetterPlusSubBlock(num_features, 256, 11, 2, 1, 0.2, 1),
            Wav2LetterPlusSubBlock(256, 256, 11, 1, 1, 0.2, 3),
            Wav2LetterPlusSubBlock(256, 384, 13, 1, 1, 0.2, 3),
            Wav2LetterPlusSubBlock(384, 512, 17, 1, 1, 0.2, 3),
            Wav2LetterPlusSubBlock(512, 640, 21, 1, 1, 0.3, 3),
            Wav2LetterPlusSubBlock(640, 768, 25, 1, 1, 0.3, 3),
            Wav2LetterPlusSubBlock(768, 896, 29, 1, 2, 0.4, 1),
            Wav2LetterPlusSubBlock(896, 1024, 1, 1, 1, 0.4, 1),
        ])
        self.output_layers = nn.ModuleList([
            Wav2LetterPlusSubBlock(1024, num_classes, 1, 1, 1, activation=False, batch_norm=False)
            for _ in range(1+n_left_context_heads+n_right_context_heads)
        ])
        self.n_left_context_heads = n_left_context_heads
        self.n_right_context_heads = n_right_context_heads
    
        self.ctc_criterion = nn.CTCLoss(blank=blank_index, reduction='mean', zero_infinity=True)
        self.ct_criterion = ct_loss.CTLoss(blank_index=blank_index, version='numpy')


    ##### hyper parameters #####
        ctc_loss_weight = base_config.get('ctc_loss_weight', 1)
        ct_loss_left_weight = base_config.get('ct_loss_left_weight', 0)
        ct_loss_right_weight = base_config.get('ct_loss_right_weight', 0)
        n_left_context_heads = len(ct_loss_left_weight)
        n_right_context_heads = len(ct_loss_right_weight)
        eval_ct_steps = base_config.get('eval_steps') * base_config.get('eval_ct_steps', 0)



    def forward(self, batch, training=True):
        """Forward pass through Wav2Letter network than 
            takes log probability of output
        Args:
            batch (int): mini batch of data
            shape (batch, num_features, frame_len)
        Returns:
            log_probs (torch.Tensor):
                shape  (batch_size, num_classes, output_len)
        """
        # y_pred shape (batch_size, num_classes, output_len)
        for layer in self.layers:
            batch = layer(batch)

        # compute log softmax probability on graphemes
        # output_layers[0] is reserved for middle prediction
        return F.log_softmax(self.output_layers[0](batch))


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward

        for layer in self.layers:
            batch = layer(batch)
        # compute log softmax probability on graphemes
        # [middle, left, right]
        log_probs_mid, log_probs_left, log_probs_right = [F.log_softmax(layer(batch), dim=1) for layer in self.output_layers]
        log_probs_mid = log_probs_mid.permute(2,0,1)
        log_probs_left = [prob.permute(2,0,1) for prob in log_probs_left]
        log_probs_right = [prob.permute(2,0,1) for prob in log_probs_right]

        maxlen = log_probs_mid.size(0)
        targets = batch['labs'].to(device)
        tgt_lengths = batch['labs_len'].to(device)
        input_lengths = torch.IntTensor([(maxlen * l) for l in batch['feats_len']]).to(device)

        # CTC loss
        ctc_loss = self.ctc_criterion(log_probs_mid, targets, input_lengths, tgt_lengths)
        # CT loss
        if( ((self.ct_loss_left_weight > 0) & (self.ct_loss_right_weight > 0)).any() ):
            ct_loss_left, ct_loss_right = self.ct_criterion(log_probs_mid, \
                log_probs_left, log_probs_right,input_lengths)
        else:
            ct_loss_left = torch.zeros(ct_loss_left_weight.size()).to(device)
            ct_loss_right = torch.zeros(ct_loss_right_weight.size()).to(device)
        # CCTC loss
        loss = (self.ctc_loss_weight * ctc_loss) + (self.ct_loss_left_weight * ct_loss_left).sum() \
                + (self.ct_loss_right_weight * ct_loss_right).sum()

        self.log('train_loss', loss)
        self.log('train_ctc', ctc_loss)
        self.log('train_ct_left', ct_loss_left)
        self.log('train_ct_right', ct_loss_right)
        return loss


    def configure_optimizers(self):
        base_config = self.base_config

        optimizer = base_config['optimizer'](model.parameters(), **base_config.get('optimizer_params', {}))
        optimizer_wrapper = base_config.get('optimizer_wrapper', None)
        if(optimizer_wrapper != None):
            optimizer = optimizer_wrapper(optimizer, **base_config.get('optimizer_wrapper_params', {}))
        scheduler = base_config.get('scheduler', None)
        
        if(scheduler != None): 
            lr_scheduler = scheduler(optimizer, **base_config.get('scheduler_params', {}))
            warmup_params = base_config.get('warmup_params', None)
            if(warmup_params != None):
                scheduler = GradualWarmupScheduler(optimizer, **warmup_params, after_scheduler=lr_scheduler)
            else:
                scheduler = lr_scheduler 
        
        return [optimizer], [scheduler]
        

    def validation_step(self, eval_batch, batch_idx):
        inputs = eval_batch['feats'].transpose(1, 2).to(device)
        for layer in self.layers:
            inputs = layer(inputs)
        log_probs_mid, log_probs_left, log_probs_right = [F.log_softmax(layer(inputs), dim=1) for layer in self.output_layers] 
        log_probs_mid = log_probs_mid.permute(2,0,1)
        log_probs_left = [prob.permute(2,0,1) for prob in log_probs_left]
        log_probs_right = [prob.permute(2,0,1) for prob in log_probs_right]

        maxlen = log_probs_mid.size(0)
        targets = eval_batch['labs']
        tgt_lengths = eval_batch['labs_len']
        input_lengths = torch.IntTensor([(maxlen * l) for l in eval_batch['feats_len']]))
        
        ctc_loss = self.ctc_criterion(log_probs_mid, targets, input_lengths, tgt_lengths)
        # eval_running_loss += loss
        left_labs, right_labs = self.ct_criterion.get_ct_label(log_probs_mid, input_lengths, blank_index, \
            len(log_probs_left), len(log_probs_right), version='numpy')
        ct_loss_left, ct_loss_right = self.ct_criterion.compute_ct_loss(log_probs_left, log_probs_right, left_labs, right_labs, input_lengths)
        loss = (self.ctc_loss_weight * ctc_loss) + (self.ct_loss_left_weight * ct_loss_left).sum() \
                + (self.ct_loss_right_weight * ct_loss_right).sum()

            
        # metric calculation
        left_labs = [x.cpu() for x in left_labs]
        right_labs = [x.cpu() for x in right_labs]
        log_probs_left = [x.cpu() for x in log_probs_left]
        log_probs_right = [x.cpu() for x in log_probs_right]
        input_lengths = input_lengths.cpu()

        results = pool.starmap(eval_batch_ct_mp, [ (
            log_probs_mid[:input_lengths[i],i,:].cpu().detach(),
            log_probs_left,
            log_probs_right,
            eval_batch['labs'][i, :eval_batch['labs_len'][i].item()],
            left_labs,
            right_labs,
            input_lengths,
            i,
            alphabet.label_from_string(' '),
            blank_index,
            ) 
            for i in range(log_probs_mid.size(1))])

        for i, result in enumerate(results):
            error += result[0]
            error_div += result[1]
            for j in range(len(result[2])):
                left_ct_error[j] += result[2][j]
                left_ct_div[j] += result[3][j]
            for j in range(len(result[4])):
                right_ct_error[j] += result[4][j]
                right_ct_div[j] += result[5][j]
            txt = alphabet.decode(eval_batch['labs'][i, :eval_batch['labs_len'][i].item()])
            if(is_cs(txt)):
                error_cs += result[0]
                error_cs_div += result[1]

        # else:
        #     results = pool.starmap(eval_batch_mp, [ (
        #         log_probs_mid[:input_lengths[i],i,:].cpu().detach(),
        #         eval_batch['labs'][i, :eval_batch['labs_len'][i].item()],
        #         alphabet.label_from_string(' '),
        #         blank_index
        #         ) for i in range(log_probs_mid.size(1))])
            
        #     for i, result in enumerate(results):
        #         error += result[0]
        #         error_div += result[1]
        #         txt = alphabet.decode(eval_batch['labs'][i, :eval_batch['labs_len'][i].item()])
        #         if(is_cs(txt)):
        #             error_cs += result[0]
                    # error_cs_div += result[1]

        return {
            'error_mid' : error,
            'eror_mid_div' : error_div,
            'error_left' : left_ct_error,
            'error_left_div' : left_ct_div,
            'error_right' : right_ct_error,
            'error_right_div' : right_ct_div,

            'loss_ctc' : ctc_loss,
            'loss_ct_left' : ct_loss_left,
            'loss_ct_right' : ct_loss_right,
            'loss_eval' : loss,
            
    
            
        }


    def unfreeze_softmax(self):
        for param in self.output_layers.parameters():
            param.requires_grad = True 


    def unfreeze_projection(self):
        pass


    def freeze_backbone(self):
        for param in self.layers.parameters():
            param.requires_grad = False


    def unfreeze_backbone(self, layer_idx=None, sublayer_idx=None):
        if (layer_idx == None or sublayer_idx == None):
            raise NotImplementedError

        for param in self.layers[layer_idx].conv1d[sublayer_idx].parameters():
            param.requires_grad = True
        for param in self.layers[layer_idx].batch_norm[sublayer_idx].parameters():
            param.requires_grad = True


    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False




