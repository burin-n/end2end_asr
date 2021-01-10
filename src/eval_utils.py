import os
from datetime import datetime
import torch
from tqdm import tqdm
from wav2letter.decoder import GreedyDecoder, f_merge_repeated, levenshtein
from wav2letter.lossV2 import CTLoss
from .utils import clear_checkpoints
import csv

def print_samples_steps(step, log_probs_mid, log_probs_left, log_probs_right, input_lengths, batch_size, blank_index, batch, alphabet, writer, nsample=2):
    print('sample steps')
    acc_error = 0.0
    acc_div = 0.0
    acc_error_left_ct = [0.0] * len(log_probs_left)
    acc_error_right_ct = [0.0] * len(log_probs_right)
    acc_div_left_ct = [0.0] * len(log_probs_left)
    acc_div_right_ct = [0.0] * len(log_probs_right)
    ct_loss = CTLoss(blank_index)
    # log_probs_left = [x.cpu() for x in log_probs_left]
    # log_probs_right = [x.cpu() for x in log_probs_right]
    left_labs, right_labs = ct_loss.get_ct_label(log_probs_mid, input_lengths, blank_index, \
        len(log_probs_left), len(log_probs_right), version='numpy')

    for i in range(min(batch_size, nsample)):
        unmerged_sample_result = GreedyDecoder(log_probs_mid[:input_lengths[i],i:i+1,:].permute(1,2,0), merge_repeated=False, blank_index=blank_index)

        sample_result = f_merge_repeated(unmerged_sample_result, blank_index=blank_index)
        sample_result_text = alphabet.decode(sample_result)
        sample_lab_text = alphabet.decode(batch['labs'][i, :batch['labs_len'][i].item()])

        error = levenshtein(sample_result_text.replace(' ',''), sample_lab_text.replace(' ',''))
        print('prediction: {}.'.format(sample_result_text))
        print('label: {}.'.format(sample_lab_text))
        print('CER:', round(error/len(sample_lab_text.replace(' ', '')) , 4))
        acc_error += error
        acc_div += len(sample_lab_text.replace(' ', '')) 

        print("ct sample")
        print('alignment  : {}.'.format(alphabet.decode(unmerged_sample_result)))

        for j in range(len(left_labs)): # for head
            ct_lab_left = left_labs[j][i, :input_lengths[i]]
            ct_lab_right = right_labs[j][i, :input_lengths[i]]

            ct_prd_left = log_probs_left[j][:input_lengths[i],i,:].argmax(dim=-1)
            ct_prd_right = log_probs_right[j][:input_lengths[i],i,:].argmax(dim=-1)

            print('prd_left_{} : {}.'.format(j,alphabet.decode(ct_prd_left)))
            print('lab_left_{} : {}.'.format(j,alphabet.decode(ct_lab_left)))
            print()
            error_left_ct = (ct_lab_left != ct_prd_left).sum().item()
            error_div_left_ct = len(ct_lab_left)


            print('prd_right_{}: {}.'.format(j,alphabet.decode(ct_prd_right)))
            print('lab_right_{}: {}.'.format(j,alphabet.decode(ct_lab_right)))
            error_right_ct = (ct_lab_right != ct_prd_right).sum().item()
            error_div_right_ct = len(ct_lab_right) 
            print('left_cer_{} : {}'.format(j,round(error_left_ct/len(ct_lab_left), 4)))
            print('right_cer_{}: {}'.format(j,round(error_right_ct/len(ct_lab_right), 4)))
            print()
        
            acc_error_left_ct[j] += error_left_ct
            acc_div_left_ct[j] += error_div_left_ct
            acc_error_right_ct[j] += error_right_ct
            acc_div_right_ct[j] += error_div_right_ct
            assert len(alphabet.decode(ct_lab_left)) == len(alphabet.decode(ct_prd_left))
            assert len(alphabet.decode(ct_lab_right)) == len(alphabet.decode(ct_prd_right))
            assert len(alphabet.decode(unmerged_sample_result)) == len(alphabet.decode(ct_lab_right))
            assert len(alphabet.decode(unmerged_sample_result)) == len(alphabet.decode(ct_lab_left))

    writer.add_scalar('metric/sample cer', acc_error/acc_div, step)
    for j in range(len(left_labs)):
        writer.add_scalar(f'metric/sample left ct_{j} cer', acc_error_left_ct[j]/acc_div_left_ct[j], step)
        writer.add_scalar(f'metric/sample right ct_{j} cer', acc_error_right_ct[j]/acc_div_right_ct[j], step)


def is_cs(txt):
    eng = 'abcdefghijklmnopqrstuvwxyz'
    for ch in txt:
        if(ch in eng):
            return True
    return False


def eval_batch(model, eval_loader, ctc_criterion, ct_criterion, step, is_eval_ct, alphabet, pool, 
        csv_file=None, device='cuda', writer=None, blank_index=0, normed_ctc=True, 
        l_head=2, r_head=2, infer=False, out_dir=None, nbest_ct=1):
    print('run evaluation', datetime.now().strftime("%d/%m/%y %H:%M:%S"))
    print('eval ct', is_eval_ct)
    model.eval()
    with torch.no_grad():
        eval_running_loss = 0.0
        eval_running_ct_loss_left = [0.0] * l_head 
        eval_running_ct_loss_right = [0.0] * r_head
        error, error_div = 0.0, 0.0
        error_cs, error_cs_div = 0.0, 0.0
        left_ct_error, left_ct_div, right_ct_error, right_ct_div = [0.0] * l_head, [0.0] * l_head, [0.0] * r_head, [0.0] * r_head 

        if(infer):
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            csv_infer_file = open(os.path.join(out_dir, "out.csv"), 'w')
            csv_infer_writer = csv.writer(csv_infer_file)
            csv_infer_writer.writerow(["wav_name", "CER", "prediction", "reference"])

            if(is_eval_ct):
                csv_ct_file = open(os.path.join(out_dir, "out_ct.csv"), 'w')
                csv_ct_writer = csv.writer(csv_ct_file)
                csv_header = ["wav_name", "mid_pred"]

                for i in range(l_head):
                    csv_header.extend([f"ct_left_lab_{i}", f"CER_left_{i}"])
                    for j in range(1,nbest_ct+1):
                        csv_header.extend([
                            f"ct_left_pred_{i}_best_{j}",
                        ])

                for i in range(r_head):
                    csv_header.extend([f"ct_right_lab_{i}", f"CER_right_{i}"])
                    for j in range(1,nbest_ct+1):
                        csv_header.extend([
                            f"ct_right_pred_{i}_best_{j}",
                        ])

                csv_ct_writer.writerow(csv_header)

        for _, eval_batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            inputs = eval_batch['feats'].transpose(1, 2).to(device)
            log_probs_mid, log_probs_left, log_probs_right = model(inputs) 
            log_probs_mid = log_probs_mid.permute(2,0,1)
            log_probs_left = [prob.permute(2,0,1) for prob in log_probs_left]
            log_probs_right = [prob.permute(2,0,1) for prob in log_probs_right]

            maxlen = log_probs_mid.size(0)
            targets = eval_batch['labs'].to(device)
            tgt_lengths = eval_batch['labs_len'].to(device)
            input_lengths = torch.IntTensor([(maxlen * l) for l in eval_batch['feats_len']]).to(device)
            
            if(normed_ctc):
                loss = ctc_criterion(log_probs_mid, targets, input_lengths, tgt_lengths)
            else:
                loss = ctc_criterion(log_probs_mid, targets, input_lengths, tgt_lengths).mean()
            
            # loss calculation
            eval_running_loss += loss
            left_labs, right_labs = ct_criterion.get_ct_label(log_probs_mid, input_lengths, blank_index, \
                len(log_probs_left), len(log_probs_right), version='numpy')
            loss_left, loss_right = ct_criterion.compute_ct_loss(log_probs_left, log_probs_right, left_labs, right_labs, input_lengths)

            for j in range(len(loss_left)):
                eval_running_ct_loss_left[j] += loss_left[j]
                eval_running_ct_loss_right[j] += loss_right[j]
            # end loss calculation 
            
            # metric calculation
            if(is_eval_ct):
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

            else:
                results = pool.starmap(eval_batch_mp, [ (
                    log_probs_mid[:input_lengths[i],i,:].cpu().detach(),
                    eval_batch['labs'][i, :eval_batch['labs_len'][i].item()],
                    alphabet.label_from_string(' '),
                    blank_index
                    ) for i in range(log_probs_mid.size(1))])
                
                for i, result in enumerate(results):
                    error += result[0]
                    error_div += result[1]
                    txt = alphabet.decode(eval_batch['labs'][i, :eval_batch['labs_len'][i].item()])
                    if(is_cs(txt)):
                        error_cs += result[0]
                        error_cs_div += result[1]

            if(infer):
                for i in range(log_probs_mid.size(1)):
                    csv_infer_writer.writerow([
                        eval_batch['wav_path'][i], 
                        round(results[i][0]/results[i][1],5),
                        alphabet.decode(f_merge_repeated(log_probs_mid[:input_lengths[i],i,:].argmax(dim=-1))),
                        alphabet.decode(eval_batch['labs'][i, :eval_batch['labs_len'][i].item()]),
                    ])
                    if(is_eval_ct):
                        tobe_write = [eval_batch['wav_path'][i], alphabet.decode(log_probs_mid[:input_lengths[i],i,:].argmax(dim=-1))]
                        for j in range(l_head):
                            tobe_write.extend([
                                alphabet.decode(left_labs[j][i][:input_lengths[i]]),
                                round((results[i][2][j]/results[i][3][j]).item(),5),
                            ])   
                            # timestep x nbatch x nbest
                            topk = torch.topk(log_probs_left[j], nbest_ct, dim=-1)[1]
                            for k in range(nbest_ct):
                                tobe_write.extend([
                                    alphabet.decode(topk[:input_lengths[i], i, k])
                                ])                
                        for j in range(r_head):
                            tobe_write.extend([
                                alphabet.decode(right_labs[j][i][:input_lengths[i]]),
                                round((results[i][4][j]/results[i][5][j]).item(),5),
                                #alphabet.decode(log_probs_right[j][:input_lengths[i],i,:].argmax(dim=-1)),
                            ])
                            # timestep x nbatch x nbest
                            topk = torch.topk(log_probs_right[j], nbest_ct, dim=-1)[1]
                            for k in range(nbest_ct):
                                tobe_write.extend([
                                    alphabet.decode(topk[:input_lengths[i], i, k])
                                ])

                        csv_ct_writer.writerow(tobe_write)

    eval_log_time = datetime.now().strftime("%d/%m/%y %H:%M:%S")

    if(writer != None):
        writer.add_scalar('loss/eval', eval_running_loss/len(eval_loader), step) 
        for i in range(l_head):
            writer.add_scalar(f'loss/eval_ct_left_{i}', eval_running_ct_loss_left[i]/len(eval_loader), step) 
        for i in range(r_head):
            writer.add_scalar(f'loss/eval_ct_right_{i}', eval_running_ct_loss_right[i]/len(eval_loader), step) 
        
        writer.add_scalar('metric/eval_cer', error/error_div, step) 
        if(error_cs_div > 0):
            writer.add_scalar('metric/eval_cer_cs', error_cs/error_cs_div, step)

        if(is_eval_ct):
            for i in range(l_head):
                writer.add_scalar(f'metric/eval_left_ct_cer_{i}', left_ct_error[i]/left_ct_div[i], step) 
            for i in range(r_head):
                writer.add_scalar(f'metric/eval_right_ct_cer_{i}', right_ct_error[i]/right_ct_div[i], step) 

    if(csv_file != None):
        with open(csv_file, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['eval', step, loss.item(), eval_log_time, round(error/error_div,5)])
            if(error_cs_div > 0 ):
                csv_writer.writerow(['eval_cs', step, loss.item(), eval_log_time, round(error_cs/error_cs_div,5)])
    
    print("eval loss: {},\teval CER: {}".format(eval_running_loss/len(eval_loader), error/error_div))
    if(is_eval_ct):
        for i in range(l_head):
            print('left ct error rate_{} :{} right ct error rate_{} :{}'.format(i, left_ct_error[i]/left_ct_div[i], i, right_ct_error[i]/right_ct_div[i])) 

    print('finish evaluation at', eval_log_time)
    print()
    model.train()
    return round(error/error_div,5)


def eval_batch_ct_mp(logp_mid, logp_left, logp_right, mid_lab, left_lab, right_lab, input_lengths, i, space_index, blank_index=0):
    pred = f_merge_repeated(logp_mid.argmax(dim=-1), blank_index)
    error = levenshtein(pred[pred != space_index], mid_lab[mid_lab != space_index]) 
    error_div = len(mid_lab[mid_lab != space_index])
    left_ct_error  = [0.0]*len(logp_left)
    left_ct_div    = [input_lengths[i]]*len(logp_left) 
    right_ct_error = [0.0]*len(logp_right)
    right_ct_div   = [input_lengths[i]]*len(logp_right)

    for j in range(len(logp_left)):
        ct_pred_left = logp_left[j][:input_lengths[i],i, :].argmax(dim=-1)
        ct_pred_right = logp_right[j][:input_lengths[i],i, :].argmax(dim=-1)
        left_ct_error[j] = (left_lab[j][i, :input_lengths[i]] != ct_pred_left).sum().item()
        right_ct_error[j] = (right_lab[j][i, :input_lengths[i]] != ct_pred_right).sum().item()

    return error, error_div, left_ct_error, left_ct_div, right_ct_error, right_ct_div


#def eval_batch_ct_mp(logp_mid, logp_left, logp_right, mid_lab, left_lab, right_lab, space_index, blank_index=0):
#    # result = GreedyDecoder(logp_mid, blank_index, merge_repeated=True)
#    pred = f_merge_repeated(logp_mid.argmax(dim=-1), blank_index)
#    error = levenshtein(pred[pred != space_index], mid_lab[mid_lab != space_index]) 
#    error_div = len(mid_lab[mid_lab != space_index])
#    ct_pred_left = logp_left.argmax(dim=-1)
#    ct_pred_right = logp_right.argmax(dim=-1)
#    left_ct_error = (left_lab != ct_pred_left).sum().item()
#    left_ct_div = len(left_lab)
#    right_ct_error = (right_lab != ct_pred_right).sum().item()
#    right_ct_div = len(right_lab)
#    return error, error_div, left_ct_error, left_ct_div, right_ct_error, right_ct_div


def eval_batch_mp(logp_mid, mid_lab, space_index, blank_index=0):
    pred = f_merge_repeated(logp_mid.argmax(dim=-1), blank_index)
    error = levenshtein(pred[pred != space_index], mid_lab[mid_lab != space_index]) 
    error_div = len(mid_lab[mid_lab != space_index])
    return error, error_div
