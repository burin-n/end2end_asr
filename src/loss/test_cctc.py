from .cctc_py import CTLoss

def unit_case(mid_prediction, gt_left, gt_right):
  blank_index=0
  ct_loss = CTLoss(blank_index)
  
  p = ct_loss.get_p_fast(mid_prediction) # M's index list
  M = ct_loss.merge_repeated_batch(mid_prediction) # character list

  left_labs = []; omega_st = p
  for i in range(2):
      c, omega_st = ct_loss._get_ct_label_fast_numpy(omega_st, M, blank_index=blank_index)
      left_labs.append(c.numpy())
      if not (left_labs[i] == gt_left[i]).all():
        raise AssertionError("\n{}\n\n{}".format(left_labs[i], gt_left[i]))

      
  right_labs = []; omega_st = p
  for i in range(2):
      c, omega_st = ct_loss._get_ct_label_fast_numpy(omega_st, M, blank_index=blank_index, right=True)
      right_labs.append(c.numpy())
      if not (right_labs[i] == gt_right[i]).all():
        raise AssertionError("\n{}\n\n{}".format(right_labs[i], gt_right[i]))

      
if __name__ == '__main__':
  mid_prediction = np.array([[1,0,0,0,2,4,4,4],
                             [4,4,4,2,0,0,0,1]])

  gt_left = [      np.array([[0,1,1,1,1,2,2,2],
                             [0,0,0,4,2,2,2,2]]),
                   np.array([[0,0,0,0,0,1,1,1],
                             [0,0,0,0,4,4,4,4]]) ]

  gt_right = [     np.array([[2,2,2,2,4,0,0,0],
                             [2,2,2,1,1,1,1,0]]),
                   np.array([[4,4,4,4,0,0,0,0],
                             [1,1,1,0,0,0,0,0]]) ]
  print('Case#1')
  unit_case(mid_prediction, gt_left, gt_right)


mid_prediction = np.array([[1,0,1,0,2,2,0,3],
                           [1,0,7,0,2,2,0,3],
                           [0,5,5,5,0,9,0,0],
                           [0,0,9,0,5,5,5,0]])

gt_left = [      np.array([[0,1,1,1,1,1,2,2],
                           [0,1,1,7,7,7,2,2],
                           [0,0,0,0,5,5,9,9],
                           [0,0,0,9,9,9,9,5]]),
                 np.array([[0,0,0,1,1,1,1,1],
                           [0,0,0,1,1,1,7,7],
                           [0,0,0,0,0,0,5,5],
                           [0,0,0,0,0,0,0,9]]) ]

gt_right = [     np.array([[1,1,2,2,3,3,3,0],
                           [7,7,2,2,3,3,3,0],
                           [5,9,9,9,9,0,0,0],
                           [9,9,5,5,0,0,0,0]]),
                 np.array([[2,2,3,3,0,0,0,0],
                           [2,2,3,3,0,0,0,0],
                           [9,0,0,0,0,0,0,0],
                           [5,5,0,0,0,0,0,0]]) ]
print('Case#2')
unit_case(mid_prediction, gt_left, gt_right)                                                 