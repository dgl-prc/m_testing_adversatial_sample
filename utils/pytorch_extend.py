import torch
import numpy as np
def batch_l2Norm_suqare(batch_tensor):
    '''
    pytorch do not support the batch input computation, so I devised this method
    :param batch_tensor: batchxchannelxWxH
    :return: batch x l2norm
    '''
    batch_size = batch_tensor.size()[0]
    ele_square = batch_tensor ** 2
    return torch.tensor([torch.sum(ele_square[i]) for i in range(batch_size)]).view(-1, 1)



if __name__ == '__main__':
    weight_data = torch.zeros(4,1,2,3).to('cuda')
    W = [torch.tensor(weight_data[i], requires_grad=True) for i in range(4)]



