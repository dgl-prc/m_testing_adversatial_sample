'''
Papernot, Nicolas, et al. "The limitations of deep learning in adversarial settings." Security and Privacy (EuroS&P)
'''

import sys

sys.path.append('../')
import torch
import math
from torch.autograd.gradcheck import *
from scipy import ndimage
from models import *
from torchvision.utils import save_image

from utils.data_manger import *
import itertools
import time
from attacks.attack_util import *
import numpy as np

'''
1. implement CPU version
2. implement GPU version
'''

class JSMA(object):

    def __init__(self, model, max_distortion, dim_features, num_out, theta=1,
                 increasing=True, optimal=True, save_img=False,verbose=True,device='cpu',
                 save_path=None,shape=None):

        self.model = model.to(device)
        self.model.eval()
        self.gama = max_distortion
        self.dim_features = dim_features  # the dim is the flattened result
        self.increasing = increasing
        self.optimal = optimal
        denominator = 1 if optimal else 2
        self.max_iter = int(math.floor(self.dim_features * self.gama / denominator))
        self.num_out = num_out
        self.theta = theta
        self.save_img = save_img
        self.verbose = verbose
        self.device = device
        self.save_path = save_path
        self.sample_shape = shape


    def saliency_map_optimal(self, jacobian, search_space, target):
        '''
         This version is according to the equation 8
            S[i] = 0 if alpha[i] < 0 or beta[i] > 0 else -1.* alpha[i] * beta[i]
        :param jacobian: a tensor,
        :param search_space: a ByteTensor
        :param target:
        :return:
        '''

        alpha = jacobian[:, target]
        beta_mat = torch.sum(jacobian, 1).squeeze() # the sum of all labels' gradient of each pixel, input_dim x 1
        beta = beta_mat - jacobian[:, target]

        mask_alpha = alpha.ge(0.)
        mask_beta = beta.le(0.)

        new_alpha = torch.mul(mask_alpha.float(), alpha)
        new_beta = torch.mul(mask_beta.float(), beta)

        saliency_map = torch.mul(new_alpha, torch.abs(new_beta))
        saliency_map = torch.mul(saliency_map, search_space.float())


        idx = torch.argmax(saliency_map)
        return idx

    def saliency_map_pair(self, jacobian, search_space, target):
        '''
        This version is according to the equation 10,but it is quite inefficient in terms of running time
        :param jacobian:
        :param search_space:
        :param target:
        :return:
        '''
        search_indices = torch.tensor(range(len(search_space)))
        search_indices =  torch.mul(search_space.float(),search_indices)
        search_indices = search_indices[search_indices.nonzero()].squeeze().numpy()

        max_change = 0
        p, q = -1, -1
        beta_mat = torch.sum(jacobian, 1).squeeze()  # decrease 2s
        beta_mat = beta_mat - jacobian[:, target]
        for p1, p2 in itertools.combinations(search_indices, 2):
            alpha = jacobian[p1, target] + jacobian[p2, target]
            beta = beta_mat[p1] + beta_mat[p2]
            if alpha > 0 and beta < 0 and -1. * alpha * beta > max_change:
                p, q = p1, p2
                max_change = -1. * alpha * beta
        return p, q

    def do_craft(self, x, target):
        '''
        NOTE:
        :param x: the raw img,it should be flattened and each element of which is between 0 and 255, that is ,
                  x should not be normalized
        :param target:
        :return: adv_sample, normal_predict, adc_predict
        '''

        target = target.item() if isinstance(target, torch.Tensor) else target

        output = self.model(x) # (batch_size,num_out)
        normal_predict = torch.argmax(output).item()
        adv_predict = normal_predict
        iter = 0
        search_space = torch.ones(self.dim_features).byte().to(self.device)
        adv_sample = x.clone()
        while adv_predict != target and iter < self.max_iter and len(search_space) != 0:
            jacobian = get_jacobian(adv_sample,self.model,self.num_out,self.device)
            if self.optimal:
                p = self.saliency_map_optimal(jacobian, search_space, target)
            else:
                p1, p2 = self.saliency_map_pair(jacobian, search_space, target)

            adv_sample = adv_sample.view(-1, 1)

            if self.optimal:
                adv_sample[p] += self.theta
            else:
                adv_sample[p1] += self.theta
                adv_sample[p2] += self.theta

            adv_sample = adv_sample.view(1,self.sample_shape['C'],self.sample_shape['H'],self.sample_shape['W'])
            # adv_sample = adv_sample.view(1, 1, 28, 28)

            # remove from search_space
            if self.optimal:
                search_space[p]=0
            else:
                search_space[p1]=0
                search_space[p2]=0

            adv_predict = torch.argmax(self.model(adv_sample)).item()
            iter += 1
            if self.verbose:
                print('Current label->{}'.format(adv_predict))

        return adv_sample,normal_predict,adv_predict

    def uniform_smaple(self,true_lable,all_labels):
        '''

        :param true_lable: tensor,single value
        :param all_labels: tensor
        :return:
        '''
        true_lable = true_lable.item()
        all_labels = all_labels.numpy() if isinstance(all_labels,torch.Tensor) else all_labels
        target_lable = np.random.choice(all_labels,1)
        while target_lable == true_lable:
            target_lable = np.random.choice(all_labels, 1)
        return torch.tensor(target_lable)










