import abc
import torch
import torchvision
import logging
from attacks.attack_util import *
class AbstractAdversary(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def do_craft(self, inputs, targets):
        ''' model should return the pre-softmax layer '''
        pass

    def check_adversarial_samples(self,model,inputs,normal_labels):
        '''
        :param model:
        :param inputs: batch x channel x H X W
        :param normal_labels: batch x 1
        :return:
        '''
        adv_output = model(inputs) # batch x dim_out
        adv_label = torch.argmax(adv_output,dim=1,keepdim=True) # batch x 1
        mask = torch.ne(adv_label,normal_labels) # batcc x 1
        mask = mask.view(-1)
        idx=torch.nonzero(mask).view(-1)
        if len(idx) == len(adv_label):
            logging.info('All elements in batch success!')
        else:
            logging.info('Success {},total {}'.format(len(idx),adv_label.size()[0]))
        # return inputs[idx]
        return inputs[idx], normal_labels[idx], adv_label[idx]



