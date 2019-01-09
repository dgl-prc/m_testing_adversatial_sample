import torch
import sys

sys.path.append('../')
from utils.data_manger import random_seed
from torch.utils.data import DataLoader
from attacks.attack_type.abstractAdversary import AbstractAdversary
from utils.pytorch_extend import batch_l2Norm_suqare


class CarliniL2(AbstractAdversary):
    '''
    only implement the L2 norm
    '''

    def __init__(self, target_model, max_iter, c, k=0, lr=1e-3, device='cpu', targeted=False):
        self.target_model = target_model.to(device)
        self.target_model.eval()
        self.c = c
        self.k = torch.tensor(1.0 * k).to(device)
        self.max_iter = max_iter
        self.lr = lr
        self.device = device
        self.TARGETED = targeted

    def __get_gap(self, adv_inputs, targets):
        '''

        :param adv_inputs:
        :param targets: if self.TARGETED, the targets is the target label, else the targets
                        is the true label(the label predicted by target model)
        :return:
        '''

        outputs = self.target_model(adv_inputs)  # batch x num_label
        target_output = []
        # todo: optimize the following code. the performance will be portion with the batch size
        # start=time.clock()
        for i, col in enumerate(targets):
            target_output.append(outputs[i][col.item()].item())
            outputs[i][col.item()] = -10000
        # print('get gap time:{}'.format(time.clock()-start))
        target_output = torch.tensor(target_output).view(-1, 1)
        others_maxout, index = torch.max(outputs, dim=1,keepdim=True)
        assert target_output.size() == others_maxout.size()

        others_maxout = others_maxout.to('cpu')
        if self.TARGETED:
            gap = others_maxout - target_output  # batch x 1
        else:
            gap = target_output - others_maxout

        return gap

    def obj_F(self, modifier, inputs, targets):
        '''
        Here, the author's implement version is slight different from the paper
        in paper, new_mg = 0.5 * (1 + torch.tanh(w))
        but in his implement, new_mg = 0.5 * (torch.tanh(raw_img+w))
        >>>>>>>self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus<<<<<<<<
        so,we follow the authors method
        :param w:
        :param x:
        :param target:
        :return:
        '''
        adv_inputs = 0.5 * torch.tanh(inputs + modifier)  # batch x channel x W x H
        # should be softmax
        adv_inputs = adv_inputs.to(self.device)
        l2dist = batch_l2Norm_suqare(adv_inputs - torch.tanh(inputs) * 0.5)
        gap = self.__get_gap(adv_inputs, targets)
        loss = l2dist + self.c * (torch.ge(gap, 0.0).float() * gap)
        return loss

    # def do_craft(self, x, target):
    #     torch.manual_seed(random_seed)
    #     x = x.to(self.device)
    #     w = torch.tensor(torch.zeros(*x.size()).to(self.device), requires_grad=True)
    #     # print self.device
    #     # print 'w:',w.get_device()
    #     optimizer = torch.optim.Adam([w], lr=self.lr)
    #     for iter in range(self.max_iter):
    #         optimizer.zero_grad()
    #         obj_loss = self.obj_F(w, x, target)
    #         obj_loss.backward()
    #         optimizer.step()
    #         # print "loss = {}".format(obj_loss)
    #     # print 'w:',w.get_device()
    #     craft_x = 0.5 * (1 + torch.tanh(w))
    #     return craft_x

    def do_craft(self, inputs, targets):
        '''

        :param inputs: batch x channel x weight x height
        :param targets: batch x labels
        :return:
        '''
        batch_size = inputs.size()[0]
        torch.manual_seed(random_seed)
        inputs = inputs.to(self.device)
        W = torch.zeros(*inputs.size(),device=self.device,requires_grad=True)
        optimizer = torch.optim.Adam([W], lr=self.lr)
        for iter in range(self.max_iter):
            optimizer.zero_grad()
            obj_loss = self.obj_F(W, inputs, targets)
            grad_to_compute = torch.zeros(*obj_loss.size()) # batch x 1
            for i in range(batch_size):
                grad_to_compute[i] = 1
                obj_loss.backward(grad_to_compute,retain_graph=True)
                grad_to_compute.zero_()
            optimizer.step()
        craft_x = 0.5 * torch.tanh(W + inputs)
        return craft_x

