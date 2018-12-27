# coding:utf8
'''
This module only supports neuron and weight mutation.
Ma, Lei, et al. "DeepMutation: Mutation Testing of Deep Learning Systems." arXiv preprint arXiv:1805.05206 (2018).
'''
from __future__ import division
import torch
import copy
from models import *
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import model_trainer
import logging
import math


class OpType(object):
    NAI = 'NAI'
    GF  = 'GF'
    WS = 'WS'
    NS = 'NS'


class MutaionOperator(object):
    def __init__(self, ration, model, acc_tolerant=0.90, verbose=True,test=True,test_data_laoder=None,device='cpu'):
        '''

        :param ration:
        :param model:
        :param acc_tolerant:
        :param verbose: print the mutated detail or not. like the number of weights to be mutated with layer
        :param test:
        '''
        self.ration = ration
        self.original_model = model.to(device)
        self.verbose = verbose
        self.test_data_laoder = test_data_laoder
        self.device = device

        if test:
            premier_acc = model_trainer.test(self.original_model, self.test_data_laoder,device=self.device)
            logging.info('orginal model acc={0}'.format(premier_acc))
            self.acc_threshold = round(premier_acc * acc_tolerant, 2)
            logging.info('acc_threshold:{}%'.format(100 * self.acc_threshold))

    def gaussian_fuzzing(self, std=None):
        '''
        Gaussian Fuzzing is a model mutation method in weight level
        :param std: the scale parameter of Gaussian Distribution
        :return: a mutated model
        '''
        mutation_model = copy.deepcopy(self.original_model)
        num_weights = 0
        num_layers = 0  # including the bias
        std_layers = [] # store each the standard deviation of paramets of a layer
        for param in mutation_model.parameters():
            num_weights += (param.data.view(-1)).size()[0]
            num_layers += 1
            std_layers.append(param.data.std().item())

        indices = np.random.choice(num_weights, int(num_weights * self.ration), replace=False)
        logging.info('{}/{} weights to be fuzzed'.format(len(indices),num_weights))
        weights_count = 0
        for idx_layer, param in enumerate(mutation_model.parameters()):
            shape = param.data.size()
            num_weights_layer = (param.data.view(-1)).size()[0]
            mutated_indices = set(indices) & set(
                np.arange(weights_count, weights_count + num_weights_layer))

            if mutated_indices:
                mutated_indices = np.array(list(mutated_indices))
                #########################
                # project the global index to the index of current layer
                #########################
                mutated_indices = mutated_indices - weights_count

                current_weights = param.data.cpu().view(-1).numpy()
                #####################
                #  Note: there is a slight difference from the original paper,in which a new
                #  value is generated via Gaussian distribution with the original single weight as the expectation,
                #  while we use the mean of all potential mutated weights as the expectation considering the time-consuming.
                #  In a nut shell, we yield all the mutated weights at once instead of one by one
                #########################
                avg_weights = np.mean(current_weights)
                current_std = std if std else std_layers[idx_layer]
                mutated_weights = np.random.normal(avg_weights, current_std, mutated_indices.size)

                current_weights[mutated_indices] = mutated_weights
                new_weights = torch.Tensor(current_weights).reshape(shape)
                param.data = new_weights.to(self.device)
            if self.verbose:
                print(">>:mutated weights in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_indices),
                                                                         num_weights_layer))
            weights_count += num_weights_layer

        return mutation_model

    def ws(self):
        '''
        Weight Shuffling. Shuffle selected weights
        Randomly select neurons and shuffle the weights of each neuron.
        The key point is to select the neurons and record the weights of its connection with previous layer
        For a regular layer,say full connected layer, it is a easy task, but it may be not straight to select the
        neurons in convolutional layer. we could make follow assumptions:
        1. The number of neurons in convolutional layer is equal to the number of its output elements
        2. Given the parameter sharing in conv layer,  the neuron of each
            slice in output volume has the same weights(i.e, the corresponding slice of the conv kernel)
        Hence, it is impossible to shuffle the weights of a neuron without changing others' weights which are in the same
        slice.
        To this end, instead of neurons, we shuffle the weights of certain slices.
        Note: we don't take the bias into account.
        :return: a mutated model
        '''
        unique_neurons = 0
        mutation_model = copy.deepcopy(self.original_model)
        ####################
        # there are three types of layers in parameters: conv(4-dim), bias layer(1-dim), linear layer(2-dim)
        # for example, (32, 16, 5, 5), (32,), (10, 1568) respectively
        ####################
        for param in mutation_model.parameters():
            shape = param.size()
            dim = len(shape)
            if dim > 1:
                unique_neurons += shape[0]

        indices = np.random.choice(unique_neurons, int(unique_neurons * self.ration), replace=False)
        logging.info('{}/{} weights to be shuffle'.format(len(indices),unique_neurons))
        neurons_count = 0
        for idx_layer, param in enumerate(mutation_model.parameters()):
            shape = param.size()
            dim = len(shape)
            # skip the bias
            if dim > 1:
                unique_neurons_layer = shape[0]
                mutated_neurons = set(indices) & set(np.arange(neurons_count, neurons_count + unique_neurons_layer))
                if mutated_neurons:
                    mutated_neurons = np.array(list(mutated_neurons)) - neurons_count
                    for neuron in mutated_neurons:
                        ori_shape = param.data[neuron].size()
                        old_data = param.data[neuron].view(-1).cpu().numpy()
                        # shuffle
                        shuffle_idx = np.arange(len(old_data))
                        np.random.shuffle(shuffle_idx)
                        new_data = old_data[shuffle_idx]
                        new_data = torch.Tensor(new_data).reshape(ori_shape)
                        param.data[neuron] = new_data.to(self.device)
                if self.verbose:
                    print(">>:mutated neurons in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_neurons),
                                                                             unique_neurons_layer))
                neurons_count += unique_neurons_layer

        return mutation_model

    def ns(self, skip=10):
        '''
        Neuron Switch.
        The NS operator switches two neurons within a layer to exchange their roles and inﬂuences for next layers.
        Note: we don't take the bias into account and set a constraint that the number of neurons( for regular layer)
        or filters( for convolution layer) of a layer should be at least greater than a given threshold since at least two
        neurons or filters are involved in a switch. We set 10 as the default value.
        The switch process is limited within a layer.
        :param skip: the threshold of amount of neurons in layer
        :return:
        '''
        unique_neurons = 0
        mutation_model = copy.deepcopy(self.original_model)
        ####################
        # there are three types of layers in parameters: conv(4-dim), bias layer(1-dim), linear layer(2-dim)
        # for example, (32, 16, 5, 5), (32,), (10, 1568) respectively
        ####################
        for idx_layer, param in enumerate(mutation_model.parameters()):
            shape = param.size()
            dim = len(shape)
            unique_neurons_layer = shape[0]
            # skip the bias
            if dim > 1 and unique_neurons_layer >= skip:
                import math
                temp = unique_neurons_layer * self.ration
                num_mutated = math.floor(temp) if temp > 2. else math.ceil(temp)
                mutated_neurons = np.random.choice(unique_neurons_layer,
                                                   int(num_mutated), replace=False)
                switch = copy.copy(mutated_neurons)
                np.random.shuffle(switch)
                param.data[mutated_neurons] = param.data[switch]
                if self.verbose:
                    print(">>:mutated neurons in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_neurons),
                                                                             unique_neurons_layer))
        return mutation_model

    def nai(self):
        '''
        The NAI operator tries to invert the activation status of a neuron,
        which can be achieved by changing the sign of the output value of
        a neuron before applying its activation function.
        Note: In this operator, we take the bias into account,but we don't regard the bias unit as a neuron
        :return:
        '''
        unique_neurons = 0
        mutation_model = copy.deepcopy(self.original_model)
        ####################
        # there are three types of layers in parameters: conv(4-dim), bias layer(1-dim), linear layer(2-dim)
        # for example, (32, 16, 5, 5), (32,), (10, 1568) respectively
        ####################
        for param in mutation_model.parameters():
            shape = param.size()
            dim = len(shape)
            if dim > 1:
                unique_neurons += shape[0]
        # select which neurons should be to inverted.
        indices = np.random.choice(unique_neurons, int(unique_neurons * self.ration), replace=False)
        logging.info('{}/{} neurons to be mutated'.format(len(indices),unique_neurons))
        neurons_count = 0
        last_mutated_neurons = []
        for idx_layer, param in enumerate(mutation_model.parameters()):
            shape = param.size()
            dim = len(shape)
            if dim > 1:
                unique_neurons_layer = shape[0]
                mutated_neurons = set(indices) & set(np.arange(neurons_count, neurons_count + unique_neurons_layer))
                if mutated_neurons:
                    mutated_neurons = np.array(list(mutated_neurons)) - neurons_count
                    param.data[mutated_neurons] = -1 * param.data[mutated_neurons]
                    last_mutated_neurons = mutated_neurons
                neurons_count += unique_neurons_layer
                if self.verbose:
                    print(">>:mutated neurons in {0}th layer:{1}/{2}".format(idx_layer, len(mutated_neurons),
                                                                             unique_neurons_layer))
            else:
                # invert the bias
                param.data[last_mutated_neurons] = -1 * param.data[last_mutated_neurons]
                last_mutated_neurons = []

        return mutation_model

    def afr(self, act_type):
        '''

        :param act_type: the type of activation func
        :return:
        '''
        model = copy.deepcopy(self.original_model)
        ActFun = nn.ReLU if act_type == 'relu' else nn.ELU

        num_actlayers = 0
        for module in model.modules():
            if isinstance(module, ActFun):
                num_actlayers += 1

        if num_actlayers == 0:
            raise Exception('No [{}] layer found'.format(ActFun))

        temp = num_actlayers * self.ration
        num_remove = 1 if temp < 1 else math.floor(temp)
        num_remove = int(num_remove)
        idces_remove = np.random.choice(num_actlayers, num_remove, replace=False)
        print('>>>>>>>idces_remove:{}'.format(idces_remove))
        idx = 0
        for name, module in model.named_children():
            # outer relu
            if isinstance(module, nn.ReLU):
                if idx in idces_remove:
                    model.__delattr__(name)
                idx += 1
            else:
                for grand_name, child in module.named_children():
                    if isinstance(child, nn.ReLU):
                        if idx in idces_remove:
                            module.__delattr__(grand_name)
                        idx += 1
        print(model)
        return model

    def __is_qualified(self, mutated_model):
        '''
        :param mutated_model:
        :return:
        '''
        acc = model_trainer.test(mutated_model, self.test_data_laoder,device=self.device)
        if round(acc,2) < self.acc_threshold:
            logging.info('Warning: bad accurate {0},reproduce mutated model'.format(acc))
            return False
        logging.info('Mutated model: accurate {0}'.format(acc))
        return True

    def filter(self, f, **kwargs):
        '''
        Make sure that the accuracy of the mutated model generated via 'f' satisfy the threshold.
        :param f: the function to generate mutated model,e.g. operator.afr,operator.nai
        :param kwargs:  the parameters passed in the 'f' function
        :return: a qualified model
        '''
        qualified = False
        while not qualified:
            mutated_model = f(**kwargs)
            qualified = self.__is_qualified(mutated_model)
        return mutated_model




########
# test methods
########
def _test():
    '''
    1. count the total weights
    2. random choice 1% weights and record their index in a flatten weights
    3. use gaussian distribution generate new value
    4. reset the value
    :return:
    '''
    model = torch.load('/home/npudgl/github/SafeDNN/bgDNN/model-storage/heter_samedata/3layers_model.pkl')
    operator = MutaionOperator(ration=0.1, model=model)
    # mutate_model = operator.ws()
    # mutate_model = operator.ns(4)
    mutate_model = operator.nai()


def _test_accuracy():
    test_data = torchvision.datasets.MNIST('~/github/SafeDNN/datasets/mnist/raw',
                                           train=False,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]))
    data_laoder = DataLoader(dataset=test_data)

    model = torch.load('/home/npudgl/github/SafeDNN/bgDNN/model-storage/heter_samedata/leNet_model.pkl')
    operator = MutaionOperator(ration=0.1, model=model)
    print("ori model >>>>>>>>")
    model_trainer.test(model, test_loader=data_laoder, verbose=True)
    model_trainer.test(model, test_loader=data_laoder)

    # model_trainer.test(operator.gaussian_fuzzing(), test_loader=data_laoder)
    # model_trainer.test(operator.ws(), test_loader=data_laoder)
    # model_trainer.test(operator.ns(), test_loader=data_laoder)
    # model_trainer.test(operator.nai(), test_loader=data_laoder)

    # print('>>>>>gaussian_fuzzing')
    # model_trainer.test(operator.filter(operator.gaussian_fuzzing), test_loader=data_laoder, verbose=True)
    # print('>>>>>ws')
    # model_trainer.test(operator.filter(operator.ws), test_loader=data_laoder, verbose=True)
    # print('>>>>>ns')
    # model_trainer.test(operator.filter(operator.ns), test_loader=data_laoder, verbose=True)
    # print('>>>>>nai')
    # model_trainer.test(operator.filter(operator.nai), test_loader=data_laoder, verbose=True)

    print('>>>>>AFR')
    model_trainer.test(operator.afr('relu'), test_loader=data_laoder, verbose=True)


def _test_layermodify():
    '''
    # 1. select the activation function of which layer should be remove.
    # 2. remove the activation function
    There are two ways to do layer mutated. An intuitive manner is to modify the architecture of the premier model.
    Another is to re-construct a new model according to the architecture of the premier model and initialize it with the
    trained model's weight
    :return:
    '''
    model = torch.load('/home/npudgl/github/SafeDNN/bgDNN/model-storage/heter_samedata/leNet_model.pkl')
    print(model)

    count_relu = 0
    for idx, module in enumerate(model.modules()):
        if isinstance(module, nn.ReLU):
            count_relu += 1
    print("Relu:{}".format(count_relu))

    stop = False
    for name, module in model.named_children():
        if not stop:
            for grand_name, grad_module in module.named_children():
                if isinstance(grad_module, nn.ReLU):
                    module.__delattr__(grand_name)
                    stop = True
    print model

    return model


if __name__ == '__main__':
    # __test__()
    # test_accuracy()
    # _test_layermodify()
    pass
