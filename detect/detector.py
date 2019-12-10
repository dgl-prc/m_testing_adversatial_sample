import sys
sys.path.append('../')
import torch
import numpy as np
from utils.model_manager import fetch_models
from utils.data_manger import load_cifar10, load_data_set, DATA_MNIST, DATA_CIFAR10, MyDataset, normalize_cifar10, \
    normalize_mnist
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils.data_manger import random_seed, datasetMutiIndx
from utils.data_manger import *
from attacks.attack_util import *
<<<<<<< HEAD
from model_mutation.mutationOperator import OpType
=======
from model_mutation.operator import OpType
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3
from utils.logging_util import setup_logging
import logging
import copy
import argparse
from models.googlenet import GoogLeNet
from models.lenet import MnistNet4


class Detector(object):
    '''
        A statistical detector for adversaries
        :param alpha : error bound of false negative
        :param beta : error bound of false positive
        :param sigma : size of indifference region
        :param kappa_nor : ratio of label change of a normal input
        :param mu : hyper parameter reflecting the difference between kappa_nor and kappa_adv
    '''

    def __init__(self, threshold, sigma, beta, alpha, seed_name, max_mutated_numbers, data_type,
                 device='cpu', models_folder=None):
        self.threshold = threshold
        self.sigma = sigma
        self.beta = beta
        self.alpha = alpha
        self.device = device
        self.data_type = data_type
        self.models_folder = models_folder
        self.seed_name = seed_name
        self.max_mutated_numbers = max_mutated_numbers
        self.start_no = 1
        self.seed_model_shell = GoogLeNet() if seed_name == "googlenet" else MnistNet4()
        if data_type == DATA_MNIST:
            self.max_models_in_memory = self.max_mutated_numbers
            self.mutated_models = fetch_models(models_folder, self.max_models_in_memory, self.device,self.seed_model_shell,
                                               start_no=self.start_no)
        else:
            self.max_models_in_memory = 20
            self.mutated_models = fetch_models(models_folder, self.max_models_in_memory,self.device, self.seed_model_shell,
                                               start_no=self.start_no)
            self.start_no += self.max_models_in_memory

    def calculate_sprt_ratio(self, c, n):
        '''
        :param c: number of model which lead to label changes
        :param n: total number of mutations
        :return: the sprt ratio
        '''
        p1 = self.threshold + self.sigma
        p0 = self.threshold - self.sigma

        return c * np.log(p1 / p0) + (n - c) * np.log((1 - p1) / (1 - p0))

    def fetch_single_model(self, t):
        '''
        :param t: fetch the t th model. from 1-index
        :return:
        '''
        if self.data_type == DATA_MNIST:
            return self.mutated_models[t - 1]
        else:
            if t <= self.start_no:
                return self.mutated_models[t % 20 - 1]
            else:
                self.mutated_models = fetch_models(self.models_folder, 20,  self.device,self.seed_model_shell,
                                                   start_no=self.start_no)
                self.start_no += self.max_models_in_memory
                return self.mutated_models[t % 20 - 1]

    def detect(self, img, origi_label):
        '''
        just judge img is an adversarial sample or not
        :param img: the adv sample
        :param origi_label: the adv label of the img
        :return:
        '''
        img = img.to(self.device)
        accept_pr = np.log((1 - self.beta) / self.alpha)
        deny_pr = np.log(self.beta / (1 - self.alpha))

        if isinstance(origi_label, torch.Tensor):
            origi_label = origi_label.item()
        stop = False
        deflected_mutated_model_count = 0
        total_mutated_model_count = 0
        while (not stop):
            total_mutated_model_count += 1
            if total_mutated_model_count > self.max_mutated_numbers:
                return False, deflected_mutated_model_count, total_mutated_model_count
            mutated_model = self.fetch_single_model(total_mutated_model_count)
            mutated_model.eval()
            new_score = mutated_model(img)
            new_lable = torch.argmax(new_score.cpu()).item()
<<<<<<< HEAD
            pr = self.calculate_sprt_ratio(deflected_mutated_model_count, total_mutated_model_count)
            if new_lable != origi_label:
                deflected_mutated_model_count += 1
                # pr = self.calculate_sprt_ratio(deflected_mutated_model_count, total_mutated_model_count)
=======
            if new_lable != origi_label:
                deflected_mutated_model_count += 1
                pr = self.calculate_sprt_ratio(deflected_mutated_model_count, total_mutated_model_count)
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3
                if pr >= accept_pr:
                    return True, deflected_mutated_model_count, total_mutated_model_count
                if pr <= deny_pr:
                    return False, deflected_mutated_model_count, total_mutated_model_count

