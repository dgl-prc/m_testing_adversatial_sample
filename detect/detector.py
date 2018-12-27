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
from attack_manners.attack_util import *
from model_mutation.operator import OP_NAME, OpType
from utils.logging_util import setup_logging
import logging
import copy
import argparse


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

        if data_type == DATA_MNIST:
            self.max_models_in_memory = self.max_mutated_models
            self.mutated_models = fetch_models(models_folder, self.max_models_in_memory, seed_name, self.device,
                                               start_no=self.start_no)
        else:
            self.max_models_in_memory = 20
            self.mutated_models = fetch_models(models_folder, self.max_models_in_memory, seed_name, self.device,
                                               start_no=self.start_no)
            self.start_no += self.max_models_in_memory

    def calculate_sprt_ratio(self, c, n):
        '''
        :param c: number of model which lead to label changes
        :param n: total number of mutations
        :return: the sprt ratio
        '''

        p1 = self.threshold - self.sigma
        p0 = self.threshold + self.sigma

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
                self.mutated_models = fetch_models(self.models_folder, 20, self.seed_name, self.device,
                                                   start_no=self.start_no)
                self.start_no += self.max_models_in_memory
                return self.mutated_models[t % 20 - 1]

    def detect_model_mutation(self, img, origi_label):
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
            new_score = mutated_model(img)
            new_lable = torch.argmax(new_score.cpu()).item()
            if new_lable != origi_label:
                deflected_mutated_model_count += 1
                pr = self.calculate_sprt_ratio(deflected_mutated_model_count, total_mutated_model_count)
                if pr >= accept_pr:
                    return True, deflected_mutated_model_count, total_mutated_model_count
                if pr <= deny_pr:
                    return False, deflected_mutated_model_count, total_mutated_model_count





def main(detect_type, data_loader, models_folder, seed_model_name, threshold, sigma, operator_type, attack_type,
         device, data_type):
    ## mnist
    alpha = 0.05
    beta = 0.05
    detector = Detector(threshold=threshold, sigma=sigma, beta=beta, alpha=alpha, models_folder=models_folder,
                        seed_name=seed_model_name,
                        max_mutated_models=500, device=device, data_type=data_type)
    adv_success = 0
    progress = 0
    avg_mutated_used = 0
    if detect_type == 'adv':
        for img, true_label, adv_label in data_loader:
            rst, success_mutated, total_mutated = detector.detect(img, origi_label=adv_label)
            if rst:
                adv_success += 1
            avg_mutated_used += total_mutated
    else:
        for img, true_label in data_loader:
            rst, success_mutated, total_mutated = detector.detect(img, origi_label=true_label)
            if rst:
                adv_success += 1
            progress += 1
            sys.stdout.write('\r Processed:%d' % (progress))
            sys.stdout.flush()
            avg_mutated_used += total_mutated
    avg_mutated_used = avg_mutated_used * 1. / len(data_loader.dataset)

    total_data = len(data_loader.dataset)
    if detect_type == 'adv':
        logging.info(
            '{},{}-Adv Accuracy:{}/{},{:.4f},,avg_mutated_used:{:.4f}'.format(operator_type, attack_type, adv_success,
                                                                              total_data,
                                                                              adv_success * 1. / total_data,
                                                                              avg_mutated_used))
        avg_accuracy = adv_success * 1. / len(data_loader.dataset)
    else:
        logging.info(
            '{},Normal Accuracy:{}/{},{:.4f},,avg_mutated_used:{:.4f}'.format(operator_type, total_data - adv_success,
                                                                              total_data,
                                                                              1 - adv_success * 1. / total_data,
                                                                              avg_mutated_used))
        avg_accuracy = 1 - adv_success * 1. / total_data

    return avg_accuracy, avg_mutated_used

def show_progress(**kwargs):
    sys.stdout.write('\r Processed:%d' % (kwargs['progress']))
    sys.stdout.flush()


def get_data_loader(data_path, is_adv_data, data_type):
    if data_type == DATA_MNIST:
        img_mode = 'L'
        normalize = normalize_mnist
    else:
        img_mode = None
        normalize = normalize_cifar10

    if is_adv_data:

        tf = transforms.Compose([transforms.ToTensor(), normalize])
        dataset = MyDataset(root=data_path, transform=tf, img_mode=img_mode, max_size=TEST_SMAPLES)  # mnist
        dataloader = DataLoader(dataset=dataset)
    else:
        dataset, channel = load_data_set(data_type, data_path, False)
        random_indcies = np.arange(10000)
        np.random.seed(random_seed)
        np.random.shuffle(random_indcies)
        random_indcies = random_indcies[:TEST_SMAPLES]
        data = datasetMutiIndx(dataset, random_indcies)
        dataloader = DataLoader(dataset=data)
    return dataloader


def get_wrong_label_data_loader(data_path, seed_model, data_type,device):
    dataset, channel = load_data_set(data_type, data_path, False)
    dataloader = DataLoader(dataset=dataset)
    wrong_labeled = samples_filter(seed_model, dataloader, return_type='adv', name='seed model',device=device)
    data = datasetMutiIndx(dataset, [idx for idx, _, _ in wrong_labeled][:TEST_SMAPLES])
    wrong_labels = [wrong_label for idx, true_label, wrong_label in wrong_labeled[:TEST_SMAPLES]]
    data = TensorDataset(data.tensors[0], data.tensors[1], torch.LongTensor(wrong_labels))
    return DataLoader(dataset=data)


def get_threshold_relax(a, t_scale, r_scale):
    return a * t_scale, a * r_scale


def get_mnist_parameters(adv_type):
    '''
    model mutation ration 0.03
    :param adv_type:
    :return:
    '''
    path_prefix = '/home/dong/gitgub/SafeDNN/datasets/mnist/adversarial'
    if adv_type == Adv_Tpye.FGSM:
        data_path = os.path.join(path_prefix, 'fgsm/single/non-pure/mnist4-eta3')
        attack_type = Adv_Tpye.FGSM
    if adv_type == Adv_Tpye.CW:
        data_path = os.path.join(path_prefix, 'cw/single/non-pure/mnist4-c8-i1w/')
        attack_type = Adv_Tpye.CW
    if adv_type == Adv_Tpye.JSMA:
        data_path = os.path.join(path_prefix, 'jsma/single/non-pure/mnist4-d12/')
        attack_type = Adv_Tpye.JSMA
    if adv_type == Adv_Tpye.DEEPFOOL:
        data_path = os.path.join(path_prefix, 'deepfool/single/non-pure/mnist4-0.02-50/')
        attack_type = Adv_Tpye.DEEPFOOL
    if adv_type == Adv_Tpye.BB:
        data_path = os.path.join(path_prefix, 'bb/single/non-pure/mnist4-fgsm35/')
        attack_type = Adv_Tpye.BB

    return data_path, attack_type


def get_cifar10_parameters(adv_type):
    '''
    model mutation ration 0.03
    :param adv_type:
    :return:
    '''
    path_prefix = '/home/dong/gitgub/SafeDNN/datasets/cifar10/adversarial-pure/'
    if adv_type == Adv_Tpye.FGSM:
        data_path = os.path.join(path_prefix, 'fgsm/single/pure/googlenet-eps-0.03/')
        attack_type = Adv_Tpye.FGSM
    if adv_type == Adv_Tpye.CW:
        data_path = os.path.join(path_prefix, 'cw/single/pure/googlenet-0.6-1000/')
        attack_type = Adv_Tpye.CW
    if adv_type == Adv_Tpye.JSMA:
        data_path = os.path.join(path_prefix, 'jsma/single/pure/googlenet-0.12/')
        attack_type = Adv_Tpye.JSMA
    if adv_type == Adv_Tpye.DEEPFOOL:
        data_path = os.path.join(path_prefix, 'deepfool/single/pure/googlenet-0.02-50/')
        attack_type = Adv_Tpye.DEEPFOOL
    if adv_type == Adv_Tpye.BB:
        data_path = os.path.join(path_prefix, 'bb/single/pure/googlenet-eps-0.3/')
        attack_type = Adv_Tpye.BB

    return data_path, attack_type


def detect_mnist(THRESHOLD_SCALE, RELAX_SCALE):
    # THRESHOLD_SCALE = 1.2
    # RELAX_SCALE = 0.2
    mutated_type = sys.argv[1]
    device = 'cuda:' + sys.argv[2]
    # mutated_type = 'ns'
    # device = 'cuda:2'
    logging.info(
        "OP:{},THRESHOLD_SCALE:{}, RELAX_SCALE:{},Mutated_Ration:{}".format(mutated_type, THRESHOLD_SCALE, RELAX_SCALE,
                                                                            0.05))
    if mutated_type == 'nai':
        #############
        # NAI, ration 0.05
        ############
        seed_model_name = 'MnistNet4'
        operator_type = OP_NAME[OpType.NAI]
        models_folder = '/home/dong/gitgub/SafeDNN/bgDNN/model-storage/mnist/mutaed-models/nai/5p/'
        # up_bound = (3.06 + 0.44) * 0.01  # 3p normal LCR
        up_bound = (3.88 + 0.53) * 0.01  # 5p normal LCR
    elif mutated_type == 'ns':
        #############
        # NS, ration 0.05
        ############
        seed_model_name = 'lenet'
        operator_type = OP_NAME[OpType.NS]
        models_folder = '/home/dong/gitgub/SafeDNN/bgDNN/model-storage/mnist/mutaed-models/ns/5e-2p/'
        # up_bound = (0.37 + 0.19) * 0.01  # 3p, normal LCR
        # up_bound = 0.004
        up_bound = (0.89 + 0.35) * 0.01  # 5p
    elif mutated_type == 'ws':
        #############
        # WS, ration 0.05
        ############
        seed_model_name = 'lenet'
        operator_type = OP_NAME[OpType.WS]
        models_folder = '/home/dong/gitgub/SafeDNN/bgDNN/model-storage/mnist/mutaed-models/ws/5e-2p/'
        # up_bound = (3.03+0.35)*0.01    # 3p, normal LCR
        # up_bound = 0.04
        up_bound = (3.83 + 0.42) * 0.01  # 5p, normal LCR

    elif mutated_type == 'gf':
        #############
        # GF, ration 0.03
        ############
        seed_model_name = 'MnistNet4'
        operator_type = OP_NAME[OpType.GF]
        models_folder = '/home/dong/gitgub/SafeDNN/bgDNN/model-storage/mnist/mutaed-models/gf/5e-2p/'
        # up_bound = (1.39+0.46)*0.01    # 3p, normal LCR
        # up_bound = 0.034
        up_bound = (2.49 + 0.59) * 0.01  # 5p, normal LCR

    else:
        raise Exception('Unknown mutated operator:{}'.format(mutated_type))

    threshold, sigma = get_threshold_relax(up_bound, THRESHOLD_SCALE, RELAX_SCALE)
    ##########
    # Normal
    #########
    data_path = '/home/dong/gitgub/SafeDNN/datasets/mnist/raw'
    data_loader = get_data_loader(data_path, is_adv_data=False, data_type=DATA_MNIST)
    main('normal', data_loader, models_folder, seed_model_name, threshold, sigma, operator_type, 'normal',
         device=device, data_type=DATA_MNIST)

    avg_acc_list = []
    avg_mutated_list = []
    #########
    # WL
    #########
    data_path = '/home/dong/gitgub/SafeDNN/datasets/mnist/raw'
    seed_model = torch.load('/home/dong/gitgub/SafeDNN/bgDNN/model-storage/mnist/hetero-base/MnistNet4.pkl')
    data_loader = get_wrong_label_data_loader(data_path, seed_model, DATA_MNIST)
    avg_accuracy, avg_mutated_used = main('adv', data_loader, models_folder, seed_model_name, threshold, sigma,
                                          operator_type, 'wl', device=device,
                                          data_type=DATA_MNIST)
    avg_acc_list.append(avg_accuracy)
    avg_mutated_list.append(avg_mutated_used)
    #######
    ##  adversarial samples
    #########
    for adv_type in [Adv_Tpye.FGSM, Adv_Tpye.JSMA, Adv_Tpye.CW, Adv_Tpye.BB, Adv_Tpye.DEEPFOOL]:
        data_path, attack_type = get_mnist_parameters(adv_type)
        data_loader = get_data_loader(data_path, is_adv_data=True, data_type=DATA_MNIST)
        avg_accuracy, avg_mutated_used = main('adv', data_loader, models_folder, seed_model_name, threshold, sigma,
                                              operator_type, attack_type,
                                              device=device, data_type=DATA_MNIST)
        avg_acc_list.append(avg_accuracy)
        avg_mutated_list.append(avg_mutated_used)

    print "adv-avg-accuracy:{:.4f}, adv-avg_mutated_used:{:.4f}".format(np.average(np.array(avg_acc_list)),
                                                                        np.average(np.array(avg_mutated_list)))


def detect_cifar10(THRESHOLD_SCALE, RELAX_SCALE):
    # mutated_type = sys.argv[1]
    # device = 'cuda:' + sys.argv[2]
    mutated_type = 'ns'
    device = 'cuda:0'

    logging.info("OP:{},THRESHOLD_SCALE:{}, RELAX_SCALE:{}".format(mutated_type, THRESHOLD_SCALE, RELAX_SCALE))
    seed_model_name = 'googlenet'
    if mutated_type == 'nai':
        #############
        # NAI, ration 0.005
        ############
        operator_type = OP_NAME[OpType.NAI]
        models_folder = '/home/dong/gitgub/SafeDNN/bgDNN/model-storage/cifar10/mutaed-models/nai/5e-3p/'
        up_bound = 0.01 * (7.28 + 1.12)  # normal LCR
    elif mutated_type == 'ns':
        #############
        # NS, ration 0.005
        ############
        operator_type = OP_NAME[OpType.NS]
        # models_folder = '/home/dong/gitgub/SafeDNN/bgDNN/model-storage/cifar10/mutaed-models/ns/5e-3p/'
        # up_bound = 0.01 * (0.94 + 0.4)  # normal LCR
        # 0.006 ration test
        models_folder = '/home/dong/gitgub/SafeDNN/bgDNN/model-storage/cifar10/mutaed-models/ns/7e-3p/'
        up_bound = 0.01 * (1.88 + 0.55)  # normal LCR
    elif mutated_type == 'ws':
        #############
        # WS, ration 0.005
        ############
        operator_type = OP_NAME[OpType.WS]
        models_folder = '/home/dong/gitgub/SafeDNN/bgDNN/model-storage/cifar10/mutaed-models/ws/5e-3p/'
        up_bound = 0.01 * (2.69 + 0.65)  # normal LCR
    elif mutated_type == 'gf':
        #############
        # GF, ration 0.005
        ############
        operator_type = OP_NAME[OpType.GF]
        models_folder = '/home/dong/gitgub/SafeDNN/bgDNN/model-storage/cifar10/mutaed-models/gf/5e-3p/'
        up_bound = 0.01 * (4.09 + 0.91)  # normal LCR
    else:
        raise Exception('Unknown mutated operator:{}'.format(mutated_type))

    threshold, sigma = get_threshold_relax(up_bound, THRESHOLD_SCALE, RELAX_SCALE)

    #########
    # Normal
    ########
    # data_path = '/home/dong/gitgub/SafeDNN/datasets/cifar10/raw'
    # data_loader = get_data_loader(data_path, is_adv_data=False, data_type=DATA_CIFAR10)
    # main('normal', data_loader, models_folder, seed_model_name, threshold, sigma, operator_type, 'normal',
    #      device=device, data_type=DATA_CIFAR10)
    #
    # avg_acc_list = []
    # avg_mutated_list = []
    # #########
    # # WL
    # #########
    # data_path = '/home/dong/gitgub/SafeDNN/datasets/cifar10/raw'
    # seed_model = torch.load('/home/dong/gitgub/SafeDNN/bgDNN/model-storage/cifar10/hetero-base/googlenet.pkl')
    # data_loader = get_wrong_label_data_loader(data_path, seed_model, DATA_CIFAR10)
    # avg_accuracy, avg_mutated_used = main('adv', data_loader, models_folder, seed_model_name, threshold, sigma,
    #                                       operator_type, 'wl', device=device,
    #                                       data_type=DATA_CIFAR10)
    # avg_acc_list.append(avg_accuracy)
    # avg_mutated_list.append(avg_mutated_used)

    #######
    ##  adversarial samples
    #########
    # just for nai to proceed
    # avg_acc_list = [0.5983,0.5370]
    # avg_mutated_list = [249.9474,259.0620]
    # 2018-08-23 01:38:58,965 - INFO - NAI,wl-Adv Accuracy:569/951,0.5983,,avg_mutated_used:249.9474
    # 2018-08-23 02:19:24,115 - INFO - NAI,fgsm-Adv Accuracy:537/1000,0.5370,,avg_mutated_used:259.0620
    # for adv_type in [Adv_Tpye.JSMA, Adv_Tpye.CW, Adv_Tpye.BB, Adv_Tpye.DEEPFOOL]:

    first_group = [Adv_Tpye.FGSM, Adv_Tpye.JSMA]
    second_group = [Adv_Tpye.CW, Adv_Tpye.BB]
    third_group = [Adv_Tpye.DEEPFOOL]
    for adv_type in third_group:
        data_path, attack_type = get_cifar10_parameters(adv_type)
        data_loader = get_data_loader(data_path, is_adv_data=True, data_type=DATA_CIFAR10)
        avg_accuracy, avg_mutated_used = main('adv', data_loader, models_folder, seed_model_name, threshold, sigma,
                                              operator_type, attack_type,
                                              device=device, data_type=DATA_CIFAR10)
        # avg_acc_list.append(avg_accuracy)
        # avg_mutated_list.append(avg_mutated_used)

    # print "adv-avg-accuracy:{:.4f}, adv-avg_mutated_used:{:.4f}".format(np.average(np.array(avg_acc_list)),
    #                                                           np.average(np.array(avg_mutated_list)))





if __name__ == '__main__':
    setup_logging()
    TEST_SMAPLES = 1000

    # THRESHOLD_SCALE = 1.5
    # RELAX_SCALE = 0.3
    # detect_cifar10(THRESHOLD_SCALE, RELAX_SCALE)

    # RELAX_SCALE = 0.1
    # for THRESHOLD_SCALE in [1.0,1.5,2]:
    #         detect_mnist(THRESHOLD_SCALE,RELAX_SCALE)
    # detect_mnist()


    # detect_type = sys.argv[1]
    # device = 'cuda:' + sys.argv[2]

    detect_type = 'adv'
    device = 'cuda:1'

    # RELAX_SCALE = 0.1
    # for THRESHOLD_SCALE in [3.0]:
    #         detect_cifar10(THRESHOLD_SCALE,RELAX_SCALE)
