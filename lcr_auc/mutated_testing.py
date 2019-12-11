import sys
sys.path.append("../")
import torch
import os
from torch.utils.data import DataLoader
from attacks.craft_adversarial_img import *
from models.ensemble_model import EnsembleModel
from utils import logging_util
import logging
from utils.data_manger import *
from attacks.attack_util import Adv_Tpye as ADV
import argparse
import re
from models.googlenet import *
from models.lenet import *
import copy
MAX_NUM_SAMPLES = 1000

from utils.model_manager import fetch_models

def parseAdvType(advPath):
    adv_types = set(["bb", "cw", "deepfool", "fgsm","jsma"])
    eles = set(ele.lower() for ele in advPath.split("/") if ele.strip() != "")
    adv_type = adv_types & eles
    assert len(adv_type) == 1
    return adv_type.pop()


def step_mutated_vote(models_folder, model_name_list, target_samples, samples_folder, useAttackSeed=True,
                      dataloader=None):
    '''
    step=10,up to 100
    :param model_folder:
    :param model_name_list:
    :param target_samples:
    :param samples_folder:
    :return:
    '''

    for i, targe_sample in enumerate(target_samples):

        # i += 3 just for mnist4, mnist5

        if not dataloader:
            adv_file_path = os.path.join(samples_folder, targe_sample)
            torch.manual_seed(random_seed)
            dataset = MyDataset(root=adv_file_path,
                                transform=transforms.Compose([transforms.ToTensor(), normalize_mnist]))
            dataloader = DataLoader(dataset=dataset, shuffle=True)

        print('>>>Progress: Test attacked samples of {} '.format(targe_sample))
        logging.info('>>>Progress: Test attacked samples of {} '.format(targe_sample))

        # for num_models in range(10, 110, 10):
        for num_models in [100]:
            # to do
            # 1. for each seed model, select the top [num_models] models
            # 2. ensemble 5*[num_models] models

            num_seed_models = len(model_name_list)
            models_list = []
            for i2, seed_name in enumerate(model_name_list):
                if useAttackSeed:
                    models_list.extend(fetch_models(models_folder, num_models, seed_name))
                elif i != i2:
                    models_list.extend(fetch_models(models_folder, num_models, seed_name))

            logging.info('>>>Progress: {} models for {}'.format(len(models_list), targe_sample))
            print('>>>Progress: {} models for {}'.format(len(models_list), targe_sample))

            vote_model = EnsembleModel(models_list)
            logging.info('>>Test-Details-start-{}>>>{}'.format(num_seed_models * num_models, targe_sample))
            samples_filter(vote_model, dataloader, '{} >> {} '.format(len(models_list), targe_sample), size=-1,
                           show_progress=True)
            logging.info('>>Test-Details-end-{}>>>{}'.format(num_seed_models * num_models, targe_sample))


def inspect_adv_lale(real_labels, adv_labels, img_files):
    for real_label, adv_label, file_name in zip(real_labels, adv_labels, img_files):
        print ('real:{},adv:{},files:{}'.format(real_label, adv_label, file_name))


def batch_adv_tetsing(device, num_models, seed_data,
                      adv_folder,
                      mutated_models_path, model_start_num,seed_model):
    if seed_data == 'mnist':
        normalization = normalize_mnist
        img_mode = 'L'  # 8-bit pixels, black and white
    elif seed_data == 'cifar10':
        normalization = normalize_cifar10
        img_mode = None
    elif seed_data == 'ilsvrc12':
        normalization = normalize_imgNet
        img_mode = None
    else:
        raise Exception('Unknown data soure!')

    adv_type = parseAdvType(adv_folder)

    tf = transforms.Compose([transforms.ToTensor(), normalization])

    logging.info('>>>>>>>>>>>seed data:{},mutated_models:{}<<<<<<<<<<'.format(seed_data, mutated_models_path))
    mutated_models = fetch_models(mutated_models_path, num_models, device=device,
                                  start_no=model_start_num,seed_model=seed_model)
    ensemble_model = EnsembleModel(mutated_models)

    dataset = MyDataset(root=adv_folder, transform=tf, img_mode=img_mode)
    dataloader = DataLoader(dataset=dataset)
    logging.info(
        '>>>Progress: {} mutated models for {}, samples {}'.format(len(mutated_models), adv_type,
                                                                   adv_folder))
    logging.info('>>Test-Details-start-{}>>>{}>>>{}'.format(num_models, adv_type,seed_data))
    samples_filter(ensemble_model, dataloader, '{} >> {} '.format(num_models, adv_type), size=-1,
                   show_progress=False, device=device, is_verbose=True)
    logging.info('>>Test-Details-end-{}>>>{}>>>{}'.format(num_models, adv_type,seed_data))

def batch_legitimate_testing(device, num_models, seed_data, raw_data_path, seed_model,
                             mutated_models_path, model_start_num, use_train=True):
    if seed_data == 'mnist':
        data_type = DATA_MNIST
    elif seed_data == 'cifar10':
        data_type = DATA_CIFAR10
    data = load_natural_data(True,data_type, raw_data_path, use_train=use_train, seed_model=seed_model, device=device, MAX_NUM_SAMPLES=MAX_NUM_SAMPLES)

    logging.info(
        '>>>>>>>>>>>For {}({}) randomly choose {} with randomseed {}. mutated_models:{}<<<<<<<<<<'.format(
            seed_data,"Training" if use_train else "Testing",MAX_NUM_SAMPLES,random_seed,mutated_models_path))

    mutated_models = fetch_models(mutated_models_path, num_models,device=device,
                                  start_no=model_start_num,seed_model=seed_model)

    ensemble_model = EnsembleModel(mutated_models)
    logging.info(
        '>>>Progress: {} mutated models for normal samples, samples path: {}'.format(len(mutated_models),
                                                                                     raw_data_path))
    logging.info('>>Test-Details-start-{}>>>{}'.format(num_models, seed_data))
    samples_filter(ensemble_model, DataLoader(dataset=data), 'legitimate {} >>'.format(seed_data), size=-1,
                   show_progress=False, device=device, is_verbose=True)
    logging.info('>>Test-Details-end-{}>>>{}'.format(num_models, seed_data))


def batch_wl_testing(device, num_models, seed_data, raw_data_path, seed_model, mutated_models_path,
                       model_start_num, use_train=True):

    if seed_data == 'mnist':
        data_type = DATA_MNIST
    elif seed_data == 'cifar10':
        data_type = DATA_CIFAR10


    dataset, channel = load_data_set(data_type, raw_data_path, train=use_train)
    dataloader = DataLoader(dataset=dataset)

    wrong_labeled = samples_filter(seed_model, dataloader, return_type='adv', name='seed model', device=device,show_accuracy=False)
    data = datasetMutiIndx(dataset, [idx for idx, _, _ in wrong_labeled])
    wrong_labels = [wrong_label for idx, true_label, wrong_label in wrong_labeled]
    data = TensorDataset(data.tensors[0], data.tensors[1], torch.LongTensor(wrong_labels))

    logging.info(
        '>>>>>>>>>>>For {}({}),mutated Models Path: {} <<<<<<<<<<'.format(
            seed_data,"Training" if use_train else "Testing",mutated_models_path))

    mutated_models = fetch_models(mutated_models_path, num_models, device=device,
                                  start_no=model_start_num,seed_model=seed_model)

    ensemble_model = EnsembleModel(mutated_models)
    logging.info(
        '>>>Progress: {} mutated models for wl samples, '.format(len(mutated_models)))
    logging.info('>>Test-Details-start-{}>>> wrong labeled of {}'.format(num_models, seed_data))
    samples_filter(ensemble_model, DataLoader(dataset=data), 'legitimate {} >>'.format(seed_data), size=-1,
                   show_progress=False, device=device, is_verbose=True)
    logging.info('>>Test-Details-end-{}>>> wrong labeled of {}'.format(num_models, seed_data))



def run():
    parser = argparse.ArgumentParser(description="Process of Label Change Rate Statistics")
    parser.add_argument("--dataType", type=int,
                        help="The data set that the given model is tailored to. Three types are available: mnist,0; "
                             "cifar10, 1", default=0, required=True)

    parser.add_argument("--device", type=int,
                        help="The index of GPU used. If -1 is assigned,then only cpu is available",
                        required=True)

    parser.add_argument("--testType", type=str,
                        help="Tree types are available: [adv], advesarial data; [normal], test on normal data; [wl],test on wrong labeled data",
                        required=True)

    parser.add_argument("--useTrainData", type=str,
                        help="Use training data (True) or test data (False). This is just for normal and wrong labeled testing",
                        default="False",# False
                        required=False)

    parser.add_argument("--startNo", type=int,
                        help="The start No. of mutated model be loaded. This parameter works with the following parameter "
                             "\"batchModelSize\". This program will load [batchModelSize] mutated models from the [startNo]th"
                             " mutated model. ",
                        required=True)

    parser.add_argument("--batchModelSize", type=int,
                        help="The number of mutated models to be loaded in this test.",
                        required=True)

    parser.add_argument("--mutatedModelsPath", type=str,
                        help="The path of mutated models",
                        required=True)

    parser.add_argument("--testSamplesPath", type=str,
                        help="The path of mutated models",
                        required=True)

    parser.add_argument("--seedModelName", type=str,
                        help="The model's name,e.g, googlenet, lenet",
                        default="lenet")
    parser.add_argument("--seedModelPath", type=str,
                        help="This parameter is just for normal testing and wl testing",
                        default="lenet",
                        required=False)


    args = parser.parse_args()
    args.useTrainData = False if args.useTrainData.lower()=="false" else True

    if args.dataType == DATA_CIFAR10:
        data_name = 'cifar10'
    elif args.dataType == DATA_MNIST:
        data_name = 'mnist'

    if not torch.cuda.is_available():
        assert args.device == -1, "cuda is not available"
    device = "cuda:" + str(args.device) if args.device >= 0 else "cpu"

    if args.seedModelName == "googlenet":
        seed_model = GoogLeNet()
    elif args.seedModelName == "lenet":
        seed_model = MnistNet4()

    if args.testType == 'adv':
        batch_adv_tetsing(device=device,
                          num_models=args.batchModelSize,
                          model_start_num=args.startNo,
                          seed_data=data_name,
                          adv_folder=args.testSamplesPath,
                          mutated_models_path=args.mutatedModelsPath,
                          seed_model=seed_model)

    elif args.testType == 'normal':
        seed_model.load_state_dict(torch.load(args.seedModelPath))
        batch_legitimate_testing(device=device,num_models=args.batchModelSize, seed_data=data_name, raw_data_path=args.testSamplesPath,
                                 seed_model = seed_model,mutated_models_path=args.mutatedModelsPath,
                                 model_start_num=args.startNo, use_train=args.useTrainData)

    elif args.testType == 'wl':
        seed_model.load_state_dict(torch.load(args.seedModelPath))
        batch_wl_testing(device=device,num_models=args.batchModelSize, seed_data=data_name, raw_data_path=args.testSamplesPath,
                                 seed_model = seed_model,mutated_models_path=args.mutatedModelsPath,
                                 model_start_num=args.startNo, use_train=args.useTrainData)
    else:
        raise Exception('Unknown test type:{}'.format(args.testType))

if __name__ == '__main__':
    logging_util.setup_logging()
    run()
    # models_folder = "../build-in-resource/mutated_models/mnist/lenet/ns/1e-2p/"
    # start_no = 1
    # num_models = 10
    # device = 1
    # seed_model = MnistNet4()
    # a=fetch_models(models_folder, num_models, device, seed_model, start_no=start_no)
    # print(len(a))