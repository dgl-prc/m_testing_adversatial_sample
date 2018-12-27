import sys
sys.path.append('./cifar10models/')
import torch
import os
from torch.utils.data import DataLoader
from attack.craft_adversarial_img import *
from models import EnsembleModel
from util import logging_util
import logging
from util.data_manger import *

from attack.attack_util import Adv_Tpye as ADV

MAX_NUM_SAMPLES=1000

def fetch_models(models_folder, num_models, seed_name, device, start_no=1):
    '''
    :param models_folder:
    :param num_models: the number of models to be load
    :param start_no: the start serial number from which loading  "num_models" models
    :return: the top [num_models] models in models_folder
    '''
    No_models = range(start_no, start_no + num_models)
    target_models_name = [seed_name + '-m' + str(i) for i in No_models]
    target_models = []
    for model_name in target_models_name:
        target_models.append(torch.load(os.path.join(models_folder, seed_name, model_name + '.pkl')).to(device))
        # logging.warning("loading {} ...".format(model_name))
    return target_models


def hybrid_seed_vote(model_folder, model_name_list, target_samples, samples_folder):
    '''
    This is for 2.1 in RQ2-large-vote.md
    1. get 5 ensembel model
    2. run 5 times test
    3. # NOTE: model_name_list and target_samples should point-to-point align
    :param model_folder:
    :param model_name_list:
    :param target_samples:
    :param samples_folder:
    :return:
    '''
    model_list = []
    for model_name in model_name_list:
        model_list.append(torch.load(os.path.join(model_folder, model_name + '.pkl')))

    for i, targe_sample in enumerate(target_samples):
        adv_file_path = os.path.join(samples_folder, targe_sample)
        torch.manual_seed(5566)
        dataset = MyDataset(root=adv_file_path, transform=transforms.Compose([transforms.ToTensor(), normalize_mnist]),
                            img_mode='L')
        dataloader = DataLoader(dataset=dataset, shuffle=True)
        target_model = model_list[i]
        model_list.pop(i)
        vote_model = EnsembleModel(model_list)
        logging.info('>>>Start: Test attacked samples of {} '.format(targe_sample))
        samples_filter(vote_model, dataloader, '{} >> '.format(targe_sample), size=2500)
        logging.info('>>>End: Test attacked samples of {} '.format(targe_sample))

        model_list.insert(i, target_model)


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


def adv_samples_vote(device, num_models, seed_data, seed_model_name, path_prefix, adversarial_types,
                     adversarial_folders,
                     mutated_model_folers,
                     mutated_ration_list, model_start_num):
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

    tf = transforms.Compose([transforms.ToTensor(), normalization])
    for mutated_ration in mutated_ration_list:
        logging.info('>>>>>>>>>>>seed data:{},model mutated ration:{},mutated_models:{}<<<<<<<<<<'.format(seed_data,
                                                                                                          mutated_ration,
                                                                                                          mutated_model_folers))
        mutated_model_foler = os.path.join(mutated_model_folers, mutated_ration)
        mutated_models = fetch_models(mutated_model_foler, num_models, seed_name=seed_model_name, device=device,
                                      start_no=model_start_num)
        vote_model = EnsembleModel(mutated_models)
        for adversarial_type, adversarial_folder in zip(adversarial_types, adversarial_folders):
            adv_path = os.path.join(path_prefix, adversarial_type, 'single/pure', adversarial_folder)
            dataset = MyDataset(root=adv_path, transform=tf, img_mode=img_mode)
            dataloader = DataLoader(dataset=dataset)
            logging.info(
                '>>>Progress: {} mutated models for {}, samples path: {}'.format(len(mutated_models), adversarial_type,
                                                                                 adv_path))
            logging.info('>>Test-Details-start-{}>>>{}'.format(num_models, adversarial_type))
            samples_filter(vote_model, dataloader, '{} >> {} '.format(num_models, adversarial_type), size=-1,
                           show_progress=False, device=device, is_verbose=True)
            logging.info('>>Test-Details-end-{}>>>{}'.format(num_models, adversarial_type))


def adv_samples_vote_imagenet(device, num_models, seed_data, seed_model_name, path_prefix, adversarial_types,
                              adversarial_folders,
                              mutated_model_folers,
                              mutated_ration_list,
                              model_start_num):
    '''
    Since the model of imagenet is usually prodigious compared with the model on other dataset, we can not assemble
    all the mutated models once. So we
    :param device:
    :param num_models:
    :param seed_data:
    :param seed_model_name:
    :param path_prefix:
    :param adversarial_types:
    :param adversarial_folders:
    :param mutated_model_folers:
    :param mutated_ration_list:
    :return:
    '''
    normalization = normalize_imgNet
    img_channels = 3
    img_h = 224
    img_w = 224
    for mutated_ration in mutated_ration_list:
        logging.info('>>>>>>>>>>>seed data:{},model mutated ration:{}<<<<<<<<<<'.format(seed_data, mutated_ration))
        mutated_model_foler = os.path.join(mutated_model_folers, mutated_ration)
        mutated_models = fetch_models(mutated_model_foler, num_models, seed_name=seed_model_name, device=device,
                                      start_no=model_start_num)
        vote_model = EnsembleModel(mutated_models, num_out=1000)

        for adversarial_type, adversarial_folder in zip(adversarial_types, adversarial_folders):
            adv_path = os.path.join(path_prefix, adversarial_type, 'single/pure', adversarial_folder)
            [image_list, img_files, real_labels, adv_labels] = load_adversary_data(adv_path, normalization,
                                                                                   img_channels, img_h, img_w)
            # torch.manual_seed(random_seed)
            dataloader = DataLoader(TensorDataset(image_list, real_labels, adv_labels), batch_size=1, shuffle=False)
            logging.info(
                '>>>Progress: {} mutated models for {}, samples path: {}'.format(len(mutated_models), adversarial_type,
                                                                                 adv_path))
            logging.info('>>Test-Details-start-{}>>>{}'.format(num_models, adversarial_type))
            samples_filter(vote_model, dataloader, '{} >> {} '.format(num_models, adversarial_type), size=-1,
                           show_progress=False, device=device, is_verbose=True)
            logging.info('>>Test-Details-end-{}>>>{}'.format(num_models, adversarial_type))


def legitimate_samples_vote(device, num_models, seed_data, raw_data_path, seed_model_name, mutated_ration_list,
                            mutated_model_folers, model_start_num, use_train=True):
    if seed_data == 'mnist':
        data_type = DATA_MNIST
    elif seed_data == 'cifar10':
        data_type = DATA_CIFAR10

    data, channel = load_data_set(data_type, raw_data_path, train=use_train)

    seed_model_path = './model-storage/' + seed_data + '/hetero-base/' + seed_model_name + '.pkl'
    seed_model = torch.load(seed_model_path)
    correct_labeled = samples_filter(seed_model, DataLoader(dataset=data), return_type='normal', name='seed model',
                                     device=device)
    random_indcies = np.arange(len(correct_labeled))
    np.random.seed(random_seed)
    np.random.shuffle(random_indcies)
    random_indcies = random_indcies[:MAX_NUM_SAMPLES]
    data = datasetMutiIndx(data, np.array([idx for idx, _, _ in correct_labeled])[random_indcies])
    assert len(mutated_ration_list) > 0
    for mutated_ration in mutated_ration_list:
        logging.info(
            '>>>>>>>>>>>seed data:{},model mutated ration:{},mutated_models:{},random choose 2000 <<<<<<<<<<'.format(
                seed_data,
                mutated_ration,
                mutated_model_folers))
        mutated_model_foler = os.path.join(mutated_model_folers, mutated_ration)
        mutated_models = fetch_models(mutated_model_foler, num_models, seed_name=seed_model_name, device=device,
                                      start_no=model_start_num)
        vote_model = EnsembleModel(mutated_models)
        logging.info(
            '>>>Progress: {} mutated models for normal samples, samples path: {}'.format(len(mutated_models),
                                                                                         raw_data_path))
        logging.info('>>Test-Details-start-{}>>>{}'.format(num_models, seed_data))
        samples_filter(vote_model, DataLoader(dataset=data), 'legitimate {} >>'.format(seed_data), size=-1,
                       show_progress=False, device=device, is_verbose=True)
        logging.info('>>Test-Details-end-{}>>>{}'.format(num_models, seed_data))


def wrong_labeled_vote(device, num_models, seed_data, raw_data_path, mutated_ration_list, mutated_model_folers,
                       seed_model_name, model_start_num, use_train=True):
    if seed_data == 'mnist':
        data_type = DATA_MNIST
    elif seed_data == 'cifar10':
        data_type = DATA_CIFAR10

    seed_model_path = './model-storage/' + seed_data + '/hetero-base/' + seed_model_name + '.pkl'
    seed_model = torch.load(seed_model_path)
    dataset, channel = load_data_set(data_type, raw_data_path, train=use_train)
    dataloader = DataLoader(dataset=dataset)

    wrong_labeled = samples_filter(seed_model, dataloader, return_type='adv', name='seed model', device=device)
    data = datasetMutiIndx(dataset, [idx for idx, _, _ in wrong_labeled])
    wrong_labels = [wrong_label for idx, true_label, wrong_label in wrong_labeled]
    data = TensorDataset(data.tensors[0], data.tensors[1], torch.LongTensor(wrong_labels))

    for mutated_ration in mutated_ration_list:
        logging.info('>>>>>>>>>>>seed data:{},model mutated ration:{},mutated_models:{}<<<<<<<<<<'.format(seed_data,
                                                                                                          mutated_ration,
                                                                                                          mutated_model_folers))
        mutated_model_foler = os.path.join(mutated_model_folers, mutated_ration)
        mutated_models = fetch_models(mutated_model_foler, num_models, seed_name=seed_model_name, device=device,
                                      start_no=model_start_num)
        vote_model = EnsembleModel(mutated_models)
        logging.info(
            '>>>Progress: {} mutated models for normal samples, '.format(len(mutated_models)))
        logging.info('>>Test-Details-start-{}>>> wrong labeled of {}'.format(num_models, seed_data))
        samples_filter(vote_model, DataLoader(dataset=data), 'legitimate {} >>'.format(seed_data), size=-1,
                       show_progress=False, device=device, is_verbose=True)
        logging.info('>>Test-Details-end-{}>>> wrong labeled of {}'.format(num_models, seed_data))


def run(sample_type='adv'):
    if sys.argv[1] == 'help':
        print('[1] daat_type: 0(mnist),1(cifar10),2(ilsvrc12)\n' +
              '[2] cuda_no\n' +
              '[3] vote_type: [adv], test on advesarial data; [normal], test on normal data; [wl],test on wrong labeled data\n' +
              '[4] use_train: use training data or test data, just for normal and wrong labeled voting\n' +
              '[5] model_start_no: from the which the mutated model to be loaded\n' +
              '[6] num_models: the number of mutated models to be involved\n' +
              '[7] mutated_type: [nai],[ws],[gf],[ns]\n' +
              '[8] scal_mutated_ration: 1:1e-1,2:1e-2, 3:1e-3')
        exit(0)

    daat_type = int(sys.argv[1])
    cuda_no = sys.argv[2]
    if len(sys.argv) > 3:
        vote_type = sys.argv[3]
        use_train = True if int(sys.argv[4]) else False
        model_start_no = int(sys.argv[5])
        num_models = int(sys.argv[6])
        mutated_type = sys.argv[7]
        scal_mutated_ration = int(sys.argv[8])

    #######
    # for debug
    # ########
    # daat_type = 1
    # cuda_no = '2'
    # vote_type = 'normal'
    # use_train = False
    # model_start_no = 1
    # num_models = 25 # we should not
    # mutated_type = 'ns'
    # scal_mutated_ration = 3

    if cuda_no == '-1':
        device = 'cpu'
    else:
        device = 'cuda:' + cuda_no if torch.cuda.is_available() else 'cpu'

    if daat_type == 0:
        seed_data = 'mnist'
        if mutated_type in ['nai','gf']:
            seed_model_name = 'MnistNet4'
        else:
            seed_model_name = 'lenet'
        adversarial_types = [ADV.FGSM, ADV.JSMA, ADV.CW, ADV.BB, ADV.DEEPFOOL]
        adversarial_folders = [MNIST_ADV_FOLDERs[adv] for adv in adversarial_types]

        if mutated_type == 'nai':
            mutated_ration_list = ['1p', '3p', '5p']
        else:
            mutated_ration_list = ['1e-2p', '3e-2p', '5e-2p']
    elif daat_type == 1:
        seed_data = 'cifar10'
        # seed_model_name = 'lenet'
        seed_model_name = 'googlenet'
        adversarial_types = [ADV.FGSM, ADV.JSMA, ADV.CW, ADV.BB, ADV.DEEPFOOL]
        adversarial_folders = [CIFAR10_ADV_FOLDERs_GooGlNet[adv] for adv in adversarial_types]

        if mutated_type == 'nai' and scal_mutated_ration == 2:
            mutated_ration_list = ['1p', '3p', '5p']
        elif scal_mutated_ration == 2:
            mutated_ration_list = ['1e-2p', '3e-2p', '5e-2p']
        elif scal_mutated_ration == 3:
            # mutated_ration_list = ['1e-3p', '3e-3p', '5e-3p']
            mutated_ration_list = ['7e-3p']
        # elif scal_mutated_ration == 7:
        #     mutated_ration_list = ['7e-3p']
        else:
            raise Exception('Unknown mutated ration magnitude!')

        # if mutated_type == "ns":
        #     mutated_ration_list=mutated_ration_list[1:]

    elif daat_type == DATA_IMAGENET:
        seed_data = 'ilsvrc12'
        seed_model_name = 'densenet121'
        # adversarial_folders = ['densenet121-eta0.035'(fgsm),'densenet121-0.6-1000'(c&w)]
        adversarial_folders = ['densenet121-0.6-1000']
        mutated_ration_list = ['1e-3p', '3e-3p', '5e-3p']
        adversarial_types = ['fgsm', 'jsma', 'bb', 'cw', 'deepfool']

    else:
        raise Exception('Unknown Data type:' + daat_type)

    if sample_type == 'adv':
        path_prefix = '../datasets/' + seed_data + '/adversarial/'
    else:
        path_prefix = '../datasets/' + seed_data + '/adversarial-pure/'

    mutated_model_folers = os.path.join('model-storage', seed_data, 'mutaed-models', mutated_type)

    if vote_type == 'adv':
        if daat_type != DATA_IMAGENET:
            adv_samples_vote(device=device,
                             num_models=num_models,
                             seed_data=seed_data,
                             seed_model_name=seed_model_name,
                             path_prefix=path_prefix,
                             adversarial_types=adversarial_types,
                             adversarial_folders=adversarial_folders,
                             mutated_model_folers=mutated_model_folers,
                             mutated_ration_list=mutated_ration_list,
                             model_start_num=model_start_no)
        else:
            adv_samples_vote_imagenet(device=device,
                                      num_models=num_models,
                                      seed_data=seed_data,
                                      seed_model_name=seed_model_name,
                                      path_prefix=path_prefix,
                                      adversarial_types=adversarial_types,
                                      adversarial_folders=adversarial_folders,
                                      mutated_model_folers=mutated_model_folers,
                                      mutated_ration_list=mutated_ration_list,
                                      model_start_num=model_start_no)
    elif vote_type == 'normal':
        raw_data = '../datasets/' + seed_data + '/raw'
        legitimate_samples_vote(device=device,
                                num_models=num_models, seed_data=seed_data, seed_model_name=seed_model_name,
                                raw_data_path=raw_data, use_train=use_train,
                                mutated_ration_list=mutated_ration_list,
                                mutated_model_folers=mutated_model_folers,
                                model_start_num=model_start_no)
    elif vote_type == 'wl':
        raw_data = '../datasets/' + seed_data + '/raw'
        wrong_labeled_vote(device=device,
                           num_models=num_models,
                           seed_data=seed_data,
                           raw_data_path=raw_data,
                           seed_model_name=seed_model_name,
                           mutated_ration_list=mutated_ration_list,
                           mutated_model_folers=mutated_model_folers,
                           use_train=use_train,
                           model_start_num=model_start_no)
    else:
        raise Exception('Unknown test type:{}'.format(vote_type))


if __name__ == '__main__':
    logging_util.setup_logging()
    run(sample_type='adv-pure') # for cifar10
    # run(sample_type='adv')
    torch.cuda.empty_cache()
    import time
    time.sleep(3)
