from __future__ import print_function
import sys

sys.path.append('../')
sys.path.append('../cifar10models/')
from fgsm import FGSM
from carlinl2 import CarliniL2
from util.logging_util import *
from jsma import *
from deepfool import DeepFool
from util.data_manger import DATA_CIFAR10
from util.data_manger import load_cifar10, load_data_set
from cifar10models import lenet

'''
This component has three levels

The genereate_?_samples is the kernel generator which calls a specific adversary to generate the adversarial samples by 
batch or by single sample.

do_craft_? is designed to do attack for multil-models.

The two levels are not bound with certain data

The level, like craft_mnist,craft_cifar10,craft_imagenet,  is the application level,which is bound with certain level.

'''


def genereate_fgsm_samples_for_large_data(model_path, source_data, save_path, eps, is_save=False, is_exclude_wr=True,
                                          data_type=DATA_MNIST):
    '''

    :param model_path:
    :param source_data:
    :param save_path:
    :param eps:
    :param is_save:
    :param is_exclude_wr:  exclude the wrong labeled or not
    :return:
    '''
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    raw_test_data, channel = load_data_set(data_type, source_data)
    batch_size = 1000

    model = torchvision.models.densenet121(pretrained=True)
    random_samples = np.arange(10000)
    np.random.seed(random_seed)
    np.random.shuffle(random_samples)

    start = 0
    for batch_no in range(10):
        dataset = datasetMutiIndx(raw_test_data, random_samples[start:start + batch_size])
        test_data = exclude_wrong_labeled(model, dataset, device)
        test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=1)
        fgsm = FGSM(model, eps=eps, device=device)
        adv_samples, y = fgsm.do_craft_batch(test_loader)
        adv_loader = DataLoader(TensorDataset(adv_samples, y), batch_size=1, shuffle=False)
        succeed_adv_samples = samples_filter(model, adv_loader, "Eps={}".format(eps))
        num_adv_samples = len(succeed_adv_samples)
        print('batch:{},successful samples:{}'.format(batch_no, num_adv_samples))
        if is_save:
            save_imgs(succeed_adv_samples, TensorDataset(adv_samples, y), save_path, 'fgsm', channel, batch_size,
                      batch_no)
        start += batch_size
    print('Done!')


def genereate_jsma_samples_for_large_data(model_path, source_data, save_path, is_save=False, max_distortion=0.12,
                                          dim_features=784,
                                          num_out=10, data_type=DATA_MNIST, img_shape={'C': 3, 'H': 32, 'W': 32}):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(random_seed)
    # train_data, _ =  load_data_set(data_type=data_type,source_data=source_data,train=True)
    # complete_data = ConcatDataset([test_data,train_data])

    test_data, channels = load_data_set(data_type=data_type, source_data=source_data)
    model = torchvision.models.densenet121(pretrained=True)
    random_samples = np.arange(10000)
    np.random.seed(random_seed)
    np.random.shuffle(random_samples)
    test_data = datasetMutiIndx(test_data, random_samples[:100])

    complete_data = exclude_wrong_labeled(model, test_data, device)
    test_data_laoder = DataLoader(dataset=complete_data, batch_size=1, shuffle=True)

    jsma = JSMA(model, max_distortion, dim_features, num_out=num_out, theta=1, optimal=True, verbose=False,
                device=device, shape=img_shape)
    success = 0
    progress = 0
    all_lables = range(num_out)
    for data, label in test_data_laoder:
        data, label = data.to(device), label.to(device)
        target_label = jsma.uniform_smaple(label, all_lables)
        adv_sample, normal_predit, adv_label = jsma.do_craft(data, target_label)
        if adv_label == target_label:
            success += 1
            if is_save:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_imgs_tensor([adv_sample.to('cpu')], [label], [adv_label], save_path, 'jsma', no_batch=success,
                                 channels=channels)

        progress += 1
        if success > 3000:
            break
    print(success * 1. / progress)


def genereate_cw_samples_for_large_data(model_path, source_data, save_path, is_save=False, c=0.8, iter=10000,
                                        batch_size=1,
                                        data_type=DATA_MNIST):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    raw_test_data, channels = load_data_set(data_type, source_data)
    target_model = torchvision.models.densenet121(pretrained=True)
    l2Attack = CarliniL2(target_model=target_model, max_iter=iter, c=c, k=0, device=device, targeted=False)

    random_samples = np.arange(10000)
    np.random.seed(random_seed)
    np.random.shuffle(random_samples)

    step_size = 100
    start = 0
    for batch_no in range(10):
        dataset = datasetMutiIndx(raw_test_data, random_samples[start:start+step_size])
        test_data = exclude_wrong_labeled(target_model, dataset, device=device)
        test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0)
        for i, data_pair in enumerate(test_loader):
            i += 1
            data, real_label = data_pair
            data, real_label = data.to(device), real_label.to(device)
            scores = target_model(data)
            normal_preidct = torch.argmax(scores, dim=1, keepdim=True)
            adv_samples = l2Attack.do_craft(data, normal_preidct)
            success_samples, normal_labels, adv_label = l2Attack.check_adversarial_samples(l2Attack.target_model,
                                                                                           adv_samples, normal_preidct)
            if is_save:
                save_imgs_tensor(success_samples, normal_labels, adv_label, save_path, 'cw', no_batch=batch_no*step_size+i,
                                 batch_size=batch_size, channels=channels)
            logging.info('batch:{}'.format(i))
        start +=step_size



def genereate_deepfool_samples_for_large_data(model_path, source_data_path, save_path=None, overshoot=0.02, num_out=10,
                                              max_iter=50,
                                              data_type=DATA_MNIST):
    '''
    Single data only!Do not Support batch
    :return:
    '''
    #######
    # this part of code is identical with the genereate_jsma_samples
    ######
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    raw_test_data, channels = load_data_set(data_type=data_type, source_data=source_data_path)

    model = torchvision.models.densenet121(pretrained=True)
    random_samples = np.arange(10000)
    np.random.seed(random_seed)
    np.random.shuffle(random_samples)

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    start = 4600
    batch_size = 100
    count = 881
    for batch_no in range(9,24):
        test_data = datasetMutiIndx(raw_test_data, random_samples[start:start + batch_size])
        complete_data = exclude_wrong_labeled(model, test_data, device)
        test_data_laoder = DataLoader(dataset=complete_data, batch_size=1, shuffle=True)
        deepfool = DeepFool(target_model=model, num_classes=num_out, overshoot=overshoot, max_iter=max_iter,
                            device=device)
        for data, label in test_data_laoder:
            data = data.squeeze(0)
            data, label = data.to(device), label.to(device)
            adv_img, normal_label, adv_label = deepfool.do_craft(data)
            assert adv_label != normal_label
            assert label.item() == normal_label
            if is_save:
                save_imgs_tensor([adv_img.to('cpu')], [label], [adv_label], save_path, 'deepfool', no_batch=batch_no,
                                 batch_size=batch_size,
                                 channels=channels,adv_count=count)
            logging.info('{}th finished'.format(count))
            count += 1
        start += batch_size


def genereate_fgsm_samples(model_path, source_data, save_path, eps, is_save=False, is_exclude_wr=True,
                           data_type=DATA_MNIST):
    '''

    :param model_path:
    :param source_data:
    :param save_path:
    :param eps:
    :param is_save:
    :param is_exclude_wr:  exclude the wrong labeled or not
    :return:
    '''
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    test_data, channel = load_data_set(data_type, source_data,train=False)
    if data_type == DATA_IMAGENET:
        model = torchvision.models.densenet121(pretrained=True)
        random_samples = np.arange(10000)
        np.random.seed(random_seed)
        np.random.shuffle(random_samples)
        dataset = datasetMutiIndx(test_data, random_samples[:1000])
        test_data = exclude_wrong_labeled(model, dataset, device)
        test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=1)
    else:
        model = torch.load(model_path)
        test_data = exclude_wrong_labeled(model, test_data, device)
        test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    fgsm = FGSM(model, eps=eps, device=device)
    adv_samples, y = fgsm.do_craft_batch(test_loader)
    adv_loader = DataLoader(TensorDataset(adv_samples, y), batch_size=1, shuffle=False)
    succeed_adv_samples = samples_filter(model, adv_loader, "Eps={}".format(eps),device=device)
    num_adv_samples = len(succeed_adv_samples)
    print('successful samples', num_adv_samples)
    if is_save:
        save_imgs(succeed_adv_samples, TensorDataset(adv_samples, y), save_path, 'fgsm', channel)
    print('Done!')



def genereate_cw_samples(model_path, source_data, save_path, is_save=False, c=0.8, iter=10000, batch_size=1,
                         data_type=DATA_MNIST,device='cpu'):
    # at present, only  cuda0 suopport
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data, channels = load_data_set(data_type, source_data)
    if data_type == DATA_IMAGENET:
        target_model = torchvision.models.densenet121(pretrained=True)
        random_samples = np.arange(10000)
        np.random.seed(random_seed)
        np.random.shuffle(random_samples)
        dataset = datasetMutiIndx(test_data, random_samples)
        test_data = exclude_wrong_labeled(target_model, dataset, device=device)
        test_loader = DataLoader(dataset=test_data, batch_size=128, num_workers=3)
    else:
        target_model = torch.load(model_path)
        test_data = exclude_wrong_labeled(target_model, test_data,device=device)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    l2Attack = CarliniL2(target_model=target_model, max_iter=iter, c=c, k=0, device=device, targeted=False)

    for i, data_pair in enumerate(test_loader):
        i += 1
        data, real_label = data_pair
        data, real_label = data.to(device), real_label.to(device)
        scores = target_model(data)
        normal_preidct = torch.argmax(scores, dim=1, keepdim=True)
        adv_samples = l2Attack.do_craft(data, normal_preidct)
        success_samples, normal_labels, adv_label = l2Attack.check_adversarial_samples(l2Attack.target_model,
                                                                                       adv_samples, normal_preidct)
        if is_save:
            save_imgs_tensor(success_samples, normal_labels, adv_label, save_path, 'cw', no_batch=i,
                             batch_size=batch_size, channels=channels)
            # save_imgs_tensor(success_samples, normal_labels, adv_label, save_path, 'cw-org', no_batch=i,
            #                  batch_size=batch_size, channels=channels)
        logging.info('batch:{}'.format(i))
        if i > 1500:
            break


def genereate_jsma_samples(model_path, source_data, save_path, is_save=False, max_distortion=0.12, dim_features=784,
                           num_out=10, data_type=DATA_MNIST, img_shape={'C': 3, 'H': 32, 'W': 32},device = 'cpu'):
    torch.manual_seed(random_seed)
    # train_data, _ =  load_data_set(data_type=data_type,source_data=source_data,train=True)
    # complete_data = ConcatDataset([test_data,train_data])

    test_data, channels = load_data_set(data_type=data_type, source_data=source_data)
    if data_type == DATA_IMAGENET:
        model = torchvision.models.densenet121(pretrained=True)
        random_samples = np.arange(10000)
        np.random.seed(random_seed)
        np.random.shuffle(random_samples)
        test_data = datasetMutiIndx(test_data, random_samples)
    else:
        model = torch.load(model_path)
    complete_data = exclude_wrong_labeled(model, test_data, device)
    test_data_laoder = DataLoader(dataset=complete_data, batch_size=1, shuffle=True)

    jsma = JSMA(model, max_distortion, dim_features, num_out=num_out, theta=1, optimal=True, verbose=False,
                device=device, shape=img_shape)
    success = 0
    progress = 0
    all_lables = range(num_out)
    for data, label in test_data_laoder:
        data, label = data.to(device), label.to(device)
        target_label = jsma.uniform_smaple(label, all_lables)
        adv_sample, normal_predit, adv_label = jsma.do_craft(data, target_label)
        if adv_label == target_label:
            success += 1
            if is_save:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_imgs_tensor([adv_sample.to('cpu')], [label], [adv_label], save_path, 'jsma', no_batch=success,
                                 channels=channels)

        progress += 1
        sys.stdout.write(
            '\rprogress:{:.2f}%,success:{:.2f}%'.format(100. * progress / len(test_data_laoder),
                                                        100. * success / progress))
        sys.stdout.flush()

        if success > 5000:
            break

    print(success * 1. / progress)


def genereate_deepfool_samples(model_path, source_data_path, save_path=None, overshoot=0.02, num_out=10, max_iter=50,
                               data_type=DATA_MNIST):
    '''
    Single data only!Do not Support batch
    :return:
    '''
    #######
    # this part of code is identical with the genereate_jsma_samples
    ######
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    test_data, channels = load_data_set(data_type=data_type, source_data=source_data_path)
    if data_type == DATA_IMAGENET:
        model = torchvision.models.densenet121(pretrained=True)
        random_samples = np.arange(10000)
        np.random.seed(random_seed)
        np.random.shuffle(random_samples)
        test_data = datasetMutiIndx(test_data, random_samples)
    else:
        model = torch.load(model_path)

    complete_data = exclude_wrong_labeled(model, test_data, device)
    test_data_laoder = DataLoader(dataset=complete_data, batch_size=1, shuffle=True)

    deepfool = DeepFool(target_model=model, num_classes=num_out, overshoot=overshoot, max_iter=max_iter, device=device)
    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    count = 1
    for data, label in test_data_laoder:
        data = data.squeeze(0)
        data, label = data.to(device), label.to(device)
        adv_img, normal_label, adv_label = deepfool.do_craft(data)
        assert adv_label != normal_label
        assert label.item() == normal_label
        if is_save:
            save_imgs_tensor([adv_img.to('cpu')], [label], [adv_label], save_path, 'deepfool', no_batch=count,
                             channels=channels)
        logging.info('{}th success!'.format(count))
        if count > 4000:
            break
        count += 1


def model_cross_valid(prefix_model_path, prefix_save_path, model_list, adv_folder_list):
    '''
    :return:
    '''
    # Here we use a random seed to make sure that the result repeatable when run this function again.
    # It mainly control the shuffle process in DataLoader
    # torch.manual_seed(5566)
    for model_name_A, save_folder in zip(model_list, adv_folder_list):
        for model_name_B in model_list:
            model = torch.load(os.path.join(prefix_model_path, model_name_B))
            adv_file_path = os.path.join(prefix_save_path, save_folder)
            [image_list, image_files, real_labels, predicted_labels] = load_adversary_data(adv_file_path,
                                                                                           normalize_mnist)
            torch.manual_seed(5566)
            dataloader = DataLoader(TensorDataset(image_list, real_labels), batch_size=1, shuffle=True)
            samples_filter(model, dataloader, '{} >> {}'.format("adv_" + model_name_A, model_name_B), size=2500)


def do_craft_fgsm(prefix_model_path, prefix_save_path, model_list, eps=0.5, is_save=False,
                  data_type=DATA_MNIST, source_data=None):
    for model_name in model_list:
        save_folder = model_name.split('.')[0]+'-eps-'+str(eps)
        model_path = os.path.join(prefix_model_path, model_name)
        save_path = os.path.join(prefix_save_path, save_folder)
        print('>>>>Generate {} >>>>>>>>>>'.format(model_name))
        if data_type == DATA_IMAGENET:
            genereate_fgsm_samples_for_large_data(model_path, source_data, save_path, eps, is_save=is_save,
                                                  data_type=data_type)
        else:
            genereate_fgsm_samples(model_path, source_data, save_path, eps, is_save=is_save, data_type=data_type)


def do_craft_cw(prefix_model_path, prefix_save_path, model_list,is_save=False, c=0.8, iter=10000,
                batch_size=1, source_data=None, data_type=None,device='cpu'):
    for model_name in model_list:
        save_folder = model_name.split('.')[0]+'-'+str(c)+'-'+str(iter)
        model_path = os.path.join(prefix_model_path, model_name)
        save_path = os.path.join(prefix_save_path, save_folder)
        print('>>>>Generate {} >>>>>>>>>>'.format(model_name))
        if data_type == DATA_IMAGENET:
            genereate_cw_samples_for_large_data(model_path, source_data, save_path, is_save=is_save, c=c, iter=iter,
                                                batch_size=batch_size,
                                                data_type=data_type)
        else:
            genereate_cw_samples(model_path, source_data, save_path, is_save=is_save, c=c, iter=iter,
                                 batch_size=batch_size,
                                 data_type=data_type,device=device)


def do_craft_jsma(prefix_model_path, prefix_save_path, model_list,is_save=False, max_distortion=0.12,
                  dim_features=784,
                  num_out=10, source_data=None, data_type=None,device='cpu'):
    for model_name in model_list:
        save_folder = model_name.split('.')[0]+'-'+str(max_distortion)
        model_path = os.path.join(prefix_model_path, model_name)
        save_path = os.path.join(prefix_save_path, save_folder)
        print('>>>>Generate {} >>>>>>>>>>'.format(model_name))
        if data_type == DATA_IMAGENET:
            genereate_jsma_samples_for_large_data(model_path, source_data, save_path, is_save, max_distortion,
                                                  dim_features,
                                                  num_out, data_type=data_type, img_shape={'C': 3, 'H': 224, 'W': 224})
        else:
            genereate_jsma_samples(model_path, source_data, save_path, is_save, max_distortion, dim_features,
                                   num_out, data_type=data_type,device=device)


def do_craft_deepfool(prefix_model_path, prefix_save_path, model_list, source_data, data_type):
    for model_name in model_list:
        save_folder = model_name.split('.')[0]+'-'+str(0.02)+'-'+str(50)
        model_path = os.path.join(prefix_model_path, model_name)
        save_path = os.path.join(prefix_save_path, save_folder)
        logging.info('>>>>Generate {} >>>>>>>>>>'.format(model_name))
        if data_type == DATA_IMAGENET:
            genereate_deepfool_samples_for_large_data(model_path, source_data_path=source_data, save_path=save_path,
                                                      data_type=data_type,
                                                      num_out=1000)
        else:
            genereate_deepfool_samples(model_path, source_data_path=source_data, save_path=save_path,
                                       data_type=data_type,
                                       num_out=10)


def craft_mnist():
    source_data = '../../datasets/mnist/raw'
    data_type = DATA_MNIST
    setup_logging()
    prefix_model_path = '../model-storage/mnist/hetero-base/'
    model_list = ['MnistNet4.pkl']

    is_save = True
    # prefix_save_path = '../../datasets/mnist/adversarial/fgsm/single/pure/'
    # eps = 0.35
    # adv_folder_list = ['mnist4-eta3']
    # do_craft_fgsm(prefix_model_path, prefix_save_path, model_list, adv_folder_list, eps=eps,is_save=True)

    adv_folder_list = ['mnist4-c8-i1w']
    prefix_save_path = '../../datasets/mnist/adversarial/cw/single/pure'
    do_craft_cw(prefix_model_path, prefix_save_path, model_list, save_folder_list=adv_folder_list,
                is_save=True, c=0.6, iter=80000, batch_size=2,device='cuda')

    # adv_folder_list = ['mnist4-d12']
    # prefix_save_path = '../../datasets/mnist/adversarial/jsma/single/pure'
    # do_craft_jsma(prefix_model_path, prefix_save_path, model_list, adv_folder_list, is_save=True, max_distortion=0.12,
    #               dim_features=784,
    #               num_out=10)

    # adv_folder_list = ['mnist4-0.02-50']
    # prefix_save_path = '../../datasets/mnist/adversarial/deepfool/single/pure'
    # do_craft_deepfool(prefix_model_path, prefix_save_path, model_list, adv_folder_list, source_data,
    #                   data_type=data_type)
    # recheck_advsamples(prefix_model_path, prefix_save_path, model_list, adv_folder_list)

    # save_path_list  = ['../../datasets/mnist/adversarial/fgsm/single/pure/',
    #                    '../../datasets/mnist/adversarial/cw/single/pure',
    #                    '../../datasets/mnist/adversarial/jsma/single/pure',
    #                    '../../datasets/mnist/adversarial/bb/single/pure/']

    # save_path_list = ['../../datasets/mnist/adversarial/fgsm/single/non-pure/',
    #                   '../../datasets/mnist/adversarial/cw/single/non-pure/',
    #                   '../../datasets/mnist/adversarial/jsma/single/non-pure/',
    #                   '../../datasets/mnist/adversarial/bb/single/non-pure/']
    #
    # folder_list = [['mnist4-eta3'], ['mnist4-c8-i1w'], ['mnist4-d12'], ['mnist4-fgsm35']]
    #
    # for prefix_save_path, adv_folder_list in zip(save_path_list, folder_list):
    #     print('delete invalid data:{}'.format(adv_folder_list[0]))
    #     recheck_advsamples(prefix_model_path, prefix_save_path, model_list, adv_folder_list)


def craft_cifar10(device):
    source_data = '../../datasets/cifar10/raw'
    prefix_model_path = '../model-storage/cifar10/hetero-base/'
    # model_list = ['lenet.pkl']
    model_list = ['googlenet.pkl']
    data_type = DATA_CIFAR10
    is_save = True

def craft_imagenet():
    # source_data = '../../datasets/cifar10/raw'
    # prefix_model_path = '../model-storage/cifar10/hetero-base/'
    # model_list = ['lenet.pkl']
    # data_type = DATA_CIFAR10

    source_data = '../../datasets/ilsvrc12/raw'
    data_type = DATA_IMAGENET
    prefix_model_path = 'pretrained'
    model_list = ['densenet121']

    # prefix_save_path = '../../datasets/ilsvrc12/adversarial/fgsm/single/pure/'
    # eps = 0.035
    # adv_folder_list = ['densenet121-eta0.035']
    # do_craft_fgsm(prefix_model_path, prefix_save_path, model_list, adv_folder_list, eps=eps, is_save=True,
    #               source_data=source_data, data_type=data_type)
    # recheck_advsamples(prefix_model_path, prefix_save_path, model_list, adv_folder_list, channels=3,data_type=DATA_IMAGENET)

    # prefix_save_path = '../../datasets/ilsvrc12/adversarial/jsma/single/pure/'
    # max_distortion = 0.0012
    # adv_folder_list = ['densenet121-' + str(max_distortion)]
    # do_craft_jsma(prefix_model_path, prefix_save_path, model_list, adv_folder_list, is_save=True,
    #               max_distortion=max_distortion,
    #               dim_features=3 * 224 * 224,
    #               num_out=1000, source_data=source_data, data_type=DATA_IMAGENET)

    # prefix_save_path = '../../datasets/ilsvrc12/adversarial/cw/single/pure/'
    # c = 0.6
    # iter = 1000
    # adv_folder_list = ['densenet121-' + str(c) + '-' + str(iter)]
    # do_craft_cw(prefix_model_path, prefix_save_path, model_list, save_folder_list=adv_folder_list,
    #             is_save=True, c=c, iter=iter, batch_size=2, source_data=source_data, data_type=DATA_IMAGENET)

    # prefix_save_path = '../../datasets/ilsvrc12/adversarial/deepfool/single/pure/'
    # adv_folder_list = ['densenet121-0.02-50']
    # do_craft_deepfool(prefix_model_path, prefix_save_path, model_list, adv_folder_list, source_data,
    #                   data_type=data_type)


def mnist_advesaril_refine():
    model = torch.load('../model-storage/mnist/hetero-base/MnistNet4.pkl')

    #####
    # bb 2006
    #####
    # file_path = '/home/dong/gitgub/SafeDNN/datasets/mnist/adversarial/bb/single/non-pure/mnist4-fgsm35/'

    #####
    # cw  724
    #####
    # file_path = '/home/dong/gitgub/SafeDNN/datasets/mnist/adversarial/cw/single/non-pure/mnist4-c8-i1w'

    ######
    # df 1116
    #######
    file_path  = '/home/dong/gitgub/SafeDNN/datasets/mnist/adversarial/deepfool/single/non-pure/mnist4-0.02-50'
    ######
    # fgsm  1934
    ########
    # file_path = '/home/dong/gitgub/SafeDNN/datasets/mnist/adversarial/fgsm/single/non-pure/mnist4-eta3'


    ##########
    # jsma 1072
    #########
    # file_path =  '/home/dong/gitgub/SafeDNN/datasets/mnist/adversarial/jsma/single/non-pure/mnist4-d12'

    rename_advlabel_deflected_img(model, file_path, data_description='raw dgl-mnist', img_mode=None, device='cuda',
                                      data_type=DATA_MNIST)

if __name__ == '__main__':
    setup_logging()
    device = 'cuda:0'
    # craft_cifar10(device)
    # craft_mnist()
    # craft_imagenet()
    mnist_advesaril_refine()