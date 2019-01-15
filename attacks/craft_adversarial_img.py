from __future__ import print_function
import sys
sys.path.append('../')
from attacks.attack_type.fgsm import FGSM
from attacks.attack_type.carlinl2 import CarliniL2
from attacks.attack_type.blackBox import BlackBox
from attacks.attack_type.jsma import *
from attacks.attack_type import blackBox
from attacks.attack_type.deepfool import DeepFool
from utils.data_manger import *
from models.googlenet import *
from models import lenet
import argparse
from utils.time_util import current_timestamp


'''
This component has three levels

The genereate_?_samples is the kernel generator which calls a specific adversary to generate the adversarial samples by 
batch or by single sample.

do_craft_? is designed to do attack for multil-models.

The two levels are not bound with certain data

The level, like craft_mnist,craft_cifar10,craft_imagenet,is the application level,which is bound with certain level.

'''


def genereate_fgsm_samples_for_ImageNet(source_data, save_path, eps, is_save=False, is_exclude_wr=True,
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


def genereate_jsma_samples_for_ImageNet(source_data, save_path, is_save=False, max_distortion=0.12,
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


def genereate_cw_samples_for_ImageNet(source_data, save_path, is_save=False, c=0.8, iter=10000,
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
        dataset = datasetMutiIndx(raw_test_data, random_samples[start:start + step_size])
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
                save_imgs_tensor(success_samples, normal_labels, adv_label, save_path, 'cw',
                                 no_batch=batch_no * step_size + i,
                                 batch_size=batch_size, channels=channels)
            logging.info('batch:{}'.format(i))
        start += step_size


def genereate_deepfool_samples_for_ImageNet(source_data_path, save_path=None, overshoot=0.02, num_out=10,
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
    for batch_no in range(9, 24):
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
                                 channels=channels, adv_count=count)
            logging.info('{}th finished'.format(count))
            count += 1
        start += batch_size


def genereate_fgsm_samples(model, source_data, save_path, eps, is_exclude_wr=True,
                           data_type=DATA_MNIST, device="cpu"):
    '''

    :param model_path:
    :param source_data:
    :param save_path:
    :param eps:
    :param is_save:
    :param is_exclude_wr:  exclude the wrong labeled or not
    :return:
    '''

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    test_data, channel = load_data_set(data_type, source_data, train=False)
    if data_type == DATA_IMAGENET:
        model = torchvision.models.densenet121(pretrained=True)
        random_samples = np.arange(10000)
        np.random.seed(random_seed)
        np.random.shuffle(random_samples)
        dataset = datasetMutiIndx(test_data, random_samples[:1000])
        test_data = exclude_wrong_labeled(model, dataset, device)
        test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=1)
    else:
        test_data = exclude_wrong_labeled(model, test_data, device)
        test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    fgsm = FGSM(model, eps=eps, device=device)
    adv_samples, y = fgsm.do_craft_batch(test_loader)
    adv_loader = DataLoader(TensorDataset(adv_samples, y), batch_size=1, shuffle=False)
    succeed_adv_samples = samples_filter(model, adv_loader, "Eps={}".format(eps), device=device)
    num_adv_samples = len(succeed_adv_samples)
    print('successful samples', num_adv_samples)
    if is_save:
        save_imgs(succeed_adv_samples, TensorDataset(adv_samples, y), save_path, 'fgsm', channel)
    print('Done!')


def genereate_cw_samples(target_model, source_data, save_path,c=0.8, iter=10000, batch_size=1,
                         data_type=DATA_MNIST, device='cpu'):
    # at present, only  cuda0 suopport

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

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
        test_data = exclude_wrong_labeled(target_model, test_data, device=device)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    l2Attack = CarliniL2(target_model=target_model, max_iter=iter, c=c, k=0, device=device, targeted=False)
    print("Generating adversarial sampels...")
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


def genereate_jsma_samples(model, source_data, save_path,max_distortion=0.12, dim_features=784,
                           num_out=10, data_type=DATA_MNIST, img_shape={'C': 3, 'H': 32, 'W': 32}, device='cpu'):

    # train_data, _ =  load_data_set(data_type=data_type,source_data=source_data,train=True)
    # complete_data = ConcatDataset([test_data,train_data])

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    test_data, channels = load_data_set(data_type=data_type, source_data=source_data)
    if data_type == DATA_IMAGENET:
        model = torchvision.models.densenet121(pretrained=True)
        random_samples = np.arange(10000)
        np.random.seed(random_seed)
        np.random.shuffle(random_samples)
        test_data = datasetMutiIndx(test_data, random_samples)

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


def genereate_deepfool_samples(model, source_data_path, save_path=None, overshoot=0.02, num_out=10, max_iter=50,
                               data_type=DATA_MNIST, device="cpu"):
    '''
    Single data only!Do not Support batch
    :return:
    '''
    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    test_data, channels = load_data_set(data_type=data_type, source_data=source_data_path)
    if data_type == DATA_IMAGENET:
        model = torchvision.models.densenet121(pretrained=True)
        random_samples = np.arange(10000)
        np.random.seed(random_seed)
        np.random.shuffle(random_samples)
        test_data = datasetMutiIndx(test_data, random_samples)

    complete_data = exclude_wrong_labeled(model, test_data, device)
    test_data_laoder = DataLoader(dataset=complete_data, batch_size=1, shuffle=True)
    deepfool = DeepFool(target_model=model, num_classes=num_out, overshoot=overshoot, max_iter=max_iter, device=device)
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


def genereate_balck_box_samples(target_model, source_data_path, save_path=None,eps=0.03,max_iter=6,
                                submodel_epoch=10,seed_data_size=200,rnd_seed=random_seed,
                                data_type=DATA_MNIST, step_size=1,device="cpu"):
    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    if data_type == DATA_MNIST:
        subModel = blackBox.ArchA()
        test_data, channel = load_data_set(data_type=DATA_MNIST, source_data=source_data_path, train=False)
        train_data, channel = load_data_set(data_type=DATA_MNIST, source_data=source_data_path, train=True)

    elif data_type == DATA_CIFAR10:
        subModel = lenet.Cifar10Net()
        test_data, channel = load_data_set(data_type=DATA_MNIST, source_data=source_data_path, train=False)


    indices = np.arange(10000)
    np.random.seed(rnd_seed)
    np.random.shuffle(indices)
    seed_data = datasetMutiIndx(test_data, indices[:seed_data_size])
    test_data = datasetMutiIndx(test_data, indices[seed_data_size:])

    bb = BlackBox(target_model=target_model, substitute_model=subModel,
                  seed_data=seed_data,
                  test_data=test_data,
                  step_size=step_size,
                  max_iter=max_iter, submodel_epoch=submodel_epoch,device=device)

    print("substitute training....")
    bb.substitute_training()
    adversary = FGSM(bb.substitute_model, eps=eps, target_type=blackBox.TRUELABEL)
    succeed_adv_samples = bb.do_craft(adversary, ConcatDataset([train_data, test_data]), save_path=save_path)
    adv_laoder = DataLoader(dataset=succeed_adv_samples)
    samples_filter(model=bb.substitute_model, loader=adv_laoder, name='substitue model')
    samples_filter(model=target_model, loader=adv_laoder, name='source model')


def parametersParse(arg):

    if isinstance(arg,list):
        return [parametersParse(ele) for ele in arg if ele.strip()!=""]

    if arg.lower() == "true":
        return True
    if arg.lower() == "false":
        return False
    import re
    int_pattern = re.compile("^(\d)+$")
    float_patter = re.compile("^(\d)+\.(\d)+$")

    if int_pattern.match(arg) is not None:
        return  int(arg)
    if float_patter.match(arg) is not None:
        return float(arg)
    raise Exception("Warinings:Unsupported data type. {}".format(arg))

def run():

    parser = argparse.ArgumentParser(description="The required parameters of mutation process")

    parser.add_argument("--modelName", type=str,
                        help="The model's name,e.g, googlenet, lenet",
                        default="lenet")
    parser.add_argument("--modelPath", type=str,
                        help="The the path of pretrained targeted model.Note, the model should be saved as the form model.stat_dict")
    parser.add_argument("--dataType", type=int,
                        help="The data set that the given model is tailored to. Three types are available: mnist,0; "
                             "cifar10, 1", default=0, required=True)
    parser.add_argument("--sourceDataPath", type=str,
                        help="The path of source data which is used to yield adversarial sampels",
                        required=True)
    parser.add_argument("--attackType", type=str,
                        help="Five attacks are available: fgsm, jsma, bb, deepfool, cw",
                        required=True)
    parser.add_argument("--attackParameters", type=str,
                        help="The parameters for specific attack.fgsm:",
                        required=True)
    parser.add_argument("--savePath", type=str, help="The path where the adversarial samples to be stored",
                        required=True)
    parser.add_argument("--device", type=int,
                        help="The index of GPU used. If -1 is assigned,then only cpu is available",
                        required=True)

    args = parser.parse_args()
    attackParameters = parametersParse(args.attackParameters.split(","))

    if args.modelName == "googlenet":
        seed_model = GoogLeNet()
    elif args.modelName == "lenet":
        seed_model = lenet.MnistNet4()
    seed_model.load_state_dict(torch.load(args.modelPath))

    device = "cuda:" + str(args.device) if args.device >= 0 else "cpu"

    if args.dataType == DATA_CIFAR10:
        data_name = 'cifar10'
        dim_features = 3 * 32 * 32
        num_out = 10
        img_shape = {'C': 3, 'H': 32, 'W': 32}
    elif args.dataType == DATA_MNIST:
        data_name = 'mnist'
        dim_features = 28 * 28
        num_out = 10
        img_shape = {'C': 1, 'H': 28, 'W': 28}
    else:
        raise Exception("{} data set is not supported".format(args.dataType))

    args.savePath = os.path.join(args.savePath,current_timestamp().replace(" ",'_'))

    if args.attackType == Adv_Tpye.FGSM:
        numParas=len(attackParameters)
        assert numParas == 2, "FGSM just need two parameters,but {} found:{}".format(numParas,attackParameters)
        eps = attackParameters[0]
        is_exclude_wr = args.attackParameters[1]
        genereate_fgsm_samples(seed_model, args.sourceDataPath, args.savePath, data_type=args.dataType,
                               device=device, eps=eps, is_exclude_wr=is_exclude_wr)

    elif args.attackType == Adv_Tpye.JSMA:
        numParas = len(attackParameters)
        assert numParas == 1, "JSMA just need one parameter,but {} found:{}".format(numParas,attackParameters)
        max_distortion = attackParameters[0]
        genereate_jsma_samples(seed_model, args.sourceDataPath, args.savePath, data_type=args.dataType,
                               max_distortion=max_distortion, dim_features=dim_features,
                               num_out=num_out,device=device,img_shape=img_shape)

    elif args.attackType == Adv_Tpye.BB:
        numParas = len(attackParameters)
        assert numParas == 5, "BlackBox just need five parameter,but {} found:{}".format(numParas, attackParameters)
        eps  = attackParameters[0]
        max_iter = attackParameters[1]
        submodel_epoch = attackParameters[2]
        seed_data_size = attackParameters[3]
        step_size = attackParameters[4]
        genereate_balck_box_samples(seed_model, args.sourceDataPath,  args.savePath, data_type=args.dataType,eps=eps,
                                    max_iter=max_iter,submodel_epoch=submodel_epoch, seed_data_size=seed_data_size,
                                    rnd_seed=random_seed, device=device,step_size=step_size)

    elif args.attackType == Adv_Tpye.CW:
        numParas = len(attackParameters)
        assert numParas == 2, "CW just need two parameter,but {} found:{}".format(numParas, attackParameters)
        scaleCoefficient =attackParameters[0]
        itertaions = attackParameters[1]

        genereate_cw_samples(seed_model, args.sourceDataPath, args.savePath, data_type=args.dataType, device=device,
                             c=scaleCoefficient, iter=itertaions, batch_size=2)

    elif args.attackType == Adv_Tpye.DEEPFOOL:
        numParas = len(attackParameters)
        assert numParas == 2, "DEEPFOOL just need two parameter,but {} found:{}".format(numParas, attackParameters)
        overshoot = attackParameters[0]
        max_iter = attackParameters[1]
        genereate_deepfool_samples(seed_model, args.sourceDataPath, args.savePath,data_type=args.dataType, overshoot=overshoot,
                                   num_out=num_out,
                                   max_iter=max_iter,
                                   device=device)
    else:
        raise Exception("{} is not supported".format(args.attackType))

    ########
    # remove the saved deflected adversarial samples
    ########
    rename_advlabel_deflected_img(seed_model, args.savePath, data_description='icse19-eval-attack-{}'.format(args.attackType), img_mode=None, device=device,
                                  data_type=args.dataType)

    print("Adversarial samples are saved in {}".format(args.savePath))

if __name__ == '__main__':
    run()
    # parametersParse("[]")



