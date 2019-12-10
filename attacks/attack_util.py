import sys
<<<<<<< HEAD
=======

>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3
sys.path.append('../')
import logging
import torch.nn.functional  as F
from torch.utils.data import TensorDataset
from torch.autograd.gradcheck import *
from scipy.misc import imsave
from utils.data_manger import *
import argparse
from models.lenet import MnistNet4
from models.googlenet import GoogLeNet


class Adv_Tpye(object):
    FGSM = 'fgsm'
    JSMA = 'jsma'
    BB = 'bb'
    DEEPFOOL = 'deepfool'
    CW = 'cw'

<<<<<<< HEAD
def load_natural_data(is_normal, data_type, raw_data_path, use_train, seed_model,device='cpu',MAX_NUM_SAMPLES=1000):
    data, channel = load_data_set(data_type, raw_data_path, train=use_train)
    if is_normal:
        selected_data = samples_filter(seed_model, DataLoader(dataset=data), return_type='normal', name='seed model',
                                        device=device, show_accuracy=False)
    else:
        selected_data = samples_filter(seed_model, DataLoader(dataset=data), return_type='adv', name='seed model',
                                        device=device, show_accuracy=False)

    random_indcies = np.arange(len(selected_data))
    np.random.seed(random_seed)
    np.random.shuffle(random_indcies)
    random_indcies = random_indcies[:MAX_NUM_SAMPLES]
    data = datasetMutiIndx(data, np.array([idx for idx, _, _ in selected_data])[random_indcies])
    return data




=======
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3

def adv_samples_filter(model, loader, name, size=0,
                       device='cpu', file_path=False):
    '''
    :param model:
    :param loader:
    :param name:
    :param return_type:
    :param use_adv_ground:
    :param size:
    :param show_progress:
    :param device:
    :param is_verbose:
    :return:
    '''
    assert loader.batch_size == 1
    model.eval()
    test_loss = 0
    correct = 0
    index = 0
    model = model.to(device)
    remove_count = 0
    rename_count = 0
    success_count = 0
    for data_tuple in loader:
        if len(data_tuple) == 2:
            data, target = data_tuple
        elif len(data_tuple) == 3:
            data, target, adv_label = data_tuple
        elif len(data_tuple) == 4:
            data, target, adv_label, file_name = data_tuple
            file_name = file_name[0]
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        rst = pred.eq(target).sum().item()
        correct += rst

        if pred.item() == target.item():
            # remove invalid adversarial samples
            os.remove(os.path.join(file_path, file_name))
            remove_count += 1
        elif pred.item() != adv_label.item():
            # rename deflected adversarial samples
            new_name = get_file_name(file_name, pred.item())
            os.rename(os.path.join(file_path, file_name), os.path.join(file_path, new_name))
            rename_count += 1
        else:
            success_count += 1

        index += 1

        if size > 0 and index == size:
            break

    print('{}: rename {}, remove {},success {}'.format(name, rename_count, remove_count, success_count))


def samples_filter(model, loader, name, return_type="adv", size=0, show_progress=False,
                   device='cpu', is_verbose=False, show_accuracy=True):
    '''
    :param model:
    :param loader:
    :param name:
    :param return_type:
    :param use_adv_ground: use the adv label as the desired label
    :param size:
    :param show_progress:
    :param device:
    :param is_verbose:
    :return:
    '''
    assert loader.batch_size == 1
    model.eval()
    test_loss = 0
    correct = 0
    index = 0
    adv_samples = []  # index:pred
    normal_sample = []

    total_sample = size if size > 0 else len(loader.dataset)
    model = model.to(device)
    for data_tuple in loader:
        if len(data_tuple) == 2:
            data, target = data_tuple
        elif len(data_tuple) == 3:
            data, target, adv_label = data_tuple
        elif len(data_tuple) == 4:
            data, target, adv_label, file_name = data_tuple

        data, target = data.to(device), target.to(device)

        if is_verbose:
            if len(data_tuple) == 4:
                logging.info(
                    '{}>>>True Label:{},adv_label:{}'.format(file_name.item(), target.item(), adv_label.item()))
            elif len(data_tuple) == 3:
                logging.info('>>>True Label:{},adv_label:{}'.format(target.item(), adv_label.item()))
            else:
                logging.info('>>>True Label:{}'.format(target.item()))
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if is_verbose:
            logging.info('Predicted Label:{}<<<'.format(pred.item()))
        rst = pred.eq(target).sum().item()
        correct += rst
        is_adv_success = 0 if rst == 1 else 1  # if the predict label is equal to the target label,then the attack is failed
        if is_adv_success:
            adv_samples.append((index, target.item(), pred.item()))
        else:
            normal_sample.append((index, target.item(), pred.item()))
        index += 1

        if size > 0 and index == size:
            break

        #####
        # progress
        #####
        if show_progress:
            sys.stdout.write("\r Test: %d of %d" % (index + 1, total_sample))
            sys.stdout.flush()

    size = size if size > 0 else len(loader.dataset)
    test_loss /= size
    if show_accuracy:
        print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            name, test_loss, correct, size, 100. * correct / size))
    if return_type == 'adv':
        return adv_samples
    else:
        return normal_sample


def exclude_wrong_labeled(model, dataset, device):
    dataloader = DataLoader(dataset=dataset)
    correct_labeled = samples_filter(model, dataloader, return_type='normal', name='targeted model', device=device)
    return datasetMutiIndx(dataset, [idx for idx, _, _ in correct_labeled])


def get_jacobian(x, model, num_out, device='cpu'):
    input = torch.tensor(x.cpu().clone().cuda(), requires_grad=True)
    input_cp = input.to(device)
    model = model.eval()
    model = model.to(device=device)
    output = model(input_cp)
    jacobian = make_jacobian(input, num_out=num_out)  # input_dim x num_out
    grand_out = torch.zeros(*output.size()).to(device)
    # Note: the first axis denotes the batch size,for single example,it's 1
    for axis in range(num_out):
        grand_out.zero_()
        grand_out[:, axis] = 1
        zero_gradients(input)
        output.backward(grand_out, retain_graph=True)
        grad = input.grad.data  # (1, 1, 28, 28)
        grad = torch.squeeze(grad.view(-1, 1))
        jacobian[:, axis] = grad

    return jacobian.to(device)


def save_imgs(adv_samples, dataset, save_path, file_prefix, channels=1, batch_size=1, batch_no=0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    adv_count = batch_size * batch_no
    for idx, normal_pred, adv_pred in adv_samples:
        adv_count += 1
        data, true_label = dataset[idx]
        assert true_label.item() == normal_pred
        adv_path = os.path.join(save_path,
                                file_prefix + '_' + str(adv_count) + '_' + str(true_label.item()) + '_' + str(
                                    adv_pred) + '_.png')
        if channels == 1:
            img = data.cpu().numpy()
            img = img.reshape(28, 28)
            imsave(adv_path, img)
        elif channels == 3:
            img = data.cpu().numpy()
            img = np.transpose(img, axes=(1, 2, 0))
            imsave(adv_path, img)


def save_imgs_tensor(adv_samples, normal_preds, adv_preds, save_path, file_prefix, no_batch=1, batch_size=1, channels=1,
                     adv_count=-1):
    '''
    The difference from the "save_imgs" is that the adv_samples are image tensors, not the indicies
    :param adv_samples:
    :param normal_preds:
    :param adv_preds:
    :param save_path:
    :param file_prefix:
    :param no_batch: The number of batch. 1-index
    :return:
    '''

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if adv_count == -1:
        adv_count = (no_batch - 1) * batch_size
    for data, normal_pred, adv_pred in zip(adv_samples, normal_preds, adv_preds):
        adv_count += 1
        if isinstance(normal_pred, torch.Tensor):
            normal_pred = normal_pred.item()
        if isinstance(adv_pred, torch.Tensor):
            adv_pred = adv_pred.item()
        adv_path = os.path.join(save_path,
                                file_prefix + '_' + str(adv_count) + '_' + str(normal_pred) + '_' + str(
                                    adv_pred) + '_.png')

        if channels == 1:
            img = data.detach().cpu().numpy()
            img = img.reshape(28, 28)
            imsave(adv_path, img)
        elif channels == 3:
            img = data.squeeze().detach().cpu().numpy()
            img = np.transpose(img, axes=(1, 2, 0))
            imsave(adv_path, img)


def rename_advlabel_deflected_img(model, file_path, data_description='raw dgl-cifar10', img_mode=None, device='cpu',
                                  data_type=DATA_CIFAR10):
    '''
    This function just remove those adversarial samples whose preicted label is not identical to its adversarial label
    :param model:
    :param file_path:
    :param data_soure:
    :param img_mode:
    :return:
    '''

    if data_type == DATA_CIFAR10:
        normalize = normalize_cifar10
    elif data_type == DATA_MNIST:
        normalize = normalize_mnist
    elif data_type == DATA_IMAGENET:
        normalize = normalize_imgNet

    dataset = MyDataset(root=file_path, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]), show_file_name=True, img_mode=img_mode, max_size=10000)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # here, 'adv' means that those sampels whose predict label is not identical to the adv_lable
    adv_samples_filter(model, loader, data_description, 'adv', file_path=file_path,
                       device=device)


def test_adv_samples():
    parser = argparse.ArgumentParser("Test Adversarial samples")
    parser.add_argument("--dataType", type=int,
                        help="The data set that the given model is tailored to. Three types are available: mnist,0; "
                             "cifar10, 1", default=0, required=True)
    parser.add_argument("--filePath", type=str,
                        help="The path of samples to be tested", required=True)

    parser.add_argument("--device", type=int,
                        help="The index of GPU used. If -1 is assigned,then only cpu is available",
                        required=True)

    parser.add_argument("--advGround", type=int,
                        help="1,if use adversarial label as ground truth;0,otherwise",
                        required=True)

    args = parser.parse_args()
    use_adv_ground = True if args.advGround == 1 else False
    device = "cuda:" + str(args.device) if args.device >= 0 else "cpu"

    if args.dataType == DATA_CIFAR10:
        data_name = 'cifar10'
        dim_features = 3 * 32 * 32
        num_out = 10
        img_shape = {'C': 3, 'H': 32, 'W': 32}
        model = GoogLeNet()
        modelPath = "../build-in-resource/pretrained-model/googlenet.pkl"
        normalize = normalize_cifar10
        img_mode = None
    elif args.dataType == DATA_MNIST:
        data_name = 'mnist'
        dim_features = 28 * 28
        num_out = 10
        img_shape = {'C': 1, 'H': 28, 'W': 28}
        model = MnistNet4()
        normalize = normalize_mnist
        modelPath = "../build-in-resource/pretrained-model/lenet.pkl"
        img_mode = "L"
    else:
        raise Exception("{} data set is not supported".format(args.dataType))
    model.load_state_dict(torch.load(modelPath))
    model = model.to(device)
    model.eval()
    dataset = MyDataset(root=args.filePath, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]), show_file_name=True, img_mode=img_mode, max_size=10000)
    dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct = 0
    for data_tuple in dataLoader:
        data, target, adv_label, file_name = data_tuple
        if use_adv_ground:
            target = adv_label
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        rst = pred.eq(target).sum().item()
        correct += rst
        print("{},Adversarial Label:{},Current Predict Label:{}").format(file_name, adv_label.item(), pred.item())
    print("Total:{},Success:{}".format(len(dataLoader), correct))


def get_file_name(old_file_name, new_adv_label):
    img_file_split = old_file_name.split('_')
    img_file_split[-2] = str(new_adv_label)
    return img_file_split[0] + '_' + img_file_split[1] + '_' + img_file_split[2] + '_' + img_file_split[3] + '_' + \
           img_file_split[4]


def refine_mnist():
    modelPath = "../build-in-resource/pretrained-model/lenet.pkl"
    model = MnistNet4()
    model.load_state_dict(torch.load(modelPath))

    bb_file_path = "../build-in-resource/dataset/mnist/adversarial/bb/"
    rename_advlabel_deflected_img(model, bb_file_path, data_description='bb-mnist', img_mode=None, device='cuda:1',
                                  data_type=DATA_MNIST)

    cw_file_path = "../build-in-resource/dataset/mnist/adversarial/cw/"
    rename_advlabel_deflected_img(model, cw_file_path, data_description='cw-mnist', img_mode=None, device='cuda:1',
                                  data_type=DATA_MNIST)

    jsma_file_path = "../build-in-resource/dataset/mnist/adversarial/jsma/"
    rename_advlabel_deflected_img(model, jsma_file_path, data_description='jsma-mnist', img_mode=None, device='cuda:2',
                                  data_type=DATA_MNIST)

    dp_file_path = "../build-in-resource/dataset/mnist/adversarial/deepfool/"
    rename_advlabel_deflected_img(model, dp_file_path, data_description='dp-mnist', img_mode=None, device='cuda:2',
                                  data_type=DATA_MNIST)


    fgsm_file_path = "../build-in-resource/dataset/mnist/adversarial/fgsm/"
    rename_advlabel_deflected_img(model, fgsm_file_path, data_description='fgsm-mnist', img_mode=None, device='cuda:2',
                                  data_type=DATA_MNIST)

if __name__ == '__main__':
    test_adv_samples()
