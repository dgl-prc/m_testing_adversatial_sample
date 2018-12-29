import sys

sys.path.append('../')
sys.path.append('../cifar10models')
import logging
import torch.nn.functional  as F
from torch.utils.data import TensorDataset
from torch.autograd.gradcheck import *
from util.data_manger import *
from scipy.misc import imsave


class Adv_Tpye(object):
    FGSM = 'fgsm'
    JSMA = 'jsma'
    BB = 'bb'
    DEEPFOOL = 'deepfool'
    CW = 'cw'


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

    print('{}: rename {}, remove {},success {}'.format(name,rename_count,remove_count,success_count))

def samples_filter(model, loader, name, return_type="adv", use_adv_ground=False, size=0, show_progress=False,
                   device='cpu', is_verbose=False):
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

        if use_adv_ground:
            target = adv_label

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
        is_adv_success = 0 if rst == 1 else 1
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
            img = data.numpy()
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
        normalize = normalize_cifar10
    elif data_type == DATA_IMAGENET:
        normalize = normalize_imgNet

    dataset = MyDataset(root=file_path, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]), show_file_name=True, img_mode=img_mode,max_size=10000)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # here, 'adv' means that those sampels whose predict label is not identical to the adv_lable

    adv_samples_filter(model, loader, data_description, 'adv', file_path=file_path,
                                    device=device)
    # samples_filter(model, loader, data_description, 'adv', use_adv_ground=True, is_verbose=False,
    #                                     device=device)

# def recheck_advsamples():
#     from util.logging_util import setup_logging
#     setup_logging()
#
#     ########
#     # mnsit
#     ########
#     data_soure = 'mnist'
#     adversarial_folders = ['mnist4-eta3', 'mnist4-d12', 'mnist4-fgsm35', 'mnist4-c8-i1w', 'mnist4-0.02-50']
#     adversarial_types = ['fgsm', 'jsma', 'bb', 'cw', 'deepfool']
#     model_path = '../model-storage/mnist/hetero-base/MnistNet4.pkl'
#     img_mode = 'L'
#
#     #########
#     # cifar10
#     #########
#     # data_soure = 'cifar10'
#     # adversarial_folders = ['lenet-eps-0.03', 'lenet-0.12', 'lenet-fgsm-eps-0.03', 'lenet-0.6-10000','mnist4-0.02-50']
#     # adversarial_types = ['fgsm', 'jsma', 'bb', 'cw', 'deepfool']
#     # model_path = '../model-storage/cifar10/hetero-base/lenet.pkl'
#     img_mode = None
#
#     #########
#     # general settings
#     #########
#     prefix_path = '../../datasets/' + data_soure + '/adversarial-pure/'
#     midfix_path = 'single/pure/'
#     model = torch.load(model_path)
#
#     for adv_type, folder in zip(adversarial_types, adversarial_folders):
#         print('check {}-{}'.format(data_soure, adv_type))
#         file_path = os.path.join(prefix_path, adv_type, midfix_path, folder)
#         # remove_invalid_img(model, file_path, data_soure, img_mode=img_mode)
#         remove_advlabel_deflected_img(model, file_path, data_soure, img_mode=img_mode)


def get_file_name(old_file_name, new_adv_label):
    img_file_split = old_file_name.split('_')
    img_file_split[-2] = str(new_adv_label)
    return img_file_split[0] + '_' + img_file_split[1] + '_' + img_file_split[2] + '_' + img_file_split[3] + '_' + \
           img_file_split[4]


if __name__ == '__main__':
    # recheck_advsamples()
    file_name = 'bb_3351_5_6_.png'
    print get_file_name(file_name, new_adv_label=1)
