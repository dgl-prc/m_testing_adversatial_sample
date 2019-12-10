import torch
import torch.nn.functional as F
from models.lenet import MnistNet4
import time
import numpy as np


import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import os
import numpy as np
from scipy import ndimage
import torch
from scipy.misc import imsave
from PIL import Image

normalize_imgNet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

normalize_mnist = transforms.Normalize((0.1307,), (0.3081,))

normalize_cifar10 = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])



def test(model, test_loader, verbose=False,device='cpu'):
    model = model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    progress = 0
    batch_size = test_loader.batch_size
    data_size = len(test_loader.dataset)
    time_count = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            start = time.clock()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            time_count.append(time.clock()-start)
            progress +=1

    test_loss /= len(test_loader.dataset)
    acc = 1. * correct / len(test_loader.dataset)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%) time:{:.6f}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100 * acc, np.average(time_count)))
    return acc


DATA_MNIST = 0
DATA_CIFAR10 = 1
DATA_IMAGENET = 2

def load_data_set(data_type, source_data, train=False):
    if data_type == DATA_MNIST:
        data = torchvision.datasets.MNIST(root=source_data, train=train, transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                normalize_mnist
            ]
        ))
        channels = 1
    elif data_type == DATA_CIFAR10:
        data = torchvision.datasets.CIFAR10(root=source_data, train=train,
                                                 transform=torchvision.transforms.Compose(
                                                     [
                                                         torchvision.transforms.ToTensor(),
                                                         normalize_cifar10
                                                     ]
                                                 ))
        channels = 3
    elif data_type == DATA_IMAGENET:
        data = torchvision.datasets.ImageFolder(
            source_data,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_imgNet,
            ]))
        channels = 3
    else:
        raise Exception('Unknown data source')

    return data, channels


if __name__ == '__main__':
    seed_model = MnistNet4()
    datatype = "mnist"
    modelPath = "./build-in-resource/pretrained-model/mnist/lenet.pkl"
    seed_model.load_state_dict(torch.load(modelPath))

    source_data = './build-in-resource/dataset/' + datatype + '/raw'
    test_data, channel = load_data_set(DATA_MNIST, source_data=source_data)
    test_data_laoder = DataLoader(dataset=test_data, batch_size=64, num_workers=4)
    test(seed_model, test_data_laoder, verbose=True, device='cpu')

