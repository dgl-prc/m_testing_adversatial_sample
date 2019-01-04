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

random_seed = 5566

DATA_MNIST = 0
DATA_CIFAR10 = 1
DATA_IMAGENET = 2

MNIST_ADV_FOLDERs = {'fgsm':'mnist4-eta3', 'jsma':'mnist4-d12', 'bb':'mnist4-fgsm35', 'cw':'mnist4-c8-i1w', 'deepfool':'mnist4-0.02-50'}

CIFAR10_ADV_FOLDERs_GooGlNet = {'cw': 'googlenet-0.6-1000', 'fgsm': 'googlenet-eps-0.03', 'jsma': 'googlenet-0.12',
                                'deepfool': 'googlenet-0.02-50', 'bb': 'googlenet-eps-0.3'}


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


def load_cifar10(data_path, split, normalize=normalize_cifar10):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    train_data = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True,
                                              transform=transforms.Compose(
                                                  [
                                                      transforms.RandomCrop(32, padding=4),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      normalize]))

    test_data = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 normalize]))

    if split:
        return train_data, test_data
    else:
        dataset = ConcatDataset([train_data, test_data])
        return dataset


def load_mnist(data_path, split, normalize=normalize_mnist):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    train_data = torchvision.datasets.MNIST(data_path,
                                            train=True,
                                            download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize
                                            ]))

    test_data = torchvision.datasets.MNIST(data_path,
                                           train=False,
                                           download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               normalize
                                           ]))

    if split:
        return train_data, test_data
    else:
        dataset = ConcatDataset([train_data, test_data])
        return dataset


def load_imagenet():
    traindir = '../../datasets/ilsvrc12/raw'
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_imgNet,
        ]))
    print len(train_dataset)
    loader = DataLoader(dataset=train_dataset, shuffle=True)
    count = 1
    for data, lable in loader:
        print lable, type(data), data.size()
        img = data.squeeze().permute(1, 2, 0)
        imsave('img_' + str(count) + '_l_' + str(lable.item()) + '.JPEG', img)
        count += 1

    print 'ok'




def create_data_loader(batch_size, test_batch_size, train_data, test_data, use_cuda=False, seed=-1):
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=test_batch_size, shuffle=True
    )
    return train_loader, test_loader


def bootstrap(dataset, verbose=False):
    '''
    sample dataset with replacement.
    :param dataset:
    :return: new_dataset: the sampling data, used as train data
             oob_dataset: out of bag data, used as tets data
    '''
    size = len(dataset)
    new_dataset_indices = np.random.choice(size, size)
    unique = set(new_dataset_indices)
    oob_indices = set(np.arange(size)) - unique
    new_dataset = []
    for idx in new_dataset_indices:
        new_dataset.append(dataset[idx])

    oob_dataset = []
    for idx in oob_indices:
        oob_dataset.append(dataset[idx])
    if verbose:
        print "train coverage:{0}".format(len(unique) * 1. / size)
    return new_dataset, oob_dataset


def load_adversary_data(adv_file_path, normalization, channels=1, img_height=28, img_width=28, max_size=1000,
                        filter=True):
    '''
    :param file_path: the file path for the adversary images
    :return: the formatted data for mutation test, the actual label of the images, and the predicted label of the images
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalization
    ])

    image_list = []
    real_labels = []
    predicted_labels = []
    image_files = []
    data_count = 1
    for img_file in os.listdir(adv_file_path):
        if img_file.endswith('.png'):
            img_file_split = img_file.split('_')
            real_label = int(img_file_split[-3])
            predicted_label = int(img_file_split[-2])

            if filter:
                if real_label == predicted_label:
                    continue
            else:  # only extract those successfully generated adversaries
                real_labels.append(real_label)
                predicted_labels.append(predicted_label)
                img = ndimage.imread(adv_file_path + os.sep + img_file)
                if channels == 1:
                    img = np.expand_dims(img, axis=2)
                    img = transform(img)
                    image_list.append(img.view(1, 1, img_height, img_width))
                elif channels == 3:
                    img = Image.fromarray(img)
                    img = transform(img)
                    image_list.append(img)
                    # img = transform(img)
                    # image_list.append(img.permute(2,0,1).contiguous().view(1,3, img_height, img_width))
                data_count += 1
                image_files.append(img_file)
        if data_count > max_size:
            break
    # print('--- Total number of adversary images: ', len(image_list))
    return torch.cat(image_list, 0), image_files, torch.LongTensor(real_labels), torch.LongTensor(predicted_labels)


# def default_loader(path):
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)

# todo: enable imageNet read. reference:  torchvision.datasets.ImageFolder
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, img_mode=None, show_extral_label=True,
                 show_file_name=False, max_size=1000):

        image_list = []
        real_labels = []
        predicted_labels = []
        img_names = []
        data_count = 1
        self.max_size = max_size
        all_files = np.array([img_file for img_file in os.listdir(root)])

        if len(all_files) < self.max_size:
            load_files = all_files
        else:
            np.random.seed(random_seed)
            load_indices = np.random.choice(len(all_files), self.max_size, replace=False)  # not put back
            load_files = all_files[load_indices]

        assert len(load_files) <= self.max_size
        for img_file in load_files:
            if img_file.endswith('.png'):
                img_file_split = img_file.split('_')
                real_label = int(img_file_split[-3])
                predicted_label = int(img_file_split[-2])
                real_labels.append(real_label)
                predicted_labels.append(predicted_label)
                img = ndimage.imread(root + os.sep + img_file)
                image_list.append(img)
                img_names.append(img_file)

        self.data = np.array(image_list)
        self.labels = np.array(real_labels)
        self.predicted_labels = np.array(predicted_labels)
        self.transform = transform
        self.target_transform = target_transform
        self.show_extral_label = show_extral_label
        self.show_file_name = show_file_name
        self.img_names = img_names
        self.img_mode = img_mode  # ref: https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode=self.img_mode)

        if self.transform is not None:
            img = self.transform(img)

        if self.show_extral_label:
            predict = self.predicted_labels[index]

        if self.target_transform is not None:
            target = self.target_transform(target)
            predict = self.target_transform(predict)

        if self.show_extral_label:
            if self.show_file_name:
                return img, target, predict, self.img_names[index]
            else:
                return img, target, predict
        else:
            return img, target

    def __len__(self):
        return len(self.data)


def get_img_transforms_to_save():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv * torch.ones(A.shape))
        A = torch.min(A, maxv * torch.ones(A.shape))
        return A

    clip = lambda x: clip_tensor(x, 0, 255)

    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
                             transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
                             transforms.Lambda(clip),
                             transforms.ToPILImage(),
                             transforms.CenterCrop(224)])
    return tf


def datasetMutiIndx(dataset, indices):
    x_list = []
    y_list = []
    for idx in indices:
        x, y = dataset[idx]
        old_shape = [d for d in x.size()]
        old_shape.insert(0, 1)
        x = x.view(old_shape)
        x_list.append(x)
        y_list.append(y)
    return torch.utils.data.TensorDataset(torch.cat(x_list, 0), torch.LongTensor(y_list))


if __name__ == '__main__':
    file_path = '../../datasets/cifar10/adversarial-pure/fgsm/single/pure/googlenet-eps-0.03'
    dataset = MyDataset(root=file_path, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize_cifar10
    ]), show_file_name=True, img_mode=None)

    # train_loader,test_loader = load_dataloader(64,1000,"../nMutant/datasets/mnist")
    # print test_loader.__len__()
    # dataloader = load_complete_dataset("../nMutant/datasets/mnist")
    # print dataloader.__len__()
    # train_data, test_data = load_dataset('../../datasets/mnist/raw', split=True)
    # a =datasetMutiIndx(train_data, [1, 12, 4])
    # print len(a),a.size()

    # size = len(train_data)
    # idx  = np.arange(size)
    # np.random.shuffle(idx)
    # p1,p2,p3,p4,p5 = np.split(idx,5)
    # p1d = datasetMutiIndx(train_data,p1)
    # from model_trainer import test
    #
    # dir = '../../datasets/ilsvrc12/raw'
    # # model = torchvision.models.resnet34(pretrained=True)
    # model = torchvision.models.densenet121(pretrained=True)
    # # model = torchvision.models.inception_v3(pretrained=True)
    # dataset, channel = load_data_set(DATA_IMAGENET, source_data=dir)
    #
    # for i in range(10):
    #     random_samples = np.arange(10)
    #     np.random.seed(random_seed)
    #     np.random.shuffle(random_samples)
    #     print random_samples
    # dataset = datasetMutiIndx(dataset, random_samples)
    #
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # # device = 'cpu'
    # from logging_util import setup_logging
    # setup_logging()
    # import logging
    #
    # dataset = datasetMutiIndx(dataset, random_samples)
    # logging.info('start eval')
    # test(model=model,test_loader=DataLoader(dataset=dataset,batch_size=128,num_workers=5),device=device,verbose=True)
    # logging.info('end eval')
