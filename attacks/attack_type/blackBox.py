import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from utils.data_manger import *
import math
import torch.nn as nn
from utils.model_trainer import train
from jsma import get_jacobian
from models import *
import logging
from utils.logging_util import setup_logging
from attacks.attack_util import *
from fgsm import *
import torch.nn as nn
import copy
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from models.lenet import Cifar10Net




class ArchA(torch.nn.Module):
    def __init__(self):
        super(ArchA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 2),  # 32x27X27
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32x13x13, the default stride of max pool is 2. (27-2)/2+1=13
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 2),  # 64x25X25
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x6x6
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 6 * 6, 200),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class ArchB(torch.nn.Module):
    def __init__(self):

        super(ArchB,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,2), # 32-2+1 = 3*31*31
            nn.MaxPool2d(2) # (31-2)/2+1 = 3*15*15
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,2), # 15-2+1 = 14
            nn.MaxPool2d(2) # (14-2)/2 + 1 = 7
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*7*7,256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(256,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class BlackBox(object):
    def __init__(self, target_model, substitute_model, max_iter, seed_data,step_size,test_data,output_type=1, out_dim=10,
                 submodel_epoch=15,device='cpu'):
        '''
        :param target_model:
        :param max_iter: control the max iterations for substitute model's training
        :param output_type: 0,the target model outputs a label;1,the target model outputs a probability vector
        '''

        # OUTPUT TYPE
        self.OUT_LABEL = 0
        self.OUT_PVECTOR = 1

        self.max_iter = max_iter
        self.seed_data = seed_data
        self.test_data = test_data
        self.step=step_size
        self.output_type = output_type
        self.device = device
        self.target_model = target_model.to(self.device)
        self.target_model.eval()
        self.substitute_archi = substitute_model
        self.out_dim = out_dim
        self.sub_model_epoch = submodel_epoch
        self.substitute_model = None

        if not isinstance(self.seed_data, Dataset):
            raise Exception('Type of seed_data must be orch.utils.data.Dataset')

    def substitute_training(self):
        for iter in range(self.max_iter):
            '''NOTE: the substitute model is trained from scratch at each mimic model'''
            logging.info('Substitute Training-iter:{}'.format(iter))
            mirror_data = self.mimic_data(self.seed_data)
            if not self.substitute_model:
                del self.substitute_model # prevent out of memory
            self.substitute_model = copy.deepcopy(self.substitute_archi).to(self.device)
            logging.info('Mimic model on {} samples'.format(len(mirror_data)))
            acc = self.mimic_model(mirror_data)
            logging.info('Accuracy of mimic model:{}'.format(acc))
            self.seed_data = self.update_seed_data(mirror_data)
            torch.cuda.empty_cache()


    def mimic_data(self, seed_data):
        # batch_size = int(math.ceil(len(seed_data) / 10))
        batch_size = 1
        data_loader = DataLoader(dataset=seed_data, batch_size=batch_size)
        mirror_data = []
        mirror_labels = []
        for data, _ in data_loader:
            data = data.to(self.device)
            outputs = self.target_model(data)
            if self.output_type == self.OUT_PVECTOR:
                # outputs: (batch,output_dim)
                assert outputs.size()[0] == batch_size
                outputs = torch.argmax(outputs, dim=1)
            mirror_data.append(data.to('cpu'))
            mirror_labels.append(outputs.to('cpu'))

        return TensorDataset(torch.cat(mirror_data, 0), torch.cat(mirror_labels, 0))

    def mimic_model(self, train_set):
        train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True,num_workers=4)
        test_loader = DataLoader(dataset=self.test_data, batch_size=64, shuffle=True,num_workers=4)
        kwargs_train = {"model_name": 'my_model', 'epochs': self.sub_model_epoch}
        verbose_info = {"verbose": False, "log_interval": 10}
        optim = torch.optim.SGD(self.substitute_model.parameters(), lr=0.01, momentum=0.5)
        loss = nn.CrossEntropyLoss()
        accuracy = train(train_loader=train_loader, test_loader=test_loader, epochs=kwargs_train["epochs"],
                         model=self.substitute_model,
                         optimizer=optim, loss_function=loss,
                         device=self.device,
                         verbose=verbose_info)
        return accuracy

    def update_seed_data(self, mirror_data):
        dataloader = DataLoader(dataset=mirror_data, shuffle=True, batch_size=1)
        new_data_list = []
        new_data_targets = []
        for input, target in dataloader:
            input = input.to(self.device)
            jacobian = get_jacobian(input, self.substitute_model, self.out_dim, self.device)
            new_input = input + self.step * torch.sign(jacobian[:, target].view(*input.size()))
            new_data_list.append(new_input)
            new_data_targets.append(target)
        new_set = TensorDataset(torch.cat(new_data_list, 0), torch.cat(new_data_targets, 0))
        mirror_data = mirror_data.__add__(new_set)
        return mirror_data

    def do_craft(self, adversary, dataset, save_path=None,channels=1):

        data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)
        # only use the samples who can be correctly predicted by the substitute model
        dataset = exclude_wrong_labeled(self.substitute_model,dataset,device=self.device)
        adv_samples, y = adversary.do_craft_batch(DataLoader(
                                    dataset=dataset,
                                    shuffle=False, batch_size=64))

        adv_loader = DataLoader(dataset=TensorDataset(adv_samples, y), shuffle=False, batch_size=1)
        succeed_adv_samples = samples_filter(self.substitute_model, adv_loader, "Eps={}".format(0.35))

        premier_adv_set = datasetMutiIndx(TensorDataset(adv_samples, y), [idx for idx, _, _ in succeed_adv_samples])
        premier_adv_laoder = DataLoader(dataset=premier_adv_set)

        refined_adv_sampels = samples_filter(model=self.target_model, loader=premier_adv_laoder, name='source model',device=self.device)
        refined_adv_set = datasetMutiIndx(premier_adv_set, [idx for idx, _, _ in refined_adv_sampels])

        if save_path:
            save_imgs_tensor(refined_adv_set.tensors[0], refined_adv_set.tensors[1],
                             [adv_label for _, _, adv_label in refined_adv_sampels], save_path, file_prefix='bb',channels=channels)

        return refined_adv_set






