import sys
sys.path.append('../cifar10models/')
from jsma import JSMA
import torchvision
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from util.data_manger import normalize_mnist, random_seed, datasetMutiIndx
import math
import torch.nn as nn
from model_trainer import train
from jsma import get_jacobian
from models import *
import logging
from util.logging_util import setup_logging
from attack_util import *
from fgsm import *
import torch.nn as nn
import copy
from cifar10models.lenet import LeNet
from cifar10models.vgg import VGG



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
    def __init__(self, target_model, substitute_model, max_iter, seed_data, test_data, step, output_type=1, out_dim=10,
                 submodel_epoch=15,device='cpu'):
        '''
        :param target_model:
        :param max_iter:
        :param step: control the magnitude of the perturbation added to the seed samples
        :param output_type: 0,the target model outputs a label;1,the target model outputs a probability vector
        '''

        # OUTPUT TYPE
        self.OUT_LABEL = 0
        self.OUT_PVECTOR = 1

        self.max_iter = max_iter
        self.seed_data = seed_data
        self.test_data = test_data
        self.step = step
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



def craft_mnist():
    source_data = '../../datasets/mnist/raw'
    torch.manual_seed(random_seed)
    train_data,test_data = load_dataset(source_data, split=True)
    target_model = torch.load('../model-storage/mnist/hetero-base/MnistNet4.pkl')
    save_path = '../../datasets/mnist/adversarial/bb/single/non-pure/mnist4-fgsm35'
    target_type = TRUELABEL
    for step in range(1, 2):
        print('#########step:{}#########'.format(step * 0.1))
        subModel = ArchA()
        bb = BlackBox(target_model=target_model, substitute_model=subModel,
                      seed_data=datasetMutiIndx(test_data, range(150)),
                      test_data=datasetMutiIndx(test_data, range(150, 10000)),
                      max_iter=6, submodel_epoch=10, step=step * 0.1)
        bb.substitute_training()
        adversary = FGSM(bb.substitute_model, eps=0.35, target_type=target_type)
        succeed_adv_samples = bb.do_craft(adversary, ConcatDataset([train_data, test_data]), save_path=save_path)
        adv_laoder = DataLoader(dataset=succeed_adv_samples)
        samples_filter(model=bb.substitute_model, loader=adv_laoder, name='substitue model')
        samples_filter(model=target_model, loader=adv_laoder, name='source model')


def craft_cifar10(device):

    torch.manual_seed(random_seed)
    source_data = '../../datasets/cifar10/raw'
    target_model_name = 'googlenet.pkl'
    test_data,channel = load_data_set(data_type=DATA_CIFAR10,source_data=source_data,train=False)
    target_model = torch.load('../model-storage/cifar10/hetero-base/'+target_model_name)
    eps = 0.03
    adv_folder = target_model_name.split('.')[0]+'-eps-'+str(eps)
    save_path = '../../datasets/cifar10/adversarial-pure/bb/single/pure/'+adv_folder
    step = 0.3
    # subModel = VGG('VGG11')
    # subModel = ArchB()
    subModel = LeNet()
    seed_size = 200
    indices = np.arange(10000)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    bb = BlackBox(target_model=target_model, substitute_model=subModel,
                  seed_data=datasetMutiIndx(test_data, indices[:seed_size]),
                  test_data=datasetMutiIndx(test_data, indices[seed_size:]),
                  max_iter=4, submodel_epoch=100, step=step * 0.05,device=device)
    bb.substitute_training()
    model_save_name = target_model_name.split('.')[0] + '-lenet-mimic.pkl'
    torch.save(subModel,os.path.join(save_path, model_save_name))
    adversary = FGSM(bb.substitute_model, eps=eps,device=device)
    bb.do_craft(adversary, test_data, save_path=save_path,channels=3)
    logging.info('Black-Box Done!')



class gpu_test(nn.Module):
    def __init__(self):
        self.feature

if __name__ == '__main__':
    setup_logging()
    # print('00000')
    craft_cifar10(device='cuda:0')




