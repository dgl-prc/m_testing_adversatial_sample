from __future__ import print_function
import sys
sys.path.append('./*')
sys.path.append('./cifar10models')
from models import *
from utils.data_manger import *
import os

from models.googlenet import *
from models.lenet import MnistNet4
import time
<<<<<<< HEAD
from models.temp_lenet import *
=======
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3

'''
Todo List:
1. make optim and loss beacome optional

Given that different model may have different optimizer and loss function, we config these parameters in
each train process.
'''


def __kernel_trainer(args, model, train_loader, optimizer, epoch, loss_f=F.nll_loss,device='cpu'):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        if args["verbose"]:
            if batch_idx % args["log_interval"] == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()


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
            if verbose:
                sys.stdout.write('\r progress:{:.2f}%'.format((1.*batch_size*progress*100)/data_size))

    test_loss /= len(test_loader.dataset)
    acc = 1. * correct / len(test_loader.dataset)

    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%) time:{:.6f}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100 * acc, np.average(time_count)))
    return acc


def train(train_loader, test_loader, epochs, model, optimizer, loss_function, verbose,device='cpu',stop_precision=1e-5):
    last_loss = 0
    accuracy = 0.0
    for epoch in range(1, epochs + 1):
        loss = __kernel_trainer(verbose, model, train_loader, optimizer, epoch, loss_function,device=device)
        if test_loader:
            accuracy = test(model, test_loader, verbose["verbose"],device=device)
        if last_loss==0:
            last_loss = loss
        else:
            if abs(loss-last_loss) < stop_precision:
                break
            else:
                last_loss = loss
    return accuracy




#######################
# Example
#######################
def train_ad_hoc_model_example(kwargs_train, source_data, save_info={"save": False, "save_path": './'},
                               verbose_info={"verbose": True, "log_interval": 10}):
    '''
    This is an example about how to train a model use these artifacts
    1. load data
    2. create data loader
    3. create model
    3. create optimizer
    4. create loss function
    5. train
    6. save model
    :return:
    '''

    print("------------Train " + kwargs_train["model_name"] + " model------------------")
    model = MnistNet4()
    # train_data, test_data = load_dataset(data_path, split=True)
    train_data, _ = load_data_set(data_type=DATA_MNIST,source_data=source_data,train=True)
    test_data, _ = load_data_set(data_type=DATA_MNIST,source_data=source_data,train=False)
    train_loader, test_loader = create_data_loader(
        batch_size=64,
        test_batch_size=1000,
        train_data=train_data,
        test_data=test_data
    )
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss = nn.CrossEntropyLoss()
    train(train_loader=train_loader, test_loader=test_loader, epochs=kwargs_train["epochs"], model=model,
          optimizer=optim, loss_function=loss,
          verbose=verbose_info)
    if save_info.save:
        if not os.path.exists(save_info.save_path):
            os.makedirs(save_info.save_path)
        model_name = kwargs_train["model_name"] + '.pkl'
        path = os.path.join(save_info.save_path, model_name)
        torch.save(model, path)

    print("------------Train End------------------")

##############
# Train
##############

def train_mnist_model(kwargs_train, data_path, save_info={"save": False, "save_path": './'},
                      verbose_info={"verbose": True, "log_interval": 10}):
    '''
    This is an example about how to train a model use these artifacts
    1. load data
    2. create data loader
    3. create model
    3. create optimizer
    4. create loss function
    5. train
    6. save model
    :return:
    '''
    model_name = kwargs_train["model_name"]

    # if model_name == 'MnistNet1':
    #     model = MnistNet1()
    # elif model_name == 'MnistNet2':
    #     model = MnistNet2()
    # elif model_name == 'MnistNet3':
    #     model = MnistNet3()
    # elif model_name == 'MnistNet4':
    #     model = MnistNet4()
    # elif model_name == 'MnistNet5':
    #     model = MnistNet5()
<<<<<<< HEAD
    # model = MnistNet4()
    model = JingyiNet()
=======
    model = MnistNet4()

>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3
    print("------------Train " + kwargs_train["model_name"] + " model------------------")
    train_data, _ = load_data_set(data_type=DATA_MNIST,source_data=data_path,train=True)
    test_data, _ = load_data_set(data_type=DATA_MNIST,source_data=data_path,train=False)
    train_loader, test_loader = create_data_loader(
        batch_size=64,
        test_batch_size=1000,
        train_data=train_data,
        test_data=test_data
    )
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loss = nn.CrossEntropyLoss()
    train(train_loader=train_loader, test_loader=test_loader, epochs=kwargs_train["epochs"], model=model,
          optimizer=optim, loss_function=loss,
          verbose=verbose_info)
    if save_info["save"]:
        if not os.path.exists(save_info["save_path"]):
            os.makedirs(save_info["save_path"])
        model_name = kwargs_train["model_name"] + '.pkl'
        path = os.path.join(save_info["save_path"], model_name)
<<<<<<< HEAD
        torch.save(model.state_dict(), path)
=======
        torch.save(model, path)
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3

    print("------------Train End------------------")


def train_cifar_model(kwargs_train, data_path, save_info={"save": False, "save_path": './'},
                      verbose_info={"verbose": True, "log_interval": 10}):
    model_name = kwargs_train["model_name"]

    # if model_name == 'vgg':
    #     model = VGG('VGG16')
    # elif model_name == 'googlenet':
    #     model = GoogLeNet()
    # elif model_name == 'resnet':
    #     model = ResNet18()
    # elif model_name == 'densenet':
    #     model = DenseNet121()
    # elif model_name == 'lenet':
    #     model = LeNet()
    model = GoogLeNet()

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model,2)

    print("------------Train " + kwargs_train["model_name"] + " model------------------")

    train_data, tets_data = load_cifar10(data_path)
    train_loader, test_loader = create_data_loader(batch_size=128,
                                                   test_batch_size=100,
                                                   train_data=train_data,
                                                   test_data=tets_data)

    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=5e-4)
    loss = nn.CrossEntropyLoss()
    train(train_loader=train_loader, test_loader=test_loader, epochs=kwargs_train["epochs"], model=model,
          optimizer=optim, loss_function=loss,
          verbose=verbose_info,device=device,stop_precision=1e-6)

    if save_info["save"]:
        if not os.path.exists(save_info["save_path"]):
            os.makedirs(save_info["save_path"])
        model_name = kwargs_train["model_name"] + '.pkl'
        path = os.path.join(save_info["save_path"], model_name)
        torch.save(model, path)

    print("------------Train End------------------")


def train_mnist_disjoint_data():
    '''
    We use MnistNet4 as the base model and train it on 5 disjoint datasets,each with 12000 samples.

    1. split datasets p1,p2,p3,p4,p5
    2. train 5 instance model
    :return:
    '''
    seed = 5599
    torch.manual_seed(seed)
    data_path='../datasets/mnist/raw'
    ori_train_data, _ = load_data_set(data_type=DATA_MNIST,source_data=data_path,train=True)
    test_data, _ = load_data_set(data_type=DATA_MNIST,source_data=data_path,train=False)

    rState = np.random.RandomState(seed)
    size = len(ori_train_data)
    idx = np.arange(size)
    rState.shuffle(idx)
    p1, p2, p3, p4, p5 = np.split(idx,5)

    verbose_info = {"verbose": True, "log_interval": 10}
    save_info = {"save": True, "save_path": './model-storage/mnist/home-base'}
    kwargs_train = {"model_name": 'my_model', 'epochs': 8}
    for i, p in enumerate([p1, p2, p3, p4, p5 ]):
        kwargs_train['model_name'] = 'MnistNet4-p{}'.format(i+1)
        model = MnistNet4()
        train_data = datasetMutiIndx(ori_train_data,p)
        train_loader, test_loader = create_data_loader(
            batch_size=64,
            test_batch_size=1000,
            train_data=train_data,
            test_data=test_data
        )
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        loss = nn.CrossEntropyLoss()
        train(train_loader=train_loader, test_loader=test_loader, epochs=kwargs_train["epochs"], model=model,
              optimizer=optim, loss_function=loss,
              verbose=verbose_info)
        if save_info["save"]:
            if not os.path.exists(save_info["save_path"]):
                os.makedirs(save_info["save_path"])
            model_name = kwargs_train["model_name"] + '.pkl'
            path = os.path.join(save_info["save_path"], model_name)
            torch.save(model, path)

    print("------------Train End------------------")


#############
# Test
#############

def main():
<<<<<<< HEAD
    kwargs_train = {"model_name": 'my_model', 'epochs': 1}
    verbose_info = {"verbose": True, "log_interval": 10}
    data_path = '../build-in-resource/dataset/mnist/raw/'
    save_info = {"save": True, "save_path": './'}
    model_list = ['MnistNet4']

    # save_info = {"save": True, "save_path": './model-storage/cifar10/hetero-base/'}
    # data_path = '../datasets/cifar10'
    # model_list = ['vgg', 'lenet', 'googlenet', 'resnet', 'densenet']
    # model_list = ['resnet', 'densenet']

    for model_name in model_list:
        start = time.clock()
        kwargs_train['model_name'] = model_name
        train_mnist_model(kwargs_train=kwargs_train, data_path=data_path, save_info=save_info,
                                   verbose_info=verbose_info)
        # train_cifar_model(kwargs_train, data_path, save_info, verbose_info)
=======
    kwargs_train = {"model_name": 'my_model', 'epochs': 100}
    verbose_info = {"verbose": True, "log_interval": 10}
    # data_path = '../datasets/mnist/raw'
    # save_info = {"save": False, "save_path": './model-storage/mnist/hetero-base/'}
    # model_list = ['MnistNet1', 'MnistNet2', 'MnistNet3', 'MnistNet4', 'MnistNet5']

    save_info = {"save": True, "save_path": './model-storage/cifar10/hetero-base/'}
    data_path = '../datasets/cifar10'
    # model_list = ['vgg', 'lenet', 'googlenet', 'resnet', 'densenet']
    model_list = ['resnet', 'densenet']

    for model_name in model_list:
        kwargs_train['model_name'] = model_name
        # train_mnist_model(kwargs_train=kwargs_train, data_path=data_path, save_info=save_info,
        #                            verbose_info=verbose_info)
        start = time.clock()
        train_cifar_model(kwargs_train, data_path, save_info, verbose_info)
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3
        print({'total time:{} min'.format((time.clock() - start) / 60.)})


if __name__ == '__main__':
    main()


