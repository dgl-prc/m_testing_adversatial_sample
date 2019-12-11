from __future__ import division, absolute_import, print_function
import sys
sys.path.append("../")
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import scale
from models.googlenet import GoogLeNet
from models.lenet import MnistNet4
from baseline.ModelAdapter import *
from utils.data_manger import *
import torch
from collections import defaultdict
from detect.adv_detect import get_data_loader
from sklearn import metrics
from attacks.attack_util import load_natural_data
from utils.data_manger import *
from models.temp_lenet import JingyiNet
from sklearn.linear_model import LogisticRegressionCV

BANDWIDTHS = {'mnist': 1.2, 'cifar': 0.26, 'svhn': 1.00}
MAX_NUM_SAMPLES = 1000 #follow the same setting with lcr.


def load_data_model(model_name, dataType, attack_type):

    print("======={}======{}======={}=======".format(model_name, dataType, attack_type))
    #############
    # load model
    #############
    model_path = "../build-in-resource/pretrained-model/" + dataType + "/lenet.pkl"
    # model_path = "../utils/MnistNet4.pkl"
    advDataPath = "../build-in-resource/dataset/" + dataType + "/adversarial/" + attack_type  # under lenet

    ###################
    # training data
    ###################
    if dataType == "mnist":
        data_path = "../build-in-resource/dataset/mnist/raw"
    else:
        data_path = "../build-in-resource/dataset/cifar10/raw"

    print("load model.....")
    target_model = GoogLeNet() if model_name == "googlenet" else MnistNet4()
    # target_model = JingyiNet()
    target_model.load_state_dict(torch.load(model_path))
    target_model.eval()
    model_adapter = MnistNet4Adapter(target_model)
    # model_adapter = JingyiNetAdapter(target_model)


    print("load train data.....")
    #########################################################
    # load data. use training data to denote the submanifold.
    #########################################################
    train_data, _ = load_data_set(data_type=DATA_MNIST, source_data=data_path, train=True)
    test_data, _ = load_data_set(data_type=DATA_MNIST, source_data=data_path, train=False)
    adv_data_loader = get_data_loader(advDataPath, is_adv_data=True, data_type=dataType)
    train_loader, test_loader = create_data_loader(
        batch_size=1,
        test_batch_size=1,
        train_data=train_data,
        test_data=test_data
    )

    print("load normal data.....")
    normal_data = load_natural_data(True, 0 if dataType == "mnist" else 1, data_path, use_train=True, seed_model=target_model, device='cpu', MAX_NUM_SAMPLES=MAX_NUM_SAMPLES)
    normal_loader = DataLoader(dataset=normal_data)

    print("load wl data.....")
    wl_data = load_natural_data(False, 0 if dataType == "mnist" else 1, data_path, use_train=True, seed_model=target_model, device='cpu', MAX_NUM_SAMPLES=MAX_NUM_SAMPLES)
    wl_loader = DataLoader(dataset=wl_data)


    ################
    # test model
    ################
    print("check model.....")
    acc = 0
    i = 0
    for x_test, y_text in test_loader:

        pred, h = model_adapter.get_predict_lasth(x_test)
        acc += 1 if pred == y_text.item() else 0
        i += 1
        print("progress i={}".format(i))
    print("model acc:{:.4f}".format(acc / len(test_loader)))

    #########################
    # test selected normal
    #########################
    print("check normal.....")
    acc = 0
    i = 0
    for x_normal, y_norma in normal_loader:
        pred, h = model_adapter.get_predict_lasth(x_normal)
        acc += 1 if pred == y_norma.item() else 0
        i += 1
        print("progress i={}".format(i))
    print("expected: 1.0, valid normal acc:{:.4f}".format(acc / len(normal_loader)))

    ################
    # test adv data
    ################
    print("check adv.....")
    acc = 0
    adv_acc = 0
    i = 0
    for x_adv, y_true,y_adv in adv_data_loader:
        pred, h = model_adapter.get_predict_lasth(x_adv)
        acc += 1 if pred == y_true.item() else 0
        adv_acc +=1 if pred == y_adv.item() else 0
        i += 1
        print("progress i={}".format(i))
    print("acc:{:.4f}, adv_acc:{:.4f}".format(acc / len(adv_data_loader), adv_acc / len(adv_data_loader)))

    ################
    # test wl data
    ################
    acc = 0
    adv_acc = 0
    for x_adv, y_true in wl_loader:
        pred, h = model_adapter.get_predict_lasth(x_adv)
        acc += 1 if pred == y_true.item() else 0
    print("acc:{:.4f}".format(acc / len(wl_loader)))

    return train_loader,normal_loader,adv_data_loader, wl_loader, model_adapter


def get_uncerts(data_loader,iters,model_adapter):
    uncerts = []
    for item in data_loader:
        x_test = item[0]
        uncert = []
        for i in range(iters):
            h = model_adapter.get_dropout_ouput(x_test)
            h = torch.squeeze(h).detach().numpy().tolist()
            uncert.append(h)
        uncert = np.asarray(uncert)
        uncert = uncert.var(axis=0)
        uncert = uncert.mean(axis=0)
        uncerts.append(uncert)
    return uncerts

def get_last_hidden_layer(train_loader,model_adapter):
    last_layer_output = defaultdict(list)
    for x, label in train_loader:
        last_h_layer = torch.squeeze(model_adapter.last_hd_layer_output(x)).detach().numpy()
        last_layer_output[label.item()].append(last_h_layer)
    return last_layer_output

def get_kde_score(data_loader,model_adapter,kdes):
    kde_score = []
    for item in data_loader:
        x_test = item[0]
        pred, h = model_adapter.get_predict_lasth(x_test)
        h = np.reshape(torch.squeeze(h).detach().numpy(), (1, -1))
        score = kdes[pred].score_samples(h)[0]
        kde_score.append(score)
    return kde_score

def detect_uncerts(model_name="lenet",dataType = "mnist",attack_type = "jsma",p=0.5,iters=50):

    train_loader, test_loader, adv_data_loader, model_adapter = load_data_model(model_name, dataType, attack_type)
    model_adapter.model.train() # open dropout
    #########################################
    # calculate the uncertainty of models
    #########################################
    print("calculating uncert of  normal...")
    normal_uncerts = get_uncerts(test_loader,iters,model_adapter)

    print("calculating uncert of adversarial...")
    adv_uncerts = get_uncerts(adv_data_loader,iters,model_adapter)

    y_score = normal_uncerts+adv_uncerts
    y_label = [0]*len(normal_uncerts)+[1]*len(adv_uncerts)
    fpr, tpr, thresholds = metrics.roc_curve(y_label, y_score)
    auc_score = metrics.auc(fpr, tpr)
    print('Detector ROC-AUC score: {:.4f}'.format(auc_score))


def detect_kde(model_name="lenet",dataType = "mnist",attack_type = "jsma"):

    train_loader, test_loader, adv_data_loader, wl_loader, model_adapter = load_data_model(model_name, dataType, attack_type)

    last_layer_output = get_last_hidden_layer(train_loader,model_adapter)

    kdes = {}
    print("getting kde models...")
    for key in last_layer_output.keys():
        kdes[key] = KernelDensity(kernel='gaussian',
                                  bandwidth=BANDWIDTHS[dataType]) \
            .fit(last_layer_output[key])

    kde_score = defaultdict(list)
    ####################################################################################
    # test normal. positive. since density estimate of normal sample is typically bigger.
    #####################################################################################
    print("getting kde score of normal...")
    print("================ked-normal=================")
    kde_score["normal"] = get_kde_score(test_loader,model_adapter,kdes)
    print(kde_score["normal"])
    ########################################################################
    # test adv. negative. density estimate of adversarial sample is smaller.
    ########################################################################
    print("getting kde score of adversarial...")
    kde_score["adv"] = get_kde_score(adv_data_loader,model_adapter,kdes)
    print("================ked-adv=================")
    print(kde_score["adv"])

    y_score = kde_score["adv"] + kde_score["normal"]
    y_label = [0] * len(kde_score["adv"]) + [1] * len(kde_score["normal"])
    fpr, tpr, thresholds = metrics.roc_curve(y_label, y_score)
    auc_score = metrics.auc(fpr, tpr)
    print('{}------>Detector ROC-AUC score: {:.4f}'.format(attack,auc_score))

    ##################################
    # wl data
    ##################################
    print("getting kde score of wl...")
    kde_score["wl"] = get_kde_score(wl_loader,model_adapter,kdes)
    print("================ked-adv=================")
    print(kde_score["wl"])

    normal_kde,w_kde = normalize(kde_score["normal"],kde_score["wl"] )
    y_score = w_kde.tolist() + normal_kde.tolist()
    y_label = [0] * len(kde_score["wl"]) + [1] * len(kde_score["normal"])
    fpr, tpr, thresholds = metrics.roc_curve(y_label, y_score)
    auc_score = metrics.auc(fpr, tpr)
    print('wl------>Detector ROC-AUC score: {:.4f}'.format(auc_score))

def normalize(normal, adv):
    """
    On effects on the result.!!!!!!!
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv)))
    return total[:n_samples], total[n_samples:]


def train_lr(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr

def logistic_regression(model_name="lenet",dataType = "mnist",attack_type = "jsma",p=0.5,iters=50):

    train_loader, normal_loader, adv_data_loader, wl_loader, model_adapter = load_data_model(model_name, dataType, attack_type)
    model_adapter.model.train()  # open dropout
    #########################################
    # calculate the uncertainty of models
    #########################################
    print("calculating uncert of  normal...")
    normal_uncerts = get_uncerts(normal_loader,iters,model_adapter)

    print("calculating uncert of adversarial...")
    adv_uncerts = get_uncerts(adv_data_loader,iters,model_adapter)

    #########################################
    # calculate the kde
    #########################################
    last_layer_output = get_last_hidden_layer(train_loader, model_adapter)
    kdes = {}
    print("getting kde models...")
    for key in last_layer_output.keys():
        kdes[key] = KernelDensity(kernel='gaussian',
                                  bandwidth=BANDWIDTHS[dataType]) \
            .fit(last_layer_output[key])

    kde_score = defaultdict(list)
    kde_score["normal"] = get_kde_score(normal_loader, model_adapter, kdes)
    kde_score["adv"] = get_kde_score(adv_data_loader, model_adapter, kdes)

    normal_uncerts, adv_uncerts = normalize(normal_uncerts,adv_uncerts)
    normal_kde, adv_kde = normalize(kde_score["normal"], kde_score["adv"])
    values, labels, lr = train_lr(
            densities_pos=adv_kde,
            densities_neg=normal_kde,
            uncerts_pos=adv_uncerts,
            uncerts_neg=normal_uncerts
        )

    probs = lr.predict_proba(values)[:, 1]
    probs_neg =probs[:len(normal_uncerts)]
    probs_pos =probs[len(normal_uncerts):]

    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    auc_score = metrics.auc(fpr, tpr)
    print('LR------>Detector ROC-AUC score: {:.4f}'.format(auc_score))



if __name__ == "__main__":
     # for attack in ["bb","cw","deepfool","jsma","fgsm"]:
     for attack in ["fgsm"]:
        # detect_kde(model_name="lenet",dataType="mnist",attack_type=attack)
        # detect_uncerts(model_name="lenet",dataType="mnist",attack_type=attack)
        logistic_regression(model_name="lenet", dataType="mnist", attack_type="jsma", p=0.5, iters=50)