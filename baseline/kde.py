from __future__ import division, absolute_import, print_function
import sys
sys.path.append("../")
from sklearn.neighbors import KernelDensity
from models.googlenet import GoogLeNet
from models.lenet import MnistNet4
from baseline.ModelAdapter import *
from utils.data_manger import *
import torch
from collections import defaultdict
from detect.adv_detect import get_data_loader
from sklearn import metrics

BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26, 'svhn': 1.00}
def detect_kde(model_name):
    #############
    #load model
    #############
    data_path = ""
    model_path = "../build-in-resource/pretrained-model/mnist/lenet.pkl"
    dataType = "mnist"
    advDataPath = "../build-in-resource/dataset/mnist/adversarial/jsma/"

    target_model = GoogLeNet() if model_name == "googlenet" else MnistNet4()
    target_model.load_state_dict(torch.load(model_path))
    target_model.eval()
    model_adapter = MnistNet4Adapter(target_model)

    #########################################################
    #load data. use training data to denote the submanifold.
    #########################################################
    train_data, _ = load_data_set(data_type=DATA_MNIST,source_data=data_path,train=True)
    test_data, _ = load_data_set(data_type=DATA_MNIST,source_data=data_path,train=False)
    adv_data_loader = get_data_loader(advDataPath, is_adv_data=True, data_type=dataType)
    train_loader, test_loader = create_data_loader(
        batch_size=64,
        test_batch_size=1000,
        train_data=train_data,
        test_data=test_data
    )

    last_layer_output=defaultdict(list)
    for x, label in train_loader:
        last_layer_output[label].append(model_adapter.last_hd_layer_output(x))
    kdes = {}
    for key in last_layer_output.keys():
        kdes[key] = KernelDensity(kernel='gaussian',
                                bandwidth=BANDWIDTHS[dataType]) \
            .fit(last_layer_output[key])

    kde_score = defaultdict(list)
    ##########################
    # test normal. negative
    ##########################
    for x_test,y_text in test_loader:
        pred,h = model_adapter.get_predict_lasth(x_test)
        score = kdes[pred].score_samples(np.reshape(x, (1, -1)))[0]
        kde_score[0].append(score)

    ##########################
    # test normal. positive
    ##########################
    for x_adv,y_adv in adv_data_loader:
        pred,h = model_adapter.get_predict_lasth(x_adv)
        score = kdes[pred].score_samples(np.reshape(x_adv, (1, -1)))[0]
        kde_score[1].append(score)
    y_score = kde_score[0] + kde_score[1]
    y_label = [0]*len(kde_score[0]) + [1]*len(kde_score[1])
    fpr, tpr, thresholds = metrics.roc_curve(y_label, y_score)
    auc_score = metrics.auc(fpr, tpr)
    print('Detector ROC-AUC score: {:.4f}'.format(auc_score))


def detect_uncerts():
    pass

def detect_combine():
    pass









#
#
# def detect_kde():
#     pass
#
#
#
# def getBayesianUncertainty(model,normal_data,adv_data):
#
#     ## Get Bayesian uncertainty scores
#     print('Getting Monte Carlo dropout variance predictions...')
#     uncerts_normal = get_mc_predictions(model, normal_data,
#                                         batch_size=args.batch_size) \
#         .var(axis=0).mean(axis=1)
#     uncerts_adv = get_mc_predictions(model, adv_data,
#                                      batch_size=args.batch_size) \
#         .var(axis=0).mean(axis=1)
#
#     return uncerts_normal,uncerts_adv
#
#
# def getKDE():
#     ## Get KDE scores
#     # Get deep feature representations
#     print('Getting deep feature representations...')
#     X_train_features = get_deep_representations(model, X_train,
#                                                 batch_size=args.batch_size)
#     X_test_normal_features = get_deep_representations(model, X_test,
#                                                       batch_size=args.batch_size)
#     X_test_noisy_features = get_deep_representations(model, X_test_noisy,
#                                                      batch_size=args.batch_size)
#     X_test_adv_features = get_deep_representations(model, X_test_adv,
#                                                    batch_size=args.batch_size)
#
#     # Train one KDE per class
#     print('Training KDEs...')
#     class_inds = {}
#     for i in range(Y_train.shape[1]):
#         class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
#     kdes = {}
#     warnings.warn("Using pre-set kernel bandwidths that were determined "
#                   "optimal for the specific CNN models of the paper. If you've "
#                   "changed your model, you'll need to re-optimize the "
#                   "bandwidth.")
#     for i in range(Y_train.shape[1]):
#         kdes[i] = KernelDensity(kernel='gaussian',
#                                 bandwidth=BANDWIDTHS[args.dataset]) \
#             .fit(X_train_features[class_inds[i]])
#
#
#
#
# def main(args):
#
#     print('Loading the data and model...')
#     # Load the model
#     model = load_model('../data/model_%s.h5' % args.dataset)
#     # Load the dataset
#     X_train, Y_train, X_test, Y_test = get_data(args.dataset)
#     # Check attack type, select adversarial and noisy samples accordingly
#     print('Loading noisy and adversarial samples...')
#     if args.attack == 'all':
#         # TODO: implement 'all' option
#         #X_test_adv = ...
#         #X_test_noisy = ...
#         raise NotImplementedError("'All' types detector not yet implemented.")
#     else:
#         # Load adversarial samples
#         X_test_adv = np.load('../data/Adv_%s_%s.npy' % (args.dataset,
#                                                         args.attack))
#         # Craft an equal number of noisy samples
#         X_test_noisy = get_noisy_samples(X_test, X_test_adv, args.dataset,
#                                          args.attack)
#
#     # Refine the normal, noisy and adversarial sets to only include samples for
#     # which the original version was correctly classified by the model
#     preds_test = model.predict_classes(X_test, verbose=0,
#                                        batch_size=args.batch_size)
#     inds_correct = np.where(preds_test == Y_test.argmax(axis=1))[0]
#     X_test = X_test[inds_correct]
#     X_test_noisy = X_test_noisy[inds_correct]
#     X_test_adv = X_test_adv[inds_correct]
#
#
#
#
#
#     # Get model predictions
#     print('Computing model predictions...')
#     preds_test_normal = model.predict_classes(X_test, verbose=0,
#                                               batch_size=args.batch_size)
#     preds_test_noisy = model.predict_classes(X_test_noisy, verbose=0,
#                                              batch_size=args.batch_size)
#     preds_test_adv = model.predict_classes(X_test_adv, verbose=0,
#                                            batch_size=args.batch_size)
#     # Get density estimates
#     print('computing densities...')
#     densities_normal = score_samples(
#         kdes,
#         X_test_normal_features,
#         preds_test_normal
#     )
#     densities_adv = score_samples(
#         kdes,
#         X_test_adv_features,
#         preds_test_adv
#     )
#
#     ## Z-score the uncertainty and density values
#     uncerts_normal_z, uncerts_adv_z, uncerts_noisy_z = normalize(
#         uncerts_normal,
#         uncerts_adv,
#         uncerts_noisy
#     )
#     densities_normal_z, densities_adv_z, densities_noisy_z = normalize(
#         densities_normal,
#         densities_adv,
#         densities_noisy
#     )
#
#     ## Build detector
#     values, labels, lr = train_lr(
#         densities_pos=densities_adv_z,
#         densities_neg=np.concatenate((densities_normal_z, densities_noisy_z)),
#         uncerts_pos=uncerts_adv_z,
#         uncerts_neg=np.concatenate((uncerts_normal_z, uncerts_noisy_z))
#     )
#
#     ## Evaluate detector
#     # Compute logistic regression model predictions
#     probs = lr.predict_proba(values)[:, 1]
#     # Compute AUC
#     n_samples = len(X_test)
#     # The first 2/3 of 'probs' is the negative class (normal and noisy samples),
#     # and the last 1/3 is the positive class (adversarial samples).
#     _, _, auc_score = compute_roc(
#         probs_neg=probs[:2 * n_samples],
#         probs_pos=probs[2 * n_samples:]
#     )
#     print('Detector ROC-AUC score: %0.4f' % auc_score)

#
# if __name__ == "__main__":
#     main("lenet")
