"""
This tutorial shows how to generate adversarial examples
using JSMA in white-box setting.
The original paper can be found at:
https://arxiv.org/abs/1511.07528
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")
import torch
from input_detect import utils
from mutation import MutationTest
from data_manger import *
from attack_util import load_natural_data
from models.googlenet import GoogLeNet
from models.lenet import MnistNet4
from baseline.ModelAdapter import *

FLAGS = flags.FLAGS

MAX_NUM_SAMPLES = 1000
def mutation_tutorial(datasets, attack_type, store_path, model_name, test_num=100, mutated=False):
    model_path = "../build-in-resource/pretrained-model/" + datasets + "/" + model_name + ".pkl"
    advDataPath = "../build-in-resource/dataset/" + datasets + "/adversarial/" + attack_type

    target_model = GoogLeNet() if model_name == "googlenet" else MnistNet4()

    target_model.load_state_dict(torch.load(model_path))
    target_model.eval()
    if datasets == "mnist":
        model_adapter = MnistNet4Adapter(target_model)
    else:
        model_adapter = Cifar10NetAdapter(target_model)

    if datasets == "mnist":
        data_path = "../build-in-resource/dataset/mnist/raw"
        train_data, _ = load_data_set(data_type=DATA_MNIST, source_data=data_path, train=True)
        test_data, _ = load_data_set(data_type=DATA_MNIST, source_data=data_path, train=False)
    else:
        data_path = "../build-in-resource/dataset/cifar10/raw"
        train_data, _ = load_data_set(data_type=DATA_CIFAR10, source_data=data_path, train=True)
        test_data, _ = load_data_set(data_type=DATA_CIFAR10, source_data=data_path, train=False)

    train_loader, test_loader = create_data_loader(
        batch_size=1,
        test_batch_size=1,
        train_data=train_data,
        test_data=test_data
    )

    if attack_type == "normal":
        print("load normal data.....")
        normal_data = load_natural_data(True, 0 if datasets == datasets else 1, data_path, use_train=True, seed_model=target_model, device='cpu', MAX_NUM_SAMPLES=MAX_NUM_SAMPLES)
        loader = DataLoader(dataset=normal_data)
    elif attack_type == "wl":
        print("load wl data.....")
        wl_data = load_natural_data(False, 0 if datasets == datasets else 1, data_path, use_train=True, seed_model=target_model, device='cpu', MAX_NUM_SAMPLES=MAX_NUM_SAMPLES)
        loader = DataLoader(dataset=wl_data)
    else:
        advDataPath = "../build-in-resource/dataset/" + datasets + "/adversarial/" + attack_type  # under lenet
        loader = get_data_loader(advDataPath, is_adv_data=True, data_type=datasets)

    # Generate random matution matrix for mutations
    store_path = store_path + attack + '/' + datasets + '/'
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    result = ''

    if datasets == 'mnist':
        img_rows = 28
        img_cols = 28
        mutation_test = MutationTest(img_rows, img_cols)
        mutation_test.mutation_generate(mutated, store_path, utils.generate_value_1)
    elif datasets == 'cifar10' or datasets == 'svhn':
        img_rows = 32
        img_cols = 32
        mutation_test = MutationTest(img_rows, img_cols)
        mutation_test.mutation_generate(mutated, store_path, utils.generate_value_3)

    store_string, result = mutation_test.mutation_test_adv(loader, attack_type, result, test_num, model_adapter)

    with open(store_path + "/" + attack_type + "_result.csv", "w") as f:
        f.write(store_string)

    # store_string, result = mutation_test.mutation_test_ori(normal_loader, result, test_num, model_adapter)
    #
    # with open(store_path + "/ori_result.csv", "w") as f:
    #     f.write(store_string)

    with open(store_path + "/result.csv", "w") as f:
        f.write(result)

    print('Finish.')


def main(argv=None):
    mutation_tutorial(datasets=FLAGS.datasets,
                      attack_type=FLAGS.attack,
                      store_path=FLAGS.store_path,
                      model_name=FLAGS.model_name,
                      test_num=FLAGS.test_num,
                      mutated=FLAGS.mutated)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('attack', 'jsma', 'The type of generating adversaries')
    flags.DEFINE_boolean('mutated', False, 'The mutation list is already generate.')  # default:False
    flags.DEFINE_string('model_name', 'lenet4','name of the model')
    flags.DEFINE_string('store_path', '../results/', 'The path to store results.')
    flags.DEFINE_integer('test_num', 100, 'Number of mutation test targets')

    tf.app.run()
