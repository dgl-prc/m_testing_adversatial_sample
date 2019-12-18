from __future__ import print_function

import sys
sys.path.append("../")
import math
import random

import numpy as np
import torch

from input_detect.utils import c_occl, c_light, c_black

attack_dict = ["fgsm", "jsma", "cw", "df", "bb"]
class MutationTest:

    '''
        Mutation testing for the training dataset
        :param img_rows:
        :param img_cols:
        :param seed_number:
        :param mutation_number:
    '''

    img_rows = 28
    img_cols = 28
    seed_number = 500
    mutation_number = 1000
    mutations = []
    level = 1

    # def __init__(self, img_rows, img_cols, seed_number, mutation_number, level):
    #     self.img_rows = img_rows
    #     self.img_cols = img_cols
    #     self.seed_number = seed_number
    #     self.mutation_number = mutation_number
    #     self.level = level

    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def mutation_matrix(self, generate_value):

        method = random.randint(1, 3)
        trans_matrix = generate_value(self.img_rows, self.img_cols)
        rect_shape = (random.randint(1, 3), random.randint(1, 3))
        start_point = (
            random.randint(0, self.img_rows - rect_shape[0]),
            random.randint(0, self.img_cols - rect_shape[1]))

        if method == 1:
            transformation = c_light(trans_matrix)
        elif method == 2:
            transformation = c_occl(trans_matrix, start_point, rect_shape)
        elif method == 3:
            transformation = c_black(trans_matrix, start_point, rect_shape)

        return np.asarray(transformation[0])
        # trans_matrix = generate_value(self.img_rows, self.img_cols, self.level)
        # return np.asarray(trans_matrix)

    def mutation_generate(self, mutated, path, generate_value):
        if mutated:
            self.mutations = np.load(path + "/mutation_list.npy")
        else:
            for i in range(self.mutation_number):
                mutation = self.mutation_matrix(generate_value)
                self.mutations.append(mutation)
            np.save(path + "/mutation_list.npy", self.mutations)

    def mutation_test_adv(self, data, attack, result, test_num, model_adapter):
        store_string = ''
        label_change_numbers = []

        # Iterate over all the test data
        # count = 0
        i = 0
        for item in data:
            if attack not in attack_dict:
                x, _ = item
            else:
                x, _, _ = item
            if i >= test_num:
                break
            orig_label,_ = model_adapter.get_predict_lasth(x)
            label_changes = 0
            ori_img = x.numpy()
            mu_labels = []
            rgb = False
            if self.mutations.shape[-1] != 1:
                rgb = True
            for j in range(self.mutation_number):
                t_img = ori_img.copy()
                if rgb:
                    mutation_matrix = (self.mutations[j].reshape((1, 32, 32, 3)).astype(np.float32) - [0.4914, 0.4822,0.4465]) / [0.247, 0.243, 0.261] * self.level
                    mutation_img = t_img + np.transpose(mutation_matrix, [0, 3, 1, 2])
                else:
                    mutation_img = t_img + (self.mutations[j].reshape((1, 1, 28, 28)).astype(np.float32) - 0.1307)/ 0.3081 * self.level  # mnist(1,1,28,28)

                # mutation_img = torch.from_numpy((mutation_img - 0.1307) / 0.3081)
                mutation_img = torch.from_numpy(mutation_img)
                mu_label, _ = model_adapter.get_predict_lasth(mutation_img)

                mu_labels.append(mu_label)
            for mu_label in mu_labels:
                if mu_label != int(orig_label):
                    label_changes += 1

            label_change_numbers.append(label_changes)
            # pxzhang
            store_string = store_string + str(i) + "," + str(orig_label) + "," + str(label_changes) + "\n"
            i+=1

        label_change_numbers = np.asarray(label_change_numbers)
        adv_average = round(np.mean(label_change_numbers), 2)
        adv_std = np.std(label_change_numbers)
        adv_99ci = round(2.576 * adv_std / math.sqrt(len(label_change_numbers)), 2)
        result = result + attack + ',' + str(adv_average) + ',' + str(round(adv_std, 2)) + ',' + str(adv_99ci) + '\n'

        return store_string, result

    # def mutation_test_ori(self, data, result, test_num, model_adapter):
    #     store_string = ''
    #     label_change_numbers = []
    #     # Iterate over all the test data
    #     i = 0
    #     for x, y in data:
    #         if i >= test_num:
    #             break
    #         orig_label,_ = model_adapter.get_predict_lasth(x)
    #         label_changes = 0
    #         ori_img = x.numpy()
    #         mu_labels = []
    #         for j in range(self.mutation_number):
    #             t_img = ori_img.copy()
    #             mutation_img = t_img + (self.mutations[j].reshape((1, 1, 28, 28)).astype(
    #                 np.float32) - 0.1307) / 0.3081 * self.level  # mnist(1,1,28,28)
    #
    #             # mutation_img = torch.from_numpy((mutation_img - 0.1307) / 0.3081)
    #             mutation_img = torch.from_numpy(mutation_img)
    #             mu_label, _ = model_adapter.get_predict_lasth(mutation_img)
    #
    #             mu_labels.append(mu_label)
    #         for mu_label in mu_labels:
    #             if mu_label != int(orig_label):
    #                 label_changes += 1
    #
    #         label_change_numbers.append(label_changes)
    #         # pxzhang
    #         store_string = store_string + str(i) + "," + str(orig_label) + "," + str(label_changes) + "\n"
    #         i += 1
    #
    #     label_change_numbers = np.asarray(label_change_numbers)
    #     adv_average = round(np.mean(label_change_numbers), 2)
    #     adv_std = np.std(label_change_numbers)
    #     adv_99ci = round(2.576 * adv_std / math.sqrt(len(label_change_numbers)), 2)
    #     result = result + 'ori,' + str(adv_average) + ',' + str(round(adv_std, 2)) + ',' + str(adv_99ci) + '\n'
    #
    #     return store_string, result

