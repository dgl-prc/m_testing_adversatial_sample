import numpy as np
import utils
import torch
from .mutation import MutationTest


class detector:

    '''
        A statistical detector for adversaries
        :param alpha : error bound of false negative
        :param beta : error bound of false positive
        :param sigma : size of indifference region
        :param kappa_nor : ratio of label change of a normal input
        :param mu : hyper parameter reflecting the difference between kappa_nor and kappa_adv
    '''

    alpha = 0.05
    beta = 0.05
    sigma = 0.05
    kappa_nor = 0.01
    mu = 1.2
    img_rows = 28
    img_cols = 28
    step_size = 1
    max_mutation = 5000
    rgb = True

    def __init__(self, kappa_nor, mu, img_rows, img_cols, step_size, rgb=False, max_mutation=5000, alpha=0.05, beta=0.05, sigma=0.01):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.kappa_nor = kappa_nor
        self.mu = mu
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.step_size = step_size
        self.max_mutation = max_mutation
        self.rgb = rgb


        assert self.mu * self.kappa_nor > self.sigma

    def print_config(self):
        attrs = vars(self)
        return attrs

    def calculate_sprt_ratio(self, c, n):
        '''
        :param c: number of label changes
        :param n: total number of mutations
        :return: the sprt ratio
        '''

        p1 = self.mu * self.kappa_nor + self.sigma
        p0 = self.mu * self.kappa_nor - self.sigma

        return c * np.log(p1 / p0) + (n - c) * np.log((1 - p1) / (1 - p0))
            # pow(p1,c)*pow(1-p1,n-c)/pow(p0,c)/pow(1-p0,n-c)


    def detect(self, orig_img, orig_label, model_adapter):
        stop = False
        label_change_mutation_count = 0
        total_mutation_count = 0
        decided = False
        sprt_ratio = 0

        while (not stop):
            total_mutation_count += 1

            if total_mutation_count>self.max_mutation:
                # print('====== Result: Can\'t make a decision in ' + str(total_mutation_count-1) + ' mutations')
                # print('Total number of mutations evaluated: ', total_mutation_count-1)
                # print('Total label changes of the mutations: ', label_change_mutation_count)
                # print('=======')
                return False, decided, total_mutation_count, label_change_mutation_count

            # print('Test mutation number ', i)
            mt = MutationTest(self.img_rows, self.img_cols)
            if self.rgb:
                mutation_matrix = mt.mutation_matrix(utils.generate_value_3)
            else:
                mutation_matrix = mt.mutation_matrix(utils.generate_value_1)

            t_img = orig_img.numpy()

            mutation_img = t_img + (mutation_matrix.reshape((1,1,28,28)).astype(np.float32)-0.1307) / 0.3081 * self.step_size #mnist(1,1,28,28)

            total_mutation_count += 1

            mutation_img = torch.from_numpy(mutation_img)
            # mu_label = model_argmax(sess, x, preds, mutation_img, feed=feed_dict)
            mu_label, _ = model_adapter.get_predict_lasth(mutation_img)
            if orig_label!=mu_label:
                # print('- Orig label: ', orig_label, ', New label: ', mu_label)
                label_change_mutation_count += 1

            sprt_ratio = sprt_ratio + self.calculate_sprt_ratio(label_change_mutation_count,total_mutation_count)

            if sprt_ratio >= np.log((1-self.beta)/self.alpha):
                # print('=== Result: Adversarial input ===')
                # print('Total number of mutations evaluated: ', total_mutation_count)
                # print('Total label changes of the mutations: ', label_change_mutation_count)
                # print('======')
                decided = True
                return True, decided, total_mutation_count, label_change_mutation_count

            elif sprt_ratio<= np.log(self.beta/(1-self.alpha)):
                # print('=== Result: Normal input ===')
                # print('Total number of mutations evaluated: ', total_mutation_count)
                # print('Total label changes of the mutations: ', label_change_mutation_count)
                # print('======')
                decided = True
                return False, decided, total_mutation_count, label_change_mutation_count




