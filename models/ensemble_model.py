
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Pool
import logging
import numpy as np

class EnsembleModel(nn.Module):
    def __init__(self, model_list,num_out=10):
        super(EnsembleModel, self).__init__()
        self.model_list = model_list
        self.num_out = num_out

    def forward(self, x):
        '''
        Current the size of x must be one. That is, this model dose not support batch data
        :param x:
        :return: the average confidence shape: Nx10, where N <=5
        '''

        predicts = [0] * self.num_out
        scores_list = []
        for model in self.model_list:
            model.eval()
            scores = model(x)
            predict = torch.argmax(scores).item()
            predicts[predict] += 1
            scores_list.append(scores)

        voting_rst = torch.argmax(torch.Tensor(predicts)).item()
        logging.info("confidence:{}/{}({:.2f}%);detail:{}".format(predicts[voting_rst],
                                                                  len(self.model_list),
                                                                  predicts[voting_rst] * 100. / len(self.model_list),
                                                                  predicts))
        members = []
        for scores in scores_list:
            if torch.argmax(scores).item() == voting_rst:
                members.append(scores)
        members = torch.stack(members)
        return members.mean(dim=0)