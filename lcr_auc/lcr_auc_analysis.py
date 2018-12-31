'''
1. true label
2. adversarial label
3. predict label
4. the vote detail
'''
import linecache
import numpy as np
import os
from sklearn import metrics


def get_confidence(std, size):
    # 95%
    # 98%
    # 99%
    c95 = 1.96 * std / np.sqrt(size)
    c98 = 2.33 * std / np.sqrt(size)
    c99 = 2.58 * std / np.sqrt(size)
    return "confidence(95%):{:.4f},confidence(98%):{:.4f},confidence(99%):{:.4f}".format(c95, c98, c99)


def get_auc(adv_lcr_list, normal_lcr_list):
    pos_score = np.array(adv_lcr_list)
    pos_label = np.ones(pos_score.size, dtype=int)

    if isinstance(normal_lcr_list,list):
        neg_score = np.array(normal_lcr_list)
    elif isinstance(normal_lcr_list,np.ndarray):
        neg_score = normal_lcr_list
    else:
        raise Exception("Unknown data type:{}".format(type(normal_lcr_list)))
    neg_label = np.zeros(neg_score.size, dtype=int)

    y_score = np.concatenate((pos_score, neg_score))
    y_label = np.concatenate((pos_label, neg_label))

    fpr, tpr, thresholds = metrics.roc_curve(y_label, y_score)
    auc = metrics.auc(fpr, tpr)
    return auc


def f_for_distribute_log(file_list, total_lines, verbose=False, is_adv=True, total_mutated_models=10,
                         lcr_save_path=None, nr_lcr_list=None):
    '''

    :param file_list: list(str). the files list of batch testing results
    :param total_lines: int. total lines contained in a test file.
    :param verbose: bool. input verbose info or not
    :param is_adv: bool. True if the samples are adversarial, otherwise,false
    :param total_mutated_models: int. total mutated models are used to yield the label change rate(lcr)
    :param lcr_save_path: str. the path to save the lcr list.
    :param nr_lcr_list: list. the lcr list of normal samples. This is just for the auc computing
    :return:
    '''
    base_file = file_list[0]
    p = 1
    line = linecache.getline(base_file, p).strip()
    count = 0
    lcr_list = []
    while p <= total_lines:
        if line.__contains__('mutated_models'):
            print(line.split('INFO -')[-1])
            adv_type = linecache.getline(base_file, p + 2).strip()
            assert adv_type.__contains__('Test-Details-start')
            print('>>>>>>>>>>>{}<<<<<<<<<<<<<<'.format(adv_type.split('>>>')[-1].strip()))
            p += 3
        if line == '':
            assert linecache.getline(base_file, p + 2).strip().__contains__('Test-Details-end')
            if len(lcr_list) != 0:
                rst = np.array(lcr_list)
                size = len(rst)
                avg = np.average(rst)
                std = np.std(rst)
                if is_adv:
                    auc = get_auc(lcr_list, nr_lcr_list)
                    print('Total Samples Used:{},auc:{:.4f},avg_lcr:{:.4f},std:{:.4f},{}'.format(size, auc, avg, std,
                                                                                                    get_confidence(std,
                                                                                                                   size)))
                else:
                    print('Total Samples Used:{},avg_lcr:{:.4f},std:{:.4f},{}'.format(size, avg, std,
                                                                                             get_confidence(std,
                                                                                                            size)))
                if lcr_save_path is not None:
                    if not os.path.exists(os.path.split(args.lcrSavePath)[0]):
                        os.makedirs(os.path.split(args.lcrSavePath)[0])
                    with open(lcr_save_path, "w") as f:
                        np.save(f, rst)
                return
        labels = linecache.getline(base_file, p)
        if is_adv:
            '''adversarial samples must contain adv labels'''
            if labels.__contains__('adv_label'):
                trure_label, adv_label = labels.strip().split('>>>')[-1].split(',')
                adv_label = int(adv_label.split(':')[1])
            else:
                adv_label = 0
                trure_label = labels.strip().split('>>>')[-1]
            trure_label = int(trure_label.split(':')[1])
            ori_label = adv_label
        else:
            trure_label = labels.strip().split('>>>')[-1]
            trure_label = int(trure_label.split(':')[1])
            ori_label = trure_label
        p += 1
        vote_detail_total = None
        for filename in file_list:
            vote_detail = linecache.getline(filename, p).split(';')[-1].split(':')[-1].strip()
            vote_detail = np.array([int(item) for item in vote_detail.split('[')[-1].split(']')[0].split(',')])
            if vote_detail_total is not None:
                vote_detail_total += vote_detail
            else:
                vote_detail_total = vote_detail
        assert np.sum(vote_detail_total) == total_mutated_models
        p += 1
        currenct_pred_label = int(linecache.getline(base_file, p).strip().split(':')[-1].split('<')[0])
        p += 1
        line = linecache.getline(base_file, p).strip()
        lcr = 1 - 1. * vote_detail_total[ori_label] / np.sum(vote_detail_total)
        if verbose:
            if is_adv:
                print('true:{},adv:{},pred:{},detail:{},lcr_auc:{},adv_votes:{}').format(trure_label, adv_label,
                                                                                         currenct_pred_label,
                                                                                         vote_detail, lcr,
                                                                                         vote_detail[adv_label])
            else:
                print('true:{},pred:{},detail:{},lcr_auc:{},adv_votes:{}').format(trure_label,
                                                                                  currenct_pred_label,
                                                                                  vote_detail, lcr,
                                                                                  vote_detail[trure_label])
        lcr_list.append(lcr)
        count += 1


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser("Pararmeters for Lcr and Auc Analysis")
    parser.add_argument("--logPath",type=str,help="The files path of batch testing results")
    parser.add_argument("--maxModelsUsed",type=int ,help="Total mutated models are used to yield the label change rate(lcr)")
    parser.add_argument("--isAdv",type=str,help="True if the samples are adversarial, otherwise,false")
    parser.add_argument("--nrLcrPath",type=str,help="The lcr list of normal samples. This is just for the auc computing",required=False)
    parser.add_argument("--lcrSavePath",type=str,help="The path to save the lcr list",required=False)
    args = parser.parse_args()

    is_adv = True if args.isAdv=="True" else False

    file_list = []
    for file_name in os.listdir(args.logPath):
        file_list.append(os.path.join(args.logPath, file_name))
    total_lines = len(open(file_list[0], 'rU').readlines())
    if is_adv:
        nr_lcr_list = np.load(args.nrLcrPath)
        f_for_distribute_log(file_list, total_lines, is_adv=is_adv, total_mutated_models=args.maxModelsUsed,
                             lcr_save_path=None, nr_lcr_list=nr_lcr_list)
    else:
        f_for_distribute_log(file_list, total_lines, is_adv=is_adv, total_mutated_models=args.maxModelsUsed,
                             lcr_save_path=args.lcrSavePath, nr_lcr_list=None)


    # print(log_folder,total_mutated_models,is_adv)
    # log_folder="../lcr_auc-testing-results/mnist/lenet/ns/3e-2p/wl/2018-12-30-14"
    # total_mutated_models=40
    # is_adv=True

    # filename = "../lcr_auc-testing-results/mnist/lenet/ns/3e-2p/normal/2018-12-30-12/1.log"
    # filename = "/home/dgl/gitgub/SafeDNN/bgDNN/log_vote/mnist_new_vote/normal/ns-normal-mnist-500-new-pure.log"
    # f(filename, verbose=False, is_adv=False, is_save_senbilit_list=False, mutated_models=20)
