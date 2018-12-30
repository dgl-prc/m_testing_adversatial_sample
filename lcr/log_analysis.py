'''
1. true label
2. adversarial label
3. predict label
4. the vote detail
'''
import linecache
import numpy as np
import os


def get_confidence(std, size):
    # 95%
    # 98%
    # 99%
    c95 = 1.96 * std / np.sqrt(size)
    c98 = 2.33 * std / np.sqrt(size)
    c99 = 2.58 * std / np.sqrt(size)
    return "confidence(95%):{:.4f},confidence(98%):{:.4f},confidence(99%):{:.4f}".format(c95,c98,c99)


def f_for_distribute_log(file_list, total_lines,verbose=False, is_adv=True, total_mutated_models=10,is_save_senbilit_list=False):
    '''
    we read 100 files at the same time. the point P is shared by all files since all the files have the same dataset
    :param filename:
    :param total_lines:
    :param verbose:
    :param is_adv:
    :return:
    '''
    base_file = file_list[0]
    p = 1
    line = linecache.getline(base_file, p).strip()
    count = 0
    sensibility_list = []
    sensibility_count = 1

    while p <= total_lines:
        if line.__contains__('mutated_models'):
            print(line.split('INFO -')[-1])
            adv_type = linecache.getline(base_file, p + 2).strip()
            assert adv_type.__contains__('Test-Details-start')
            print('>>>>>>>>>>>{}<<<<<<<<<<<<<<'.format(adv_type.split('>>>')[-1].strip()))
            p += 3
        if line == '':
            assert linecache.getline(base_file, p + 2).strip().__contains__('Test-Details-end')
            if len(sensibility_list) != 0:
                rst = np.array(sensibility_list)
                size = len(rst)
                avg = np.average(rst)
                std = np.std(rst)
                print('Total:{},avg:{:.4f},std:{:.4f},{}'.format(size, avg, std,
                                                                 get_confidence(std, size)))
            sensibility_list = []
            p += 3
            line = linecache.getline(base_file, p).strip()
            if line.__contains__('model mutated ration'):
                print(line.split('INFO -')[-1])
                p += 2
            elif line.__contains__('Progress'):
                p += 1
            if p > total_lines:
                return
            adv_type = linecache.getline(base_file, p)
            assert adv_type.__contains__('Test-Details-start-'), '{},p={}'.format(adv_type, p)
            print('>>>>>>>>>>>{}<<<<<<<<<<<<<<'.format(adv_type.split('>>>')[-1].strip()))
            p += 1

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

        sensibility = 1 - 1. * vote_detail_total[ori_label] / np.sum(vote_detail_total)

        if verbose:
            if is_adv:
                print('true:{},adv:{},pred:{},detail:{},sensibility:{},adv_votes:{}').format(trure_label, adv_label,
                                                                                             currenct_pred_label,
                                                                                             vote_detail, sensibility,
                                                                                             vote_detail[adv_label])
            else:
                print('true:{},pred:{},detail:{},sensibility:{},adv_votes:{}').format(trure_label,
                                                                                      currenct_pred_label,
                                                                                      vote_detail, sensibility,
                                                                                      vote_detail[trure_label])
        sensibility_list.append(sensibility)
        count += 1


if __name__ == '__main__':
    import sys
    log_folder=sys.argv[1]
    total_mutated_models=int(sys.argv[2])
    is_adv=False if sys.argv[3] == "False" else True
    # print(log_folder,total_mutated_models,is_adv)
    # log_folder="../lcr-testing-results/mnist/lenet/ns/3e-2p/wl/2018-12-30-14"
    # total_mutated_models=40
    # is_adv=True
    file_list = []
    for file_name in os.listdir(log_folder):
        file_list.append(os.path.join(log_folder, file_name))
    total_lines = len(open(file_list[0],'rU').readlines())
    f_for_distribute_log(file_list, total_lines, is_adv=is_adv,total_mutated_models=total_mutated_models)

    # filename = "../lcr-testing-results/mnist/lenet/ns/3e-2p/normal/2018-12-30-12/1.log"
    # filename = "/home/dgl/gitgub/SafeDNN/bgDNN/log_vote/mnist_new_vote/normal/ns-normal-mnist-500-new-pure.log"
    # f(filename, verbose=False, is_adv=False, is_save_senbilit_list=False, mutated_models=20)


