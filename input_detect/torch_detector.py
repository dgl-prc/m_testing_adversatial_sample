import sys
sys.path.append("../")
import tensorflow as tf
from tensorflow.python.platform import flags
import os
from input_detect.detection import detector
from models.googlenet import GoogLeNet
from models.lenet import MnistNet4
from baseline.ModelAdapter import *
from data_manger import *
from attack_util import load_natural_data

FLAGS = flags.FLAGS

def directory_detect(datasets, dir_path, normal, store_path, ad, target_model):
    MAX_NUM_SAMPLES = 1000
    model_adapter = MnistNet4Adapter(target_model)

    print('--- Extracting images from: ', dir_path)
    # if normal:
        # [adv_image_list, adv_image_files, real_labels, predicted_labels] = utils.get_normal_data_mutation_test(dir_path)
    # else:
        # [adv_image_list, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(dir_path)

    #########################################################
    # load data. use training data to denote the submanifold.
    #########################################################
    data_path = "../build-in-resource/dataset/mnist/raw"
    train_data, _ = load_data_set(data_type=DATA_MNIST, source_data=data_path, train=True)
    test_data, _ = load_data_set(data_type=DATA_MNIST, source_data=data_path, train=False)
    # adv_data_loader = get_data_loader(advDataPath, is_adv_data=True, data_type=dataType)
    train_loader, test_loader = create_data_loader(
        batch_size=1,
        test_batch_size=1,
        train_data=train_data,
        test_data=test_data
    )

    print("load normal data.....")
    normal_data = load_natural_data(True, 0 if datasets == "mnist" else 1, data_path, use_train=True, seed_model=target_model, device='cpu', MAX_NUM_SAMPLES=MAX_NUM_SAMPLES)
    normal_loader = DataLoader(dataset=normal_data)

    print("load wl data.....")
    wl_data = load_natural_data(False, 0 if datasets == "mnist" else 1, data_path, use_train=True, seed_model=target_model, device='cpu', MAX_NUM_SAMPLES=MAX_NUM_SAMPLES)
    wl_loader = DataLoader(dataset=wl_data)

    adv_count = 0
    not_decided_images = 0
    total_mutation_counts = []
    label_change_mutation_counts = []
    suc_total_mutation_counts = []
    suc_label_change_mutation_counts = []

    print('--- Evaluating inputs ---')

    if not os.path.exists(store_path):
        os.makedirs(store_path)
    detector_results = []
    summary_results = []
    i = 0
    for x_test, y_test in wl_loader:
        orig_label, _ = model_adapter.get_predict_lasth(x_test)
        [result, decided, total_mutation_count, label_change_mutation_count] = ad.detect(x_test, orig_label,
                                                                                         model_adapter)

        detector_results.append(str(i) + ',' + str(result) + ',' + str(decided) + ',' + str(total_mutation_count) + ',' + str(label_change_mutation_count))

        if result:
            adv_count += 1
            if not normal: # Record the counts for adversaries
                suc_total_mutation_counts.append(total_mutation_count)
                suc_label_change_mutation_counts.append(label_change_mutation_count)

        if normal and not result: # Record the counts for normals
            suc_total_mutation_counts.append(total_mutation_count)
            suc_label_change_mutation_counts.append(label_change_mutation_count)

        if not decided:
            not_decided_images += 1

        total_mutation_counts.append(total_mutation_count)
        label_change_mutation_counts.append(label_change_mutation_count)
        i += 1

    with open(store_path + "/detection_result.csv", "w") as f:
        for item in detector_results:
            f.write("%s\n" % item)

    summary_results.append('adv_num,' + str(i))
    summary_results.append('identified_num,' + str(adv_count))
    summary_results.append('undecided_num,' + str(not_decided_images))

    if normal:
        summary_results.append('accuracy,' + str(1 - float(adv_count)/len(total_mutation_counts)))
    else:
        summary_results.append('accuracy,' + str(float(adv_count)/len(total_mutation_counts)))

    if len(suc_label_change_mutation_counts) > 0 and not normal:
        summary_results.append(
            'avg_mutation_num,' + str(sum(suc_total_mutation_counts) / len(suc_total_mutation_counts)))
        summary_results.append(
            'avg_lc_num,' + str(float(sum(suc_label_change_mutation_counts)) / len(suc_label_change_mutation_counts)))

    if len(suc_label_change_mutation_counts) > 0 and normal:
        summary_results.append(
            'avg_mutation_num,' + str(sum(suc_total_mutation_counts) / len(suc_total_mutation_counts)))
        summary_results.append(
            'avg_lc_num,' + str(float(sum(suc_label_change_mutation_counts)) / len(suc_label_change_mutation_counts)))

    summary_results.append(total_mutation_counts)
    summary_results.append(label_change_mutation_counts)

    with open(store_path + "/detection_summary_result.csv", "w") as f:
        for item in summary_results:
            f.write("%s\n" % item)

    print('- Total adversary images evaluated: ', i)
    print('- Identified adversaries: ', adv_count)
    print('- Not decided images: ', not_decided_images)
    if len(suc_label_change_mutation_counts) > 0:
        print('- Average mutation needed: ', sum(suc_total_mutation_counts) / len(suc_total_mutation_counts))
        print('- Average label change mutations: ',
              float(sum(suc_label_change_mutation_counts)) / len(suc_label_change_mutation_counts))
    else:
        summary_results.append(
            'avg_mutation_num,' + str(sum(total_mutation_counts) / len(total_mutation_counts)))
        summary_results.append(
            'avg_lc_num,' + str(float(sum(label_change_mutation_counts)) / len(label_change_mutation_counts)))

def main(argv=None):
    dataType = FLAGS.datasets
    model_name = FLAGS.model_name
    attack_type = FLAGS.attack_type
    # pred, _ = model_adapter.get_predict_lasth(x_test)

    # detector config
    k_nor = FLAGS.k_nor
    mu = FLAGS.mu
    level = FLAGS.level
    max_mutations = FLAGS.max_iteration
    normal = False

    indifference_region_ratio = mu - 1
    alpha = 0.05
    beta = 0.05
    if 'mnist' == dataType:
        rgb = False
        image_rows = 28
        image_cols = 28
    elif 'cifar10' == dataType:
        rgb = True
        image_rows = 32
        image_cols = 32

    print('--- Dataset: ', dataType, 'attack type: ', attack_type)

    # model_path = "../../torch_model/" + dataType + "/lenet.pkl"
    model_path = "../build-in-resource/pretrained-model/" + dataType + "/lenet.pkl"
    advDataPath = "../build-in-resource/dataset/" + dataType + "/adversarial/" + attack_type
    #
    # if dataType == "mnist":
    #     data_path = "../build-in-resource/dataset/mnist/raw"
    # else:
    #     data_path = "../build-in-resource/dataset/cifar10/raw"

    target_model = GoogLeNet() if model_name == "googlenet" else MnistNet4()

    target_model.load_state_dict(torch.load(model_path))
    target_model.eval()

    adv_image_dir = FLAGS.sample_path + dataType + '/' + attack_type + '/test'
    if attack_type.__eq__('normal'):
        normal = True

    store_path = FLAGS.store_path + dataType + '_' + attack_type + '/level=' + str(level)+',mm=' + \
                     str(max_mutations) + '/mu=' + str(mu) + ',irr=' + str(indifference_region_ratio)

    # Detection
    ad = detector(k_nor, mu, image_rows, image_cols, level, rgb, max_mutations, alpha, beta, k_nor*indifference_region_ratio)
    print('--- Detector config: ', ad.print_config())
    directory_detect(dataType, adv_image_dir, normal, store_path, ad, target_model)

if __name__ == '__main__':
    # flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    # flags.DEFINE_string('model_path', '../models/integration/mnist', 'The path to load model.')
    # flags.DEFINE_string('sample_path', '../datasets/experiment/mnist', 'The path storing samples.')
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model_name', 'lenet5', 'The path to load model.')
    flags.DEFINE_string('epoch', 9, 'The epoch of model for use.')
    flags.DEFINE_string('sample_path', '/Users/pxzhang/Downloads/nMutant_backup_server_tacas/datasets/experiment_tacas/', 'The path storing samples.')
    flags.DEFINE_string('store_path', '../detection/', 'The path to store result.')
    flags.DEFINE_string('attack_type', 'normal', 'attack_type')
    flags.DEFINE_float('k_nor', 0.0017, 'normal ratio change')
    flags.DEFINE_float('mu', 1.2, 'mu parameter of the detection algorithm')
    flags.DEFINE_integer('level', 1, 'the level of random mutation region.')
    flags.DEFINE_integer('max_iteration', 2000, 'max iteration of mutation')

    tf.app.run()
