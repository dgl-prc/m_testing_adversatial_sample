import random

import numpy as np
import os
from scipy import ndimage
import math
import tensorflow as tf

def clip(image, clip_min, clip_max):
    shape = np.asarray(image).shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if image[i][j][k] > clip_max:
                    image[i][j][k] = clip_max
                elif image[i][j][k] < clip_min:
                    image[i][j][k] = clip_min
    return image

def c_occl(gradients, start_point, rect_shape):
    gradients = np.asarray(gradients)
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads

def c_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads

def c_black(gradients, start_point, rect_shape):
    # start_point = (
    #     random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    gradients = np.asarray(gradients)
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads

def generate_value_1(row, col):
    matrix = []
    for i in range(row):
        line = []
        for j in range(col):
            pixel = []
            for k in range(1):
                div = random.randint(1, 20)#5,100
                pixel.append((random.random() - 0.5) / (div*100)) #normal
            line.append(pixel)
        matrix.append(line)
    return [matrix]

# generate for RGB
def generate_value_3(row, col):
    matrix = []
    for i in range(row):
        line = []
        for j in range(col):
            pixel = []
            for k in range(3):
                div = random.randint(1, 20)  # 1,20
                pixel.append((random.random() - 0.5)/ (div*100)) #*4 /10
            line.append(pixel)
        matrix.append(line)
    return [matrix]

def generate_value_uniform_1(row, col, delta):
    matrix = []
    for i in range(row):
        line = []
        for j in range(col):
            pixel = []
            for k in range(1):
                # x = delta / 28
                x = delta
                pixel.append(random.uniform(-x,x))
            line.append(pixel)
        matrix.append(line)
    return matrix

# generate for RGB
def generate_value_uniform_3(row, col, delta):
    matrix = []
    for i in range(row):
        line = []
        for j in range(col):
            pixel = []
            for k in range(3):
                x = delta /(32 * np.sqrt(3))
                pixel.append(random.uniform(-x,x))
            line.append(pixel)
        matrix.append(line)
    return matrix

def generate_value_normal_1(row, col, delta):
    matrix = []
    for i in range(row):
        line = []
        for j in range(col):
            pixel = []
            for k in range(1):
                # sigma = 1/4 * pow(delta / 28, 2)
                sigma = 1 / 4 * pow(delta, 2)
                pixel.append(random.gauss(0, sigma))
            line.append(pixel)
        matrix.append(line)
    return matrix

# generate for RGB
def generate_value_normal_3(row, col, delta):
    matrix = []
    for i in range(row):
        line = []
        for j in range(col):
            pixel = []
            for k in range(3):
                sigma = 1 / 4 * pow(delta /(32 * np.sqrt(3)), 2)
                pixel.append(random.gauss(0, sigma))
            line.append(pixel)
        matrix.append(line)
    return matrix

def get_data_mutation_test(adv_file_path):
    '''
    :param file_path: the file path for the adversary images
    :return: the formatted data for mutation test, the actual label of the images, and the predicted label of the images
    '''
    image_list = []
    real_labels = []
    predicted_labels = []
    image_files =[]
    for img_file in os.listdir(adv_file_path):
        if img_file.endswith('.png'):
            img_file_split = img_file.split('_')
            real_label = int(img_file_split[-3])
            predicted_label = int(img_file_split[-2])

            if real_label!=predicted_label: # only extract those successfully generated adversaries
                real_labels.append(real_label)
                predicted_labels.append(predicted_label)
                current_img = ndimage.imread(adv_file_path + os.sep + img_file)
                image_list.append(current_img)
                image_files.append(img_file)
        # if len(image_list) >= 100:
        #     break
    print('--- Total number of adversary images: ', len(image_list))
    return image_list, image_files, real_labels, predicted_labels

def get_normal_data_mutation_test(adv_file_path):
    '''
    :param file_path: the file path for the adversary images
    :return: the formatted data for mutation test, the actual label of the images, and the predicted label of the images
    '''
    image_list = []
    real_labels = []
    predicted_labels = []
    image_files =[]
    for img_file in os.listdir(adv_file_path):
        if img_file.endswith('.png'):
            img_file_split = img_file.split('_')
            real_label = int(img_file_split[-3])
            predicted_label = int(img_file_split[-2])

            if real_label==predicted_label: # only extract those successfully generated adversaries
                real_labels.append(real_label)
                predicted_labels.append(predicted_label)
                current_img = ndimage.imread(adv_file_path + os.sep + img_file)
                image_list.append(current_img)
                image_files.append(img_file)
        # if len(image_list) >= 100:
        #     break
    print('--- Total number of normal images: ', len(image_list))
    return image_list, image_files, real_labels, predicted_labels


def extract_label_change_ratios(stats_file):

    normal_label_change_1 = []
    adv_label_change_1 = []

    normal_label_change_5 = []
    adv_label_change_5 = []

    normal_label_change_10 = []
    adv_label_change_10 = []

    with open(stats_file) as f:
        for line in f:
            lis = line.split(',')
            image_name = lis[0].split('_')
            step_size = int(lis[1])
            if step_size==1:
                if image_name[-3]==image_name[-2]:
                    normal_label_change_1.append(int(lis[2]))
                else:
                    adv_label_change_1.append(int(lis[2]))

            if step_size==5:
                if image_name[-3]==image_name[-2]:
                    normal_label_change_5.append(int(lis[2]))
                else:
                    adv_label_change_5.append(int(lis[2]))

            if step_size==10:
                if image_name[-3]==image_name[-2]:
                    normal_label_change_10.append(int(lis[2]))
                else:
                    adv_label_change_10.append(int(lis[2]))

    print('=== Result file: ', stats_file)

    print('--- Step size: 1')
    if(len(normal_label_change_1)>0):
        print('- Normal change: ', extract_ci(normal_label_change_1))
    else:
        print('- Normal change not applicable')
    print('- Adv change: ', extract_ci(adv_label_change_1))

    print('--- Step size: 5')
    if (len(normal_label_change_5) > 0):
        print('- Normal change: ', extract_ci(normal_label_change_5))
    else:
        print('- Normal change not applicable')
    print('- Adv change: ', extract_ci(adv_label_change_5))

    print('--- Step size: 10')
    if (len(normal_label_change_10) > 0):
        print('- Normal change: ', extract_ci(normal_label_change_10))
    else:
        print('- Normal change not applicable')
    print('- Adv change: ', extract_ci(adv_label_change_10))

    return normal_label_change_1, adv_label_change_1, normal_label_change_5, adv_label_change_5, normal_label_change_10, adv_label_change_10


def extract_ci(label_change_number):
    adv_average = round(np.mean(label_change_number), 2)
    adv_std = np.std(label_change_number)
    adv_99ci = round(2.576 * adv_std / math.sqrt(len(label_change_number)), 2)
    return adv_average, adv_std, adv_99ci

def process(img):
    new_img = img
    return new_img

# "ENHANCING THE RELIABILITY OF OUT-OF-DISTRIBUTION IMAGE DETECTION IN NEURAL NETWORKS(ICLR2018)"
def input_preprocessing(preds, x, eps, clip_min, clip_max):
    y = tf.reduce_max(preds, 1, keep_dims=True)
    # grad, = tf.gradients(tf.log(y), x)
    # normalized_grad = tf.sign(tf.multiply(-1.0, grad))
    # normalized_grad = tf.stop_gradient(normalized_grad)
    # scaled_grad = eps * normalized_grad
    # output = x - scaled_grad

    grad, = tf.gradients(y, x)
    normalized_grad = tf.sign(grad)
    normalized_grad = tf.stop_gradient(normalized_grad)
    scaled_grad = eps * normalized_grad
    output = x - scaled_grad

    if (clip_min is not None) and (clip_max is not None):
        output = tf.clip_by_value(output, clip_min, clip_max)

    return output

def get_data_loader(data_path, is_adv_data, data_type):
    if data_type == DATA_MNIST:
        img_mode = 'L'
        normalize = normalize_mnist
    else:
        img_mode = None
        normalize = normalize_cifar10

    if is_adv_data:
        tf = transforms.Compose([transforms.ToTensor(), normalize])
        dataset = MyDataset(root=data_path, transform=tf, img_mode=img_mode, max_size=TEST_SMAPLES)  # mnist
        dataloader = DataLoader(dataset=dataset)
    else:
        dataset, channel = load_data_set(data_type, data_path, False)
        random_indcies = np.arange(10000)
        np.random.seed(random_seed)
        np.random.shuffle(random_indcies)
        random_indcies = random_indcies[:TEST_SMAPLES]
        data = datasetMutiIndx(dataset, random_indcies)
        dataloader = DataLoader(dataset=data)
    return dataloader

# extract_label_change_ratios('/Users/jingyi/nMutant/mt_result/mnist_jsma/adv_result.csv')
# extract_label_change_ratios('/Users/jingyi/nMutant/mt_result/mnist_cw/adv_result.csv')
# extract_label_change_ratios('/Users/jingyi/nMutant/mt_result/mnist_bb/adv_result.csv')
# extract_label_change_ratios('/Users/jingyi/nMutant/mt_result/mnist_fgsm1/adv_fgsm1_result.csv')

# extract_label_change_ratios('/Users/jingyi/nMutant/mt_result/cifar10_jsma/adv_result.csv')
# extract_label_change_ratios('/Users/jingyi/nMutant/mt_result/cifar10_cw/adv_result.csv')
# extract_label_change_ratios('/Users/jingyi/nMutant/mt_result/cifar10_bb/adv_result.csv')
# extract_label_change_ratios('/Users/jingyi/nMutant/mt_result/cifar10_tf/adv_fgsm1_result.csv')