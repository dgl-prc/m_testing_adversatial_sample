
import argparse
from detector import *
from models.lenet import MnistNet4
from models.googlenet import GoogLeNet



TEST_SMAPLES=1000
def main(detect_type, data_loader, models_folder, seed_model_name, threshold, sigma, attack_type,
         device, data_type):
    ## mnist
    alpha = 0.05
    beta = 0.05
    detector = Detector(threshold=threshold, sigma=sigma, beta=beta, alpha=alpha, models_folder=models_folder,
                        seed_name=seed_model_name,
                        max_mutated_numbers=500, device=device, data_type=data_type)
    adv_success = 0
    progress = 0
    avg_mutated_used = 0
    totalSamples = len(data_loader)
    if detect_type == 'adv':
        for img, true_label, adv_label in data_loader:
            rst, success_mutated, total_mutated = detector.detect(img, origi_label=adv_label)
            if rst:
                adv_success += 1
            avg_mutated_used += total_mutated
            progress += 1
            sys.stdout.write('\r Processed:%.2f %%' % (100.*progress/totalSamples))
            sys.stdout.flush()
    else:
        for img, true_label in data_loader:
            rst, success_mutated, total_mutated = detector.detect(img, origi_label=true_label)
            if rst:
                adv_success += 1
            progress += 1
            sys.stdout.write('\r Processed:%.2f' % (100. * progress / totalSamples))
            sys.stdout.flush()
            avg_mutated_used += total_mutated
    avg_mutated_used = avg_mutated_used * 1. / len(data_loader.dataset)

    total_data = len(data_loader.dataset)
    if detect_type == 'adv':
        logging.info(
            '{},{}-Adv Accuracy:{}/{},{:.4f},,avg_mutated_used:{:.4f}'.format(models_folder, attack_type, adv_success,
                                                                              total_data,
                                                                              adv_success * 1. / total_data,
                                                                              avg_mutated_used))
        avg_accuracy = adv_success * 1. / len(data_loader.dataset)
    else:
        logging.info(
            '{},Normal Accuracy:{}/{},{:.4f},,avg_mutated_used:{:.4f}'.format(models_folder, total_data - adv_success,
                                                                              total_data,
                                                                              1 - adv_success * 1. / total_data,
                                                                              avg_mutated_used))
        avg_accuracy = 1 - adv_success * 1. / total_data

    return avg_accuracy, avg_mutated_used

def show_progress(**kwargs):
    sys.stdout.write('\r Processed:%d' % (kwargs['progress']))
    sys.stdout.flush()


def get_data_loader(data_path, is_adv_data, data_type):
    if data_type == DATA_MNIST:
        img_mode = 'L'
        normalize = normalize_mnist
    else:
        img_mode = None
        normalize = normalize_cifar10

    if is_adv_data:
<<<<<<< HEAD
=======

>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3
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


def get_wrong_label_data_loader(data_path, seed_model, data_type,device):
    dataset, channel = load_data_set(data_type, data_path, False)
    dataloader = DataLoader(dataset=dataset)
    wrong_labeled = samples_filter(seed_model, dataloader, return_type='adv', name='seed model',device=device)
    data = datasetMutiIndx(dataset, [idx for idx, _, _ in wrong_labeled][:TEST_SMAPLES])
    wrong_labels = [wrong_label for idx, true_label, wrong_label in wrong_labeled[:TEST_SMAPLES]]
    data = TensorDataset(data.tensors[0], data.tensors[1], torch.LongTensor(wrong_labels))
    return DataLoader(dataset=data)


<<<<<<< HEAD
def get_threshold_relax(threshold, extend_scale, relax_scale):
    return threshold * extend_scale, threshold * relax_scale




'''
=======
def get_threshold_relax(a, t_scale, r_scale):
    return a * t_scale, a * r_scale


'''

>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3
cifar10 threshold(ration 0.005):
     nai: up_bound = 0.01 * (7.28 + 1.12)
     ns:  up_bound = 0.01 * (1.88 + 0.55) 
     ws:  up_bound = 0.01 * (2.69 + 0.65) 
     gf:  up_boun= 0.01 * (4.09 + 0.91) 

mnist threshold(ration 0.05):
    nai: up_bound = (3.88 + 0.53) * 0.01
    ns:  up_bound = (0.89 + 0.35) * 0.01
    ws:  up_bound = (3.83 + 0.42) * 0.01
    gf:  up_bound = (2.49 + 0.59) * 0.01
'''


def run():
    parser = argparse.ArgumentParser("Prarameters for Detection")
    parser.add_argument("--threshold",type=float,help="The lcr_auc of normal samples. The value is equal to: avg+99%confidence.",required=True)
    parser.add_argument("--extendScale",type=float,help="The scale to extend the thrshold",required=True)
    parser.add_argument("--relaxScale",type=float,help="The proportion of threhold,which is useed to control the scale of indifference region",required=True)
    parser.add_argument("--mutatedModelsPath",type=str,help="The path where the pre-mutated models are",required=True)
    parser.add_argument("--alpha",type=float,help="probability od Type-I error",required=True)
    parser.add_argument("--beta",type=float,help="probability od Type-2 error",required=True)
    parser.add_argument("--testSamplesPath",type=str,help="The path where the samples to be tested are")
    parser.add_argument("--dataType", type=int,
                        help="The data set that the given model is tailored to. Three types are available: mnist,0; "
                             "cifar10, 1", default=0, required=True)
    parser.add_argument("--testType", type=str,
                        help="Tree types are available: [adv], advesarial data; [normal], test on normal data; [wl],test on wrong labeled data",
                        required=True)
    parser.add_argument("--seedModelPath", type=str,
                        help="the path of the targeted model",
                        required=True)
    parser.add_argument("--device", type=int,
                        help="The index of GPU used. If -1 is assigned,then only cpu is available",
                        required=True)

    args = parser.parse_args()
    threshold, sigma = get_threshold_relax(args.threshold, args.extendScale, args.relaxScale)
    device = "cuda:" + str(args.device) if args.device >= 0 else "cpu"
    seed_model_name = "googlenet" if args.dataType == DATA_CIFAR10 else "lenet"
    seed_model = GoogLeNet() if args.dataType == DATA_CIFAR10 else MnistNet4()

    if args.testType == "normal":
        data_loader = get_data_loader(args.testSamplesPath, is_adv_data=False, data_type=args.dataType)
        avg_accuracy, avg_mutated_used=main('normal', data_loader, args.mutatedModelsPath, seed_model_name, threshold, sigma, 'normal',
             device=device, data_type=args.dataType)
    elif args.testType == "wl":
        seed_model.load_state_dict(torch.load(args.seedModelPath))
        data_loader = get_wrong_label_data_loader(args.testSamplesPath, seed_model, args.dataType,device=device)
        avg_accuracy, avg_mutated_used = main('adv', data_loader, args.mutatedModelsPath, seed_model_name, threshold, sigma,
                                              'wl', device=device,
                                              data_type=args.dataType)
    elif args.testType == "adv":
        data_loader = get_data_loader(args.testSamplesPath, is_adv_data=True, data_type=args.dataType)
        avg_accuracy, avg_mutated_used = main('adv', data_loader, args.mutatedModelsPath, seed_model_name, threshold, sigma,"adv",
                                              device=device, data_type=args.dataType)
    else:
        raise Exception("Unsupported test type.")

<<<<<<< HEAD
    print("average accuracy:{}, average mutants used:{}".format(avg_accuracy,avg_mutated_used))
=======
    print("adverage accuracy:{}, avgerage mutated used:{}".format(avg_accuracy,avg_mutated_used))
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3


if __name__=="__main__":
    run()




