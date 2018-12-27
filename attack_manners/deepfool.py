from abstractAdversary import AbstractAdversary
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
import torchvision.transforms as transforms
from attack_util import save_imgs_tensor

class DeepFool(AbstractAdversary):
    def __init__(self, target_model, num_classes=10, overshoot=0.02, max_iter=50, device='cpu'):
        '''
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
        '''
        self.targeted_model = target_model.to(device)
        self.targeted_model.eval()
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.device = device

    def do_craft(self, image):
        """
           :param image: Image of size HxWx3

           :return: new estimated_label and perturbed image
        """
        f_image = self.targeted_model.forward(
            Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]

        I = I[0:self.num_classes]
        label = I[0]

        input_shape = image.cpu().numpy().shape
        pert_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0

        x = Variable(pert_image[None, :], requires_grad=True)
        #########
        # ????? same to the bb and jsma
        ##########
        x_cp = x.to(self.device)
        fs = self.targeted_model.forward(x_cp)

        fs_list = [fs[0, I[k]] for k in range(self.num_classes)]
        k_i = label

        while k_i == label and loop_i < self.max_iter:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, self.num_classes):
                zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            pert_image = image + (1 + self.overshoot) * torch.from_numpy(r_tot).to(self.device)

            x = Variable(pert_image, requires_grad=True)
            fs = self.targeted_model.forward(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        r_tot = (1 + self.overshoot) * r_tot

        return pert_image, label, k_i,


if __name__ == '__main__':
    from util.data_manger import load_data_set, DATA_CIFAR10
    from torch.utils.data.dataloader import DataLoader
    from cifar10models import lenet
    from cifar10models.lenet import LeNet
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('../cifar10models/')
    test_data, channel = load_data_set(data_type=DATA_CIFAR10, source_data='../../datasets/cifar10/raw')
    loader = DataLoader(dataset=test_data)
    # model = torch.load('../model-storage/cifar10/hetero-base/lenet.pkl')
    model = torch.load('../model-storage/cifar10/hetero-base/googlenet.pkl')
    deepfool = DeepFool(target_model=model)
    count = 1
    for data, label in loader:
        data = data.squeeze(0)
        adv_img, raw_predict, adv_label = deepfool.do_craft(data)
        save_imgs_tensor([adv_img.to('cpu')], [label], [adv_label], '../../datasets/cifar10/adversarial-pure/deepf/single/pure', 'df',
                         channels=channel,no_batch=count)
        print count
        count+=1
        if count > 100:
            break





