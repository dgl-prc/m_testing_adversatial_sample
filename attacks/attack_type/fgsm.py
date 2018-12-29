import torch
import torch.nn.functional as F
from abstractAdversary import AbstractAdversary
import logging

PREDICT=0
TRUELABEL=1
class FGSM(AbstractAdversary):

    def __init__(self, model, criterion=F.nll_loss, eps=0.1, device='cpu',target_type=TRUELABEL):
        self.model = model.to(device)
        self.model.eval()
        self.eps = eps
        self.criterion = criterion
        self.device = device
        self.target_type = target_type

    def do_craft(self, inputs, target_labels):
        """
        This is the untargeted attack.
        If we use the targeted attack, then the "true labels" shoule be replaced with the targeted label
        and "crafting_input" should be subtracted a constant instead of addition
        :param inputs: Clean samples (Batch X Size)
        :param targets: True labels
        :param model: Model
        :param criterion: Loss function
        :param gamma:
        :return:
        """

        crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
        crafting_target = torch.autograd.Variable(target_labels.clone())
        output = self.model(crafting_input)
        loss = self.criterion(output, crafting_target)
        if crafting_input.grad is not None:
            crafting_input.grad.data.zero_()
        loss.backward()
        crafting_output = crafting_input.data + self.eps * torch.sign(crafting_input.grad.data)

        return crafting_output

    def do_craft_batch(self, data_loader,is_verbose=True):
        adv_samples = []
        target_tensor = []
        # switch to evaluation mode
        total_batch = (1.*len(data_loader.dataset))/data_loader.batch_size
        for bi, batch in enumerate(data_loader):
            inputs, y = batch
            inputs, y = inputs.to(self.device), y.to(self.device)
            if self.target_type == PREDICT:
                predict_scores = self.model(inputs)
                y = torch.argmax(predict_scores,dim=1,keepdim=False)
            crafted = self.do_craft(inputs, y)
            adv_samples.append(crafted)
            target_tensor.append(y)
            if is_verbose:
                logging.info('fgsm====>{}th batch,total batches:{}'.format(bi,total_batch))
        return torch.cat(adv_samples, 0), torch.cat(target_tensor, 0)
