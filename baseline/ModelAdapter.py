import torch.nn as nn
import torch.nn.functional as F


class Adapter(object):
    def last_hd_layer_output(self, x):
        pass
    def get_predict_lasth(self,input):
        h = self.last_hd_layer_output(input)
        output = self.model.fc3(h)
        pred = output.max(1, keepdim=True)[1]
        assert isinstance(pred,int),"return type error"
        return pred,h


class MnistNet4Adapter(Adapter):
    def __init__(self,model):
        self.model = model
    def last_hd_layer_output(self, input):
        x = self.model.conv1(input)
        x = self.model.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.model.fc1(x))
        x = F.relu(self.model.fc2(x))
        return x


class Cifar10NetAdapter(Adapter):
    def __init__(self,model):
        self.model = model
    def last_hd_layer_output(self, input):
        out = F.relu(self.modelconv1(input))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.model.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.model.fc1(out))
        out = F.relu(self.model.fc2(out))
        return out

