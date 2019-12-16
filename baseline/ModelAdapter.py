import torch.nn as nn
import torch.nn.functional as F
import torch

class Adapter(object):
    def get_dropout_ouput(self,input):
        pass
    def last_hd_layer_output(self, x):
        pass

    def get_predict_lasth(self,input):
        h = self.last_hd_layer_output(input)
        output = self.output_layer(h)
        pred = torch.squeeze(output.max(1, keepdim=True)[1]).item()
        assert isinstance(pred,int),"return type error"
        return pred,h


class MnistNet4Adapter(Adapter):
    def __init__(self,model,dp=0.5):
        self.model = model
        self.dropout = nn.Dropout(dp)
        self.dropout2d = nn.Dropout2d(dp)
        self.softmax = nn.Softmax()
        self.output_layer = self.model.fc3
    def last_hd_layer_output(self, input):
        x = self.model.conv1(input)
        x = self.model.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.model.fc1(x))
        x = F.relu(self.model.fc2(x))
        return x

    def get_dropout_ouput(self,input):
        '''
        with a dropout rate of 0.5 after last pooling layer and after the inner-product layer
        '''
        x = self.model.conv1(input)
        x = self.model.conv2(x)
        x = self.dropout2d(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.model.fc1(x))
        x = F.relu(self.model.fc2(x))
        x = self.dropout(x)
        x = self.model.fc3(x)
        x = self.softmax(x)
        return x



class JingyiNetAdapter(Adapter):
    def __init__(self,model,dp=0.5):
        self.model = model
        self.dropout = nn.Dropout(dp)
        self.dropout2d = nn.Dropout2d(dp)
        self.softmax = nn.Softmax()
    def last_hd_layer_output(self, input):
        x = self.model.conv1(input)
        x = self.model.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc1(x)
        return x
    def get_predict_lasth(self,input):
        h = self.last_hd_layer_output(input)
        output = self.model.fc2(h)
        pred = torch.squeeze(output.max(1, keepdim=True)[1]).item()
        assert isinstance(pred,int),"return type error"
        return pred,h

    def get_dropout_ouput(self,input):
        '''
        with a dropout rate of 0.5 after last pooling layer and after the inner-product layer
        '''
        x = self.model.conv1(input)
        x = self.model.conv2(x)
        x = self.dropout2d(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.model.fc1(x))
        x = F.relu(self.model.fc2(x))
        x = self.dropout(x)
        x = self.model.fc3(x)
        x = self.softmax(x)
        return x


class Cifar10NetAdapter(Adapter):
    def __init__(self,model,dp=0.5):
        self.model = model
        self.dropout = nn.Dropout(dp)
        self.dropout2d = nn.Dropout2d(dp)
        self.softmax = nn.LogSoftmax()
    def last_hd_layer_output(self, input):
        out = F.relu(self.modelconv1(input))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.model.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.model.fc1(out))
        out = F.relu(self.model.fc2(out))
        return out




class GooglenetAdapter(Adapter):
    def __init__(self,model):
        self.model = model
        self.output_layer = self.model.linear

    def last_hd_layer_output(self, input):
        out = self.model.pre_layers(input)
        out = self.model.a3(out)
        out = self.model.b3(out)
        out = self.model.maxpool(out)
        out = self.model.a4(out)
        out = self.model.b4(out)
        out = self.model.c4(out)
        out = self.model.d4(out)
        out = self.model.e4(out)
        out = self.model.maxpool(out)
        out = self.model.a5(out)
        out = self.model.b5(out)
        out = self.model.avgpool(out)
        out = out.view(out.size(0), -1)
        return out





