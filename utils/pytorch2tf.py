import sys
sys.path.append("../")
import torch
from models.lenet import MnistNet4
from models.googlenet import GoogLeNet
import os
import pickle
from utils.model_trainer import test
from utils.data_manger import *

def extract_parametes_pytorch(model):
    parameters = {}
    ordered_keys = []
    for item in model.named_parameters():
        parameters[item[0]] = item[1].data.numpy()
        ordered_keys.append(item[0])
    return parameters,ordered_keys

def transfer2tf_minst(modelName):
    datatype = "mnist"
    modelPath = "../build-in-resource/pretrained-model/" + "/" + datatype + "/" + +modelName + ".pkl"
    tmp_to_save = os.path.join("./", "parameter_" + modelName + ".pkl")
    if modelName == "googlenet":
        seed_model = GoogLeNet()
    elif modelName == "lenet":
        seed_model = MnistNet4()
    seed_model.load_state_dict(torch.load(modelPath))
    parameters, ordered_keys = extract_parametes_pytorch(seed_model)
    with open(tmp_to_save, "w") as f:
        pickle.dump({"parameters": parameters, "ordered_keys": ordered_keys}, f)

if __name__ == '__main__':
    modelName = "lenet"
    datatype = "mnist"
    modelPath = "../build-in-resource/pretrained-model/"+"/"+datatype+"/"+modelName+".pkl"
    # tmp_to_save = os.path.join("./","parameter_"+modelName+".pkl")
    # if modelName == "googlenet":
    #     seed_model = GoogLeNet()
    # elif modelName == "lenet":
    #     seed_model = MnistNet4()
    # seed_model.load_state_dict(torch.load(modelPath))
    # parameters, ordered_keys = extract_parametes_pytorch(seed_model)
    # with open(tmp_to_save,"w") as f:
    #     pickle.dump({"parameters":parameters,"ordered_keys":ordered_keys},f)
    seed_model = MnistNet4()
    seed_model.load_state_dict(torch.load(modelPath))

    source_data = '../build-in-resource/dataset/' + datatype + '/raw'
    test_data, channel = load_data_set(DATA_MNIST, source_data=source_data)
    test_data_laoder = DataLoader(dataset=test_data, batch_size=64, num_workers=4)
    test(seed_model, test_data_laoder, verbose=True, device='cpu')





