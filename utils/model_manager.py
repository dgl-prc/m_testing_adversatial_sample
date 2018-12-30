
import re
import os
import copy
import torch
def NoModel(name):
    '''
     Get the serial No. of a given model's name
    :param name: the model name.
    :return:
    '''
    pattern = re.compile(".*-m(\d+).pkl")
    match = pattern.match(name)
    return int(match.group(1))

def fetch_models(models_folder, num_models,device, seed_model,start_no=1):
    '''
    :param models_folder:
    :param num_models: the number of models to be load
    :param start_no: the start serial number from which loading  "num_models" models. 1-index
    :return: the top [num_models] models in models_folder
    '''
    files = os.listdir(models_folder)
    files.sort(key=lambda x: NoModel(x))
    if num_models + start_no <= len(files):
        batch_models_name = files[start_no - 1:num_models + start_no-1]
    else:
        batch_models_name = files[start_no - 1:]
    target_models = []
    for model_name in batch_models_name:
        model = copy.deepcopy(seed_model)
        model.load_state_dict(torch.load(os.path.join(models_folder, model_name)))

        target_models.append(model.to(device))
    return target_models