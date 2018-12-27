
import torch
import os

def fetch_models(models_folder, num_models, seed_name, device, start_no=1):
    '''
    :param models_folder:
    :param num_models: the number of models to be load
    :param start_no: the start serial number from which loading  "num_models" models
    :return: the top [num_models] models in models_folder
    '''
    No_models = range(start_no, start_no + num_models)
    target_models_name = [seed_name + '-m' + str(i) for i in No_models]
    target_models = []
    for model_name in target_models_name:
        target_models.append(torch.load(os.path.join(models_folder, seed_name, model_name + '.pkl')).to(device))
        # logging.warning("loading {} ...".format(model_name))
    return target_models