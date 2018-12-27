import sys
sys.path.append("../")
from model_mutation.operator import *
from numpy.random import RandomState
from utils import logging_util
from utils.data_manger import *
from utils.time_util import current_timestamp
from model_mutation.operator import OpType as OP_TYPE
import argparse
from models.googlenet import *
from models.lenet import *

def yield_mutated_model(operator, op_type):

    if op_type == OP_TYPE.GF:
        mutated_model = operator.filter(operator.gaussian_fuzzing)
    elif op_type == OP_TYPE.NAI:
        mutated_model = operator.filter(operator.nai)
    elif op_type == OP_TYPE.NS:
        mutated_model = operator.filter(operator.ns)
    elif op_type == OP_TYPE.WS:
        mutated_model = operator.filter(operator.ws)
    else:
        raise Exception("No such Operator!")
    return mutated_model


def mutated_single_op(num_models, operator, op_type, save_path=None, seed_md_name=None):
    '''
    Generate mutated models via a specific operator
    :param num_models: the number of mutated models to yield
    :param operator: MutaionOperator object
    :param op_type: the operator type. Four types are available: OpType.NAI, OpType.GF, OpType.WS, OpType.NS
    :param save_path: the path where the mutated models to save
    :param seed_md_name: the name of the seed model
    :return: the list of mutated models. Be careful that if the number of mutated models is large and the size of
             seed model is big, the memory can be overflow.
    '''
    mutated_models = []
    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    for i in range(1, num_models + 1):
        mutated_name = seed_md_name + '-m' + str(i) + '.pkl'
        mutated_model = yield_mutated_model(operator, op_type)

        if is_save:
            path = os.path.join(save_path, mutated_name)
            if path and os.path.exists(path):  # proceed the progress
                continue
            else:
                torch.save(mutated_model, path)
        else:
            mutated_models.append(mutated_model)

        logging.info('Progress:{}/{}'.format(i, num_models))
    return mutated_models


def mutated_hybrid_op(num_models, operator, seed, operator_types, save_path=None, seed_md_name=None):
    '''
    Generate mutated models via a group of operator types
    :param num_models: the number of mutated models to yield
    :param operator: MutaionOperator instance
    :param seed: the random seed
    :param save_path: the path where the mutated models to save
    :param seed_md_name: the name of the seed model
    :param operator_types: the list of operator types. mutated models will be generated with randomly selected operator types
    :return: the list of mutated models. Be careful that if the number of mutated models is large and the size of
             seed model is big, the memory can be overflow.
    '''
    mutated_models = []
    rState = RandomState(seed)
    method_list = rState.choice(operator_types, num_models)

    is_save = False
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_save = True

    for i, op_type in enumerate(method_list):
        mutated_name = seed_md_name + '-m' + str(i) + op_type + '.pkl'
        mutated_model = yield_mutated_model(operator, op_type)
        if is_save:
            path = os.path.join(save_path, mutated_name)
            if path and os.path.exists(path):  # proceed the progress
                continue
            else:
                torch.save(mutated_model, path)
        else:
            mutated_models.append(mutated_model)

        logging.info('Progress:{}/{}'.format(i, num_models))
    return mutated_models


def batch_mutated_model(model, model_name, test_loader, op_type, acc_tolerant, mutated_ration, num_mutated_models,
                        save_path, device):
    '''

    :param model: nn.Moduel, the seed model
    :param model_name: str, the name of seed model
    :param test_loader: DataLoader, the loader of test dataset
    :param op_type: str,   NAI, GF , WS , NS
    :param acc_tolerant: float, which proportion of the accuracy of the seed model that the mutated model need to reach
    :param mutated_ration: float, the num of neurons to be involved
    :param num_mutated_models: int, the num of mutated model needed to yield
    :param save_path: str or os.path,  where the mutated model to save
    :return:
    '''

    logging.info('>>>>>>>>>>>>Start-new-experiment>>>>>>>>>>>>>>>>')
    operator = MutaionOperator(ration=mutated_ration, model=model, verbose=False, acc_tolerant=acc_tolerant,
                               test_data_laoder=test_loader, device=device)
    logging.info(
        'seed_md_name:{},op_type:{},ration:{},acc_tolerant:{},num_mutated:{}'.format(model_name, op_type,
                                                                                     mutated_ration, acc_tolerant,
                                                                                     num_mutated_models))
    save_path = os.path.join(save_path, model_name)
    mutated_single_op(num_mutated_models, operator, op_type, save_path, model_name)


def main():
    parser = argparse.ArgumentParser(description="The required parameters of mutation process")

    parser.add_argument("--modelName", type=str,
                        help="The model's name,e.g, googlenet, lenet, or the self-defined name by users",
                        default="lenet")
    parser.add_argument("--modelPath", type=str, help="The the path of pretrained model(seed model).Note, the model should be saved as the form model.stat_dict")
    parser.add_argument("--accRation", type=float,
                        help="The ration of the seed model's accuracy that the mutated model should achieve",
                        default=0.9)
    parser.add_argument("--dataType", type=int,
                        help="The data set that the given model is tailored to. Three types are available: mnist,0; "
                             "cifar10, 1",default=0,required=True)
    parser.add_argument("--numMModels", type=int, help="The number of mutated models to be generated",
                        default=10,
                        required=True)
    parser.add_argument("--mutatedRation", type=float, help="Model mutation ration.The percent of neurons to be mutated",
                        required=True)
    parser.add_argument("--opType", type=str, help="The type of operator used. Four types are availabel: NAI,GF,WS,NS",
                        required=True)
    parser.add_argument("--savePath",type=str,help="The path where the mutated models are stored",
                        required=True)
    parser.add_argument("--device", type=int, help="The index of GPU used. If -1 is assigned,then only cpu is available",
                        required=True)

    args = parser.parse_args()

    if args.modelName == "googlenet":
        seed_model = GoogLeNet()
    elif args.modelName == "lenet":
        seed_model = MnistNet4()
    seed_model.load_state_dict(torch.load(args.modelPath))

    device = "cuda:"+str(args.device) if args.device >=0 else "cpu"

    if args.dataType == DATA_CIFAR10:
        data_name = 'cifar10'
    elif args.dataType == DATA_MNIST:
        data_name = 'mnist'

    logging_util.setup_logging()
    logging.info("data type:{}".format(data_name))

    source_data = '../build-in-resource/dataset/' + data_name + '/raw'
    test_data, channel = load_data_set(args.dataType, source_data=source_data)
    test_data_laoder = DataLoader(dataset=test_data, batch_size=64, num_workers=4)

    save_path = os.path.join(args.savePath, current_timestamp().replace(" ", "_"),
                             args.opType.lower() + str(args.mutatedRation))

    batch_mutated_model(model=seed_model,
                        model_name=args.modelName,
                        test_loader=test_data_laoder,
                        num_mutated_models=args.numMModels,
                        mutated_ration=args.mutatedRation,
                        acc_tolerant=args.accRation,
                        op_type=args.opType,
                        save_path=save_path,
                        device=device)
    print("The mutated models are stored in {}/{}".format(save_path,args.modelName))


def hard_code():
    torch.manual_seed(random_seed)

    ###########
    # CIFAR0 setting
    ############
    data_type = DATA_CIFAR10
    data_name = 'cifar10'
    seed_md_name = 'googlenet'

    ###########
    # mnsit setting
    ############
    # data_type = DATA_MNIST
    # data_name = 'mnist'
    # seed_md_name = 'MnistNet4'

    ###########
    # general setting
    ###########
    acc_tolerant = 0.9
    num_mutated_models = 500

    source_data = '../datasets/' + data_name + '/raw'
    save_path = './model-storage/' + data_name + '/mutaed-models/'
    model_path = './model-storage/' + data_name + '/hetero-base/'

    ###########
    # general
    ###########
    logging_util.setup_logging()
    logging.info("data type:{}".format(data_name))
    seed_model = torch.load(os.path.join(model_path, seed_md_name + '.pkl'))
    test_data, channel = load_data_set(data_type, source_data=source_data)
    test_data_laoder = DataLoader(dataset=test_data, batch_size=64, num_workers=4)
    # OP_TYPE.NS,OP_TYPE.GF
    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    device = sys.argv[1]
    device = 'cuda:' + device
    for op_type in [OP_TYPE.GF]:
        for ration, folder in zip([0.007], ['7e-3p']):
            # for ration, folder in zip([0.01, 0.03, 0.05], ['1e-2p', '3e-2p', '5e-2p']):
            # for ration, folder in zip([0.001, 0.003, 0.005], ['1e-3p', '3e-3p', '5e-3p']):
            op_type_name = op_type
            logging.info('operator type:{}'.format(op_type_name))
            batch_mutated_model(model=seed_model,
                                model_name=seed_md_name,
                                test_loader=test_data_laoder,
                                num_mutated_models=num_mutated_models,
                                mutated_ration=ration,
                                acc_tolerant=acc_tolerant,
                                op_type=op_type,
                                save_path=os.path.join(save_path, op_type_name.lower(), folder),
                                device=device)

if __name__ == '__main__':
    main()
