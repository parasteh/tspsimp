#load the models

import json
import os
import pickle
# torch.cuda.empty_cache()
import warnings
from os import listdir
from os.path import isfile, join

import torch
from tensorboard_logger import Logger as TbLogger
from torch.optim.swa_utils import AveragedModel

from nets.attention_model import AttentionModel
from options import get_options
from train import validate
from utils import torch_load_cpu, load_problem, get_inner_model

warnings.filterwarnings("ignore", category=Warning)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
Select randomly 10 models from the last 20 or 30 models 

"""


# load test_dataset


# load the run args
opts = get_options()





# Set the random seed
torch.manual_seed(opts.seed)

# Optionally configure tensorboard
tb_logger = None
if not opts.no_tensorboard:
    tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)

# Save arguments so exact configuration can always be found
with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
    json.dump(vars(opts), f, indent=True)

 # Set the device
opts.device = torch.device("cuda" if opts.use_cuda else "cpu")

problem = load_problem(opts.problem)(
                        p_size = opts.graph_size, # tsp size
                        with_assert = not opts.no_assert) # if checking feasibiliy
model_class = {
    'attention': AttentionModel,
}.get(opts.model, None)
assert model_class is not None, "Unknown model: {}".format(model_class)




def load_model(model_path):
    model = model_class(
        problem=problem,
        embedding_dim=opts.embedding_dim,
        hidden_dim=opts.hidden_dim,
        n_heads=1,  # can increase for better performance
        n_layers=opts.n_encode_layers,
        dropout=opts.dropout,
        normalization=opts.normalization,
        device=opts.device,
    ).to(opts.device)
    model = AveragedModel(model)
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Load data from load_path
    load_data = {}

    if model_path is not None:
        print('  [*] Loading data from {}'.format(model_path))
        load_data = torch_load_cpu(model_path)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    return model

def get_list_of_trained_models(dir_path, swa_start):
    # load the run args


    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

    condidate_models = {}
    ordered = []
    for name in onlyfiles:


        if name != 'args.json' and name[0] == 'e':
            id = int(name.split('-')[2].split('.')[0])

            ordered.append(id)
            condidate_models[id]= name
    temp = []
    ordered = sorted(ordered)
    for i in ordered:
        temp.append(condidate_models[i])
    return temp


if __name__ == '__main__':
    val_file = "datasets/tsp_{}_10000.pkl".format(opts.graph_size)
    with open(val_file,'rb') as input_file:
        val_dataset = pickle.load(input_file)

    TRAINED_DIRECORY = "./outputs/tsp_50/run_name_20220427T155200/"
    models_name = get_list_of_trained_models(TRAINED_DIRECORY,99)


    for name in models_name:
        # Load the validation datasets
        model_swa = load_model(TRAINED_DIRECORY + name)
        # Validating the model
        validate(problem, model_swa, val_dataset, tb_logger, opts, _id=int(name.split('-')[2].split('.')[0]) )