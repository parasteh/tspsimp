import json
import os
import pickle
from os import listdir
from os.path import isfile, join
from random import sample
from tensorboard_logger import Logger as TbLogger
from torch import optim
from torch.utils.data import DataLoader

from nets.attention_model import AttentionModel
from nets.critic_network import CriticNetwork
from nets.reinforce_baselines import CriticBaseline
from train import validate
from utils import load_problem, torch_load_cpu, get_inner_model

import torch
from torch.optim.swa_utils import AveragedModel, SWALR



#given the trained directory of the model
from options import get_options


trained_directory=  "./outputs/tsp_100/run_name_20220429T201155/"

# with open('datasets/tsp_20_10000.pkl', 'rb') as input_file:
#     test_set = pickle.load(input_file)

opts = get_options()

# Save arguments so exact configuration can always be found
with open(os.path.join(trained_directory, "args.json"), 'w') as f:
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

test_set = problem.make_dataset(size=opts.graph_size,
                                       num_samples=opts.val_size,
                                       filename=opts.val_dataset)

# Load baseline from data, make sure script is called with same type of baseline





# Set the random seed
torch.manual_seed(opts.seed)
# Optionally configure tensorboard
tb_logger = None
if not opts.no_tensorboard:
    tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)



def get_list_of_trained_models(dir_path, swa_start):
    # load the run args

    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]

    condidate_models = {}
    ordered = []
    for name in onlyfiles:

        if name != 'args.json' and name[0] == 'e' and int(name.split('-')[1].split('.')[0]) > swa_start:
            ordered.append(int(name.split('-')[1].split('.')[0]))
            condidate_models[int(name.split('-')[1].split('.')[0])] = name
    temp = []
    ordered = sorted(ordered)
    for i in ordered:
        temp.append(condidate_models[i])
    return temp

def init_model():
    model_temp = model_class(
        problem=problem,
        embedding_dim=opts.embedding_dim,
        hidden_dim=opts.hidden_dim,
        n_heads=2,  # can increase for better performance
        n_layers=opts.n_encode_layers,
        dropout=opts.dropout,
        normalization=opts.normalization,
        device=opts.device,
    ).to(opts.device)

    # Initialize baseline
    baseline = CriticBaseline(
        CriticNetwork(
            problem=problem,
            embedding_dim=opts.embedding_dim,
            hidden_dim=opts.hidden_dim,
            n_heads=opts.n_heads_decoder,
            n_layers=opts.n_encode_layers,
            dropout=opts.dropout,
            normalization=opts.normalization,
            device=opts.device
        ).to(opts.device)
    )

    return model_temp, baseline

def load_models(models_path):
    models = []
    optimizers= []
    base_lines = []
    for path  in models_path:

        model_temp, baseline  = init_model()
        load_data = {}

        if path is not None:
            print('  [*] Loading data from {}'.format(path))
            load_data = torch_load_cpu(trained_directory+path)

        # Overwrite model parameters by parameters to load
        model_ = get_inner_model(model_temp)
        model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
        models.append(model_temp)
        if 'baseline' in load_data:
            baseline.load_state_dict(load_data['baseline'])
        if opts.use_cuda and torch.cuda.device_count() > 1:
            model_temp = torch.nn.DataParallel(model_temp)

        optimizer = optim.Adam(
            [{'params': model_temp.parameters(), 'lr': opts.lr_model}]
            + (
                [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
                if len(baseline.get_learnable_parameters()) > 0 else []
            )
        )
        # Load optimizer state
        if 'optimizer' in load_data:
            optimizer.load_state_dict(load_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    # if isinstance(v, torch.Tensor):
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)
        optimizers.append(optimizer)
        base_lines.append(baseline)

    return models, optimizers, base_lines


training_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size)
training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)

def do_swa(models, optimizers, baselines, k):

    # Add the SWA strategy
    model = models[0]
    optimizer = optimizers[0]
    baseline = baselines[0]

    #custome avg function
    # ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
    #     (num_averaged * averaged_model_parameter + model_parameter) / ( num_averaged + 1)

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    c=1

    for model_index in range(1,len(models)):
        optimizer = optimizers[model_index]
        baseline = baselines[model_index]
        if (model_index % c) == 0:
            swa_model.update_parameters(models[model_index])
            swa_scheduler.step()

            torch.save(
                    {
                        'model': get_inner_model(swa_model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state_all(),
                        'baseline': baseline.state_dict()
                    },
                    os.path.join(opts.save_dir, 'epoch-swa-{}.pt'.format(model_index + swa_start))
                )

        # now we need to continue one more step to run batch normalization
    print
    update_bn(training_dataloader, swa_model, problem, opts)
    torch.save(
        {
            'model': get_inner_model(swa_model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'baseline': baseline.state_dict()
        },
        os.path.join(opts.save_dir, 'epoch-swa-{}.pt'.format(model_index + swa_start+1))
    )
    # validate the model
    validate(problem, swa_model, test_set[:1000], tb_logger, opts, _id=model_index+swa_start, is_swa=True)





@torch.no_grad()
def update_bn(loader, model,problem,opts, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for batch_id, batch in enumerate(loader):
        from utils import move_to
        solution = move_to(problem.get_initial_solutions(opts.init_val_met, batch), opts.device)
        if problem.NAME == 'tsp':
            x = batch
        x_input = move_to(x, opts.device)  # batch_size, graph_size, 2
        batch = move_to(batch, opts.device)  # batch_size, graph_size, 2
        T = opts.T_train
        t = 0
        exchange = None
        while t < T:
            exchange, log_lh,_ = model(x_input,
                                 solution,
                                 exchange,
                                 do_sample=True)
            t+=1
            solution  = problem.step(solution, exchange)
            solution = move_to(solution, opts.device)


    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

####################
swa_start = 50


####################

if __name__== '__main__':

    models_name = get_list_of_trained_models(trained_directory,swa_start)
    models, optimizers, baselines = load_models(models_name)
    do_swa(models, optimizers, baselines,1)
    print()

