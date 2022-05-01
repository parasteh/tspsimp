import json
import os
import pickle
import time
from os import listdir
from os.path import isfile, join
from random import sample

import torch
# load the models
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

from options import get_options
from utils.logger import log_to_tb_val_swa, log_to_tb_val, log_to_screen
from tensorboard_logger import Logger as TbLogger
from torch.utils.data import DataLoader

from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem, get_inner_model, move_to

import warnings

warnings.filterwarnings("ignore", category=Warning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

"""
Select randomly 10 models from the last 20 or 30 models 

"""


# load test_dataset

def validate(problem, models, val_dataset, tb_logger, opts, _id=None, is_swa=False):
    # Validate mode
    print('\nValidating...', flush=True)

    for model_ in models:
        model_.eval()

    init_value = []
    best_value = []
    improvement = []
    reward = []
    time_used = []
    swa_str = ''
    if is_swa:
        swa_str = '_swa'
    for batch in tqdm(DataLoader(val_dataset, batch_size=opts.eval_batch_size),
                      disable=opts.no_progress_bar or opts.val_size == opts.eval_batch_size,
                      desc='validate' + swa_str, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):

        # initial solutions
        initial_solution = move_to(
            problem.get_initial_solutions(opts.init_val_met, batch), opts.device)

        if problem.NAME == 'tsp':
            x = batch

        else:
            assert False, "Unsupported problem: {}".format(problem.NAME)

        x_input = move_to(x, opts.device)  # batch_size, graph_size, 2
        batch = move_to(batch, opts.device)  # batch_size, graph_size, 2

        initial_value = problem.get_costs(batch, initial_solution)
        init_value.append(initial_value)

        # run the model
        s_time = time.time()
        bv, improve, r, _ = rollout(ensemble_random,
                                    problem,
                                    models,
                                    x_input,
                                    batch,
                                    initial_solution,
                                    initial_value,
                                    opts,
                                    T=opts.T_max,
                                    do_sample=True)

        duration = time.time() - s_time
        time_used.append(duration)
        best_value.append(bv.clone())
        improvement.append(improve.clone())
        reward.append(r.clone())

    best_value = torch.cat(best_value, 0)
    improvement = torch.cat(improvement, 0)
    reward = torch.cat(reward, 0)
    init_value = torch.cat(init_value, 0).view(-1, 1)
    time_used = torch.tensor(time_used)

    # log to screen
    log_to_screen(time_used,
                  init_value,
                  best_value,
                  reward,
                  improvement,
                  batch_size=opts.eval_batch_size,
                  dataset_size=len(val_dataset),
                  T=opts.T_max)

    # log to tb
    if (not opts.no_tb):
        if not is_swa:
            log_to_tb_val(tb_logger,
                          time_used,
                          init_value,
                          best_value,
                          reward,
                          improvement,
                          batch_size=opts.eval_batch_size,
                          dataset_size=len(val_dataset),
                          T=opts.T_max,
                          epoch=_id)
        else:
            log_to_tb_val_swa(tb_logger,
                              time_used,
                              init_value,
                              best_value,
                              reward,
                              improvement,
                              batch_size=opts.eval_batch_size,
                              dataset_size=len(val_dataset),
                              T=opts.T_max,
                              epoch=_id)

    # save to file
    swa_str = ''
    if is_swa:
        swa_str = "_swa"
    if _id is not None:
        torch.save(
            {
                'init_value': init_value,
                'best_value': best_value,
                'improvement': improvement,
                'reward': reward,
                'time_used': time_used,
            },
            os.path.join(opts.save_dir, 'validate{}-{}.pt'.format(swa_str, _id)))


def ensemble_average(models, x_input, solutions, exchange, do_sample):
    sum_ms = torch.zeros(x_input.size()[0], problem.size ** 2)
    sum_ms = move_to(sum_ms, device=opts.device)
    for model in models:
        ij, _, m_values = model(x_input, solutions, exchange, do_sample=do_sample)

        sum_ms = sum_ms.add(m_values)
    return sum_ms / len(models)


def ensemble_random(models, x_input, solutions, exchange, do_sample):
    sum_ms = torch.zeros(x_input.size()[0], problem.size ** 2)
    sum_ms = move_to(sum_ms, device=opts.device)
    model_ = sample(models, 1)[0]
    ij, _, m_values = model_(x_input, solutions, exchange, do_sample=False)
    sum_ms = sum_ms.add(m_values)
    return sum_ms / len(models)

def ensemble_weighted_random(models, weights,  x_input, solutions, exchange, do_sample):

    pass

def ensemble_weighted_avg(models, weights,  x_input, solutions, exchange, do_sample):
    pass


def rollout(ensem_func, problem, models, x_input, batch, solution, value, opts, T, do_sample=False, record=False):
    solutions = solution.clone()
    best_so_far = solution.clone()
    cost = value

    exchange = None
    best_val = cost.clone()
    improvement = []
    reward = []
    solution_history = [best_so_far]

    for t in tqdm(range(T), disable=opts.no_progress_bar, desc='rollout',
                  bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        sum_ms = torch.zeros(x_input.size()[0], problem.size ** 2)
        sum_ms = move_to(sum_ms, device=opts.device)

        sum_ms = ensem_func(models, x_input, solutions, exchange, do_sample)
        # sample or select current pair for actions
        pair_index = sum_ms.multinomial(1) if do_sample else sum_ms.max(-1)[1].view(-1, 1)
        col_selected = pair_index % problem.size
        row_selected = pair_index // problem.size
        exchange = torch.cat((row_selected, col_selected), -1)  #

        # new solution
        solutions = problem.step(solutions, exchange)
        solutions = move_to(solutions, opts.device)

        obj = problem.get_costs(batch, solutions)

        # calc improve
        improvement.append(cost - obj)
        cost = obj

        # calc reward
        new_best = torch.cat((best_val[None, :], obj[None, :]), 0).min(0)[0]
        r = best_val - new_best
        reward.append(r)

        # update best solution
        best_val = new_best
        best_so_far[(r > 0)] = solutions[(r > 0)]

        # record solutions
        if record: solution_history.append(best_so_far.clone())

    return best_val.view(-1, 1), torch.stack(improvement, 1), torch.stack(reward,
                                                                          1), None if not record else torch.stack(
        solution_history, 1)


# load the run args
opts = get_options()
mypath = './outputs/tsp_20/run_name_20220329T140348/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

condidate_models = []
last_model_id = 0
last_model_name = ''
for name in onlyfiles:

    if name != 'args.json' and name[0] == 'e' and int(name.split('-')[-1].split('.')[0]) > 80:
        id = int(name.split('-')[-1].split('.')[0])
        if last_model_id < id:
            last_model_id = id
            last_model_name = name
        condidate_models.append(name)

selected_models = sample(condidate_models, 4)
selected_models.append(last_model_name)
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
    p_size=opts.graph_size,  # tsp size
    with_assert=not opts.no_assert)  # if checking feasibiliy
model_class = {
    'attention': AttentionModel,
}.get(opts.model, None)
assert model_class is not None, "Unknown model: {}".format(model_class)




def load_model(model_path, is_swa):
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

    if is_swa:
        model =  AveragedModel(model)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    # load_path = opts.load_path if opts.load_path is not None else opts.resume
    if model_path is not None:
        print('  [*] Loading data from {}'.format(model_path))
        load_data = torch_load_cpu(model_path)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    return model


models = []

for path in selected_models:
    models.append(load_model(mypath + path, is_swa=True))

with open('datasets/tsp_20_10000.pkl', 'rb') as input_file:
    test_set = pickle.load(input_file)

# Validating the model
validate(problem, models, test_set, tb_logger, opts)
