import os
import json
import pprint

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

from options import get_options

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from train import train_epoch, validate
from nets.reinforce_baselines import CriticBaseline
from nets.attention_model import AttentionModel
from utils import torch_load_cpu, load_problem, get_inner_model


def run(opts):


    torch.cuda.empty_cache()
    # Pretty print the run args
    pprint.pprint(vars(opts))

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

    # Figure out what's the problem
    problem = load_problem(opts.problem)(
                            p_size = opts.graph_size, 
                            with_assert = not opts.no_assert)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
            problem = problem,
            embedding_dim = opts.embedding_dim,
            hidden_dim = opts.hidden_dim,
            n_heads = opts.n_heads_encoder,
            n_layers = opts.n_encode_layers,
            normalization = opts.normalization,
            device = opts.device,
        dropout=opts.dropout
        ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # model2 = AveragedModel(model)
    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    
    # Load the validation datasets
    val_dataset = problem.make_dataset(size=opts.graph_size,
                                       num_samples=opts.val_size,
                                       filename = opts.val_dataset)
    
    # Do validation only

    if opts.eval_only:
        validate(problem, model, val_dataset, tb_logger, opts, _id = 0)
        
    else:
        
        # Initialize baseline
        baseline = CriticBaseline(
                CriticNetwork(
                    problem = problem,
                    embedding_dim = opts.embedding_dim,
                    hidden_dim = opts.hidden_dim,
                    n_heads = opts.n_heads_decoder,
                    n_layers = opts.n_encode_layers,
                    normalization = opts.normalization,
                    device = opts.device,
                    dropout=opts.dropout
                ).to(opts.device)
        )
    
        # Load baseline from data, make sure script is called with same type of baseline
        if 'baseline' in load_data:
            baseline.load_state_dict(load_data['baseline'])
    
        # Initialize optimizer
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': opts.lr_model}]
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
    
        # Initialize learning rate scheduler, decay by lr_decay once per epoch!
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    
        if opts.resume:
            epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
    
            torch.set_rng_state(load_data['rng_state'])
            if opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
            # Set the random states
            # Dumping of state was done before epoch callback, so do that now (model is loaded)
            print("Resuming after {}".format(epoch_resume))
            opts.epoch_start = epoch_resume + 1

        # Start the actual training loop
        swa_start = 160
        # Add the SWA strategy
        swa_model = AveragedModel(model)

        # swa_scheduler = SWALR(optimizer,swa_lr=0.05)

        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            # Generate new training data for each epoch
            training_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size)
            training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)

            train_epoch(
                problem,
                model,
                training_dataloader,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                tb_logger,
                opts
            )
            # if epoch > swa_start:
            #     # swa_model.update_parameters(model)
            #     # swa_scheduler.step()
            #     if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
            #         print('Saving model and state...')
            #         torch.save(
            #             {
            #                 'model': get_inner_model(swa_model).state_dict(),
            #                 'optimizer': optimizer.state_dict(),
            #                 'rng_state': torch.get_rng_state(),
            #                 'cuda_rng_state': torch.cuda.get_rng_state_all(),
            #                 'baseline': baseline.state_dict()
            #             },
            #             os.path.join(opts.save_dir, 'epoch-swa-{}.pt'.format(epoch))
            #         )
            #     if epoch % 10 == 0:
            #         #now we need to continue one more step to run batch normalization
            #         update_bn(training_dataloader, swa_model,problem,opts)
            #         # validate the model
            #         validate(problem, swa_model,val_dataset,tb_logger,opts,_id=epoch,is_swa= True)

        # update_bn(training_dataloader, swa_model,problem,opts)
        # # validate the model
        # validate(problem, swa_model, val_dataset, tb_logger, opts, _id=epoch, is_swa=True)


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
            exchange, log_lh = model(x_input,
                                 solution,
                                 exchange,
                                 do_sample=True)
            t+=1
            solution  = problem.step(solution, exchange)
            solution = move_to(solution, opts.device)


    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=Warning)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    run(get_options())