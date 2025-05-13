import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from data_io import run_subdir_setup, runinfo_append
from data_tools import (data_train_test_split_linear, data_train_test_split_clusters, data_train_test_split_manifold,
                        DatasetWrapper, plot_one_prediction_batch, plot_batch_predictions)
from nn_loss_baselines import (loss_if_predict_zero, loss_if_predict_mostrecent, loss_if_predict_average,
                               loss_if_predict_linalg, loss_if_predict_linalg_shrunken,
                               loss_if_predict_subsphere_baseline, loss_if_predict_clustering_baseline,
                               theory_linear_expected_error, report_dataset_loss)
from nn_model_base import (MODEL_CLASS_FROM_STR, load_modeltype_from_fname)
from settings import *
from visualize import vis_weights_kq_pv


def train_model(restart_nn_instance=None,
                restart_dataset=None,
                nn_model=None,
                context_len=None,
                dim_n=None,
                datagen_case=0,
                datagen_kwargs=None,
                seed_torch=None,
                epochs=40,
                batch_size=100,
                optimizer_choice='adam',
                optimizer_lr=0.01,
                scheduler_kwargs=None,
                train_plus_test_size=1000,
                full_loss_sample_interval=4,
                flag_save_dataset=False,
                flag_vis_loss=True,
                flag_vis_weights=True,
                flag_vis_batch_perf=True,
                dir_runs=DIR_RUNS):
    """
    :param restart_nn_instance:         instance of a sequence model class (e.g. TransformerModelV1)
    :param restart_dataset:             (None) or 6-tuple (X_train, y_train, X_test, y_test, train_dict, test_dict) for dataset
    :param nn_model:                    string for a sequence model class (e.g. 'TransformerModelV1')
    :param context_len:                 (default: 500) int, context length of input token sequence (X.shape[-1])
    :param dim_n:                       (default: 32) int, ambient dimension
    :param datagen_case:                (default: settings.py) int in {0, 1, 2} for 'Linear', 'Clustering', or 'Manifold'
    :param datagen_kwargs:              (default: settings.py) dict of kwargs for data generation
    :param seed_torch:                  (default: None) None or an int (sets net weights; possibly batch order/shuffle)
    :param seed_dataset:                (default: None) None or an int (sets dataset rng) -> moved into datagen_kwargs
    :param flag_vis_loss:               bool
    :param dir_runs:                    (default: DIR_RUNS) aka 'experiment_dir' place where io_dict points/stores run
    :return:
    """

    # model parameters and class
    if context_len is None:
        context_len = 500     # this will induce the synthetic dataset via data_train_test_split_linear
    if dim_n is None:
        dim_n = 32            # this will induce the synthetic dataset ^^ (try 32 to 128)
    if nn_model is None:
        # select a model class:
        # V1 models (linear self-attention)
        #   - TransformerModelV1
        #   - TransformerModelV1nores
        #   - TransformerModelV1noresForceDiag
        #   - TransformerModelV1noresOmitLast
        #   - TransformerModelV1noresForceDiagAndOmitLast
        # V2 models (softmax self-attention)
        #   - TransformerModelV2
        #   - TransformerModelV2nores
        #   - TransformerModelV2noresOmitLast
        nn_model = 'TransformerModelV1noresOmitLast'
        print('warning: defaulting nn_model=None to', nn_model)
    if seed_torch is not None:
        torch.manual_seed(seed_torch)

    assert datagen_case in DATASET_CASES.keys()
    datagen_case_label = DATASET_CASES[datagen_case]
    if datagen_kwargs is None:
        datagen_kwargs = DATAGEN_GLOBALS[datagen_case]

    # check the kwargs for the three cases
    dgkw = datagen_kwargs
    if datagen_case == 0:
        data_train_test_split_fncall = data_train_test_split_linear
        # these keys are assumed for this case (list all the ones used later in this training function)
        for k in ['sigma2_corruption', 'sigma2_pure_context',
                  'style_corruption_orthog', 'style_origin_subspace', 'style_subspace_dimensions']:
            if k not in datagen_kwargs.keys():
                print('missing key:', k, 'in datagen_kwargs - will set to default (', DATAGEN_GLOBALS[datagen_case][k], ') in settings.py')
                datagen_kwargs[k] = DATAGEN_GLOBALS[datagen_case][k]

        # specific to this case but used in function calls below
        linear_sigma2_pure_context = datagen_kwargs['sigma2_pure_context']
        sigma2_pure_context = linear_sigma2_pure_context  # alias

        data_suffix = 'case%d_s2z%.2f_s2n%.2f_ortho%d_origin%d_d-%s' % (
            datagen_case, dgkw['sigma2_pure_context'], dgkw['sigma2_corruption'], dgkw['style_corruption_orthog'],
            dgkw['style_origin_subspace'], dgkw['style_subspace_dimensions'])

    elif datagen_case == 1:
        data_train_test_split_fncall = data_train_test_split_clusters
        # these keys are assumed for this case
        for k in ['sigma2_corruption', 'cluster_var', 'num_cluster', 'style_cluster_mus', 'style_cluster_vars',
                  'style_corruption_orthog', 'style_origin_subspace', 'style_subspace_dimensions']:
            if k not in datagen_kwargs.keys():
                print('missing key:', k, 'in datagen_kwargs - will set to default (', DATAGEN_GLOBALS[datagen_case][k], ') in settings.py')
                datagen_kwargs[k] = DATAGEN_GLOBALS[datagen_case][k]

        data_suffix = 'case%d_s2z%.2f_s2n%.2f_ortho%d_origin%d_d-%s' % (
            datagen_case, dgkw['cluster_var'], dgkw['sigma2_corruption'], dgkw['style_corruption_orthog'],
            dgkw['style_origin_subspace'], dgkw['style_subspace_dimensions'])

        # specific to this case but used in function calls below
        cluster_var = datagen_kwargs['cluster_var']
        sigma2_pure_context = cluster_var  # alias

    else:
        assert datagen_case == 2
        # these keys are assumed for this case
        for k in ['sigma2_corruption', 'radius_sphere',
                  'style_corruption_orthog', 'style_origin_subspace', 'style_subspace_dimensions']:
            if k not in datagen_kwargs.keys():
                print('missing key:', k, 'in datagen_kwargs - will set to default (', DATAGEN_GLOBALS[datagen_case][k],
                      ') in settings.py')
                datagen_kwargs[k] = DATAGEN_GLOBALS[datagen_case][k]

        data_train_test_split_fncall = data_train_test_split_manifold

        data_suffix = 'case%d_s2z%.2f_s2n%.2f_ortho%d_origin%d_d-%s' % (
            datagen_case, dgkw['radius_sphere'], dgkw['sigma2_corruption'], dgkw['style_corruption_orthog'],
            dgkw['style_origin_subspace'], dgkw['style_subspace_dimensions'])

        # specific to this case but used in function calls below
        radius_sphere = datagen_kwargs['radius_sphere']
        sigma2_pure_context = radius_sphere  # alias

    # shorthand aliases used below
    sigma2_corruption = dgkw['sigma2_corruption']                  # float
    style_corruption_orthog = dgkw['style_corruption_orthog']      # True/False (orthog OR ball)
    style_origin_subspace = dgkw['style_origin_subspace']          # True/False
    style_subspace_dimensions = dgkw['style_subspace_dimensions']  # int or 'random' (or just 'full' in clustering case)

    """ How many samples per in-context subspace? - We use option (A) (explained below) throughout 
    # (X, y) samples style A
    num_W_in_dataset = train_plus_test_size
    context_examples_per_W = 1
    samples_per_context_example = 1
    
    # (X, y) samples style B
    num_W_in_dataset = 1
    context_examples_per_W = train_plus_test_size
    samples_per_context_example = 1
    
    # (X, y) samples style C
    num_W_in_dataset = 100
    context_examples_per_W = 1
    samples_per_context_example = 100
    """
    num_W_in_dataset = dgkw['num_W_in_dataset']  # we assert context_examples_per_W = 1, samples_per_context_example = 1
    context_examples_per_W = dgkw['context_examples_per_W']
    samples_per_context_example = dgkw['samples_per_context_example']
    assert context_examples_per_W == 1 and samples_per_context_example == 1
    train_plus_test_size == context_examples_per_W * num_W_in_dataset * samples_per_context_example  # sanity check
    # this means train_plus_test_size == context_examples_per_W   i.e. we use option (A) throughout

    test_ratio = dgkw['test_ratio']  # 0.2 means 1000 -> 800, 200 samples for train/test
    seed_dataset = dgkw['seed']

    # Data parameters
    if flag_save_dataset:
        print('Warning - flag_save_dataset - 1000 samples with n=32 still gives 60 MB, relatively big)')

    # Training loop hyperparameters
    period_save_weights = 1  # save weights every k epochs, starting from 0

    # From the provided model class string, get the shorthand model name and the class definition
    nn_fpath = MODEL_CLASS_FROM_STR[nn_model]['alias']
    nn_class = MODEL_CLASS_FROM_STR[nn_model]['class']

    # Optimizer settings (or create custom one below) | currently 'sgd' or 'adam'
    if optimizer_lr is None:
        sgd_lr = 0.05   # start: 0.001
        adam_lr = 1e-2  # default (torch) 1e-3 | 1e-2 also works
    else:
        sgd_lr = optimizer_lr
        adam_lr = optimizer_lr
    sgd_momentum = 0.0  # start: 0.9   # 0.9 momentum helped V1nores train for lr 0.01

    # Set device
    #device = device_select()
    device = 'cpu'
    print('Current device:', device)

    # Output settings for visualization after training
    skip_PCA_heuristic_slow =  False

    ################################################################################
    # Setup io dict
    ################################################################################
    io_dict = run_subdir_setup(dir_runs=dir_runs, run_subfolder=None, timedir_override=None, minimal_mode=False)

    if flag_save_dataset:
        dataset_savez_fname = io_dict['dir_base'] + os.sep + 'training_dataset_split.npz'  # if none, do not dave
        datagen_kwargs['savez_fname'] = dataset_savez_fname
    else:
        dataset_savez_fname = None
        datagen_kwargs['savez_fname'] = dataset_savez_fname

    ################################################################################
    # Build or load data
    ################################################################################
    if restart_dataset is not None:
        x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = restart_dataset
        print('x_train.shape', x_train.shape)

        # specify training and testing datasets
        train_size = x_train.shape[0]
        assert train_size == int(train_plus_test_size * (1 - test_ratio))  # sanity check

        # fname suffix for io
        data_suffix = 'RESTART-SAME-DATASET'  # appended to fname
    else:
        x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = data_train_test_split_fncall(
            **datagen_kwargs)
        print('x_train.shape', x_train.shape)

        # specify training and testing datasets
        train_size = x_train.shape[0]
        print('train_size', train_size)
        print('int(train_plus_test_size * (1 - test_ratio))', int(train_plus_test_size * (1 - test_ratio)))
        assert train_size == int(train_plus_test_size * (1 - test_ratio))  # sanity check

        data_suffix += '_tx%d_xpw%d_spx%d' % (
            train_size, context_examples_per_W, samples_per_context_example)

    train_dataset = DatasetWrapper(x_train, y_train)
    test_dataset = DatasetWrapper(x_test, y_test)

    ################################################################################
    # Initialize network class
    ################################################################################
    if restart_nn_instance is not None:
        net = restart_nn_instance
        assert net.rho == context_len  # clunky deprecation guard; need to generalize
        assert net.W_KQ.size()[0] == dim_n
    else:
        net = nn_class(context_len, dim_n)

    # inspect network class
    params = list(net.parameters())
    print('\nNum of params matrices to train:', len(params))
    print('\tparams[0].size():', params[0].size())

    # first dimension is generally batch dimension when feeding to nn layers
    sample_input = torch.randn(1, train_dataset.dim_n, train_dataset.context_length)
    sample_input.to(device)
    sample_out = net(sample_input)
    print('sample_input.size()', sample_input.size())
    print('sample_out.size()', sample_out.size())

    ################################################################################
    # Choose loss
    ################################################################################
    criterion = nn.MSELoss()

    sample_out = net(sample_input)
    sample_target = torch.randn(1, train_dataset.dim_n)              # a single vector of length n
    loss = criterion(sample_out, sample_target)
    print('\nloss', loss)
    print('\tloss.grad_fn', loss.grad_fn)                            # MSELoss
    print('\tloss.grad_fn.next_functions[0][0]',
          loss.grad_fn.next_functions[0][0])                         # Linear
    print('\tloss.grad_fn.next_functions[0][0].next_functions[0][0]',
          loss.grad_fn.next_functions[0][0].next_functions[0][0])    # Linear ... etc.

    ################################################################################
    # Optimization (training loop)
    ################################################################################
    assert optimizer_choice in ['sgd', 'adam']
    if optimizer_choice == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=adam_lr)
        #optimizer = optim.Adam(net.parameters(), lr=adam_lr, weight_decay=0.01)
        #optimizer = optim.AdamW(net.parameters(), lr=adam_lr, weight_decay=0.01)
        runinfo_optimizer_lines = ['\tadam_lr, %.2e' % adam_lr]
        opt_suffix = 'adam%.1e' % adam_lr  # appended to fname
    else:
        optimizer = optim.SGD(net.parameters(), lr=sgd_lr, momentum=sgd_momentum)
        runinfo_optimizer_lines = ['\tsgd_lr, %.2e' % sgd_lr,
                                   '\tsgd_momentum, %.2e' % sgd_momentum]
        opt_suffix = 'sgd%.1e-%.1e' % (sgd_lr, sgd_momentum)  # appended to fname
    if scheduler_kwargs is None:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[epochs + 1], gamma=1.0)  # dummy scheduler, no effect
    else:
        #scheduler_lr = torch.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_kwargs['milestones'], gamma=scheduler_kwargs['gamma'])
        opt_suffix = opt_suffix + '_sched'  # appended to fname
        runinfo_optimizer_lines.append('\tscheduler, %s' % scheduler_kwargs)

    # Setup data batching
    nwork = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nwork)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=nwork)

    net.to(device)

    ################################################################################
    # Model ID string
    ################################################################################
    model_fname = '%s_L%d_n%d_e%d_%s_%s' % (nn_fpath, context_len, dim_n, epochs, data_suffix,
                                            opt_suffix)  # used as specialized label for model settings and run

    if flag_vis_weights:
        # extracts weight matrices and visualize weight using utility fn
        from visualize import vis_weights_kq_pv
        if nn_model in ['TransformerModelQKVnores']:
            W_Q = net.W_Q.detach().numpy()
            W_K = net.W_K.detach().numpy()
            W_V = net.W_V.detach().numpy()

            learned_W_KQ = W_K.T @ W_Q
            learned_W_PV = W_V

            _, axarr = plt.subplots(1, 3, figsize=(10, 5))
            imK = axarr[0].imshow(W_K, cmap='viridis')
            axarr[0].set_title('W_K (init)'), plt.colorbar(imK, ax=axarr[0])
            imQ = axarr[1].imshow(W_Q, cmap='viridis')
            axarr[1].set_title('W_Q (init)'), plt.colorbar(imQ, ax=axarr[1])

            imKtQ = axarr[2].imshow(W_K.T @ W_Q, cmap='viridis')
            axarr[2].set_title('W_K.T @ W_Q (init)'), plt.colorbar(imKtQ, ax=axarr[2])

            plt.savefig(io_dict['dir_vis'] + os.sep + 'W_K_and_W_Q_init.png')

        else:
            learned_W_KQ = net.W_KQ.detach().numpy()
            learned_W_PV = net.W_PV.detach().numpy()

        if learned_W_KQ.size == 1:  # in this case we are training 1-param weights (scaled identity)
            vis_weights_kq_pv(learned_W_KQ * np.eye(dim_n),
                              learned_W_PV * np.eye(dim_n), model_fname,
                              dir_out=io_dict['dir_vis'], fname='weights_init', flag_show=False)
        else:
            vis_weights_kq_pv(learned_W_KQ, learned_W_PV, model_fname,
                              dir_out=io_dict['dir_vis'], fname='weights_init', flag_show=False)

    ################################################################################
    # prep loss curves (x, y arrays)
    ################################################################################
    nbatches_per_epoch = np.ceil(x_train.shape[0] / batch_size)

    curve_x_losstrain_batch = np.arange(1, epochs * nbatches_per_epoch + 1) / nbatches_per_epoch
    curve_x_losstrain_epochs_avg = np.arange(1, epochs + 1)  # average over batches in the epoch (fast but inaccurate estimate)
    curve_x_losstest_interval = np.arange(0, epochs + 1e-5, full_loss_sample_interval / nbatches_per_epoch)
    curve_x_losstrain_interval = np.arange(0, epochs + 1e-5, full_loss_sample_interval / nbatches_per_epoch)

    # monitor the train error on the full test set every k batches (could be more/less than once per epoch)
    train_full_mse_loss = report_dataset_loss(net, criterion, train_loader, 'train')
    # monitor the test error on the full test set every k batches (could be more/less than once per epoch)
    test_full_mse_loss = report_dataset_loss(net, criterion, test_loader, 'test')

    curve_y_losstrain_epochs_avg = []  # will append to this each epoch
    curve_y_losstrain_batch = []  # we begin tracking AFTER the first batch (could get loss nograd first batch here)
    curve_y_losstest_interval = [test_full_mse_loss]  # will append to this each full_loss_sample_interval batches
    curve_y_losstrain_interval = [train_full_mse_loss]  # will append to this each epoch

    ################################################################################
    # train loop
    ################################################################################
    count = 1  # batch counter
    for epoch in range(epochs):  # loop over the dataset multiple times

        if epoch % period_save_weights == 0:
            model_path = io_dict['dir_checkpoints'] + os.sep + 'model_e%d' % epoch + '.pth'
            torch.save(net.state_dict(), model_path)

        running_loss_epoch = 0.0
        running_loss_mesoscale = 0.0
        running_batch_counter = 0
        print('\nepoch:', epoch)
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data  # get the inputs; data is a list of [inputs, targets]
            # Note: currently the last batch can be smaller than the rest (remainder)
            #print('\tbatch %d size (inputs --> targets):' % i, inputs.size(), '-->', targets.size())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize=
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            curve_y_losstrain_batch.append(loss.item())

            # print statistics
            running_loss_epoch     += curve_y_losstrain_batch[-1]  # was [count]
            running_loss_mesoscale += curve_y_losstrain_batch[-1]  # was [count]

            # (slow) periodic inspection of test error
            if count % full_loss_sample_interval == 0:  # report it every "full_loss_sample_interval" batches
                print('Epoch: %d, batch: %4d, loss (avg): %.2e' % (
                    epoch, count, running_loss_mesoscale / full_loss_sample_interval))
                print('running_loss_mesoscale, full_loss_sample_interval, count |', running_loss_mesoscale,
                      full_loss_sample_interval, count)

                loss_test = report_dataset_loss(net, criterion, test_loader, 'test')
                curve_y_losstest_interval.append(loss_test)

                loss_train = report_dataset_loss(net, criterion, train_loader, 'train')
                curve_y_losstrain_interval.append(loss_train)

                running_loss_mesoscale = 0.0

            count += 1  # count tracks number of batches which have been trained over (at this point)
            running_batch_counter += 1

        scheduler.step()  # step the learning rate scheduler
        print('\tlast LR:', scheduler.get_last_lr())
        print('end epoch:', epoch, '====================')
        curve_y_losstrain_epochs_avg.append(running_loss_epoch / running_batch_counter)

    print('Finished Training')

    ################################################################################
    # Save model
    ################################################################################
    # save a copy of final model using detailed fname label
    model_path = io_dict['dir_base'] + os.sep + model_fname + '.pth'
    torch.save(net.state_dict(), model_path)
    # save a copy of final model as 'model_final.pth'
    model_path = io_dict['dir_checkpoints'] + os.sep + 'model_final' + '.pth'
    torch.save(net.state_dict(), io_dict['dir_checkpoints'] + os.sep + 'model_final' + '.pth')
    print('\nModel checkpoint saved to', model_path)
    train_loss_end = report_dataset_loss(net, criterion, train_loader, 'train')
    test_loss_end = report_dataset_loss(net, criterion, test_loader, 'test')

    print('curve_x_losstrain_epochs_avg', curve_x_losstrain_epochs_avg)
    print('curve_y_losstrain_epochs_avg', curve_y_losstrain_epochs_avg, '\n')

    print('curve_x_losstrain_batch', curve_x_losstrain_batch)
    print('curve_y_losstrain_batch', curve_y_losstrain_batch, '\n')

    print('curve_x_lossrain_interval', curve_x_losstrain_interval)
    print('curve_y_losstrain_interval', curve_y_losstrain_interval, '\n')

    print('curve_x_losstest_interval', curve_x_losstest_interval)
    print('curve_y_losstest_interval', curve_y_losstest_interval, '\n')

    ################################################################################
    # Save info from training run to .txt
    ################################################################################
    runinfo_lines = [
        'model, %s' % nn_model,
        'fname, %s' % model_fname,
        'datagen_case, %s' % datagen_case,
        'datagen_case_label, %s' % datagen_case_label,
        'context_len, %d' % context_len,
        'dim_n, %d' % dim_n,
        'style_subspace_dimensions, %s' % style_subspace_dimensions,
        'style_corruption_orthog, %s' % style_corruption_orthog,
        'style_origin_subspace, %s' % style_origin_subspace,
        'sigma2_corruption, %s' % sigma2_corruption,
        'sigma2_pure_context, %s' % sigma2_pure_context,
        'context_examples_per_W, %s' % context_examples_per_W,
        'samples_per_context_example, %s' % samples_per_context_example,
        'num_W_in_dataset, %d' % num_W_in_dataset,
        'train_plus_test_size, %d' % train_plus_test_size,
        'test_ratio, %.2f' % test_ratio,
        'epochs, %d' % epochs,
        'batch_size, %d' % batch_size,
        'full_loss_sample_interval, %d' % full_loss_sample_interval,
        'optimizer_choice, %s' % optimizer_choice,
        'seed_dataset, %s' % seed_dataset,
        'seed_torch, %s' % seed_torch,
        *runinfo_optimizer_lines,  # can contain multiple lines, each are tabbed
    ]

    # Add case-specific parameters - only applies to case 1 (clustering)
    if datagen_case == 1:
        runinfo_lines.extend([
            'case1_specific_style_cluster_mus, %s' % datagen_kwargs['style_cluster_mus'],
            'case1_specific_style_cluster_vars, %s' % datagen_kwargs['style_cluster_vars'],
            'case1_specific_cluster_var, %s' % datagen_kwargs['cluster_var'],
            'case1_specific_num_cluster, %s' % datagen_kwargs['num_cluster'],
        ])

    # add lines to file
    runinfo_append(io_dict, runinfo_lines)

    ################################################################################
    # Plot loss dynamics against simple benchmarks
    ################################################################################
    nbatches_per_epoch = np.ceil(x_train.shape[0] / batch_size)
    loss_vals_dict = {
        'loss_train_batch': dict(
            x=curve_x_losstrain_batch,
            y=curve_y_losstrain_batch,
            label='train (one batch)',
            fname='curve_loss_train_batch',
            pltkwargs=dict(linestyle='--', marker='o', color='b', markersize=4, markerfacecolor='None', alpha=0.3)),
        'loss_train_epoch_avg': dict(
            x=curve_x_losstrain_epochs_avg,
            y=curve_y_losstrain_epochs_avg,
            label='train (epoch moving avg)',
            fname='curve_loss_train_epoch_avg',
            pltkwargs=dict(linestyle='--', marker='o', color='b', markersize=4)),
        'loss_train_interval': dict(
            x=curve_x_losstrain_interval,
            y=curve_y_losstrain_interval,
            label='train (full)',
            fname='curve_loss_train_interval',
            pltkwargs=dict(linestyle='-', marker='o', color='b')),
        'loss_test_interval': dict(
            x=curve_x_losstest_interval,
            y=curve_y_losstest_interval,
            label='test (full)',
            fname='curve_loss_test_interval',
            pltkwargs=dict(linestyle='-', marker='o', color='r')),
    }

    print('Compare to null performance and lin.alg. baselines:')
    dumb_A_mse_on_train = loss_if_predict_zero(criterion, train_loader, 'train')
    dumb_A_mse_on_test = loss_if_predict_zero(criterion, test_loader, 'test')
    dumb_B_mse_on_train = loss_if_predict_mostrecent(criterion, train_loader, 'train')
    dumb_B_mse_on_test = loss_if_predict_mostrecent(criterion, test_loader, 'test')
    dumb_C_mse_on_train = loss_if_predict_average(criterion, train_loader, 'train')
    dumb_C_mse_on_test = loss_if_predict_average(criterion, test_loader, 'test')

    # add core baselines to loss_vals_dict (will also add datagen-case-specific ones later)
    loss_vals_dict['baselines'] = {
        'loss_if_predict_zero': dict(
            alias='dumb_A',
            label=r'guess $0$',
            val_train=dumb_A_mse_on_train,
            val_test=dumb_A_mse_on_test,
            pltkwargs=dict(color='grey')),
        'loss_if_predict_mostrecent': dict(
            alias='dumb_B',
            label=r'guess $x_{k-1}$',
            val_train=dumb_B_mse_on_train,
            val_test=dumb_B_mse_on_test,
            pltkwargs=dict(color='green')),
        'loss_if_predict_average': dict(
            alias='dumb_C',
            label=r'guess mean',
            val_train=dumb_C_mse_on_train,
            val_test=dumb_C_mse_on_test,
            pltkwargs=dict(color='orange')),
    }

    # the following heuristics baselines are specific to case 0: Linear subspaces
    if datagen_case == 0:
        if not skip_PCA_heuristic_slow:
            print('Warning: not skip_PCA_heuristic_slow; slow lin.alg. step...')
            heuristic_mse_on_train = loss_if_predict_linalg(criterion, train_loader, 'train')
            heuristic_mse_on_test = loss_if_predict_linalg(criterion, test_loader, 'test')

            loss_vals_dict['baselines']['loss_if_predict_linalg'] = dict(
                alias='heuristic_proj',
                label=r'$P \tilde x$',
                val_train=heuristic_mse_on_train,
                val_test=heuristic_mse_on_test,
                pltkwargs=dict(color='black'))

            # also compute shrunken predictor
            # - we assume proper subspace through origin
            # - we assume it is iid gaussian ball corruption (not orthogonal to W)
            if (style_origin_subspace) and (not style_corruption_orthog):  # we assume proper subspace through origin
                heuristic_mse_shrunken_on_train = (
                    loss_if_predict_linalg_shrunken(
                        criterion, train_loader, 'train', sigma2_pure_context, sigma2_corruption,
                        style_origin_subspace=style_origin_subspace, style_corruption_orthog=style_corruption_orthog)
                )
                heuristic_mse_shrunken_on_test = (
                    loss_if_predict_linalg_shrunken(
                        criterion, test_loader, 'test', sigma2_pure_context, sigma2_corruption,
                        style_origin_subspace=style_origin_subspace, style_corruption_orthog=style_corruption_orthog)
                )
                theory_expected_error_linalg_shrunken = theory_linear_expected_error(
                    dim_n, style_subspace_dimensions, sigma2_corruption, linear_sigma2_pure_context)

                loss_vals_dict['baselines']['loss_if_predict_linalg_shrunken'] = dict(
                    alias='heuristic_proj_shrunken',
                    label=r'$\gamma P \tilde x$',
                    val_train=heuristic_mse_shrunken_on_train,
                    val_test=heuristic_mse_shrunken_on_test,
                    pltkwargs=dict(color='mediumpurple'))

                loss_vals_dict['baselines']['theory_expected_error_linalg_shrunken'] = dict(
                    alias='theory_expected_error_linalg_shrunken',
                    label=r'$\mathbb{E}[L(\theta^*)]$',
                    val_train=theory_expected_error_linalg_shrunken,  # note train/test don't matter - theory curve
                    val_test=theory_expected_error_linalg_shrunken,
                    pltkwargs=dict(color='mediumpurple', linestyle=':'))

    elif datagen_case == 1:
        # For the baseline, we need to provide the GMM dict for each example in the dataset
        # - cluster centers (constrained to have unit norm) - n x p
        # - cluster variance (isotropic and identical)      - scalar
        # - cluster weights (can vary)                      - n x 1
        baselines_kw = dict(style_origin_subspace=style_origin_subspace,
                            style_corruption_orthog=style_corruption_orthog,
                            print_val=True)

        train_loader_bsz1_noshuffle = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=nwork)
        test_loader_bsz1_noshuffle = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nwork)

        # simplified baseline
        heuristic_mse_clustering_0var_on_train = loss_if_predict_clustering_baseline(
            criterion, train_loader_bsz1_noshuffle, 'train', sigma2_corruption, train_data_subspaces,
            force_zero_cluster_var=True, **baselines_kw)
        heuristic_mse_clustering_0var_on_test = loss_if_predict_clustering_baseline(
            criterion, test_loader_bsz1_noshuffle, 'test', sigma2_corruption, test_data_subspaces,
            force_zero_cluster_var=True, **baselines_kw)

        # full baseline
        heuristic_mse_clustering_on_train = loss_if_predict_clustering_baseline(
            criterion, train_loader_bsz1_noshuffle, 'train', sigma2_corruption, train_data_subspaces,
            force_zero_cluster_var=False, **baselines_kw)
        heuristic_mse_clustering_on_test = loss_if_predict_clustering_baseline(
            criterion, test_loader_bsz1_noshuffle, 'test', sigma2_corruption, test_data_subspaces,
            force_zero_cluster_var=False, **baselines_kw)

        loss_vals_dict['baselines']['loss_if_predict_clustering_baseline_0var'] = dict(
            alias='heuristic_gmm_0var',
            label=r'$x_{opt}$ for $\sigma_0^2=0$',
            val_train=heuristic_mse_clustering_0var_on_train,
            val_test=heuristic_mse_clustering_0var_on_test,
            pltkwargs=dict(color='pink'))

        loss_vals_dict['baselines']['loss_if_predict_clustering_baseline'] = dict(
            alias='heuristic_gmm',
            label=r'$x_{opt}$',
            val_train=heuristic_mse_clustering_on_train,
            val_test=heuristic_mse_clustering_on_test,
            pltkwargs=dict(color='mediumpurple'))

    else:
        assert datagen_case == 2

        provided_sphere_radius = None  # will be inferred from the context...
        baselines_kw = dict(style_origin_subspace=style_origin_subspace,
                            style_corruption_orthog=style_corruption_orthog,
                            print_val=True)

        heuristic_mse_subsphere_on_train = loss_if_predict_subsphere_baseline(
            criterion, train_loader, 'train', provided_sphere_radius, sigma2_corruption,
            shrunken=False, **baselines_kw)
        heuristic_mse_subsphere_on_test = loss_if_predict_subsphere_baseline(
            criterion, test_loader, 'test', provided_sphere_radius, sigma2_corruption,
            shrunken=False, **baselines_kw)

        heuristic_mse_subsphere_shrunken_on_train = loss_if_predict_subsphere_baseline(
            criterion, train_loader, 'train', provided_sphere_radius, sigma2_corruption,
            shrunken=True, **baselines_kw)
        heuristic_mse_subsphere_shrunken_on_test = loss_if_predict_subsphere_baseline(
            criterion, test_loader, 'test', provided_sphere_radius, sigma2_corruption,
            shrunken=True, **baselines_kw)

        loss_vals_dict['baselines']['loss_if_predict_subsphere_baseline'] = dict(
            alias='heuristic_proj',
            label=r'$\frac{R}{\lVert x_{\parallel} \rVert} x_{\parallel}$',
            val_train=heuristic_mse_subsphere_on_train,
            val_test=heuristic_mse_subsphere_on_test,
            pltkwargs=dict(color='black'))

        loss_vals_dict['baselines']['loss_if_predict_subsphere_shrunk_baseline'] = dict(
            alias='heuristic_proj_shrunken',
            label=r'$\frac{R}{\lVert x_{\parallel} \rVert} x_{\parallel}$',
            val_train=heuristic_mse_subsphere_shrunken_on_train,
            val_test=heuristic_mse_subsphere_shrunken_on_test,
            pltkwargs=dict(color='mediumpurple'))

    if flag_vis_loss:
        plt.close('all')
        print('\nPlotting training loss dynamics...')
        plt.figure(figsize=(6, 4), layout='tight')

        for dict_key in ['loss_train_batch', 'loss_train_epoch_avg', 'loss_train_interval', 'loss_test_interval']:
            plt.plot(loss_vals_dict[dict_key]['x'], loss_vals_dict[dict_key]['y'],
                     **loss_vals_dict[dict_key]['pltkwargs'])

        plt.axhline(dumb_A_mse_on_train, label='train (guess 0)', color='grey')
        plt.axhline(dumb_A_mse_on_test, linestyle='--', label='test (guess 0)', color='grey')
        plt.axhline(dumb_B_mse_on_train, label=r'train (guess $x_{k-1}$)', color='green')
        plt.axhline(dumb_B_mse_on_test, linestyle='--', label=r'test (guess $x_{k-1}$)', color='green')
        plt.axhline(dumb_C_mse_on_train, label=r'train (guess mean)', color='orange')
        plt.axhline(dumb_C_mse_on_test, linestyle='--', label=r'test (guess mean)', color='orange')

        # plot heuristics specific to the Case (Case 0: Linear)
        if datagen_case == 0:
            if skip_PCA_heuristic_slow:
                print('Warning: skip_PCA_heuristic_slow = True -- this is zero when sample corruption is always orthogonal')
            else:
                plt.axhline(heuristic_mse_on_train, label=r'train ($P_W$)', color='black')
                plt.axhline(heuristic_mse_on_test, linestyle='--', label=r'test ($P_W$)', color='black')
                if (style_origin_subspace) and (not style_corruption_orthog):
                    plt.axhline(heuristic_mse_shrunken_on_train, label=r'train ($P_W$ shrunk)', color='mediumpurple')
                    plt.axhline(heuristic_mse_shrunken_on_test, linestyle='--', label=r'test ($P_W$ shrunk)',
                                color='mediumpurple', linewidth=2, zorder=5)
                    plt.axhline(theory_expected_error_linalg_shrunken, linestyle=':', color='mediumpurple', linewidth=2, zorder=5,
                            label=r'$L(\theta^*)$')

        if datagen_case == 1:
            print('plotting baselines for case 1: GMM (clustering)')
            baseline_main = loss_vals_dict['baselines']['loss_if_predict_clustering_baseline']
            baseline_0var = loss_vals_dict['baselines']['loss_if_predict_clustering_baseline_0var']
            # plot full baseline
            plt.axhline(baseline_main['val_train'], label=r'train %s' % baseline_main['label'],
                        color=baseline_main['pltkwargs']['color'], zorder=10, linewidth=2)
            plt.axhline(baseline_main['val_test'], label=r'test %s' % baseline_main['label'],
                        color=baseline_main['pltkwargs']['color'], zorder=10, linewidth=2, linestyle='--')
            # now plot the 0var simplified baseline
            plt.axhline(baseline_0var['val_train'], label=r'train %s' % baseline_0var['label'],
                        color=baseline_0var['pltkwargs']['color'], zorder=10, linewidth=2)
            plt.axhline(baseline_0var['val_test'], label=r'test %s' % baseline_0var['label'],
                        color=baseline_0var['pltkwargs']['color'], zorder=10, linewidth=2, linestyle='--')

        if datagen_case == 2:
            print('plotting baselines for case 2: Manifold (subspheres)')
            plt.axhline(heuristic_mse_subsphere_on_train, label=r'train ($P_W$ onto $S$)', color='black', zorder=10, linewidth=2)
            plt.axhline(heuristic_mse_subsphere_on_test,  label=r'test ($P_W$ onto $S$)',  color='black', linestyle='--', zorder=10, linewidth=2)
            # plot more precise baseline (shrunk)
            plt.axhline(heuristic_mse_subsphere_shrunken_on_train, label=r'train ($P_W$ onto $S$ + shrunk)', color='mediumpurple', linewidth=2, zorder=5)
            plt.axhline(heuristic_mse_subsphere_shrunken_on_test,   label=r'test ($P_W$ onto $S$ + shrunk)', color='mediumpurple', linewidth=2, zorder=5, linestyle='--')


        plt.legend(ncol=2)
        plt.grid(alpha=0.5)
        plt.title(r'Loss dynamics for context length $L=%d$, dim $n=%d$' % (context_len, dim_n)
                  + '\n%s, Case %d - %s' % (nn_model, datagen_case, datagen_case_label), fontsize=12)
        plt.ylabel('Loss (MSE)')
        plt.xlabel('Epoch')
        # big output text for the csv (run settings)
        # plt.xlabel(r'Epoch \small{' + '\n'.join(lines) + r'}')
        # plt.xlabel('\n'.join(lines), fontsize=8)

        plt.tight_layout()
        plt.savefig(io_dict['dir_vis'] + os.sep + model_fname + '_loss_dynamics.pdf')
        # plt.savefig(DIR_OUT + os.sep + model_fname + '_loss_dynamics.png')
        plt.show()

    ################################################################################
    # Save loss plot curves (for later replotting)
    ################################################################################
    with open(io_dict['dir_data_for_replot'] + os.sep + 'loss_vals_dict.pkl', 'wb') as handle:
        pickle.dump(loss_vals_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.savez(io_dict['dir_data_for_replot'] + os.sep + 'curve_losstrain_tbatch.npz',
             x=curve_x_losstrain_batch,
             y=curve_y_losstrain_batch)
    np.savez(io_dict['dir_data_for_replot'] + os.sep + 'curve_losstrain_epoch_avg.npz',
             x=curve_x_losstrain_epochs_avg,
             y=curve_y_losstrain_epochs_avg)
    np.savez(io_dict['dir_data_for_replot'] + os.sep + 'curve_losstrain_interval.npz',
             # interval = n batches between recordings
             x=curve_x_losstrain_interval,
             y=curve_y_losstrain_interval)
    np.savez(io_dict['dir_data_for_replot'] + os.sep + 'curve_losstest_interval.npz',
             x=curve_x_losstest_interval,
             y=curve_y_losstest_interval)
    with open(io_dict['dir_data_for_replot'] + os.sep + 'loss_baselines.txt', 'w') as f:
        f.write('dumb_A_mse_on_train, %s\n' % str(
            dumb_A_mse_on_train))
        f.write('dumb_A_mse_on_test,  %s\n'  % str(dumb_A_mse_on_test))
        f.write('dumb_B_mse_on_train, %s\n'  % str(dumb_B_mse_on_train))
        f.write('dumb_B_mse_on_test,  %s\n'  % str(dumb_B_mse_on_test))
        f.write('dumb_C_mse_on_train, %s\n'  % str(dumb_C_mse_on_train))
        f.write('dumb_C_mse_on_test,  %s\n'  % str(dumb_C_mse_on_test))

        if datagen_case == 0:
            if not skip_PCA_heuristic_slow:
                f.write('heuristic_mse_on_train, %s\n' % str(heuristic_mse_on_train))
                f.write('heuristic_mse_on_test, %s\n' % str(heuristic_mse_on_test))
                if (style_origin_subspace) and (not style_corruption_orthog):
                    f.write('heuristic_mse_shrunken_on_train, %s\n' % str(heuristic_mse_shrunken_on_train))
                    f.write('heuristic_mse_shrunken_on_test, %s\n' % str(heuristic_mse_shrunken_on_test))
                    f.write('theory_expected_error_linalg_shrunken, %s\n' % str(theory_expected_error_linalg_shrunken))

        elif datagen_case == 1:
            f.write('heuristic_mse_clustering_on_train, %s\n' % str(heuristic_mse_clustering_on_train))
            f.write('heuristic_mse_clustering_on_test, %s\n' % str(heuristic_mse_clustering_on_test))
            f.write('heuristic_mse_clustering_0var_on_train, %s\n' % str(heuristic_mse_clustering_0var_on_train))
            f.write('heuristic_mse_clustering_0var_on_test, %s\n' % str(heuristic_mse_clustering_0var_on_test))

        else:
            assert datagen_case == 2
            f.write('heuristic_mse_subsphere_on_train, %s\n' % str(heuristic_mse_subsphere_on_train))
            f.write('heuristic_mse_subsphere_on_test, %s\n' % str(heuristic_mse_subsphere_on_test))
            if (style_origin_subspace) and (not style_corruption_orthog):
                f.write('heuristic_mse_subsphere_shrunken_on_train, %s\n' % str(heuristic_mse_subsphere_shrunken_on_train))
                f.write('heuristic_mse_subsphere_shrunken_on_test, %s\n' % str(heuristic_mse_subsphere_shrunken_on_test))

    ################################################################################
    # Vis learned weights
    ################################################################################
    if flag_vis_weights:
        # extracts weight matrices and visualize weight using utility fn
        if nn_model in ['TransformerModelQKVnores']:
            W_Q = net.W_Q.detach().numpy()
            W_K = net.W_K.detach().numpy()
            W_V = net.W_V.detach().numpy()

            learned_W_KQ = W_K.T @ W_Q
            learned_W_PV = W_V

            _, axarr = plt.subplots(1, 3, figsize=(10, 5))
            imK = axarr[0].imshow(W_K, cmap='viridis');
            axarr[0].set_title('W_K (final)'), plt.colorbar(imK, ax=axarr[0])
            imQ = axarr[1].imshow(W_Q, cmap='viridis');
            axarr[1].set_title('W_Q (final)'), plt.colorbar(imQ, ax=axarr[1])

            imKtQ = axarr[2].imshow(W_K.T @ W_Q, cmap='viridis');
            axarr[2].set_title('W_K.T @ W_Q (final)'), plt.colorbar(imKtQ, ax=axarr[2])

            # plt.show()
            plt.savefig(io_dict['dir_vis'] + os.sep + 'W_K_and_W_Q_final.png')

        else:
            learned_W_KQ = net.W_KQ.detach().numpy()
            learned_W_PV = net.W_PV.detach().numpy()

        if learned_W_KQ.size == 1:  # in this case we are training 1-param weights (scaled identity)
            vis_weights_kq_pv(learned_W_KQ * np.eye(dim_n), learned_W_PV * np.eye(dim_n), model_fname,
                              dir_out=io_dict['dir_vis'], fname='weights_end', flag_show=False)
        else:
            vis_weights_kq_pv(learned_W_KQ, learned_W_PV, model_fname,
                              dir_out=io_dict['dir_vis'], fname='weights_end', flag_show=False)

    ################################################################################
    # Load model
    ################################################################################
    if flag_vis_batch_perf:

        print('Now re-loading final Model...')
        net, model_type, context_length, dim_n = load_modeltype_from_fname(
            model_fname + '.pth', dir_models=io_dict['dir_base'])
        net.eval()

        '''
        print('\nLooking at example batches from the train set')
        for idx in range(1):
            plot_one_prediction_batch(use_test=False)
        '''
        print('\nLooking at an example batch from the training set')
        for idx in range(1):
            plot_one_prediction_batch(
                net, criterion, train_loader, max_example_from_batch=1, show_corrupted_input=True, as_barplot=True,
                save=True, dir_out=io_dict['dir_vis'])
        plot_batch_predictions(
            net, criterion, train_loader, nbatch=1, save=True, dir_out=io_dict['dir_vis'])
        print('\nLooking at an example batch from the test set')
        for idx in range(1):
            plot_one_prediction_batch(
                net, criterion, test_loader, max_example_from_batch=1, show_corrupted_input=True, as_barplot=True,
                save=True, dir_out=io_dict['dir_vis'])
        plot_batch_predictions(
            net, criterion, test_loader, nbatch=1, save=True, dir_out=io_dict['dir_vis'])

        print('\nLooking at specific example -> target predictions from a batch of the train set')
        dataiter = iter(train_loader)
        samples, targets = next(dataiter)

        with torch.no_grad():
            outputs = net(samples)

        for idx in range(5):
            print(samples[idx, :])
            print('\ttarget:', targets[idx])
            print('\tNN prediction:', outputs[idx])
            print('\tMSE:', criterion(outputs[idx], targets[idx]))

    return net, model_fname, io_dict, loss_vals_dict, train_loader, test_loader, x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


if __name__ == '__main__':

    # these determine both (A) dataset generation and (B) the model currently
    context_len = 500   # this will induce the synthetic dataset via data_train_test_split_linear  - 500
    dim_n = 16           # this will induce the synthetic dataset ^^ (try 16/32 to 128)

    #nn_model = 'TransformerModelV1nores'
    #nn_model = 'TransformerModelV2nores'
    #nn_model = 'TransformerModelQKVnores'

    #nn_model = 'TransformerModelV1noresOmitLast'
    nn_model = 'TransformerModelV2noresOmitLast'

    datagen_case = 1  # {0, 1, 2} -> {linear, GMM clusters, manifold}

    seed_dataset = 0
    seed_torch = 0

    epochs = 100
    # defaults:
    # - if train size is 8000, set batch_size to 800
    # - if train size is 800,  set batch_size to 80
    batch_size = 80  # reminder: 'manifold' notebook is using batch size of one (1)

    # (in num. batches - default of 4 when bsz was 80, 800 size train set, so 10 batches per epoch)
    #   e.g. 1000 batch-per-epoch, then get 10 loss evals per epoch for full_loss_sample_interval = 100
    full_loss_sample_interval = 4

    optimizer_choice = 'adam'  # sgd or adam

    if optimizer_choice == 'sgd':
        optimizer_lr = 0.1  #0.01  # 0.5
        #optimizer_lr = 80*0.1  #1.  400 epochs for spheres case with original params
        # scheduler_kwargs = None
        scheduler_kwargs = dict(milestones=[0.8*int(epochs), 0.9*int(epochs)], gamma=0.1)  # mult by gamma each milestone
    else:
        assert optimizer_choice == 'adam'
        optimizer_lr = 1e-2  # default: 1e-2
        scheduler_kwargs = None

    flag_save_dataset   = False  # default: False (filesize can be large)
    flag_vis_loss       = True
    flag_vis_weights    = True
    flag_vis_batch_perf = True

    # prep dataset generation kwargs
    # ################################################################################
    num_W_in_dataset = 1000
    context_examples_per_W = 1
    samples_per_context_example = 1

    base_kwargs = dict(
        context_len=context_len,
        dim_n=dim_n,
        num_W_in_dataset=num_W_in_dataset,  # this will be train_plus_test_size (since other kwargs are 1)
        context_examples_per_W=context_examples_per_W,
        samples_per_context_example=samples_per_context_example,
        test_ratio=0.2,
        verbose=True,
        as_torch=True,
        savez_fname=None,  # will be set internally depending on flag_save_dataset
        seed=seed_dataset,
        style_origin_subspace=DATAGEN_GLOBALS[datagen_case]['style_origin_subspace'],
        style_corruption_orthog=DATAGEN_GLOBALS[datagen_case]['style_corruption_orthog'],
    )

    # manually modify these case-specific settings if desired
    if datagen_case == 0:
        datagen_kwargs = base_kwargs | dict(
            sigma2_corruption=1.0,
            sigma2_pure_context=2.0,
            style_subspace_dimensions=8,  # int or 'random'
        )
    elif datagen_case == 1:
        datagen_kwargs = base_kwargs | dict(
            sigma2_corruption=0.1,
            cluster_var=0.02,    # network becomes exact when sigma=0
            #cluster_var=0.0,    # no-res network becomes exact when sigma=0
            num_cluster=8,       # int or 'random'
        )
    else:
        assert datagen_case == 2
        datagen_kwargs = base_kwargs | dict(
            sigma2_corruption=0.1,
            radius_sphere=1.0,
            style_subspace_dimensions=8,  # int or 'random' | d=1 is circle, d=2 is sphere, etc.; default: d=8
        )

    (net, model_fname, io_dict, loss_vals_dict,
     train_loader, test_loader, x_train, y_train, x_test, y_test,
     train_data_subspaces, test_data_subspaces) = train_model(
        nn_model=nn_model,
        restart_nn_instance=None,
        restart_dataset=None,
        train_plus_test_size=num_W_in_dataset * context_examples_per_W * samples_per_context_example,
        context_len=context_len, dim_n=dim_n,
        datagen_case=datagen_case,
        datagen_kwargs=datagen_kwargs,
        batch_size=batch_size,
        seed_torch=seed_torch,
        epochs=epochs,
        optimizer_choice=optimizer_choice, optimizer_lr=optimizer_lr, scheduler_kwargs=scheduler_kwargs,
        full_loss_sample_interval=full_loss_sample_interval,
        flag_save_dataset=flag_save_dataset,
        flag_vis_loss=flag_vis_loss,
        flag_vis_weights=flag_vis_weights,
        flag_vis_batch_perf=flag_vis_batch_perf)

    if nn_model not in ['TransformerModelQKVnores']:
        learned_W_KQ = net.W_KQ.detach().numpy()
        learned_W_PV = net.W_PV.detach().numpy()
        if learned_W_KQ.size == 1:  # in this case we are training 1-param weights (scaled identity) - remake as arr
            learned_W_KQ = learned_W_KQ * np.eye(dim_n)
            learned_W_PV = learned_W_PV * np.eye(dim_n)
        vis_weights_kq_pv(learned_W_KQ, learned_W_PV, titlemod=r'$\theta$ final',
                          dir_out=io_dict['dir_vis'], fname='weights_final', flag_show=True)
