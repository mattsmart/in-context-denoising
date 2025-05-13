import matplotlib.pyplot as plt
import numpy as np
import os

from nn_model_base import MODEL_CLASS_FROM_STR
from nn_train_methods import train_model
from visualize import vis_weights_kq_pv
from settings import DIR_OUT, DIR_RUNS, DATAGEN_GLOBALS


def multirun_ensemble(
        nn_model='TransformerModelV2noresOmitLast',
        datagen_case=2,
        n_ensemble=6,
        epochs=100,
        show_loss=True,
        fixed_context_len=500
    ):
    """
    Main function for training an ensemble of models with management of datagen and pytorch seeds

    Main purpose: train N models for different seeds (datagen/weights) and plot the spread of loss curves
    - A: both IC and dataset are random (diff seed)
    - B:  fix IC, diff dataset
    - C: diff IC,  fix datase

    then show variance at each point in training (in vis)

    Two modes:
    - (1) fixed context length                              -> default (intended) mode
    - (2) context length increases each run (np.linspace)   -> bonus mode
            - note: overlap with two other 'analysis' scripts; those are not yet refactored for new "datagen" API

    Replotting - mode dependent:
    - (1) run replot_multitraj_loss.py
    - (2) run replot_multitraj_vary_contextlen.py

    Args:
        nn_model: str       e.g. 'TransformerModelV2noresOmitLast'
        datagen_case: int   {0, 1, 2} -> {linear, clusters, manifold}

    Returns:
        None
    """
    FORCE_SEED_WEIGHTS = None  # if an integer, all runs will use the random weight initialization
    FORCE_SEED_DATASET = 0     # if an integer, all runs will use the same dataset

    #seed_dataset = 4  # 4 works here
    #seed_torch = 4  # dataset 4 with seed 4 is problem for multitraj 100/16/8 gradflow sgd 0.5

    # model parameters
    # - these determine both (A) dataset generation and (B) the model currently
    dim_n = 16  # this will induce the synthetic dataset ^^ (try 16 or 32 to 128)

    #context_len_varying = np.arange(20, 501, 40, dtype=int)  # this is used if FORCE_CONTEXT_LEN is None
    #context_len_varying = np.array([20,30,100])
    context_len_varying = np.concatenate([np.arange(20, 100, 10, dtype=int), np.arange(100, 501, 40, dtype=int)])

    # fixed_context_len: (default) 100 -- if None, will use context_len_varying (force is default mode)
    if fixed_context_len is None:
        print('Warning: varying context len, forcing n_ensemble to len(context_len_varying)=%d' % n_ensemble)
        n_ensemble = len(context_len_varying)
        Lstr = 'vary'
    else:
        # check that type is int
        assert isinstance(fixed_context_len, int)
        Lstr = '%d' % fixed_context_len

    # these lists can be overridden if FORCE_SEED_X is set above
    seeds_for_weights = [i for i in range(n_ensemble)]
    seeds_for_dataset = [i for i in range(n_ensemble)]

    flag_save_dataset       = False  # in nn_train_methods.py - False
    flag_vis_loss           = False  # in nn_train_methods.py - True
    flag_vis_weights        = True
    flag_vis_batch_perf     = False

    # For gradient flow, did epochs=250, sgd 0.5, bsz 800 for dataset size 1000
    batch_size = 80                # (80 or 800) - if dataset size 1000 and TEST_RATIO is 0.2, then bsize is 100 or 800
    train_plus_test_size = 1000    # train/test ratio is 0.8/0.2
    full_loss_sample_interval = 4  # this is the interval in batches for computing full loss (not just batch loss)

    optimizer_choice = 'adam'  # sgd or adam

    if optimizer_choice == 'sgd':
        optimizer_lr = 0.2   # 10.0   # 0.5
        # scheduler_kwargs = None
        #scheduler_kwargs = dict(milestones=[int(0.8 * epochs), int(0.9 * epochs)], gamma=0.1)  # *= gamma each milestone
        scheduler_kwargs = dict(milestones=[int(0.8 * epochs), int(0.9 * epochs)], gamma=0.75)  # *= gamma each milestone
    else:
        assert optimizer_choice == 'adam'
        optimizer_lr =  0.01  # 1e-2, 0.05 1e-1
        #scheduler_kwargs = None
        scheduler_kwargs = dict(milestones=[int(0.8 * epochs), int(0.9 * epochs)], gamma=0.1)  # *= gamma each milestone

    # Specify name of runs dir
    str_runinfo = '_case%d_%s_%s_e%d_L%s' % (datagen_case, MODEL_CLASS_FROM_STR[nn_model]['alias'], optimizer_choice, epochs, Lstr)
    dir_multitraj = DIR_RUNS + os.sep + 'multitraj_ensemble' + str_runinfo
    assert not os.path.exists(dir_multitraj)  # approach: manually make new dir for each experiment

    # dataset parameters - settings for training
    assert datagen_case in [0, 1, 2]

    # manually modify these case-specific settings if desired
    if datagen_case == 0:
        datagen_kwargs_to_append = dict(
            sigma2_corruption=1.0,
            sigma2_pure_context=2.0,
            style_subspace_dimensions=8,  # subspace dimension (int or 'random' alt option)
            style_origin_subspace=DATAGEN_GLOBALS[datagen_case]['style_origin_subspace'],
            style_corruption_orthog=DATAGEN_GLOBALS[datagen_case]['style_corruption_orthog'],
        )
        # specific to this case but used in function calls below
        linear_sigma2_pure_context = datagen_kwargs_to_append['sigma2_pure_context']
        sigma2_pure_context = linear_sigma2_pure_context  # alias

        runinfo_suffix = (nn_model + '\n' + 'Datagen case: %d -' % datagen_case +
                          r'dim $n = %d$, dim $d = %s$, $\sigma_{0}^2 = %.2f, \sigma_{z}^2 = %.2f$' %
                          (dim_n,
                           datagen_kwargs_to_append['style_subspace_dimensions'],
                           datagen_kwargs_to_append['sigma2_pure_context'],
                           datagen_kwargs_to_append['sigma2_corruption']))

    elif datagen_case == 1:
        datagen_kwargs_to_append = dict(
            sigma2_corruption=0.1,
            cluster_var=0.02,
            num_cluster=8,  # int or 'random'
            style_subspace_dimensions=DATAGEN_GLOBALS[datagen_case]['style_subspace_dimensions'],  # defaults to 'full' (n)
            style_origin_subspace=DATAGEN_GLOBALS[datagen_case]['style_origin_subspace'],
            style_corruption_orthog=DATAGEN_GLOBALS[datagen_case]['style_corruption_orthog'],
        )
        # specific to this case but used in function calls below
        cluster_var = datagen_kwargs_to_append['cluster_var']
        sigma2_pure_context = cluster_var  # alias

        #data_suffix = 'case%d_s2z%.2f_s2n%.2f_ortho%d_origin%d_d-%s' % (
        #    datagen_case, dgkw2a['cluster_var'], dgkw2a['sigma2_corruption'], dgkw2a['style_corruption_orthog'],
        #    dgkw2a['style_origin_subspace'], dgkw2a['style_subspace_dimensions'])

        runinfo_suffix = (nn_model + '\n' + 'Datagen case: %d -' % datagen_case +
                          r'dim $n = %d$, $K = %s$, $\sigma_{0}^2 = %.2f$, \sigma_{z}^2 = %.2f$' %
                          (dim_n,
                           datagen_kwargs_to_append['num_cluster'],
                           datagen_kwargs_to_append['cluster_var'],
                           datagen_kwargs_to_append['sigma2_corruption']))

    else:
        assert datagen_case == 2
        datagen_kwargs_to_append = dict(
            sigma2_corruption=0.1,
            radius_sphere=1.0,
            style_subspace_dimensions=8,  # int (e.g. 8) or 'random'
            style_origin_subspace=DATAGEN_GLOBALS[datagen_case]['style_origin_subspace'],
            style_corruption_orthog=DATAGEN_GLOBALS[datagen_case]['style_corruption_orthog'],
        )
        # specific to this case but used in function calls below
        radius_sphere = datagen_kwargs_to_append['radius_sphere']
        sigma2_pure_context = radius_sphere  # alias

        #data_suffix = 'case%d_s2z%.2f_s2n%.2f_ortho%d_origin%d_d-%s' % (
        #    datagen_case, dgkw2a['radius_sphere'], dgkw2a['sigma2_corruption'], dgkw2a['style_corruption_orthog'],
        #    dgkw2a['style_origin_subspace'], dgkw2a['style_subspace_dimensions'])

        runinfo_suffix = (nn_model + '\n' + 'Datagen case: %d -' % datagen_case +
                          r'dim $n = %d$, dim $d = %s$, $R = %.2f, \sigma_{z}^2 = %.2f$' %
                          (dim_n,
                           datagen_kwargs_to_append['style_subspace_dimensions'],
                           datagen_kwargs_to_append['radius_sphere'],
                           datagen_kwargs_to_append['sigma2_corruption']))

    # shorthand aliases used below
    dgkw2a = datagen_kwargs_to_append
    sigma2_corruption = dgkw2a['sigma2_corruption']                  # float
    style_corruption_orthog = dgkw2a['style_corruption_orthog']      # True/False (orthog OR ball)
    style_origin_subspace = dgkw2a['style_origin_subspace']          # True/False
    style_subspace_dimensions = dgkw2a['style_subspace_dimensions']  # int or 'random' (or just 'full' in clustering case)

    # curves to compute by training models
    gamma_imputed = np.zeros(n_ensemble)
    vals_loss_predict_zero = np.zeros(n_ensemble)

    # these are for datagen case 0 (linear subspaces)
    vals_loss_theory_linalg = np.zeros(n_ensemble)
    vals_loss_theory_linalg_shrunken = np.zeros(n_ensemble)

    datadict_curves_loss = {k: dict() for k in range(n_ensemble)}

    # MAIN ENSEMBLE TRAINING LOOP
    # =======================================================================
    print('Entering main ensemble training loop...')
    for idx in range(n_ensemble):

        # case check in main ensemble training loop
        if FORCE_SEED_WEIGHTS is None:
            seed_torch = seeds_for_weights[idx]
        else:
            seed_torch = FORCE_SEED_WEIGHTS

        if FORCE_SEED_DATASET is None:
            seed_dataset = seeds_for_dataset[idx]
        else:
            seed_dataset = FORCE_SEED_DATASET

        if fixed_context_len is None:
            context_len = context_len_varying[idx]
            print('Warning: varying context len, set to %d' % context_len)
        else:
            context_len = fixed_context_len

        # build dataset kwargs for training run
        base_kwargs = dict(
            context_len=context_len,
            dim_n=dim_n,
            num_W_in_dataset=train_plus_test_size,  # this will be train_plus_test_size (since other kwargs are 1)
            context_examples_per_W=1,
            samples_per_context_example=1,
            test_ratio=0.2,
            verbose=True,
            as_torch=True,
            savez_fname=None,  # will be set internally depending on flag_save_dataset
            seed=seed_dataset,
        )
        datagen_kwargs = base_kwargs | datagen_kwargs_to_append  # manually modify with case-specific settings, case {0,1,2}

        print('run %d of n=%d' % (idx, n_ensemble),
              'seed_torch = %d, seed_dataset = %d' % (seed_torch, seed_dataset), '...')

        (net, model_fname, io_dict, loss_vals_dict,
         train_loader, test_loader, x_train, y_train, x_test, y_test,
         train_data_subspaces, test_data_subspaces) = train_model(
            nn_model=nn_model,
            restart_nn_instance=None,
            restart_dataset=None,
            context_len=context_len, dim_n=dim_n,

            datagen_case=datagen_case,
            datagen_kwargs=datagen_kwargs,
            train_plus_test_size=train_plus_test_size,

            epochs=epochs,
            batch_size=batch_size,
            seed_torch=seed_torch,
            optimizer_choice=optimizer_choice, optimizer_lr=optimizer_lr, scheduler_kwargs=scheduler_kwargs,

            full_loss_sample_interval=full_loss_sample_interval,
            flag_save_dataset=flag_save_dataset,
            flag_vis_loss=flag_vis_loss,
            flag_vis_weights=flag_vis_weights,
            flag_vis_batch_perf=flag_vis_batch_perf,
            dir_runs=dir_multitraj)
        print('done training model %d of %d' % (idx, n_ensemble))

        # visualize weight matrices using utility fn
        if flag_vis_weights:

            learned_W_KQ = net.W_KQ.detach().numpy()
            learned_W_PV = net.W_PV.detach().numpy()
            if learned_W_KQ.size == 1:  # in this case we are training 1-param weights (scaled identity) - remake as arr
                learned_W_KQ = learned_W_KQ * np.eye(dim_n)
                learned_W_PV = learned_W_PV * np.eye(dim_n)

            vis_weights_kq_pv(learned_W_KQ, learned_W_PV, model_fname)

        # compute baselines using train or test loader
        # - just use from loss_vals_dict['baselines'] if available

        # for each run, save the following metadata to a list of dictionaries
        datadict_curves_loss[idx] = dict(seed_W=seed_torch,
                                         seed_D=seed_dataset,
                                         loss_vals_dict=loss_vals_dict,
                                         io_dict=io_dict)

        vals_loss_predict_zero[idx] = loss_vals_dict['baselines']['loss_if_predict_zero']['val_train']

        if datagen_case == 0:
            vals_loss_theory_linalg[idx]          = loss_vals_dict['baselines']['loss_if_predict_linalg']['val_train']
            vals_loss_theory_linalg_shrunken[idx] = loss_vals_dict['baselines']['loss_if_predict_linalg_shrunken']['val_train']

        # fill in remaining loop arrays
        print('above gamma_imputed')
        gamma_imputed[idx] = np.mean(np.diag(learned_W_KQ)) * np.mean(np.diag(learned_W_PV))


    print('...ensemble train loop done...')

    # Step 1: convert all loss curves into arrays where one axis is ensemble
    print('\nPre-plot - Compiling loss curves into shared arrays...')
    example_loss_vals_dict = datadict_curves_loss[0]['loss_vals_dict']
    loss_A_x = example_loss_vals_dict['loss_train_batch']['x']
    loss_B_x = example_loss_vals_dict['loss_train_epoch_avg']['x']
    loss_C_x = example_loss_vals_dict['loss_train_interval']['x']
    loss_D_x = example_loss_vals_dict['loss_test_interval']['x']

    loss_A_yarr = np.zeros((len(loss_A_x), n_ensemble))
    loss_B_yarr = np.zeros((len(loss_B_x), n_ensemble))
    loss_C_yarr = np.zeros((len(loss_C_x), n_ensemble))
    loss_D_yarr = np.zeros((len(loss_D_x), n_ensemble))

    for k in range(n_ensemble):
        dict_lossvals = datadict_curves_loss[k]['loss_vals_dict']
        loss_A_yarr[:, k] = dict_lossvals['loss_train_batch']['y']
        loss_B_yarr[:, k] = dict_lossvals['loss_train_epoch_avg']['y']
        loss_C_yarr[:, k] = dict_lossvals['loss_train_interval']['y']
        loss_D_yarr[:, k] = dict_lossvals['loss_test_interval']['y']

    loss_A_mean  = np.mean(loss_A_yarr, axis=1)  # axis=1 -> each row converted into a mean
    loss_A_stdev = np.std(loss_A_yarr,  axis=1)

    loss_B_mean  = np.mean(loss_B_yarr, axis=1)
    loss_B_stdev = np.std(loss_B_yarr,  axis=1)

    loss_C_mean  = np.mean(loss_C_yarr, axis=1)
    loss_C_stdev = np.std(loss_C_yarr,  axis=1)

    loss_D_mean  = np.mean(loss_D_yarr, axis=1)
    loss_D_stdev = np.std(loss_D_yarr,  axis=1)

    # Step 2: the main plot itself -- loos curves averaged showing spread over different runs
    # Main plot
    print('\nMain plot......')
    # ================================
    plt.figure(figsize=(6, 4))
    plt.plot(loss_A_x, loss_A_mean, label='train loss (batch)')
    plt.plot(loss_A_x, loss_A_yarr, linestyle='--', alpha=0.5)

    plt.plot(loss_C_x, loss_C_mean, label='train loss (interval)')
    plt.plot(loss_C_x, loss_C_yarr, linestyle='--', alpha=0.5)

    plt.plot(loss_D_x, loss_D_mean, label='test loss (interval)')
    plt.plot(loss_D_x, loss_D_yarr, linestyle='--', alpha=0.5)

    # plot baselines (axhlines and axvlines) specific to each run
    for idx in range(n_ensemble):
        if idx == 0:
            plt.axhline(vals_loss_predict_zero[idx], linestyle='-', alpha=0.9, linewidth=2, c='grey', label=r'predict $0$')
        else:
            plt.axhline(vals_loss_predict_zero[idx], linestyle='-', alpha=0.9, linewidth=2, c='grey')

    if datagen_case == 0:
        for idx in range(n_ensemble):
            if idx == 0:
                plt.axhline(vals_loss_theory_linalg[idx], linestyle='--', alpha=0.5, linewidth=0.5,
                            label=r'$P \tilde x$')
                plt.axhline(vals_loss_theory_linalg_shrunken[idx], linestyle=':', alpha=0.5, linewidth=0.5,
                            label=r'$\gamma P \tilde x$')
            else:
                plt.axhline(vals_loss_theory_linalg[idx], linestyle='--', alpha=0.5, linewidth=0.5)
                plt.axhline(vals_loss_theory_linalg_shrunken[idx], linestyle=':', alpha=0.5, linewidth=0.5)

        # should only need to plot this once, e.g. for the first run
        loss_vals_dict = datadict_curves_loss[0]['loss_vals_dict']
        theory_baseline = loss_vals_dict['baselines']['theory_expected_error_linalg_shrunken']['val_train']
        plt.axhline(theory_baseline, linestyle='-', label=r'Expected error at $\theta^*$', color='pink')

    # note if datagen_case == 1 -- plots not yet supported, assert preventing this is above
    elif datagen_case == 1:
        for idx in range(n_ensemble):
            loss_vals_dict = datadict_curves_loss[idx]['loss_vals_dict']
            if idx == 0:
                plt.axhline(loss_vals_dict['baselines']['loss_if_predict_clustering_baseline_0var']['val_train'],
                            linestyle='--', alpha=0.5, linewidth=2,
                            color=loss_vals_dict['baselines']['loss_if_predict_clustering_baseline_0var']['pltkwargs']['color'],
                            label=loss_vals_dict['baselines']['loss_if_predict_clustering_baseline_0var']['label'])
                plt.axhline(loss_vals_dict['baselines']['loss_if_predict_clustering_baseline']['val_train'],
                            linestyle=':', alpha=0.5, linewidth=2,
                            color=loss_vals_dict['baselines']['loss_if_predict_clustering_baseline']['pltkwargs']['color'],
                            label=loss_vals_dict['baselines']['loss_if_predict_clustering_baseline']['label'])
            else:
                plt.axhline(loss_vals_dict['baselines']['loss_if_predict_clustering_baseline_0var']['val_train'],
                            color=loss_vals_dict['baselines']['loss_if_predict_clustering_baseline_0var']['pltkwargs']['color'],
                            linestyle='--', alpha=0.5, linewidth=2)
                plt.axhline(loss_vals_dict['baselines']['loss_if_predict_clustering_baseline']['val_train'],
                            color=loss_vals_dict['baselines']['loss_if_predict_clustering_baseline']['pltkwargs']['color'],
                            linestyle=':', alpha=0.5, linewidth=2)
    else:
        assert datagen_case == 2
        for idx in range(n_ensemble):
            loss_vals_dict = datadict_curves_loss[idx]['loss_vals_dict']
            if idx == 0:
                plt.axhline(loss_vals_dict['baselines']['loss_if_predict_subsphere_baseline']['val_train'],
                            linestyle='--', alpha=0.5, linewidth=2,
                            label=r'$P \tilde x$ (onto $S$)')
                plt.axhline(loss_vals_dict['baselines']['loss_if_predict_subsphere_shrunk_baseline']['val_train'],
                            linestyle=':', alpha=0.5, linewidth=2,
                            label=r'$\gamma P \tilde x$ (onto $S$, then shrunk)')
            else:
                plt.axhline(loss_vals_dict['baselines']['loss_if_predict_subsphere_baseline']['val_train'],
                            linestyle='--', alpha=0.5, linewidth=2)
                plt.axhline(loss_vals_dict['baselines']['loss_if_predict_subsphere_shrunk_baseline']['val_train'],
                            linestyle=':', alpha=0.5, linewidth=2)

    plt.xlabel(r'Epoch' + '\n\n' + runinfo_suffix)
    plt.ylabel(r'$\frac{1}{n}$ MSE')
    plt.title(r'Training curves for different seeds')
    plt.grid(alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    plt.savefig(DIR_OUT + os.sep + model_fname + '_ensemble-spread_loss_dynamics.svg')
    plt.savefig(DIR_OUT + os.sep + model_fname + '_ensemble-spread_loss_dynamics.png')
    if show_loss:
        plt.show()


if __name__ == '__main__':

    #models_to_run = ['TransformerModelV1noresOmitLast', 'TransformerModelV2noresOmitLast']
    models_to_run = ['TransformerModelV2noresOmitLast']

    datagen_case_to_epochs = {0: 100, 1: 200, 2: 200}  # 2: 4000

    #for datagen_case in [0, 1, 2]:
    for datagen_case in [1, 2]:
        for nn_model in models_to_run:
            multirun_ensemble(nn_model=nn_model, datagen_case=datagen_case,
                              n_ensemble=6,
                              epochs=datagen_case_to_epochs[datagen_case],
                              show_loss=False,
                              fixed_context_len=500)
