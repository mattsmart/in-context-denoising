import datetime
import numpy as np
import os
import pickle
import torch

from settings import DIR_RUNS

"""
Script: save/load models and datasets
"""


def run_subdir_setup(dir_runs=DIR_RUNS, run_subfolder=None, timedir_override=None, minimal_mode=False):
    """
    Create a new directory for the run, and save the model trajectory and dataset there

    Structure:
        /runs-dir/
            /dir_base (timestamped run folder)
                /model_checkpoints/...
                /vis/...
                /data_for_replot/...
                /model_end.pth
                /training_dataset_split.npz
                /runinfo.txt
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%I.%M.%S%p")
    experiment_dir = dir_runs

    if timedir_override is not None:
        time_folder = timedir_override
    else:
        time_folder = current_time

    if run_subfolder is None:
        dir_current_run = experiment_dir + os.sep + time_folder
    else:
        if os.path.isabs(run_subfolder):
            dir_current_run = run_subfolder + os.sep + time_folder
        else:
            dir_current_run = experiment_dir + os.sep + run_subfolder + os.sep + time_folder

    # make subfolders in the timestamped run directory:
    dir_checkpoints = os.path.join(dir_current_run, "model_checkpoints")
    dir_vis = os.path.join(dir_current_run, "vis")
    dir_data_for_replot = os.path.join(dir_current_run, "data_for_replot")

    if minimal_mode:
        dir_list = [dir_runs, dir_current_run]
    else:
        dir_list = [dir_runs, dir_current_run, dir_checkpoints, dir_vis, dir_data_for_replot]
    for dirs in dir_list:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    # io path storage to pass around
    io_dict = {'dir_base': dir_current_run,
               'dir_checkpoints': dir_checkpoints,
               'dir_vis': dir_vis,
               'dir_data_for_replot': dir_data_for_replot,
               'runinfo': dir_current_run + os.sep + 'runinfo.txt'}

    # make minimal run_info settings file with first line as the base output dir
    runinfo_append(io_dict, ['dir_base, %s' % dir_current_run])

    return io_dict


def runinfo_append(io_dict, info_list, multi=False):
    """
    append to metadata file storing parameters for the run
    """
    # multi: list of list flag
    if multi:
        with open(io_dict['runinfo'], 'a') as runinfo:
            for line in info_list:
                runinfo.write('\n'.join(line))
    else:
        with open(io_dict['runinfo'], 'a') as runinfo:
            runinfo.write('\n'.join(info_list))
            runinfo.write('\n')
    return


def pickle_load(fpath):
    """
    Generic pickle loader
    """
    with open(fpath, 'rb') as pickle_file:
        loaded_object = pickle.load(pickle_file)
    return loaded_object


def load_dataset(dataset_fpath, as_torch=True):
    """
    sample fpath:
        dataset_fpath = DIR_DATA + os.sep + 'dataset_example.npz'
    """
    # load dataset from file
    print('loading dataset at:', dataset_fpath)
    npzfile = np.load(dataset_fpath)
    print(sorted(npzfile.files))
    print('loaded x_train:', npzfile['x_train'].shape)
    print('loaded y_train:', npzfile['y_train'].shape)
    print('loaded x_test:',  npzfile['x_test'].shape)
    print('loaded y_test:',  npzfile['y_test'].shape)

    if as_torch:
        loaded_x_train = torch.from_numpy(npzfile['x_train'])
        loaded_y_train = torch.from_numpy(npzfile['y_train'])
        loaded_x_test = torch.from_numpy(npzfile['x_test'])
        loaded_y_test = torch.from_numpy(npzfile['y_test'])
    else:
        loaded_x_train = npzfile['x_train']
        loaded_y_train = npzfile['y_train']
        loaded_x_test = npzfile['x_test']
        loaded_y_test = npzfile['y_test']

    return loaded_x_train, loaded_y_train, loaded_x_test, loaded_y_test


def load_runinfo_from_rundir(dir_run, model_prefix=''):
    """
    Load runinfo.txt from a given run directory
    """
    runinfo_fpath = dir_run + os.sep + model_prefix + 'runinfo.txt'
    with open(runinfo_fpath, 'r') as runinfo:
        runinfo_lines = runinfo.readlines()

    # step: convert runinfo to a dictionary and return it
    runinfo_dict = {}
    for line in runinfo_lines:
        if line.split(',')[0].strip() == 'scheduler':
            key = line.split(',')[0]
            val = ','.join(line.split(',')[1:]).strip()
            val = eval(val)
            assert isinstance(val, dict)
        else:
            key, val = line.split(',')
            key, val = key.strip(), val.strip()
        runinfo_dict[key] = val

    # handle potentially ambiguous key -> value types
    # - style_subspace_dimensions (int or str)
    # - seed_dataset (None or int)
    # - could have diff number of keys relating to gradient descent e.g. adam_lr, sgd_lr, etc. -- keep as str
    for key in ['epochs', 'batch_size', 'dim_n', 'context_len',
                'train_plus_test_size', 'full_loss_sample_interval',
                'context_examples_per_W', 'samples_per_context_example', 'num_W_in_dataset']:
        runinfo_dict[key] = int(runinfo_dict[key])
    for key in ['style_subspace_dimensions']:
        if runinfo_dict[key] != 'full':
            runinfo_dict[key] = int(runinfo_dict[key])
    for key in ['sigma2_corruption', 'sigma2_pure_context', 'test_ratio']:
        runinfo_dict[key] = float(runinfo_dict[key])
    for key in ['style_corruption_orthog', 'style_origin_subspace']:
        runinfo_dict[key] = runinfo_dict[key] == 'True'  # if True, the bool val is True, else False

    # specific checks for particular datagen cases
    if 'case1_specific_style_cluster_mus' in runinfo_dict.keys():
        runinfo_dict['style_cluster_mus'] = runinfo_dict['case1_specific_style_cluster_mus']
        del runinfo_dict['case1_specific_style_cluster_mus']

    if 'case1_specific_style_cluster_vars' in runinfo_dict.keys():
        runinfo_dict['style_cluster_vars'] = runinfo_dict['case1_specific_style_cluster_vars']
        del runinfo_dict['case1_specific_style_cluster_vars']

    if 'case1_specific_cluster_var' in runinfo_dict.keys():
        runinfo_dict['cluster_var'] = float(runinfo_dict['case1_specific_cluster_var'])
        del runinfo_dict['case1_specific_cluster_var']

    if 'case1_specific_num_cluster' in runinfo_dict.keys():
        runinfo_dict['num_cluster'] = int(runinfo_dict['case1_specific_num_cluster'])
        del runinfo_dict['case1_specific_num_cluster']
    return runinfo_dict


def reload_lossbaseline_dict(dir_replot):
    with open(dir_replot + os.sep + 'loss_baselines.txt') as f:
        loss_baselines_dict = {}
        lines = [a.strip() for a in f.readlines()]
        for lstr in lines:
            a, b_str = lstr.split(',')
            loss_baselines_dict[a] = float(b_str)
        return loss_baselines_dict
