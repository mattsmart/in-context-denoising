import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np

from src.data_io import load_runinfo_from_rundir
from src.nn_model_base import load_modeltype_from_fname, load_model_from_rundir
from src.settings import DIR_RUNS, DIR_OUT
from src.visualize import vis_weights_kq_pv


dir_parent = DIR_RUNS
dir_run = dir_parent + os.sep + 'example_run'
dir_vis = dir_run + os.sep + 'vis'


print('Re-loading model...')

# load runinfo settings
runinfo_dict = load_runinfo_from_rundir(dir_run)
dim_n = runinfo_dict['dim_n']
context_length = runinfo_dict['context_len']
nn_model_str = runinfo_dict['model']
epochs = runinfo_dict['epochs']

for epoch_int in range(epochs):

    # load specific model(s) - model_checkpoints
    net = load_model_from_rundir(dir_run, epoch_int=epoch_int)

    learned_W_KQ = net.W_KQ.detach().numpy()
    learned_W_PV = net.W_PV.detach().numpy()
    if learned_W_KQ.size == 1:  # in this case we are training 1-param weights (scaled identity) - remake as arr
        learned_W_KQ = learned_W_KQ * np.eye(dim_n)
        learned_W_PV = learned_W_PV * np.eye(dim_n)

    # visualize weight matrices using utility fn
    vis_weights_kq_pv(learned_W_KQ, learned_W_PV, titlemod='epoch %d' % epoch_int,
                      dir_out=dir_vis, fname='weights_e%d' % epoch_int, flag_show=False)
