import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

from data_io import reload_lossbaseline_dict


def vis_weights_kq_pv(learned_W_KQ, learned_W_PV, titlemod='', dir_out=None, fname=None, flag_show=False):

    """
    learned_W_KQ = net.W_KQ.detach().numpy()
    learned_W_PV = net.W_PV.detach().numpy()
    """
    cmap ='coolwarm'
    cnorm0 = colors.CenteredNorm()
    cnorm1 = colors.CenteredNorm()

    # prepare figure and axes (gridspec)
    plt.close('all')
    fig1 = plt.figure(figsize=(8,4), constrained_layout=True)
    gs = fig1.add_gridspec(1, 4, width_ratios=[1,0.05,1,0.05])
    ax0 =      fig1.add_subplot(gs[0, 0])
    ax0_cbar = fig1.add_subplot(gs[0, 1])
    ax1 =      fig1.add_subplot(gs[0, 2])
    ax1_cbar = fig1.add_subplot(gs[0, 3])

    # plot the data
    im0 = ax0.imshow(learned_W_KQ, cmap=cmap, norm=cnorm0)
    ax0.set_title(r'$W_{KQ} \langle w_{ii} \rangle = %.3f$' % np.mean(np.diag(learned_W_KQ)))

    im1 = ax1.imshow(learned_W_PV, cmap=cmap, norm=cnorm1)
    ax1.set_title(r'$W_{PV} \langle w_{ii} \rangle = %.3f$' % np.mean(np.diag(learned_W_PV)))

    #cb0 = fig1.colorbar(im0, cax=ax0_cbar)
    #plt.colorbar(im0, cax=ax0_cbar)

    cb0 = fig1.colorbar(im0, cax=ax0_cbar)
    cb1 = plt.colorbar(im1, cax=ax1_cbar, shrink=0.5)

    title = 'Weights: %s' % titlemod
    if fname is not None:
        title += '\n%s' % fname
    plt.suptitle(title, fontsize=10)

    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

    if dir_out is not None and fname is not None:
        plt.savefig(dir_out + os.sep + fname + '_arr' + '.png', dpi=300)
        plt.savefig(dir_out + os.sep + fname + '_arr' + '.svg')
    if flag_show:
        plt.show()

    # plot diags vs rowsums as line plot
    plt.close('all')
    plt.figure(figsize=(4, 4), constrained_layout=True)
    W_KQ_diag = np.diag(learned_W_KQ)
    W_KQ_rowsum_minus_diag = np.sum(np.abs(learned_W_KQ), axis=1) - W_KQ_diag
    W_PV_diag = np.diag(learned_W_PV)
    W_PV_rowsum_minus_diag = np.sum(np.abs(learned_W_PV), axis=1) - W_PV_diag
    line1, = plt.plot(np.diag(learned_W_KQ), '-o', label='KQ diag')
    plt.plot(W_KQ_rowsum_minus_diag, '--o', label='KQ sum(abs(row)) - diag', color=line1.get_color(), markerfacecolor='None')
    line2, = plt.plot(np.diag(learned_W_PV), '-o', label='PV diag')
    plt.plot(W_PV_rowsum_minus_diag, '--o', label='PV sum(abs(row)) - diag', color=line2.get_color(), markerfacecolor='None')
    plt.axhline(0, color='k', linewidth=2)
    plt.legend()

    title = 'Weights: diagonals vs. non-diag (abs. row) sums'
    if titlemod != '':
        title += '\n%s' % titlemod
    if fname is not None:
        title += '\n%s' % fname
    plt.title(title)
    if dir_out is not None and fname is not None:
        plt.savefig(dir_out + os.sep + fname + '_diagdom' + '.png', dpi=300)
        plt.savefig(dir_out + os.sep + fname + '_diagdom' + '.svg')
    if flag_show:
        plt.show()


def vis_weights_kq_pv_multirow(learned_W_KQ_list, learned_W_PV_list,
                               titlemod='', dir_out=None, fname=None, flag_show=False):
    """
    Visualize multiple rows of weight matrices with shared colorbar limits.
    """
    global_clims = False  # False is better for typical uses
    if global_clims:
        # Calculate global min and max for all matrices
        all_data = np.concatenate([np.ravel(mat) for mat in (learned_W_KQ_list + learned_W_PV_list)])
        global_min, global_max = np.min(all_data), np.max(all_data)
        global_norm = colors.Normalize(vmin=global_min, vmax=global_max)


    # TODO could probably merge with the utility fn above vis_weights_kq_pv(....) for nrows == 1
    nrows = len(learned_W_KQ_list)

    cmap ='coolwarm'

    #col_cnorm_0 = colors.CenteredNorm()
    #col_cnorm_1 = colors.CenteredNorm()

    # prepare figure and axes (gridspec)
    plt.close('all')
    fig1 = plt.figure(figsize=(8,4), constrained_layout=True)
    gs = fig1.add_gridspec(nrows, 4, width_ratios=[1, 0.05, 1, 0.05])
    for k in range(nrows):
        ax0 =      fig1.add_subplot(gs[k, 0])
        ax0_cbar = fig1.add_subplot(gs[k, 1])
        ax1 =      fig1.add_subplot(gs[k, 2])
        ax1_cbar = fig1.add_subplot(gs[k, 3])

        if global_clims:
            row_norm = global_norm
        else:
            # Calculate row-wise vmin and vmax
            combined_data = np.concatenate([learned_W_KQ_list[k].ravel(), learned_W_PV_list[k].ravel()])
            row_vmin, row_vmax = np.min(combined_data), np.max(combined_data)
            abs_max = max(abs(row_vmin), abs(row_vmax))
            row_norm = colors.Normalize(vmin=-abs_max, vmax=abs_max)

        # plot the data
        im0 = ax0.imshow(learned_W_KQ_list[k], cmap=cmap, norm=row_norm)
        ax0.set_title(r'$W_{KQ}, \langle w_{ii} \rangle = %.3f$' % np.mean(np.diag(learned_W_KQ_list[k])))

        im1 = ax1.imshow(learned_W_PV_list[k], cmap=cmap, norm=row_norm)
        ax1.set_title(r'$W_{PV}, \langle w_{ii} \rangle = %.3f$' % np.mean(np.diag(learned_W_PV_list[k])))

        #cb0 = fig1.colorbar(im0, cax=ax0_cbar)
        #plt.colorbar(im0, cax=ax0_cbar)

        cb0 = fig1.colorbar(im0, cax=ax0_cbar)
        #cb1 = plt.colorbar(im1, cax=ax1_cbar, shrink=0.5)
        cb1 = fig1.colorbar(im1, cax=ax1_cbar, shrink=0.5)

    title = 'Weights: %s ' % titlemod + 'NORM BY MIN/MAX COL ???'
    if fname is not None:
        title += '\n%s' % fname
    plt.suptitle(title, fontsize=10)

    if dir_out is not None and fname is not None:
        plt.savefig(dir_out + os.sep + fname + '_arr_nrows%d' % nrows + '.png', dpi=300)
        plt.savefig(dir_out + os.sep + fname + '_arr_nrows%d' % nrows + '.svg')
    if flag_show:
        plt.show()

    # plot diags vs rowsums as line plot
    """
    plt.close('all')
    plt.figure(figsize=(4, 4), constrained_layout=True)
    W_KQ_diag = np.diag(learned_W_KQ)
    W_KQ_rowsum_minus_diag = np.sum(np.abs(learned_W_KQ), axis=1) - W_KQ_diag
    W_PV_diag = np.diag(learned_W_PV)
    W_PV_rowsum_minus_diag = np.sum(np.abs(learned_W_PV), axis=1) - W_PV_diag
    line1, = plt.plot(np.diag(learned_W_KQ), '-o', label='KQ diag')
    plt.plot(W_KQ_rowsum_minus_diag, '--o', label='KQ sum(abs(row)) - diag', color=line1.get_color(), markerfacecolor='None')
    line2, = plt.plot(np.diag(learned_W_PV), '-o', label='PV diag')
    plt.plot(W_PV_rowsum_minus_diag, '--o', label='PV sum(abs(row)) - diag', color=line2.get_color(), markerfacecolor='None')
    plt.axhline(0, color='k', linewidth=2)
    plt.legend()

    title = 'Weights: diagonals vs. non-diag (abs. row) sums'
    if titlemod != '':
        title += '\n%s' % titlemod
    if fname is not None:
        title += '\n%s' % fname
    plt.title(title)
    if dir_out is not None and fname is not None:
        plt.savefig(dir_out + os.sep + fname + '_diagdom' + '.png', dpi=300)
    if flag_show:
        plt.show()
    """


def vis_loss(dir_curves, titlemod='', dir_out=None, fname=None, flag_show=False):
    """
        learned_W_KQ = net.W_KQ.detach().numpy()
        learned_W_PV = net.W_PV.detach().numpy()
        """
    curve_train_eavg = np.load(dir_curves + os.sep + 'curve_losstrain_epoch_avg.npz', allow_pickle=True)
    curve_train_interval = np.load(dir_curves + os.sep + 'curve_losstrain_interval.npz', allow_pickle=True)
    curve_train_tbatch = np.load(dir_curves + os.sep + 'curve_losstrain_tbatch.npz', allow_pickle=True)
    curve_test_interval = np.load(dir_curves + os.sep + 'curve_losstest_interval.npz', allow_pickle=True)

    loss_vals_dict = {
        'loss_train_batch': dict(
            x=curve_train_tbatch['x'],
            y=curve_train_tbatch['y'],
            label='train (one batch)',
            fname='curve_loss_train_batch',
            pltkwargs=dict(linestyle='--', marker='o', color='b', markersize=4, markerfacecolor='None', alpha=0.3)),
        'loss_train_epoch_avg': dict(
            x=curve_train_eavg['x'],
            y=curve_train_eavg['y'],
            label='train (epoch moving avg)',
            fname='curve_loss_train_epoch_avg',
            pltkwargs=dict(linestyle='--', marker='o', color='b', markersize=4)),
        'loss_train_interval': dict(
            x=curve_train_interval['x'],
            y=curve_train_interval['y'],
            label='train (full)',
            fname='curve_loss_train_interval',
            pltkwargs=dict(linestyle='-', marker='o', color='b')),
        'loss_test_interval': dict(
            x=curve_test_interval['x'],
            y=curve_test_interval['y'],
            label='test (full)',
            fname='curve_loss_test_interval',
            pltkwargs=dict(linestyle='-', marker='o', color='r')),
    }

    print('Compare to null performance and lin.alg. baselines:')
    loss_baselines = reload_lossbaseline_dict(dir_out)

    print('\nPlotting training loss dynamics...')
    plt.close('all')
    plt.figure(figsize=(6, 4), layout='tight')

    for dict_key in ['loss_train_batch', 'loss_train_epoch_avg', 'loss_train_interval', 'loss_test_interval']:
        plt.plot(loss_vals_dict[dict_key]['x'], loss_vals_dict[dict_key]['y'],
                 **loss_vals_dict[dict_key]['pltkwargs'])

    if 'dumb_A_mse_on_train' in loss_baselines.keys():
        plt.axhline(loss_baselines['dumb_A_mse_on_train'], label='train (guess 0)', color='grey')
    if 'dumb_A_mse_on_test' in loss_baselines.keys():
        plt.axhline(loss_baselines['dumb_A_mse_on_test'], linestyle='--', label='test (guess 0)', color='grey')
    if 'dumb_B_mse_on_train' in loss_baselines.keys():
        plt.axhline(loss_baselines['dumb_B_mse_on_train'], label=r'train (guess $x_{k-1}$)', color='green')
    if 'dumb_B_mse_on_test' in loss_baselines.keys():
        plt.axhline(loss_baselines['dumb_B_mse_on_test'], linestyle='--', label=r'test (guess $x_{k-1}$)',
                    color='green')
    if 'dumb_C_mse_on_train' in loss_baselines.keys():
        plt.axhline(loss_baselines['dumb_C_mse_on_train'], label=r'train (guess mean)', color='orange')
    if 'dumb_C_mse_on_test' in loss_baselines.keys():
        plt.axhline(loss_baselines['dumb_C_mse_on_test'], linestyle='--', label=r'test (guess mean)', color='orange')

    if 'heuristic_mse_on_train' in loss_baselines.keys():
        plt.axhline(loss_baselines['heuristic_mse_on_train'], label=r'train ($P_W$)', color='black')
    if 'heuristic_mse_on_test' in loss_baselines.keys():
        plt.axhline(loss_baselines['heuristic_mse_on_test'], linestyle='--', label=r'test ($P_W$)', color='black')
    if 'heuristic_mse_shrunken_on_train' in loss_baselines.keys():
        plt.axhline(loss_baselines['heuristic_mse_shrunken_on_train'], label=r'train ($P_W$ shrunk)', color='purple')
    if 'heuristic_mse_shrunken_on_test' in loss_baselines.keys():
        plt.axhline(loss_baselines['heuristic_mse_shrunken_on_test'], linestyle='--', label=r'test ($P_W$ shrunk)',
                    color='purple')
    if 'theory_expected_error_linalg_shrunken' in loss_baselines.keys():
        plt.axhline(loss_baselines['theory_expected_error_linalg_shrunken'], linestyle=':', color='purple',
                    label=r'$L(\theta^*)$')

    plt.legend(ncol=2)
    plt.grid(alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')

    title = 'Loss (replot)'
    if fname is not None:
        title += '\n%s' % fname
    plt.suptitle(title, fontsize=10)

    # plt.ylim(0.14, 0.15)

    if dir_out is not None and fname is not None:
        plt.savefig(dir_out + os.sep + fname + '_loss_replot' + '.png', dpi=300)
    if flag_show:
        plt.show()
    else:
        plt.close('all')
