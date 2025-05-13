import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from src.data_io import load_runinfo_from_rundir, reload_lossbaseline_dict
from src.nn_loss_baselines import theory_linear_expected_error
from src.nn_model_base import load_modeltype_from_fname, load_model_from_rundir, MODEL_CLASS_FROM_STR
from src.settings import DIR_RUNS, DIR_OUT


dir_parent = DIR_RUNS

# Overview: select one or two multiruns (ensemble training runs) for re-plotting

# multirun - case 0  (linear)
dir_multirun_LSA = dir_parent + os.sep + 'multitraj_ensemble_case0_TV1nrOL_adam_e100_L500'
dir_multirun_S = dir_parent + os.sep + 'multitraj_ensemble_case0_TV2nrOL_adam_e100_L500'
MARKER = ['^', 'o']
multiruns_to_merge = [dir_multirun_LSA, dir_multirun_S]

# multirun - Case  (GMM), 2 (manifold/subspheres)
#dir_multirun_S = dir_parent + os.sep + 'multitraj_ensemble_case1_zpt1_p8_spt02_TV2nrOL_adam_e200_L500'
#MARKER = ['o']
#multiruns_to_merge = [dir_multirun_S]

#dir_multirun_S = dir_parent + os.sep + 'multitraj_ensemble_case2_zpt1_TV2nrOL_adam_e200_L500'
#dir_multirun_S = dir_parent + os.sep + 'multitraj_ensemble_case2_zpt1_TV2nrOL_adam_e400_L500'
#MARKER = ['o']
#multiruns_to_merge = [dir_multirun_S]

# Parameters for plotting
k_step = 15  # Only plot every k-th point for train/test curves (30 or 15)

# Mode A: replot a single multiraj ensemble
#multiruns_solo = dir_parent + os.sep + 'multitraj_ensemble_case2_zpt1_TV2nrOL_adam_e400_L500'
#multiruns_to_merge = [multiruns_solo]

# Mode B: we can show softmax and linear test/train curves on same plot
#multiruns_to_merge = [dir_multirun_LSA, dir_multirun_S]
#multiruns_to_merge = [dir_multirun_S]


COLOR_TRAIN      = ['#4C72B0', '#4C72B0']  # blue, teal
COLOR_TRAIN_MEC  = ['#3D5B8D', '#3D5B8D']
COLOR_TEST       = ['#7BC475', '#7BC475']  # green, orange
COLOR_TEST_MEC   = ['#629D5E', '#629D5E']
COLOR_PROJ       = ['#BCBEC0', '#D0D2D7']  # grey, grey
COLOR_PROJSHRUNK = ['#B56BAC', '#D78CCB']  # purple, purple

#MARKER = ['^', 'o']
#LINESTYLE = [(0, (2, 1)), '-']
LINESTYLE = ['-', '-']
plot_all_runs = []  # plot all runs for these indices
#plot_all_runs = []  # plot all runs for these indices

fill_spread_runs = [0, 1]  # fill in spread over seeds for these indices


# Step 2: the main plot itself -- loos curves averaged showing spread over different runs
print('\nMain plot......')
# ================================
plt.figure(figsize=(2.7, 2))

for idx, dir_multirun in enumerate(multiruns_to_merge):

    # find all runs in folder
    run_folders = [f for f in os.listdir(dir_multirun) if os.path.isdir(os.path.join(dir_multirun, f))]
    N_ENSEMBLE = len(run_folders)

    # load the runinfo from the first run
    dir_run = dir_multirun + os.sep + run_folders[0]
    runinfo_dict = load_runinfo_from_rundir(dir_run)
    datagen_case = int(runinfo_dict['datagen_case'])
    dim_n = runinfo_dict['dim_n']
    context_length = runinfo_dict['context_len']
    nn_model_str = runinfo_dict['model']
    epochs = runinfo_dict['epochs']
    subspace_dim_d = runinfo_dict['style_subspace_dimensions']
    sigma2_pure_context = runinfo_dict['sigma2_pure_context']
    sigma2_corruption = runinfo_dict['sigma2_corruption']
    style_corruption_orthog = runinfo_dict['style_corruption_orthog']
    style_origin_subspace = runinfo_dict['style_origin_subspace']
    full_loss_sample_interval = runinfo_dict['full_loss_sample_interval']
    optimizer_choice = runinfo_dict['optimizer_choice']

    model_fname = runinfo_dict['fname']
    nn_model = runinfo_dict['model']

    runinfo_suffix = (nn_model + '\n' + r'dim $n = %d$, dim $d = %s$, $\sigma_{0}^2 = %.2f, \sigma_{z}^2 = %.2f$' %
                      (dim_n, subspace_dim_d, sigma2_pure_context, sigma2_corruption))

    if datagen_case == 0:
        eq_expected_error_linalg_shrunken = theory_linear_expected_error(
            dim_n, subspace_dim_d, sigma2_corruption, sigma2_pure_context)

    # Build the dict: datadict_curves_loss
    datadict_curves_loss = {k: dict() for k in range(N_ENSEMBLE)}
    for k in range(N_ENSEMBLE):
        print(run_folders[k])
        dir_replot = dir_multirun + os.sep + run_folders[k] + os.sep + 'data_for_replot'

        with open(dir_replot + os.sep + 'loss_vals_dict.pkl', 'rb') as handle:
            loss_vals_dict = pickle.load(handle)
        datadict_curves_loss[k]['loss_vals_dict'] = loss_vals_dict

        lossbaseline_dict = reload_lossbaseline_dict(dir_replot)
        datadict_curves_loss[k]['vals_loss_predict_zero'] = lossbaseline_dict['dumb_A_mse_on_train']

        if datagen_case == 0:
            datadict_curves_loss[k]['vals_loss_theory'] = lossbaseline_dict['heuristic_mse_on_train']
            datadict_curves_loss[k]['vals_loss_theory_shrunken'] = lossbaseline_dict['heuristic_mse_shrunken_on_train']
            datadict_curves_loss[k]['vals_loss_theory_shrunken_test'] = lossbaseline_dict['heuristic_mse_shrunken_on_test']
            datadict_curves_loss[k]['eq_expected_error_linalg_shrunken'] = eq_expected_error_linalg_shrunken
        elif datagen_case == 1:
            print(lossbaseline_dict.keys())
            datadict_curves_loss[k]['vals_loss_theory_shrunken'] = lossbaseline_dict['heuristic_mse_clustering_on_train']
            datadict_curves_loss[k]['vals_loss_theory'] = lossbaseline_dict['heuristic_mse_clustering_0var_on_train']
            datadict_curves_loss[k]['vals_loss_theory_shrunken_test'] = lossbaseline_dict['heuristic_mse_clustering_on_test']
        elif datagen_case == 2:
            print(lossbaseline_dict.keys())
            datadict_curves_loss[k]['vals_loss_theory'] = lossbaseline_dict['heuristic_mse_subsphere_on_train']
            datadict_curves_loss[k]['vals_loss_theory_shrunken'] = lossbaseline_dict['heuristic_mse_subsphere_shrunken_on_train']
            datadict_curves_loss[k]['vals_loss_theory_shrunken_test'] = lossbaseline_dict['heuristic_mse_subsphere_shrunken_on_test']
        else:
            raise ValueError('Unsupported datagen_case; 0, 2 are supported (loaded: %s)' % datagen_case)


    # Step 1: convert all loss curves into arrays where one axis is ensemble
    print('\nPre-plot - Compiling loss curves into shared arrays...')
    example_loss_vals_dict = datadict_curves_loss[0]['loss_vals_dict']
    loss_A_x = example_loss_vals_dict['loss_train_batch']['x']
    loss_B_x = example_loss_vals_dict['loss_train_epoch_avg']['x']
    loss_C_x = example_loss_vals_dict['loss_train_interval']['x']
    loss_D_x = example_loss_vals_dict['loss_test_interval']['x']

    loss_A_yarr = np.zeros((len(loss_A_x), N_ENSEMBLE))
    loss_B_yarr = np.zeros((len(loss_B_x), N_ENSEMBLE))
    loss_C_yarr = np.zeros((len(loss_C_x), N_ENSEMBLE))
    loss_D_yarr = np.zeros((len(loss_D_x), N_ENSEMBLE))

    for k in range(N_ENSEMBLE):
        dict_lossvals = datadict_curves_loss[k]['loss_vals_dict']
        loss_A_yarr[:, k] = dict_lossvals['loss_train_batch']['y']
        loss_B_yarr[:, k] = dict_lossvals['loss_train_epoch_avg']['y']
        loss_C_yarr[:, k] = dict_lossvals['loss_train_interval']['y']
        loss_D_yarr[:, k] = dict_lossvals['loss_test_interval']['y']

    loss_A_mean  = np.mean(loss_A_yarr, axis=1)  # axis=1 -> each row converted into a mean
    loss_A_stdev = np.std(loss_A_yarr, axis=1)

    loss_B_mean  = np.mean(loss_B_yarr, axis=1)
    loss_B_stdev = np.std(loss_B_yarr, axis=1)

    loss_C_mean  = np.mean(loss_C_yarr, axis=1)
    loss_C_stdev = np.std(loss_C_yarr, axis=1)

    loss_D_mean  = np.mean(loss_D_yarr, axis=1)
    loss_D_stdev = np.std(loss_D_yarr, axis=1)

    # plot lines without any markers
    plt.plot(loss_C_x, loss_C_mean, ms=0,
             color=COLOR_TRAIN[idx], linestyle=LINESTYLE[idx], markeredgecolor='k', linewidth=2, zorder=10)
    plt.plot(loss_D_x, loss_D_mean, ms=0,
             color=COLOR_TEST[idx],  linestyle=LINESTYLE[idx], markeredgecolor='k', linewidth=2, zorder=11)

    # plot lines with markers, sparsely (k_step)
    plt.plot(loss_C_x[::k_step], loss_C_mean[::k_step], marker=MARKER[idx], ms=3, label='train loss (interval)',
             markerfacecolor=COLOR_TRAIN[idx], linestyle='', markeredgecolor=COLOR_TRAIN_MEC[idx], markeredgewidth=1, linewidth=2, zorder=10)
    plt.plot(loss_D_x[::k_step], loss_D_mean[::k_step], marker=MARKER[idx], ms=3, label='test loss (interval)',
             markerfacecolor=COLOR_TEST[idx],  linestyle='', markeredgecolor=COLOR_TEST_MEC[idx], markeredgewidth=1, zorder=11)

    # only plot fill between for selected folders
    if idx in plot_all_runs:
        plt.plot(loss_C_x[::k_step], loss_C_yarr[::k_step], linestyle=LINESTYLE[idx], alpha=0.4, color=COLOR_TRAIN[idx], linewidth=0.75, zorder=10)
        plt.plot(loss_D_x[::k_step], loss_D_yarr[::k_step], linestyle=LINESTYLE[idx], alpha=0.4, color=COLOR_TEST[idx], linewidth=0.75, zorder=10)

    # Add ± standard deviation bands
    fill_stdev = False
    if fill_stdev:
        plt.fill_between(loss_C_x, loss_C_mean - loss_C_stdev, loss_C_mean + loss_C_stdev,
                         alpha=0.2, color=COLOR_TRAIN[idx], label=r'Train $\pm$ 1 std. dev')
        plt.fill_between(loss_D_x, loss_D_mean - loss_D_stdev, loss_D_mean + loss_D_stdev,
                         alpha=0.2, color=COLOR_TEST[idx], label=r'Test $\pm$ 1 std. dev')

    # plot min/max envelope using fill between
    if idx in fill_spread_runs:

        # alt fill option (min/max)
        plt.fill_between(loss_C_x, np.min(loss_C_yarr, axis=1), np.max(loss_C_yarr, axis=1), alpha=0.07, color=COLOR_TRAIN[idx], zorder=7)
        plt.fill_between(loss_D_x, np.min(loss_D_yarr, axis=1), np.max(loss_D_yarr, axis=1), alpha=0.07, color=COLOR_TEST[idx], zorder=7)
        # plot envelope curves (bit darker)
        plt.plot(loss_C_x, np.min(loss_C_yarr, axis=1), alpha=0.2, zorder=8, color=COLOR_TRAIN_MEC[idx])
        plt.plot(loss_C_x, np.max(loss_C_yarr, axis=1), alpha=0.2, zorder=8, color=COLOR_TRAIN_MEC[idx])
        plt.plot(loss_D_x, np.min(loss_D_yarr, axis=1), alpha=0.2, zorder=8, color=COLOR_TEST_MEC[idx])
        plt.plot(loss_D_x, np.max(loss_D_yarr, axis=1), alpha=0.2, zorder=8, color=COLOR_TEST_MEC[idx])

    # plot axhlines and axvlines specific to each run
    lw_baselines = 1.5
    for run_idx in range(N_ENSEMBLE):
        subdict_loss = datadict_curves_loss[run_idx]
        if run_idx == 0:
            plt.axhline(subdict_loss['vals_loss_predict_zero'], linestyle='--', alpha=0.75, linewidth=lw_baselines, label=r'predict $0$', color='grey')
            plt.axhline(subdict_loss['vals_loss_theory'], linestyle='-', alpha=0.75, linewidth=lw_baselines, label=r'$f_\text{opt}(\tilde x)$ simple', color=COLOR_PROJ[idx])
            plt.axhline(subdict_loss['vals_loss_theory_shrunken'], linestyle='-', alpha=0.75, linewidth=lw_baselines, label=r'$f_\text{opt}(\tilde x)$', color=COLOR_PROJSHRUNK[idx])
            plt.axhline(subdict_loss['vals_loss_theory_shrunken_test'], linestyle='--', alpha=0.75, linewidth=lw_baselines,
                        label=r'$f_\text{opt}(\tilde x)$', color=COLOR_PROJSHRUNK[idx])
            if datagen_case == 0:
                plt.axhline(subdict_loss['eq_expected_error_linalg_shrunken'], linestyle=':', alpha=0.75, linewidth=lw_baselines,
                            label=r'Expected error at $\theta^*$', color=COLOR_PROJSHRUNK[idx])


# decide if epoch or batch multiplier for x axis
plt.xlabel(r'Epoch' + '\n\n' + runinfo_suffix)
plt.ylabel(r'$\mathrm{MSE}/n$')

#plt.grid(alpha=0.3)
plt.legend(ncol=2)
#plt.tight_layout()

if datagen_case == 0:
    #plt.ylim(0.0, 1.1)
    plt.ylim(0.2, 1.1)
    #plt.xlim(-4, 104)
    plt.xlim(-1, 54)

if datagen_case == 1:
    #plt.ylim(0.0, 1.1)
    #plt.ylim(0.2, 1.1)
    plt.xlim(-2, 114)

if datagen_case == 2:
    #plt.ylim(0.0, 1.1)
    plt.ylim(0.026, 0.0673)
    plt.xlim(-2, 114)

# decrease x and y tick label size
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

print('Saving loss-replot to file...')
print(DIR_OUT + os.sep + model_fname + '_ensemble-spread_loss_dynamics.svg')
plt.savefig(DIR_OUT + os.sep + model_fname + '_ensemble-spread_loss_dynamics.svg')
plt.savefig(DIR_OUT + os.sep + model_fname + '_ensemble-spread_loss_dynamics.png')
plt.show()


def plot_weights_comparison(WKQ1, WPV1, WKQ2, WPV2, left_title="Softmax", right_title="Linear", show_border=True, save_path="weights_comparison"):
    """
    Plots the comparison of final weights for two models (softmax and linear).

    Args:
        softmax_model: Model with weights to be labeled as "softmax".
        linear_model: Model with weights to be labeled as "linear".
        save_path (str): Path to save the resulting figure as an SVG file.
        left_title (str): Title for the left panels (softmax-related).
        right_title (str): Title for the right panels (linear-related).
    """
    fig, axs = plt.subplots(1, 4, figsize=(6, 2), gridspec_kw={'width_ratios': [1, 1, 1, 1], 'wspace': 0.4})

    # Shared vmin/vmax for left and right panels
    vmin_left = -np.max(np.abs([WKQ1, WPV1]))
    vmax_left = np.max(np.abs([WKQ1, WPV1]))
    vmin_right = -np.max(np.abs([WKQ2, WPV2]))
    vmax_right = np.max(np.abs([WKQ2, WPV2]))

    # Left panels
    im1 = axs[0].imshow(WKQ1, cmap='coolwarm', vmin=vmin_left, vmax=vmax_left, aspect='equal')
    axs[0].set_title(r"$W_{KQ}$", fontsize=10)
    im2 = axs[1].imshow(WPV1, cmap='coolwarm', vmin=vmin_left, vmax=vmax_left, aspect='equal')
    axs[1].set_title(r"$W_{PV}$", fontsize=10)

    # Right panels
    im3 = axs[2].imshow(WKQ2, cmap='coolwarm', vmin=vmin_right, vmax=vmax_right, aspect='equal')
    axs[2].set_title(r"$W_{KQ}$", fontsize=10)
    im4 = axs[3].imshow(WPV2, cmap='coolwarm', vmin=vmin_right, vmax=vmax_right, aspect='equal')
    axs[3].set_title(r"$W_{PV}$", fontsize=10)

    # Add text in the top-right corner of each imshow
    def add_text(ax, matrix):
        mean_diag = np.mean(np.diag(matrix))
        ax.text(0.92, 0.92, f"{mean_diag:.3f}", color="black", fontsize=10,
                ha="right", va="top", transform=ax.transAxes)

    add_text(axs[0], WKQ1)
    add_text(axs[1], WPV1)
    add_text(axs[2], WKQ2)
    add_text(axs[3], WPV2)

    # Optional: Hide border
    if not show_border:
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for ax in axs:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Add shared colorbars
    cbar_ax1 = fig.add_axes([0.25, 0.15, 0.25, 0.03])  # Bottom bar for left panels
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.ax.tick_params(labelsize=8)
    cbar1.set_label(left_title, fontsize=9)
    cbar1.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    cbar_ax2 = fig.add_axes([0.62, 0.15, 0.25, 0.03])  # Bottom bar for right panels
    cbar2 = fig.colorbar(im3, cax=cbar_ax2, orientation='horizontal')
    cbar2.ax.tick_params(labelsize=8)
    cbar2.set_label(right_title, fontsize=9)
    cbar2.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Save the figure
    plt.savefig(save_path + '.svg', format="svg", bbox_inches="tight", dpi=300)
    plt.savefig(save_path + '.png', format="svg", bbox_inches="tight", dpi=300)
    print(f"Figure saved as {save_path}")
    plt.show()


def plot_weights_comparison_pcolormesh(WKQ1, WPV1, WKQ2, WPV2,
                                       left_title="Softmax", right_title="Linear",
                                       show_border=True, save_path="weights_comparison_pcolormesh"):
    """
    Plots the comparison of final weights for two models (softmax and linear) using pcolormesh for vector graphics.

    Args:
        WKQ1, WPV1: np.ndarray
            Weight matrices for the first model (left side).
        WKQ2, WPV2: np.ndarray
            Weight matrices for the second model (right side).
        left_title, right_title: str
            Titles for the left and right models.
        show_border: bool
            If True, keeps subplot borders. If False, hides axes and borders.
        save_path: str
            Path to save the resulting figure (SVG format for vector graphics).
    """
    fig, axs = plt.subplots(1, 4, figsize=(6, 2), gridspec_kw={'width_ratios': [1, 1, 1, 1], 'wspace': 0.4})

    # Shared vmin/vmax for left and right panels
    vmin_left = -np.max(np.abs([WKQ1, WPV1]))
    vmax_left = np.max(np.abs([WKQ1, WPV1]))
    vmin_right = -np.max(np.abs([WKQ2, WPV2]))
    vmax_right = np.max(np.abs([WKQ2, WPV2]))

    # Left panels
    im1 = axs[0].pcolormesh(np.flipud(WKQ1), cmap='coolwarm', vmin=vmin_left, vmax=vmax_left, shading='auto')
    axs[0].set_title(r"$W_{KQ}$", fontsize=10)
    im2 = axs[1].pcolormesh(np.flipud(WPV1), cmap='coolwarm', vmin=vmin_left, vmax=vmax_left, shading='auto')
    axs[1].set_title(r"$W_{PV}$", fontsize=10)

    # Right panels
    im3 = axs[2].pcolormesh(np.flipud(WKQ2), cmap='coolwarm', vmin=vmin_right, vmax=vmax_right, shading='auto')
    axs[2].set_title(r"$W_{KQ}$", fontsize=10)
    im4 = axs[3].pcolormesh(np.flipud(WPV2), cmap='coolwarm', vmin=vmin_right, vmax=vmax_right, shading='auto')
    axs[3].set_title(r"$W_{PV}$", fontsize=10)

    # Enforce **perfectly square** panels
    for ax in axs:
        ax.set_aspect("equal")  # Ensures the square shape
        ax.set_xticks([])  # Hide x ticks
        ax.set_yticks([])  # Hide y ticks

    # Add text in the top-right corner of each subplot with the mean diagonal
    def add_text(ax, matrix):
        mean_diag = np.mean(np.diag(matrix))
        ax.text(0.92, 0.92, f"{mean_diag:.3f}", color="black", fontsize=10,
                ha="right", va="top", transform=ax.transAxes)

    add_text(axs[0], WKQ1)
    add_text(axs[1], WPV1)
    add_text(axs[2], WKQ2)
    add_text(axs[3], WPV2)

    # Optional: Hide border
    if not show_border:
        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

    # Add shared colorbars
    cbar_ax1 = fig.add_axes([0.25, 0.15, 0.25, 0.03])  # Bottom bar for left panels
    cbar1 = fig.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.ax.tick_params(labelsize=8)
    cbar1.set_label(left_title, fontsize=9)
    cbar1.ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    cbar1.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    cbar_ax2 = fig.add_axes([0.62, 0.15, 0.25, 0.03])  # Bottom bar for right panels
    cbar2 = fig.colorbar(im3, cax=cbar_ax2, orientation='horizontal')
    cbar2.ax.tick_params(labelsize=8)
    cbar2.set_label(right_title, fontsize=9)
    cbar2.ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
    cbar2.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Save the figure as vector graphics
    plt.savefig(save_path + '.svg', format="svg", bbox_inches="tight", dpi=300)
    plt.savefig(save_path + '.png', format="png", bbox_inches="tight", dpi=300)
    print(f"Figure saved as {save_path}.svg")
    plt.show()


def plot_alpha_beta_scatter(alphas, betas, save_path="alpha_beta_scatter"):
    """
    Plots a scatter plot of alpha = <w_ii> for W_PV and beta = <w_ii> for W_KQ across ensemble runs.

    Args:
        alphas: np.ndarray
            Array of mean diagonal values for W_PV across ensemble runs.
        betas: np.ndarray
            Array of mean diagonal values for W_KQ across ensemble runs.
        save_path: str
            Path to save the resulting figure (SVG format for vector graphics).
    """
    plt.figure(figsize=(3, 3))
    plt.scatter(betas, alphas, color='black', edgecolors='k', alpha=0.7, s=20)

    # Axes labels
    plt.xlabel(r"$\beta = \langle W_{KQ, ii} \rangle$")
    plt.ylabel(r"$\alpha = \langle W_{PV, ii} \rangle$")

    # Dashed lines at means
    mean_alpha = np.mean(alphas)
    mean_beta = np.mean(betas)
    plt.axhline(mean_alpha, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.axvline(mean_beta, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    # Annotate means
    plt.text(mean_beta, min(alphas), f"{mean_beta:.3f}", ha="center", va="bottom", fontsize=9)
    plt.text(min(betas), mean_alpha, f"{mean_alpha:.3f}", ha="left", va="center", fontsize=9)

    # Grid & limits
    plt.grid(alpha=0.3)
    plt.xlim(min(betas) - 0.02, max(betas) + 0.02)
    plt.ylim(min(alphas) - 0.02, max(alphas) + 0.02)

    # Save figure
    plt.savefig(save_path + '.svg', format="svg", bbox_inches="tight", dpi=300)
    plt.savefig(save_path + '.png', format="png", bbox_inches="tight", dpi=300)
    print(f"Figure saved as {save_path}.svg")
    plt.show()


print('Plot init weights and final weights (paired way)...')
print('------------------------')
if len(multiruns_to_merge) == 2:  # only do this for paired runs, softmax and LSA

    A_run_folders = [f for f in os.listdir(multiruns_to_merge[0]) if
                     os.path.isdir(os.path.join(multiruns_to_merge[0], f))]
    B_run_folders = [f for f in os.listdir(multiruns_to_merge[1]) if
                     os.path.isdir(os.path.join(multiruns_to_merge[1], f))]

    for k in range(N_ENSEMBLE):

        A_dir_run = multiruns_to_merge[0] + os.sep + A_run_folders[k]
        A_runinfo_dict = load_runinfo_from_rundir(A_dir_run)

        A_datagen_case = int(A_runinfo_dict['datagen_case'])
        A_nn_model_str = A_runinfo_dict['model']
        A_nn_alias = MODEL_CLASS_FROM_STR[A_nn_model_str]['alias']

        B_dir_run = multiruns_to_merge[1] + os.sep + B_run_folders[k]
        B_runinfo_dict = load_runinfo_from_rundir(B_dir_run)

        B_datagen_case = int(B_runinfo_dict['datagen_case'])
        B_nn_model_str = B_runinfo_dict['model']
        B_nn_alias = MODEL_CLASS_FROM_STR[B_nn_model_str]['alias']

        epoch_start = 0
        epoch_end = None  # none means final epoch

        epoch_labels = ['init', 'final']

        for idx, epoch_choice in enumerate([epoch_start, epoch_end]):

            # softmax assumed
            A_net = load_model_from_rundir(A_dir_run, epoch_int=epoch_choice)
            A_W_KQ = A_net.W_KQ.detach().numpy()
            A_W_PV = A_net.W_PV.detach().numpy()

            # linear assumed
            B_net = load_model_from_rundir(B_dir_run, epoch_int=epoch_choice)
            B_W_KQ = B_net.W_KQ.detach().numpy()
            B_W_PV = B_net.W_PV.detach().numpy()

            plot_weights_comparison_pcolormesh(A_W_KQ, A_W_PV, B_W_KQ, B_W_PV, left_title=A_nn_model_str,
                                    right_title=B_nn_model_str,
                                    save_path=DIR_OUT + os.sep + 'replot_paired_run%d_weights_pcolor_%s_case%d' % (
                                    k, epoch_labels[idx], datagen_case))

else:
    assert len(multiruns_to_merge) == 1  # only do this for paired runs, softmax and LSA

    A_run_folders = [f for f in os.listdir(multiruns_to_merge[0]) if
                     os.path.isdir(os.path.join(multiruns_to_merge[0], f))]

    alpha_vec = np.zeros(N_ENSEMBLE)
    beta_vec = np.zeros(N_ENSEMBLE)

    for k in range(N_ENSEMBLE):

        A_dir_run = multiruns_to_merge[0] + os.sep + A_run_folders[k]
        A_runinfo_dict = load_runinfo_from_rundir(A_dir_run)

        A_datagen_case = int(A_runinfo_dict['datagen_case'])
        A_nn_model_str = A_runinfo_dict['model']
        A_nn_alias = MODEL_CLASS_FROM_STR[A_nn_model_str]['alias']

        epoch_start = 0
        epoch_end = None  # none means final epoch

        #for idx, epoch_choice in enumerate([epoch_start, epoch_end]):
        # softmax assumed
        A_net_0 = load_model_from_rundir(A_dir_run, epoch_int=0)
        A_W_KQ_0 = A_net_0.W_KQ.detach().numpy()
        A_W_PV_0 = A_net_0.W_PV.detach().numpy()

        # linear assumed
        A_net_final = load_model_from_rundir(A_dir_run, epoch_int=None)
        A_W_KQ_final = A_net_final.W_KQ.detach().numpy()
        A_W_PV_final = A_net_final.W_PV.detach().numpy()

        plot_weights_comparison_pcolormesh(A_W_KQ_0, A_W_PV_0, A_W_KQ_final, A_W_PV_final, left_title=A_nn_model_str + 'e0',
                                right_title=A_nn_model_str + 'eFinal',
                                save_path=DIR_OUT + os.sep + 'replot_run%d_weights_pcolor_case%d' % (
                                k, datagen_case))

        alpha_vec[k] = np.mean(np.diag(A_W_PV_final))
        beta_vec[k] = np.mean(np.diag(A_W_KQ_final))

    plot_alpha_beta_scatter(alpha_vec, beta_vec, save_path=DIR_OUT + os.sep + 'replot_alpha_beta_scatter_case%d' % datagen_case)
