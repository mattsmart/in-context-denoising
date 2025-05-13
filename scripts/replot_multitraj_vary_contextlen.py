import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import matplotlib.pyplot as plt
import numpy as np
import pickle

from src.data_io import load_runinfo_from_rundir, reload_lossbaseline_dict
from src.nn_loss_baselines import theory_linear_expected_error
from src.nn_model_base import load_model_from_rundir
from src.settings import DIR_RUNS, DIR_OUT
from src.visualize import vis_weights_kq_pv, vis_weights_kq_pv_multirow


"""
Script for analysis of case 0: linear subspace task (with varying context length)
"""

def gamma_star(sigma2_pure_context, sigma2_corruption):
    return 1 / (sigma2_pure_context + sigma2_corruption)


dir_parent = DIR_RUNS
dir_multirun = dir_parent + os.sep + 'multitraj_ensemble_case0_TV1nrOL_adam_e100_Lvary'

COLOR_TRAIN      = '#4C72B0'
COLOR_TEST       = '#7BC475'
COLOR_PROJ       = '#BCBEC0'
COLOR_PROJSHRUNK = '#B56BAC'

# find all runs in folder
run_folders = [f for f in os.listdir(dir_multirun) if os.path.isdir(os.path.join(dir_multirun, f))]
npts = len(run_folders)

L_range = np.zeros(npts)
gamma_imputed = np.zeros(npts)
gamma_imputed_KQ = np.zeros(npts)
gamma_imputed_PV = np.zeros(npts)
gamma_imputed_offdiag_mean_abs_KQ = np.zeros(npts)
gamma_imputed_offdiag_mean_abs_PV = np.zeros(npts)
vals_loss_train = np.zeros(npts)
vals_loss_test = np.zeros(npts)
vals_loss_predict_zero = np.zeros(npts)
vals_loss_theory_linalg = np.zeros(npts)
vals_loss_theory_linalg_shrunken = np.zeros(npts)

# load the runinfo from the first run
dir_run = dir_multirun + os.sep + run_folders[0]
runinfo_dict = load_runinfo_from_rundir(dir_run)
dim_n = runinfo_dict['dim_n']
nn_model_str = runinfo_dict['model']
epochs = runinfo_dict['epochs']
subspace_dim_d = runinfo_dict['style_subspace_dimensions']
sigma2_pure_context = runinfo_dict['sigma2_pure_context']
sigma2_corruption = runinfo_dict['sigma2_corruption']
style_corruption_orthog = runinfo_dict['style_corruption_orthog']
style_origin_subspace = runinfo_dict['style_origin_subspace']
full_loss_sample_interval = runinfo_dict['full_loss_sample_interval']
optimizer_choice = runinfo_dict['optimizer_choice']
# this last arg should be varying across the runs
context_length_dummy = runinfo_dict['context_len']

# compute gamma_star from runinfo dict parameters
gamma_star_val = gamma_star(sigma2_pure_context, sigma2_corruption)


model_fname = runinfo_dict['fname']
nn_model = runinfo_dict['model']

runinfo_suffix = (nn_model + '\n' + r'dim $n = %d$, dim $d = %d$, $\sigma_{0}^2 = %.2f, \sigma_{z}^2 = %.2f$' %
                  (dim_n, subspace_dim_d, sigma2_pure_context, sigma2_corruption))
eq_expected_error_linalg_shrunken = theory_linear_expected_error(
    dim_n, subspace_dim_d, sigma2_corruption, sigma2_pure_context)

# Build the dict: datadict_curves_loss
datadict_curves_loss = {k: dict() for k in range(npts)}
for k in range(npts):
    print(run_folders[k])
    dir_run = dir_multirun + os.sep + run_folders[k]
    dir_replot = dir_run + os.sep + 'data_for_replot'
    dir_chkpts = dir_run + os.sep + 'model_checkpoints'

    with open(dir_replot + os.sep + 'loss_vals_dict.pkl', 'rb') as handle:
        loss_vals_dict = pickle.load(handle)
    datadict_curves_loss[k]['loss_vals_dict'] = loss_vals_dict

    lossbaseline_dict = reload_lossbaseline_dict(dir_replot)

    datadict_curves_loss[k]['vals_loss_predict_zero'] = lossbaseline_dict['dumb_A_mse_on_train']
    datadict_curves_loss[k]['vals_loss_theory_linalg'] = lossbaseline_dict['heuristic_mse_on_train']
    datadict_curves_loss[k]['vals_loss_theory_linalg_shrunken'] = lossbaseline_dict['heuristic_mse_shrunken_on_train']
    datadict_curves_loss[k]['eq_expected_error_linalg_shrunken'] = eq_expected_error_linalg_shrunken

    # addendum A: get context len for the run
    runinfo_dict = load_runinfo_from_rundir(dir_multirun + os.sep + run_folders[k])
    context_len_run_k = runinfo_dict['context_len']
    L_range[k] = context_len_run_k

    # extra info to store in data dict
    datadict_curves_loss[k]['L'] = context_len_run_k

    # addendum B: get imputed gamma from the weight matrix
    # get final weights
    load_net = load_model_from_rundir(dir_run, epoch_int=None)  # load final epoch weights
    final_W_KQ = load_net.W_KQ.detach().numpy()
    final_W_PV = load_net.W_PV.detach().numpy()
    gamma_imputed_val = np.mean(np.diag(final_W_KQ)) * np.mean(np.diag(final_W_PV))
    # now take mean of abs of off-diagonal elements
    W_KQ_absval_hollow = np.abs(np.copy(final_W_KQ))
    np.fill_diagonal(W_KQ_absval_hollow, 0)
    gamma_imputed_offdiag_mean_abs_val_KQ = np.sum(W_KQ_absval_hollow) / (dim_n ** 2 - dim_n)
    W_PV_absval_hollow = np.abs(np.copy(final_W_PV))
    np.fill_diagonal(W_PV_absval_hollow, 0)
    gamma_imputed_offdiag_mean_abs_val_PV = np.sum(W_PV_absval_hollow) / (dim_n ** 2 - dim_n)

    # fill in arrays (some are redundant with data stored in datadict_curves_loss[k] but its fine)
    gamma_imputed[k] = gamma_imputed_val
    gamma_imputed_KQ[k] = np.mean(np.abs(np.diag(final_W_KQ)))
    gamma_imputed_PV[k] = np.mean(np.abs(np.diag(final_W_PV)))
    gamma_imputed_offdiag_mean_abs_KQ[k] = gamma_imputed_offdiag_mean_abs_val_KQ
    gamma_imputed_offdiag_mean_abs_PV[k] = gamma_imputed_offdiag_mean_abs_val_PV
    vals_loss_train[k] = np.min(loss_vals_dict['loss_train_interval']['y'])
    vals_loss_test[k] = np.min(loss_vals_dict['loss_test_interval']['y'])
    vals_loss_predict_zero[k] = lossbaseline_dict['dumb_A_mse_on_train']
    vals_loss_theory_linalg[k] = lossbaseline_dict['heuristic_mse_on_train']
    vals_loss_theory_linalg_shrunken[k] = lossbaseline_dict['heuristic_mse_shrunken_on_train']


# now sort L_range, gamma_imputed, others
L_range_argsort = np.argsort(L_range)
L_range_sorted = L_range[L_range_argsort]
print('L_range', L_range)
print('L_range_argsort', L_range_argsort)
print('L_range_sorted', L_range_sorted)
# assert that its a strictly increasing list
assert np.all(np.diff(L_range_sorted) > 0)
# sort the remaining lists
gamma_imputed = gamma_imputed[L_range_argsort]
gamma_imputed_KQ = gamma_imputed_KQ[L_range_argsort]
gamma_imputed_PV = gamma_imputed_PV[L_range_argsort]
gamma_imputed_offdiag_mean_abs_KQ = gamma_imputed_offdiag_mean_abs_KQ[L_range_argsort]
gamma_imputed_offdiag_mean_abs_PV = gamma_imputed_offdiag_mean_abs_PV[L_range_argsort]
vals_loss_train = vals_loss_train[L_range_argsort]
vals_loss_test = vals_loss_test[L_range_argsort]
vals_loss_predict_zero = vals_loss_predict_zero[L_range_argsort]
vals_loss_theory_linalg = vals_loss_theory_linalg[L_range_argsort]
vals_loss_theory_linalg_shrunken = vals_loss_theory_linalg_shrunken[L_range_argsort]

# Step 1: convert all loss curves into arrays where one axis is ensemble
print('\nPre-plot - Compiling loss curves into shared arrays...')
example_loss_vals_dict = datadict_curves_loss[0]['loss_vals_dict']
loss_A_x = example_loss_vals_dict['loss_train_batch']['x']
loss_B_x = example_loss_vals_dict['loss_train_epoch_avg']['x']
loss_C_x = example_loss_vals_dict['loss_train_interval']['x']
loss_D_x = example_loss_vals_dict['loss_test_interval']['x']

loss_A_yarr = np.zeros((len(loss_A_x), npts))
loss_B_yarr = np.zeros((len(loss_B_x), npts))
loss_C_yarr = np.zeros((len(loss_C_x), npts))
loss_D_yarr = np.zeros((len(loss_D_x), npts))

for k in range(npts):
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


# Step 2: the main plot itself -- loos curves averaged showing spread over different runs
# Main plot
print('\nMain plot......')
plt.figure(figsize=(5, 4))
plt.plot(loss_C_x, loss_C_yarr, linestyle='--', alpha=0.4, color=COLOR_TRAIN)
plt.plot(loss_D_x, loss_D_yarr, linestyle='--', alpha=0.4, color=COLOR_TEST)

# plot mean +- standard deviation
fill_spread = False
if fill_spread:
    # alt fill option (min/max)
    plt.fill_between(loss_C_x, np.min(loss_C_yarr, axis=1), np.max(loss_C_yarr, axis=1), alpha=0.07, color=COLOR_TRAIN)
    plt.fill_between(loss_D_x, np.min(loss_D_yarr, axis=1), np.max(loss_D_yarr, axis=1), alpha=0.07, color=COLOR_TEST)

lw_baselines = 1.5
for idx in range(npts):
    subdict_loss = datadict_curves_loss[k]
    if idx == 0:
        plt.axhline(subdict_loss['vals_loss_predict_zero'], linestyle='--', alpha=0.75, linewidth=lw_baselines, label=r'predict $0$', color='grey')
        plt.axhline(subdict_loss['vals_loss_theory_linalg'], linestyle='-', alpha=0.75, linewidth=lw_baselines, label=r'$P \tilde x$', color=COLOR_PROJ)
        plt.axhline(subdict_loss['vals_loss_theory_linalg_shrunken'], linestyle='-', alpha=0.75, linewidth=lw_baselines, label=r'$\gamma P \tilde x$', color=COLOR_PROJSHRUNK)
        plt.axhline(subdict_loss['eq_expected_error_linalg_shrunken'], linestyle=':', alpha=0.75, linewidth=lw_baselines,
                    label=r'Expected error at $\theta^*$', color=COLOR_PROJSHRUNK)
    else:
        plt.axhline(subdict_loss['vals_loss_predict_zero'], linestyle='--', alpha=0.75, linewidth=lw_baselines, color='grey')
        plt.axhline(subdict_loss['vals_loss_theory_linalg'], linestyle='-', alpha=0.75, linewidth=lw_baselines, color=COLOR_PROJ)
        plt.axhline(subdict_loss['vals_loss_theory_linalg_shrunken'], linestyle='-', alpha=0.75, linewidth=lw_baselines, color=COLOR_PROJSHRUNK)
        plt.axhline(subdict_loss['eq_expected_error_linalg_shrunken'], linestyle=':', alpha=0.75, linewidth=lw_baselines, color=COLOR_PROJSHRUNK)

plt.xlabel(r'Epoch' + '\n\n' + runinfo_suffix)
plt.ylabel(r'$\frac{1}{n}$ MSE')
plt.title(r'Training curves for different context length')

#plt.grid(alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()

plt.ylim(0, 1.1)
plt.xlim(-2, 64)

plt.savefig(DIR_OUT + os.sep + model_fname + '_ensemble-spread_loss_dynamics.svg')
plt.savefig(DIR_OUT + os.sep + model_fname + '_ensemble-spread_loss_dynamics.png')
plt.show()

# Plot 1
# ================================
plt.figure(figsize=(6, 4))
plt.plot(L_range_sorted, gamma_imputed, '-ok', label='$\gamma$ (converged nn)')

flag_extra_curves = False
if flag_extra_curves:
    plt.plot(L_range_sorted, gamma_imputed_KQ, '--o', label=r'$W_{KQ}$ mean abs. diag')
    plt.plot(L_range_sorted, gamma_imputed_PV, '--o', label=r'$W_{PV}$ mean abs. diag')
    plt.plot(L_range_sorted, gamma_imputed_offdiag_mean_abs_KQ, '--o', label=r'$W_{KQ}$ mean abs. off-diag')
    plt.plot(L_range_sorted, gamma_imputed_offdiag_mean_abs_PV, '--o', label=r'$W_{PV}$ mean abs. off-diag')

plt.axhline(gamma_star_val, linestyle='-', label='$\gamma^*$', color='purple')
plt.xlabel(r'$L$ (# in context examples)' + '\n\n' + runinfo_suffix)
plt.ylabel(r'$\gamma = w_{PV} \times w_{KQ}$')
plt.title(r'Deviation from minimizer weights (nets trained using different $L$)'
          + '\n' + r'$\gamma^* = \frac{1}{\sigma_z^2 + \sigma_{0}^2}=%.2f$' % gamma_star_val)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(DIR_OUT + os.sep + model_fname + '_gamma_imputed.png')
plt.savefig(DIR_OUT + os.sep + model_fname + '_gamma_imputed.svg')
plt.show()

# Plot 2
# ================================
plt.figure(figsize=(6, 4))
plt.plot(L_range_sorted, vals_loss_train, '-o', label='nn (train)', color='k')
plt.plot(L_range_sorted, vals_loss_test, '--o', label='nn (test)', color='k')
plt.plot(L_range_sorted, vals_loss_theory_linalg, '--o', label='linalg')
plt.plot(L_range_sorted, vals_loss_theory_linalg_shrunken, '--o', label='linalg shrunk', color='purple')
plt.axhline(eq_expected_error_linalg_shrunken, linestyle='--', label='Eq: expected error', color='purple')
plt.xlabel(r'# in context examples (train/test)' + '\n\n' + runinfo_suffix)
plt.ylabel(r'$\frac{1}{n}$ MSE')
plt.title(r'Loss (new dataset each run)' + '\n' + r'$E[C(\theta^*)] = \frac{d}{n}\frac{\sigma_z^2 \sigma_{0}^2}{\sigma_z^2 + \sigma_{0}^2}=%.2f$' % eq_expected_error_linalg_shrunken)
plt.grid(alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# Plot 1 + 2 combined using subplots
# ================================
fig, axarr = plt.subplots(2, 1, sharex=True, squeeze=False, figsize=(4.5, 4.5))
ms = 4
lw = 1.25
ax1 = axarr[0,0]
ax0 = axarr[1,0]

# === Add fitted theory curves of the form: c + A / L^alpha ===
flag_show_scaling_fit_curves = True
if flag_show_scaling_fit_curves:
    L_fit = np.array(L_range_sorted)

    ### --- Top subplot: gamma ~ gamma* + A / L^alpha
    alpha_gamma = 0.5  # or try 0.5
    A_gamma = -0.2  # adjust to fit your data visually
    gamma_theory_curve = gamma_star_val + A_gamma / (L_fit ** alpha_gamma)
    ax1.plot(L_fit, gamma_theory_curve, '--', color='gray', linewidth=1.5,
             label=rf"$\gamma^* + \mathcal{{O}}(L^{{-{alpha_gamma:.1f}}})$")

    ### --- Bottom subplot: test loss ~ E + B / L^alpha
    alpha_loss = 0.5  # or 0.5
    B_loss = 0.5  # adjust visually
    loss_theory_curve = eq_expected_error_linalg_shrunken + B_loss / (L_fit ** alpha_loss)
    ax0.plot(L_fit, loss_theory_curve, '--', color='gray', linewidth=1.5,
             label=rf"$E + \mathcal{{O}}(L^{{-{alpha_loss:.1f}}})$")


ax1.plot(L_range_sorted, gamma_imputed, '-s', label=r'$\gamma$ (trained net)', color='black', alpha=0.9, markersize=ms, linewidth=lw, markerfacecolor='white')
# curves tracking off-diagonal element means vs. L
#ax1.plot(L_range_sorted, gamma_imputed_offdiag_mean_abs_KQ, '--^', label=r'$W_{KQ}$ mean abs. off-diag')
#ax1.plot(L_range_sorted, gamma_imputed_offdiag_mean_abs_PV, '--^', label=r'$W_{PV}$ mean abs. off-diag')

ax1.axhline(gamma_star_val,
                   label=r'$\gamma^*=\frac{1}{\sigma_z^2 + \sigma_{0}^2}$',
                   linestyle='--', color=COLOR_PROJSHRUNK, markersize=ms, linewidth=1.5*lw)
ax1.set_ylabel(r'$\gamma = w_{PV} \times w_{KQ}$')
#ax1.grid(alpha=0.3)
ax1.legend()

ax0.plot(L_range_sorted, vals_loss_train, '-o', label='train loss', color=COLOR_TRAIN, markersize=ms, linewidth=lw, zorder=11, markerfacecolor='white')
ax0.plot(L_range_sorted, vals_loss_test,  '-o', label='test loss',  color=COLOR_TEST, alpha=0.6,  markersize=ms, linewidth=lw, zorder=10, markerfacecolor='white')
ax0.plot(L_range_sorted, vals_loss_theory_linalg, '-o', label='linalg',
         color=COLOR_PROJ, markersize=ms, linewidth=1.5*lw)
#ax0.plot(L_range_sorted, vals_loss_theory_linalg_shrunken, '--o', label='linalg shrunk', color=COLOR_PROJSHRUNK, markersize=ms)
ax0.axhline(eq_expected_error_linalg_shrunken, linestyle=':', label='Eq: expected error',
            color=COLOR_PROJSHRUNK, markersize=ms, linewidth=1.5*lw)
ax0.legend()
ax0.set_ylabel(r'MSE')
ax0.set_ylim(0.299, 0.525)
#ax0.grid(alpha=0.3)

"""
flag_show_scaling_fit_curves = True
if flag_show_scaling_fit_curves:
    plt.plot(L_range_sorted, gamma_star(sigma2_pure_context, sigma2_corruption) * np.ones(npts), '--', label='$\gamma^*$', color='purple')
"""

plt.suptitle(r'Deviation from minimizer weights (nets trained using different $L$)'
          + '\n' + r'$\gamma^* = \frac{1}{\sigma_z^2 + \sigma_{0}^2}=%.2f$' % gamma_star_val)

axarr[1,0].set_xlabel(r'$L$ (num. in context examples)' + '\n\n' + runinfo_suffix)
plt.tight_layout()

plt.savefig(DIR_OUT + os.sep + model_fname + '_gamma_imputed_with_loss_subplots.png')
plt.savefig(DIR_OUT + os.sep + model_fname + '_gamma_imputed_with_loss_subplots.svg')
plt.show()

# Plot 1+2 combined on log log axis for convergence check
# =======================================
from scipy.stats import linregress

# Mask to ensure valid L and avoid log(0)
mask_valid = (L_range_sorted > 0) & (gamma_imputed < gamma_star_val)

L_log = L_range_sorted[mask_valid]
logL = np.log10(L_log)

gamma_diff = gamma_star_val - gamma_imputed[mask_valid]
log_gamma_diff = np.log10(gamma_diff)

mse_train_diff = vals_loss_train[mask_valid] - eq_expected_error_linalg_shrunken
mse_test_diff  = vals_loss_test[mask_valid]  - eq_expected_error_linalg_shrunken
log_mse_train_diff = np.log10(mse_train_diff)
log_mse_test_diff  = np.log10(mse_test_diff)

# --- Set up figure with two subplots ---
fig, axarr = plt.subplots(2, 1, sharex=True, figsize=(4.3, 4.6))
ax1 = axarr[0]
ax0 = axarr[1]

ms = 4
lw = 1.5

# === Top subplot: gamma convergence ===
ax1.plot(logL, log_gamma_diff, 'o-', label=r'$\gamma^* - \gamma$', color='black', markersize=ms, linewidth=lw)

# Fit line to gamma difference
slope1, intercept1, *_ = linregress(logL, log_gamma_diff)
fit1 = intercept1 + slope1 * logL
ax1.plot(logL, fit1, '--', color='gray', label=rf"slope ≈ {slope1:.2f}")

ax1.set_ylabel(r"$\log_{10}(\gamma^* - \gamma)$")
ax1.legend()
ax1.set_title("Log-log convergence")

# === Bottom subplot: MSE convergence ===
# Plot train
ax0.plot(logL, log_mse_train_diff, 'o-',color=COLOR_TRAIN, markersize=ms, linewidth=lw)
slope_train, intercept_train, *_ = linregress(logL, log_mse_train_diff)
fit_train = intercept_train + slope_train * logL
ax0.plot(logL, fit_train, '--', color=COLOR_TRAIN, alpha=0.7, label=rf"train slope ≈ {slope_train:.2f}")

# Plot test
ax0.plot(logL, log_mse_test_diff, 'o-', color=COLOR_TEST, markersize=ms, linewidth=lw)
slope_test, intercept_test, *_ = linregress(logL, log_mse_test_diff)
fit_test = intercept_test + slope_test * logL
ax0.plot(logL, fit_test, '--', color=COLOR_TEST, alpha=0.7, label=rf"test slope ≈ {slope_test:.2f}")

# === Theory curve for α=1.0 (i.e. log(1/L) = -logL) ===
theory_alpha = -1.0
offset = log_mse_test_diff[-1] - theory_alpha * logL[-1]  # anchor to last test point
theory_curve = theory_alpha * logL + offset
ax0.plot(logL, theory_curve, ':', color='black', markersize=2, label=r"$\propto 1/L$", zorder=5)

theory_alpha = -0.5
offset = log_mse_test_diff[-1] - theory_alpha * logL[-1]  # anchor to last test point
theory_curve = theory_alpha * logL + offset
ax0.plot(logL, theory_curve, '-.', color='purple', markersize=2, label=r"$\propto 1/\sqrt{L}$", zorder=5)


ax0.set_ylabel(r"$\log_{10}(\mathrm{MSE} - \mathrm{const.})$")
ax0.set_xlabel(r"$\log_{10}(L)$")
ax0.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(DIR_OUT + os.sep + model_fname + '_gamma_loss_loglog_convergence.png')
plt.savefig(DIR_OUT + os.sep + model_fname + '_gamma_loss_loglog_convergence.svg')
plt.show()

# Plot 3
# ================================
plt.figure(figsize=(6, 4))
for k in range(npts):
    subdict_curves = datadict_curves_loss[k]
    lval = subdict_curves['L']
    loss_vals_dict = subdict_curves['loss_vals_dict']
    ll, = plt.plot(loss_vals_dict['loss_train_interval']['x'], loss_vals_dict['loss_train_interval']['y'],
                   '-o', label='%d (train)' % lval, alpha=0.5)
    plt.plot(loss_vals_dict['loss_test_interval']['x'], loss_vals_dict['loss_test_interval']['y'],
             '--', c=ll.get_color(), label='%d (test)' % lval, markerfacecolor='none', alpha=0.5)

plt.axhline(eq_expected_error_linalg_shrunken, linestyle='--', label=r'Expected error at $\theta^*$', color='purple')
plt.xlabel(r'Epoch' + '\n\n' + runinfo_suffix)
plt.ylabel(r'$\frac{1}{n}$ MSE')
plt.title(r'Training curves for different $L$')
plt.grid(alpha=0.5)
plt.legend(ncol=4)
plt.tight_layout()

#plt.savefig(DIR_OUT + os.sep + model_fname + '_loss_dynamics.pdf')
#plt.savefig(DIR_OUT + os.sep + model_fname + '_loss_dynamics.png')
plt.show()


print('Plot init weights and final weights...')
print('------------------------')
for k in range(npts):
    epoch_start = 0
    epoch_end = None  # None means: get final epoch

    dir_run = dir_multirun + os.sep + run_folders[k]
    # get init weights
    net1 = load_model_from_rundir(dir_run, epoch_int=epoch_start)
    init_W_KQ = net1.W_KQ.detach().numpy()
    init_W_PV = net1.W_PV.detach().numpy()
    # get final weights
    net2 = load_model_from_rundir(dir_run, epoch_int=epoch_end)
    final_W_KQ = net2.W_KQ.detach().numpy()
    final_W_PV = net2.W_PV.detach().numpy()

    # visualize weight matrices using utility fn
    assert init_W_KQ.size != 1

    vis_weights_kq_pv_multirow(
        [init_W_KQ, final_W_KQ],
        [init_W_PV, final_W_PV],
        titlemod=r'$\theta$ final',
        dir_out=DIR_OUT, fname='replot_run%d_weights_final' % k, flag_show=True)

    vis_weights_kq_pv(init_W_KQ, init_W_PV, titlemod=r'$\theta$ init',
                      dir_out=DIR_OUT, fname='replot_run%d_weights_init' % k, flag_show=True)
    vis_weights_kq_pv(final_W_KQ, final_W_PV, titlemod=r'$\theta$ final',
                      dir_out=DIR_OUT, fname='replot_run%d_weights_final' % k, flag_show=True)
