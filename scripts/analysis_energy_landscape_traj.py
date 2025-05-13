import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from src.settings import DIR_OUT

"""
Example: energy landscape for circle denoising

- circle of radius R in R^2 implied by L context points X = {x_1, ..., x_L} sampled evenly or randomly
- query point q = [q_1, q_2] in R^2
- energy function 

E[X,q] = 0.5 * lambda * ||q||^2 - (1 / beta) * logsumexp(beta * c_k * X.T q)
 
where:
- lambda is a regularization parameter
- beta is a scaling parameter
- k(x_i, q) = x_i^T q
- logsumexp is the log-sum-exp function

Gradient descent:
- update q_t+1 = q_t - gamma * grad_
- grad_E[X,q] = lambda * q - X * softmax(beta * k(x_i, q))
where:
- gamma is the step size
- softmax(z) = exp(z) / sum(exp(z))
- softmax(beta * c_k * X.T q) = [exp(beta * k(x_1, q)), ..., exp(beta * k(x_L, q))] / sum(exp(beta * k(x_i, q)))

Trajectories:
- M different starting points q_0
- for t = 1, ..., T:
    - q_t+1 = q_t - gamma * grad_
- plot trajectories and energy landscape

Energy landscape:
- grid of points q = [q_1, q_2] in R^2
- compute energy E[X,q] for each point q
- plot contours of
    - E[X,q] = 0.5 * lambda * ||q||^2 - (1 / beta) * logsumexp(beta * k(x_i, q))
    - for q in grid
- plot trajectories q_t for t = 0, ..., T
- plot context points X
- plot start and end points of trajectories

Parameters:
- R: radius of the circle
- L: number of context points on the circle
- c_lambda: lambda parameter
- beta: beta parameter
- c_k: k parameter
- c_v: v parameter
- noise_magnitude: standard deviation of noise for the query point
- gamma: step size
- num_steps: number of gradient descent steps
- M: number of different trajectories
"""


# ----- Base Parameters -----
R = 1.0  # radius of the circle
L = 20 #30  # number of context points on the circle
sigma2_noise = 0.1  # noise std for query point
beta = 1.0  # beta parameter (fix to 1.0 for now...)
c_k = 1/sigma2_noise   # c_k parameter
c_v = 1.0              # c_v parameter (influences step size and update)

gamma_step = 1.0 * c_v  # gradient descent step size set to c_v
num_steps = 50     # number of gradient descent steps
M = 3              # number of different trajectories
grid_size = 100    # resolution for energy contour grid

# Parameters for lambda in rows
c_lambda = 1 / c_v

row_num_steps = [1, 50]

# Settings for context randomness in columns
col_flags_random_context = [False, True]

seed_base = 42  # 42, 45 (best), 63 (best)
np.random.seed(seed_base + 0)
angles_equidistant = np.linspace(0, 2 * np.pi, L, endpoint=False)
angles_random      = np.random.rand(L) * 2 * np.pi


# Define utility functions
def logsumexp(z):
    z_max = np.max(z)
    return z_max + np.log(np.sum(np.exp(z - z_max)))

def energy(q, X, c_lambda, beta, c_k):
    term1 = 0.5 * c_lambda * np.sum(q ** 2)
    scores = beta * c_k * (X.T @ q)
    term2 = (1 / (beta * c_k)) * logsumexp(scores)
    return term1 - term2

def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def trained_network_update(q_current, X, alpha, beta):
    q_out = alpha * X @ softmax(beta * X.T @ q_current)
    return q_out

def grad_energy(q, X, c_lambda, beta, c_k):
    scores = beta * c_k * (X.T @ q)
    a = softmax(scores)
    return c_lambda * q - X @ a

# Set up subplots: 2 rows x 2 columns
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# We'll store one contour set for the global colorbar.
global_contour = None

# Loop over rows and columns to create each subplot
for i, num_steps in enumerate(row_num_steps):  # rows: different lambda values
    for j, flag_context_random in enumerate(col_flags_random_context):  # columns: context randomness
        ax = axes[i, j]

        # ----- Sample Points on Circle for Context X -----
        if flag_context_random:
            angles = angles_random
        else:
            angles = angles_equidistant
        X = np.column_stack((R * np.cos(angles), R * np.sin(angles))).T
        print('X.shape', X.shape)

        # ----- Compute Energy Contour for current settings -----
        x_vals = np.linspace(-R * 1.5, R * 1.5, grid_size)
        y_vals = np.linspace(-R * 1.5, R * 1.5, grid_size)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X_grid)

        for m in range(grid_size):
            for n in range(grid_size):
                q_point = np.array([X_grid[m, n], Y_grid[m, n]])
                Z[m, n] = energy(q_point, X, c_lambda, beta, c_k)

        # ----- Plot Energy Contours -----
        contour = ax.contourf(X_grid, Y_grid, Z, levels=50, cmap='Spectral_r')
        # Store one contour for colorbar reference
        if global_contour is None:
            global_contour = contour
        ax.scatter(X[0, :], X[1, :], color='white', edgecolor='k', label=r'Context tokens $X$')

        # ----- Initialize and Plot Multiple Trajectories -----
        colors = ['red', 'cyan', 'magenta']
        for m in range(M):
            # Sample different starting point
            np.random.seed(seed_base * (m+1))
            angle = np.random.uniform(0, 2 * np.pi)
            true_query = np.array([R * np.cos(angle), R * np.sin(angle)])
            q_init = true_query + np.random.normal(scale=np.sqrt(sigma2_noise), size=2)

            traj = np.zeros((num_steps + 1, 2))
            traj[0] = q_init
            q_current = q_init.copy()

            for t in range(num_steps):
                grad = grad_energy(q_current, X, c_lambda, beta, c_k)
                q_current = q_current - gamma_step * grad
                traj[t + 1] = q_current

            ax.scatter(true_query[0], true_query[1], color=colors[m], marker='s', edgecolor='k', s=100)
            #ax.scatter(true_query[0], true_query[1], color='white', marker='s', edgecolor='k', s=150)

            ax.plot(traj[:, 0], traj[:, 1], marker='o', color=colors[m], label=f'Traj {m + 1}')
            ax.scatter(traj[0, 0], traj[0, 1], color=colors[m], marker='o', edgecolor='k', s=100)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[m], marker='*', s=150, edgecolor='k', zorder=10)

        # ----- Titles and Labels -----
        title = f"Num steps: {num_steps:d}, "
        title += "Random context tokens" if flag_context_random else "Equidistant context tokens"
        ax.set_title(title)
        if i == 1:
            ax.set_xlabel(r'$q_1$')
        if j == 0:
            ax.set_ylabel(r'$q_2$')
        if i == 0 and j == 0:
            ax.legend(fontsize='small')

# Create a single colorbar for all subplots using the stored contour object
energy_eq = r"$E(X,q) = \frac{1}{2 \alpha} \left\|q\right\|^2 - \frac{1}{\beta} \log \sum_{i=1}^{L} \exp(\beta x_i^T q)$"
fig.colorbar(global_contour, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.1, label='Energy: ' + energy_eq)

plt.suptitle('Energy landscape and denoising trajectories', fontsize=16)

annotation_text = r"Parameters: $R=%.1f$, $L=%d$, $\beta=%.2f$, $c_k=%.2f$, $c_v=%.2f$, $\sigma_{\eta}^2=%.2f$" % (R, L, beta, c_k, c_v, sigma2_noise)
fig.text(0.5, 0.94, annotation_text, ha='center', va='top', fontsize=10)

plt.show()


# Set up subplots: 2 rows x 2 columns
fig, axes = plt.subplots(1, 2, figsize=(10, 5), squeeze=False)
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# We'll store one contour set for the global colorbar.
global_contour = None

# Loop over rows and columns to create each subplot
i = 0
flag_context_random = True
for j, num_steps in enumerate(row_num_steps):  # columns: context randomness
    ax = axes[i, j]

    # ----- Sample Points on Circle for Context X -----
    if flag_context_random:
        angles = angles_random
    else:
        angles = angles_equidistant
    X = np.column_stack((R * np.cos(angles), R * np.sin(angles))).T
    print('X.shape', X.shape)

    # ----- Compute Energy Contour for current settings -----
    x_vals = np.linspace(-R * 1.5, R * 1.5, grid_size)
    y_vals = np.linspace(-R * 1.5, R * 1.5, grid_size)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X_grid)

    for m in range(grid_size):
        for n in range(grid_size):
            q_point = np.array([X_grid[m, n], Y_grid[m, n]])
            Z[m, n] = energy(q_point, X, c_lambda, beta, c_k)

    # ----- Plot Energy Contours -----
    contour = ax.contourf(X_grid, Y_grid, Z, levels=50, cmap='Spectral_r')
    # Store one contour for colorbar reference
    if global_contour is None:
        global_contour = contour
    ax.scatter(X[0, :], X[1, :], color='white', edgecolor='k', label=r'Context tokens $X$')

    # ----- Initialize and Plot Multiple Trajectories -----
    colors = ['red', 'cyan', 'magenta']
    for m in range(M):
        # Sample different starting point
        np.random.seed(seed_base * (m+1))
        angle = np.random.uniform(0, 2 * np.pi)
        true_query = np.array([R * np.cos(angle), R * np.sin(angle)])
        q_init = true_query + np.random.normal(scale=np.sqrt(sigma2_noise), size=2)

        traj = np.zeros((num_steps + 1, 2))
        traj[0] = q_init
        q_current = q_init.copy()


        for t in range(num_steps):
            grad = grad_energy(q_current, X, c_lambda, beta, c_k)
            q_current = q_current - gamma_step * grad
            traj[t + 1] = q_current

        ax.scatter(true_query[0], true_query[1], color=colors[m], marker='s', edgecolor='k', s=100)
        #ax.scatter(true_query[0], true_query[1], color='white', marker='s', edgecolor='k', s=150)

        ax.plot(traj[:, 0], traj[:, 1], marker='o', color=colors[m], label=f'Traj {m + 1}')
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[m], marker='o', edgecolor='k', s=100)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[m], marker='*', s=150, edgecolor='k', zorder=10)


    # ----- Titles and Labels -----
    title = f"Num steps: {num_steps:d}, "
    #title += "Random context tokens" if flag_context_random else "Equidistant context tokens"
    ax.set_title(title)
    if i == 0:
        ax.set_xlabel(r'$q_1$')
    if j == 0:
        ax.set_ylabel(r'$q_2$')
    else:
        # remove yticklabels
        ax.set_yticklabels([])
    if i == 0 and j == 1:
        ax.legend(fontsize='small')

# Create a single colorbar for all subplots using the stored contour object
energy_eq = r"$E(X,q) = \frac{1}{2 \alpha} \left\|q\right\|^2 - \frac{1}{\beta} \log \sum_{i=1}^{L} \exp(\beta x_i^T q)$"
fig.colorbar(global_contour, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.1, label='Energy: ' + energy_eq)

plt.suptitle('Energy landscape and denoising trajectories', fontsize=16)

annotation_text = r"Parameters: $R=%.1f$, $L=%d$, $\beta=%.2f$, $c_k=%.2f$, $c_v=%.2f$, $\sigma_{\eta}^2=%.2f$" % (R, L, beta, c_k, c_v, sigma2_noise)
fig.text(0.5, 0.94, annotation_text, ha='center', va='top', fontsize=10)

#plt.tight_layout()
plt.savefig(DIR_OUT + os.sep + 'energy_landscape_traj_v2.png')
plt.savefig(DIR_OUT + os.sep + 'energy_landscape_traj_v2.svg')
plt.show()
