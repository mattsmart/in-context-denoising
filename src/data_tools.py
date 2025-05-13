import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from scipy.special import iv, ive
from scipy.special import softmax as scipy_softmax
from scipy.stats import special_ortho_group

from torch.utils.data import Dataset, DataLoader

from data_io import load_dataset
from settings import *


def gen_uniform_data(size, rng=None):
    """
    Return random np.array ~ *size* with U[-1,1] elements
    - note $U[-1,1]$ has variance $1/12 (b-a)^2$ which gives $1/3$
    - scaling the arr by C increases the variance by C^2 (and thus the std.dev by C)
    """
    rng = rng or np.random.default_rng()  # Use provided RNG or create one
    arr = 2 * rng.random(size=size) - 1
    return arr


def gen_gaussian_data(mean, cov, size, rng=None):
    """
    Return random np.array ~ *size* with N(mu, cov) elements
    - if size is an integer, then returned arr is {size x dim n}
    """
    rng = rng or np.random.default_rng()
    arr = rng.multivariate_normal(mean, cov, size=size, check_valid='raise')  # size x dim n
    return arr


def generate_affine_subspace(dim_n, dim_d, rng=None):
    # from setup above:
    # - vectors live in R^n but we are given d < n of them to define an affine subspace W of R^n
    rng = rng or np.random.default_rng()
    # select offsets
    rand_m = gen_uniform_data(dim_n, rng=rng)
    # select basis (orthonormal)
    rand_V = gen_uniform_data((dim_n, dim_d), rng=rng)
    if dim_d > 1:
        rand_V, _ = np.linalg.qr(rand_V)
    else:
        rand_V = rand_V / np.linalg.norm(rand_V)
    return rand_m, rand_V


# function to sample L vectors from W = {x in R^n | x = m + V c} for some c in R^d
def sample_from_subspace(W_m, W_V, nsample, sigma_Z_sqr=1.0, uniform=True, rng=None):
    """
    Args
    - W_m - array-like - n x k
    - W_V - array-like - n x k

    For each sample
    - sample c -- U[-1, 1] vector of random coefficients in R^d (for basis W_V)
    - then x = W_m + W_V @ c is a random vector from the subspace
    Return:
        np.array of size (n x nsample) where n is the dimension of the ambient space (n >= dim W == W_V.shape[1])
    """
    rng = rng or np.random.default_rng()
    dim_n, dim_W = W_V.shape
    if uniform:
        c = gen_uniform_data(
            (dim_W, nsample), rng=rng)  # random coefficients for the basis {v_i}
        # note by default, c will have a variance of 1/3 (U[-1,1]); we can scale it by
        c = np.sqrt(3 * sigma_Z_sqr) * c  # e.g. if goal is variance of 10, then scale sqrt(3 * 10)
        rand_samples = np.expand_dims(W_m, axis=1) + W_V @ c
    else:
        # gaussian ball method
        x_blob = rng.multivariate_normal(W_m, sigma_Z_sqr * np.eye(dim_n), nsample)
        projector_exact = W_V @ np.linalg.inv(W_V.T @ W_V) @ W_V.T
        projection_b_onto_W = np.expand_dims(W_m, axis=1) + projector_exact @ (x_blob.T - np.expand_dims(W_m, axis=1))
        rand_samples = projection_b_onto_W
    return rand_samples


# function to sample L vectors from a list of K guassians in R^d
def sample_from_gmm_isotropic(arr_mu_n_by_k, arr_sigma_isotropic, nsample, rng=None):
    """
    Args
    - arr_mu_k_by_n - array - dim_n x num_cluster
    - arr_sigma_isotropic - array - dim_n x dim_n - variance of the isotropic gaussian

    Generate nsample samples from a GMM with K components

    Return:
        np.array of size (n x nsample) where n is the dimension of the ambient space (n >= dim W == W_V.shape[1])
    """
    rng = rng or np.random.default_rng()

    dim_n = arr_sigma_isotropic.shape[0]
    assert arr_mu_n_by_k.shape[0] == dim_n
    num_cluster = arr_mu_n_by_k.shape[1]

    # sample centroid indices with replacement
    indices = rng.choice(range(num_cluster), nsample, replace=True)  # list of indices i in {0, ..., k-1}
    # sample corresponding centers
    X = arr_mu_n_by_k[:, indices]                                                          # dim_n x nsample
    # generate noise around centroids and add to the centroids
    Z = rng.multivariate_normal(np.zeros(dim_n), arr_sigma_isotropic, size=nsample)  # nsample x dim_n
    rand_samples = X + Z.T                                                           # dim_n x nsample

    return rand_samples, indices


def sample_from_subsphere(dim_sphere, sphere_radius, arr_rotation, context_len, rng=None):
    """
    Gaussian ball in R^{dim_sphere + 1} clamped to sphere, padded to R^{dim_n} and rotated by arr_rotation

    Returns arr of shape (dim_n, context_len)
    """
    rng = rng or np.random.default_rng()
    dim_m = dim_sphere + 1
    dim_n = arr_rotation.shape[0]
    # Get random sample from a lower dimension (dim_d + 1) then project to sphere and pad to R^n + randomly rotate
    samples_in_lower_dim = rng.normal(0, 1, size=[dim_m, context_len])  # dim_m x L
    # alt:
    """
    samples_in_lower_dim = gen_gaussian_data(np.zeros(dim_m), np.eye(dim_m), context_len, seed=seed1)  # L x dim_m
    samples_in_lower_dim = np.transpose(samples_in_lower_dim)  # dim_m x L
    """
    # Perform L2 normalization
    l2_norm = np.linalg.norm(samples_in_lower_dim, 2, axis=0, keepdims=True)
    samples_on_sphere_in_lowdim = sphere_radius * samples_in_lower_dim / l2_norm
    # Pad to desired dimension
    samples_on_sphere = np.pad(samples_on_sphere_in_lowdim, ((0, dim_n - dim_m), (0, 0)), 'constant')
    # Rotate lower dimensional sphere in the larger dimension
    Y = arr_rotation @ samples_on_sphere  # dim_n x context_len
    return Y


# function to project onto d-dim affine subspace W(m, V) of R^n
def proj_affine_subspace(W_m, W_V, x):
    """
    Projects a vector x in R^n onto the d-dim affine subspace W specified by W_m (n x 1) and W_V (n x d)
    """
    assert x.shape == W_m.shape
    projector = W_V @ np.linalg.inv(W_V.T @ W_V) @ W_V.T
    return W_m + projector @ (x - W_m)


# function to report euclidean distance of a trial vector to the subspace
def get_dist_to_subspace(W_m, W_V, x):
    """
    - Affine subspace W = {x in R^n | x = m + V c} for some c in R^d
        - m in R^n
        - V in R^{n x d} is an orthonormal basis

    A vector lies in the affine subspace if its orthogonal projection onto it is itself.
    """
    proj_x = proj_affine_subspace(W_m, W_V, x)
    return np.linalg.norm(x - proj_x)


def subspace_offset_and_basis_estimator(X_seq, eval_cutoff=1e-4, verbose_vis=False):
    """
    See proj_affine_subspace_estimator
        - functionalized the first part of that original function
    """
    ndim, ncontext = X_seq.shape
    n_samples = ncontext  # alias

    mu = np.expand_dims(
        np.mean(X_seq, axis=1),
        axis=1)

    # Manual PCA
    A = (X_seq - mu) @ (X_seq - mu).T / n_samples

    eig_D, eig_V = np.linalg.eig(A)
    sorted_indexes = np.argsort(eig_D)
    eig_D = eig_D[sorted_indexes]
    eig_V = eig_V[:, sorted_indexes]

    indices_nonzero_evals = np.where(eig_D > eval_cutoff)[0]
    nonzero_evecs = eig_V[:, indices_nonzero_evals]
    nonzero_evals = eig_D[indices_nonzero_evals]

    # Showcase the de-corruption (by estimating offset and basis from the sequence)
    estimate_offset = mu
    estimate_basis = np.real(nonzero_evecs)
    estimate_evals = np.real(nonzero_evals)

    if verbose_vis:
        print(estimate_offset.shape, estimate_basis.shape)
        print(len(indices_nonzero_evals), 'vs', eig_D.shape)
        print('...')
        plt.figure()
        plt.plot(eig_D, '-ok', markersize=3)
        plt.plot(indices_nonzero_evals, eig_D[indices_nonzero_evals], '-or', markersize=3)
        plt.title('eigenvalues from PCA on X_seq (num above = %d)' % len(indices_nonzero_evals))
        plt.xlabel(r'rank $i$'); plt.ylabel(r'$\lambda_i$')
        plt.axhline(eval_cutoff, color='k', linestyle='--')
        plt.show()

    return estimate_offset, estimate_basis, estimate_evals


def proj_affine_subspace_estimator(X_seq, x_corrupt, eval_cutoff=1e-4, verbose_vis=False, return_basis=False):
    """
    See transformer_A.ipynb

    Args:
        X_seq     - 2-d sequence X_seq of shape ndim, ncontext - 1
        x_corrupt - 1-d arr of shape ndim

    Note:
        together X_seq and x_corrupt compose the input sequence to the NN (x_corrupt is suffix, goal is to decorrupt)
    """
    # step 1
    est_offset, est_basis, _ = subspace_offset_and_basis_estimator(X_seq, eval_cutoff=eval_cutoff, verbose_vis=verbose_vis)
    # step 2
    projection_b_onto_W = proj_affine_subspace(est_offset, est_basis, x_corrupt)

    if verbose_vis:
        print(est_offset.shape, est_basis.shape, x_corrupt.shape)
        print(projection_b_onto_W.shape)

    if return_basis:
        return projection_b_onto_W, est_basis
    else:
        return projection_b_onto_W


def modified_bessel_firstkind_scipy(n, z):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.iv.html#scipy.special.iv
    n is the order; must be integer
    z is the argument; must be real

    In case of overflow (large z), try https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html#scipy.special.ive
    "exponentially scaled modified Bessel function of the first kind"
    """
    return iv(n, z)


def modified_bessel_firstkind_scipy_expscale(n, z):
    """
    In case of overflow (large z), try https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ive.html#scipy.special.ive
    "exponentially scaled modified Bessel function of the first kind"

    ive(n, z) = iv(n, z) * exp(-abs(z.real))
    """
    return ive(n, z)


def bessel_ratio_np1_over_n(n, z):
    """
    use scipy ive for numeric stability; note the exp scalings cancel out
    """
    return modified_bessel_firstkind_scipy_expscale(n+1, z) / modified_bessel_firstkind_scipy_expscale(n, z)


def bessel_ratio_subhalf_sub3half(n, z):
    """
    use scipy ive for numeric stability; note the exp scalings cancel out
    - n = 2 case corresponds to sphere in R3, so I_{3/2} / I_{ 1/2}
    - n = 1 case corresponds to circle in R2, so I_{1/2} / I_{-1/2}
    """
    return modified_bessel_firstkind_scipy_expscale(n - 0.5, z) / modified_bessel_firstkind_scipy_expscale(n - 1.5, z)


def proj_subsphere_estimator(X_seq, x_corrupt, sig2_corrupt, eval_cutoff=1e-6, verbose_vis=False, shrunken=False,
                             style_origin_subspace=True, style_corruption_orthog=False):
    """
    Baseline predictor for the subsphere case (Manifold case)
    """
    assert style_origin_subspace        # we assume proper subspace through origin
    assert not style_corruption_orthog  # we assume it is iid gaussian ball corruption (not orthogonal to W)

    # 1) estimate subspace projection
    # ===================================================================================
    x_corrupt = np.expand_dims(x_corrupt, axis=1)  # this is needed for linalg shape inference in fn
    # TODO torch version? speedup how?
    project_xL_onto_W_np, est_basis = proj_affine_subspace_estimator(X_seq, x_corrupt,
                                                                     return_basis=True, eval_cutoff=eval_cutoff)
    project_xL_onto_W_np = np.squeeze(project_xL_onto_W_np)

    # 2) estimate sphere radius from the sequence
    # ===================================================================================
    #mu = np.expand_dims(np.mean(X_seq, axis=1), axis=1)
    #radius_est = np.mean(np.linalg.norm(X_seq[:, 0] - mu, axis=0))
    radius_est = np.mean(np.linalg.norm(X_seq[:, 0], axis=0))  # for now, we assert style_origin_subspace

    # 3) get norm of x_L_corrupt
    # ===================================================================================
    x_L_corrupt_projected_norm = np.linalg.norm(project_xL_onto_W_np)  # TODO prob subtract mu first?

    # 4) estimate sphere dimension from the sequence
    # ===================================================================================
    print(est_basis.shape)
    d_dim_infer = est_basis.shape[0]  # for a circle in 2d, this should be "2"; for the bessel ratio fn we then pass d-2

    # 5) perform shrink step (should be doing inside proj_affine_subspace_estimator but this is fine with asserts above)
    # ===================================================================================
    beta_val = radius_est * x_L_corrupt_projected_norm / sig2_corrupt
    shrink_factor = bessel_ratio_np1_over_n(d_dim_infer - 2, beta_val)

    prediction_on_subsphere = radius_est * project_xL_onto_W_np / x_L_corrupt_projected_norm

    if shrunken:
        baseline_vec = shrink_factor * prediction_on_subsphere
    else:
        baseline_vec = prediction_on_subsphere

    if verbose_vis:
        print('radius_est', radius_est)
        print('d_dim_infer', d_dim_infer)
        print('shrink_factor', shrink_factor)
        print(est_basis.shape, x_corrupt.shape)
        print(baseline_vec.shape)

    return baseline_vec


def proj_clustering_estimator(X_seq, x_corrupt, sig2_corrupt, gmm_mu_matrix, gmm_cluster_ids, gmm_sig2_0,
                              verbose_vis=False,
                              force_zero_cluster_var=False,
                              style_origin_subspace=True, style_corruption_orthog=False):
    """
    Baseline predictor for the GMM clustering case

    See loss_if_predict_clustering_baseline() in nn_loss_baselines.py
    """
    assert style_origin_subspace        # we assume proper subspace through origin
    assert not style_corruption_orthog  # we assume it is iid gaussian ball corruption (not orthogonal to W)

    num_cluster = gmm_mu_matrix.shape[1]

    # 1) calculate scalars (fixed for dataset)
    a_res      =   gmm_sig2_0 / (gmm_sig2_0 + sig2_corrupt)
    a_cluster  = sig2_corrupt / (gmm_sig2_0 + sig2_corrupt)
    beta_scale = 1 / (gmm_sig2_0 + sig2_corrupt)

    # simpler baseline for plotting/analysis
    if force_zero_cluster_var:
        a_res = 0.0
        a_cluster = 1.0
        beta_scale = 1 / sig2_corrupt

    # 2) extract subspace/submanifold/cluster parameters for this example
    # ===================================================================================

    # from the indices, infer the sampled cluster weights
    # (ALTERNATIVELY: could assume 1/p for all weights)
    gmm_w_vec = np.zeros(num_cluster)
    # TODO this caveman loop is probably not efficient
    for k in range(num_cluster):
        gmm_w_vec[k] = np.sum(gmm_cluster_ids == k)
    gmm_w_vec = gmm_w_vec / np.sum(gmm_w_vec)
    print('gmm_w_vec', gmm_w_vec)  # asymptotically, should be true cluster weights (1/p, 1/p, ..., 1/p)
    # ===================================================================================

    # 3) compute residual term (this goes to zero when cluster var goes to zero)
    x_term_residual = a_res * x_corrupt

    # 4) compute cluster term - can do this as a sum or use softmax (check both...) # TODO
    # 4) - v1 - summation method
    '''
    x_term_cluster = np.zeros(ndim)
    denominator = 0
    for k in range(num_cluster):
        x_term_cluster[:] += gmm_mu_matrix[:, k] * gmm_w_vec[k] * np.exp(beta_scale * np.dot(gmm_mu_matrix[:, k].T, x_L_corrupt))
        denominator += np.exp(np.dot(beta_scale * gmm_mu_matrix[:, k].T, x_L_corrupt)) * gmm_w_vec[k]
    x_term_cluster = a_cluster * x_term_cluster / denominator'''
    #print('value of denominator', denominator)
    # 4) - v2 - softmax method (same numeric value, maybe slightly faster and more stable)
    x_term_cluster = a_cluster * gmm_mu_matrix @ scipy_softmax(beta_scale * gmm_mu_matrix.T @ x_corrupt + np.log(gmm_w_vec))

    # 5) combine
    x_predict = x_term_residual + x_term_cluster

    if verbose_vis:
        print(x_predict.shape)

    return x_predict


def loss_if_predict_subsphere_baseline(criterion, dataloader, data_label, sphere_radius, sig2_corrupt,
                                       plot_some=True, print_val=True, shrunken=True):
    # TODO replace by DummyModel variant class (for baselines PredictMostRecent, PredictAverage, PredictZero, etc)
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            nbatch, ndim, ncontext = samples.size()
            for b in range(nbatch):
                X_seq =       samples[b, :, :-1].numpy()
                x_L_corrupt = samples[b, :,  -1].numpy()

                # 1) estimate subspace projection
                # ===================================================================================
                x_L_corrupt = np.expand_dims(x_L_corrupt, axis=1)  # this is needed for linalg shape inference in fn
                project_xL_onto_W_np, est_basis = proj_affine_subspace_estimator(X_seq, x_L_corrupt, return_basis=True)  # TODO torch version? speedup how?
                project_xL_onto_W_np = np.squeeze(project_xL_onto_W_np)

                # 1) get norm of x_L_corrupt
                # ===================================================================================
                x_L_corrupt_projected_norm = np.linalg.norm(project_xL_onto_W_np)

                # 2) estimate sphere radius from the sequence
                # ===================================================================================
                mu = np.expand_dims(np.mean(X_seq, axis=1), axis=1)
                radius_est = np.mean(np.linalg.norm(X_seq[:, 0] - mu, axis=0))

                # 3) estimate sphere dimension from the sequence
                # ===================================================================================
                print(est_basis.shape)
                d_dim_infer = est_basis.shape[0]

                # 4) perform shrink step (should be doing inside proj_affine_subspace_estimator but this is fine with asserts above)
                # ===================================================================================
                beta_val      = radius_est * x_L_corrupt_projected_norm / sig2_corrupt
                shrink_factor = bessel_ratio_np1_over_n(d_dim_infer, beta_val)
                print('radius_est',    radius_est)
                print('d_dim_infer',   d_dim_infer)
                print('shrink_factor', shrink_factor)
                prediction_on_subsphere = radius_est * project_xL_onto_W_np / x_L_corrupt_projected_norm

                if shrunken:
                    baseline_vec = shrink_factor * prediction_on_subsphere
                else:
                    baseline_vec = prediction_on_subsphere

                baseline_vec = torch.from_numpy(baseline_vec)

    return baseline_vec


def data_train_test_split_util(x_total, y_total, data_subspace_dict, context_len, dim_n, num_W_in_dataset,
                               context_examples_per_W, test_ratio, as_torch=True,
                               savez_fname=None, verbose=True, rng=None):
    """
    Used commonly by data_train_test_split_* functions - for * = {linear, clusters, manifold}

    TODO make it so test ratio 1.0 or 0.0 makes the corresponding array empty (as opposed to nasty if/else currently)
    """
    if verbose:
        print('Generating train/test data for NN training (context length = %d)...' % context_len)
        print('\tcontext_len=%d, dim_n=%d, num_W_in_dataset=%d, examples_per_W=%d, test_ratio=%s' %
              (context_len, dim_n, num_W_in_dataset, context_examples_per_W, test_ratio))
        print('\tnsample_per_subspace (context_len):', context_len)
        print('Total dataset size before split: %d (x,y) pairs' % len(y_total))

    x_total = np.array(x_total).astype(np.float32)
    y_total = np.array(y_total).astype(np.float32)

    rng = rng or np.random.default_rng()  # determines how dataset is split; if no rng passed, create one

    # now perform train test split and randomize
    ntotal = len(y_total)
    if test_ratio is None:
        x_test = None
        y_test = None
        test_data_subspaces = None

        ntrain = ntotal
        train_indices_to_shuffle = [i for i in range(ntotal)]
        train_indices = rng.choice(train_indices_to_shuffle, ntrain, replace=False)

        # grab train data
        x_train = x_total[train_indices, :, :]
        y_train = y_total[train_indices, :]
        # rebuild metadata dicts after shuffling
        train_data_subspaces = dict()
        for idx, val in enumerate(train_indices):
            train_data_subspaces[idx] = data_subspace_dict[val].copy()

        if verbose:
            print('\t x_train:', x_train.shape)
            print('\t y_train:', y_train.shape)
            print('\t x_test: None')
            print('\t y_test: None')

        if savez_fname is not None:
            assert savez_fname[-4:] == '.npz'
            # save dataset to file
            dataset_fpath = savez_fname
            print('\nSaving dataset to file...', dataset_fpath)
            np.savez_compressed(dataset_fpath,
                                x_train=x_train,
                                y_train=y_train,
                                x_test=np.empty(1),
                                y_test=np.empty(1))

        if as_torch:
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)

    else:
        ntest = int(test_ratio * ntotal)
        ntrain = ntotal - ntest

        test_indices = rng.choice(ntotal, ntest, replace=False)
        train_indices_to_shuffle = [i for i in range(ntotal) if i not in test_indices]
        train_indices = rng.choice(train_indices_to_shuffle, ntrain, replace=False)

        # grab train data
        x_train = x_total[train_indices, :, :]
        y_train = y_total[train_indices, :]
        # grab test data
        x_test = x_total[test_indices, :, :]
        y_test = y_total[test_indices, :]

        # rebuild metadata dicts after shuffling
        train_data_subspaces = dict()
        test_data_subspaces = dict()
        for idx, val in enumerate(train_indices):
            train_data_subspaces[idx] = data_subspace_dict[val].copy()
        for idx, val in enumerate(test_indices):
            test_data_subspaces[idx] = data_subspace_dict[val].copy()

        if verbose:
            print('\t x_train:', x_train.shape)
            print('\t y_train:', y_train.shape)
            print('\t x_test:', x_test.shape)
            print('\t y_test:', y_test.shape)

        if savez_fname is not None:
            assert savez_fname[-4:] == '.npz'
            # save dataset to file
            dataset_fpath = savez_fname
            print('\nSaving dataset to file...', dataset_fpath)
            np.savez_compressed(dataset_fpath,
                                x_train=x_train,
                                y_train=y_train,
                                x_test=x_test,
                                y_test=y_test)

        if as_torch:
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)
            x_test = torch.from_numpy(x_test)
            y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


def data_train_test_split_linear(
        context_len=100,
        dim_n=128,
        num_W_in_dataset=100,
        context_examples_per_W=1,
        samples_per_context_example=1,
        test_ratio=0.2,
        verbose=True,
        seed=None,
        style_subspace_dimensions=DATAGEN_GLOBALS[0]['style_subspace_dimensions'],
        style_origin_subspace=DATAGEN_GLOBALS[0]['style_origin_subspace'],      # TODO convert to float  0 < m < 1 magnitude
        style_corruption_orthog=DATAGEN_GLOBALS[0]['style_corruption_orthog'],  # TODO convert to float  0 < a < 1
        sigma2_corruption=DATAGEN_GLOBALS[0]['sigma2_corruption'],
        sigma2_pure_context=DATAGEN_GLOBALS[0]['sigma2_pure_context'],
        corr_scaling_matrix=DATAGEN_GLOBALS[0]['corr_scaling_matrix'],
        savez_fname=None,
        as_torch=True):
    """
    Args:
    - test_ratio: float      - 0 <= x <= 1
    - seed: None or int      - static randomization for the order of examples in test and train set

    Vectors x_i live in R^n

    Training sequences are of the form {x_1, ..., x_L, x_query} -> x_{L+1}
     - x_1, ..., x_L lie on affine space W of dimension d < n
     - the final input x_query lies outside W, and the task is to de-corrupt it by projecting onto W
        - corruption procedure A: kick size of random magnitude in a direction orthogonal to W
            kick_size = mu + sigma * np.random.randn()   (mu, sigma both 1.0)
        - corruption procedure B: iid gaussian var = kick_size centered

       W = {x in R^n | x = m + V c} for some c in R^d
        - we do not directly provide V [n x d] or m [n x 1]
        - these can be inferred from the mean and by PCA

    - Teacher solution:
        g(x_q) = mu + P (x_q - mu)  where  P_W = V (V^T V)^-1 V^T defines the projection onto underlying vector space
                                            mu = mean(x_1, ..., x_L)
    """
    print('data_train_test_split_linear() args...')
    print('\tstyle_subspace_dimensions:', style_subspace_dimensions)
    print('\tstyle_origin_subspace:',     style_origin_subspace)
    print('\tstyle_corruption_orthog:',   style_corruption_orthog)
    print('\tcontext_len:', context_len)
    print('\tdim_n:', dim_n)
    print('\tnum_W_in_dataset:', num_W_in_dataset)
    print('\tcontext_examples_per_W:', context_examples_per_W)
    print('\tsamples_per_context_example:', samples_per_context_example)
    print('\tsigma_corruption:', sigma2_corruption)
    print('\tsigma_pure_context:', sigma2_pure_context)
    print('\tcorr_scaling_matrix:', corr_scaling_matrix)
    print('\tseed:', seed)
    print('\tsavez_fname:', savez_fname)

    assert samples_per_context_example == 1

    rng = np.random.default_rng(seed=seed)

    # generate potentially many k = 1, ..., K affine subspaces W of different dimension d << dim_n
    # - K = num_W_in_dataset
    # dim W should be at least 2? so dim_n >= 3?
    # dim W should be at most a fraction of the context_len --OR-- dim_n - 1
    if isinstance(style_subspace_dimensions, (np.integer, int)):
        assert 1 <= style_subspace_dimensions <= min(dim_n,  context_len//2)
        dim_d_k = style_subspace_dimensions * np.ones(num_W_in_dataset, dtype=int)  # all subspace same size
    else:
        assert style_subspace_dimensions == 'random'
        dim_d_k = np.random.randint(1, min(dim_n,  context_len//2), size=num_W_in_dataset)
    print('data_train_test_split_linear(...)')
    print('\tstyle_subspace_dimensions=%s' % style_subspace_dimensions)
    print('\tdim_d_k min/max:', dim_d_k.min(), dim_d_k.max())

    nsample_per_subspace = context_len  # alias | this is the length of the input sequence for sequence model
    assert context_len > max(dim_d_k)

    # sanity check
    train_plus_test_size = num_W_in_dataset * context_examples_per_W * samples_per_context_example
    print('data_train_test_split_linear(...)')
    print('\ttrain_plus_test_size (%d) = num_W_in_dataset (%d) x context_examples_per_W (%d) x samples_per_context_example (%d)' %
          (train_plus_test_size, num_W_in_dataset, context_examples_per_W, samples_per_context_example))

    # create X, y training blob
    x_total = []
    y_total = []
    data_subspace_dict = {j: {} for j in range(train_plus_test_size)}

    j = 0
    # for each dimensionality in the list above, generate a random m, V pair and use them to sample > d_max points
    for k, dim_d in enumerate(dim_d_k):

        # select offsets
        rand_m = gen_uniform_data(dim_n, rng=rng)
        if style_origin_subspace:
            rand_m = np.zeros(dim_n)

        # select basis (orthonormal via QR decomp)
        #rand_V = gen_uniform_data((dim_n, dim_d + 1))              # d+1 is used below to get extra orthog direction
        rand_V = gen_uniform_data((dim_n, dim_n), rng=rng)  # dims d+1 to n are used below to create extra orthog directions
        rand_V, _ = np.linalg.qr(rand_V)

        # to corrupt x_query (final vector in each training sequence), we will give it a kick in orthogonal direction
        W_random_orthog_direction = rand_V[:, dim_d:]  # we leverage this in the style_corruption_orthog = True case
        rand_V =                    rand_V[:, :dim_d]

        """ 
        - (A) mode where the corruption of the last token x_L is just a gaussian centered at x_L (i.e. not necc. orthogonal to the subspace)
        - (B) alt, the corruption is orthogonal to the subspace (using the last direction of the synthetic dim_d + 1 size basis)
        - consider two separate dataset modes: (A) and (B), could also interpolate between them as 0 <= alpha <= 1 and summing a A + (1-a) B
        """
        if style_corruption_orthog:
            # samples the corruption kicks (in orthogonal directions)
            '''
            # ORIGINAL WAY
            corruption_kicks = sample_from_subspace(0 * rand_m, W_random_orthog_direction, examples_per_W, seed=seed1)
            kick_mu, kick_sigma = sigma_corruption, 1.0  # was 1.0, 1.0 orig
            np.random.seed(seed1)
            kick_size = np.random.normal(loc=kick_mu, scale=kick_sigma, size=examples_per_W)
            corruption_kicks = corruption_kicks * kick_size  # rescales corruption kicks... need to clean this up...
            #corruption_kicks = corruption_kicks * 1  # rescales corruption kicks... need to clean this up...
            '''
            # TODO cleanup this block so that it matches the one below and lengths scale with sigma_corruption
            dim_W_perp = W_random_orthog_direction.shape[1]
            assert dim_W_perp == dim_n - dim_d
            corruption_cov = sigma2_corruption * np.eye(dim_W_perp)  # TODO replace by A @ A.T, given full rank A?
            c = gen_gaussian_data(
                np.zeros(dim_W_perp),
                corruption_cov,
                (context_examples_per_W), rng=rng)  # random coefficients for the basis {v_i}; shape samples x dim_d
            corruption_kicks = W_random_orthog_direction @ c.T
        else:
            corruption_mean = np.zeros(dim_n)
            corruption_cov = sigma2_corruption * np.eye(dim_n)

            # Alternative way of having non-isotropic corruption covariance
            """
            if corr_scaling_matrix is None:
                corruption_cov = sigma2_corruption * np.eye(dim_n)
            else:
                print('WARNING - corr_scaling_matrix is not none, setting and normalizing induced covariance...')
                # we normalize to original case by forcing trace to be sigma2_corruption * n
                # - since total cov in isotropic case is sigma2_corruption * n
                # - given any full rank A, we have the following steps:
                #   1) compute frobenius norm of A
                #   2) scale A as  A' = c A  with  c = sqrt(n) / ||A||_F
                #   3) compute sigma_arr = A' @ A'.T
                # - observe that Tr(sigma_arr) = Tr(A' @ A'.T) = ||A'||_F^2 = n   - matching identity matrix
                '''
                frob_norm_val = np.linalg.norm(corr_scaling_matrix, 'fro')  # square to get Tr(A @ A.T)
                corr_scaling_matrix_normed = (np.sqrt(dim_n) / frob_norm_val) * corr_scaling_matrix
                sigma_matrix_normed = corr_scaling_matrix_normed @ corr_scaling_matrix_normed.T
                print(np.linalg.trace(sigma_matrix_normed))
                print('='*20)
                '''
                corruption_cov = sigma2_corruption * sigma_matrix_normed  # TODO new form
            """
            corruption_kicks = gen_gaussian_data(corruption_mean, corruption_cov, context_examples_per_W, rng=rng)  # shape samples x dim_n
            corruption_kicks = corruption_kicks.T  # shape dim_n x context_examples_per_W

        for sample_idx in range(context_examples_per_W):
            # generate samples from the random subspace
            X_sequence = sample_from_subspace(rand_m, rand_V, nsample_per_subspace,
                                              sigma_Z_sqr=sigma2_pure_context, rng=rng, uniform=False)
            corruption_kicks_for_sample = corruption_kicks[:, sample_idx]

            # if non-isotropic covariance modification is used, we need to apply the scaling matrix
            if corr_scaling_matrix is not None:
                '''
                print('WARNING - corr_scaling_matrix is not none, setting and normalizing induced covariance...')
                frob_norm_val = np.linalg.norm(corr_scaling_matrix, 'fro')  # square to get Tr(A @ A.T)
                corr_scaling_matrix_normed = (np.sqrt(dim_n) / frob_norm_val) * corr_scaling_matrix'''
                print('WARNING - corr_scaling_matrix is not none...')
                X_sequence                  = corr_scaling_matrix @ X_sequence
                corruption_kicks_for_sample = corr_scaling_matrix @ corruption_kicks[:, sample_idx]

            # corruption of last column
            y_target = np.copy(X_sequence[:, -1])
            X_sequence[:, -1] = y_target + corruption_kicks_for_sample

            x_total.append(X_sequence)
            y_total.append(y_target)

            data_subspace_dict[j]['W_m'] = rand_m
            data_subspace_dict[j]['W_V'] = rand_V
            data_subspace_dict[j]['W_dim'] = dim_d
            j += 1

    x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = data_train_test_split_util(
        x_total, y_total, data_subspace_dict, context_len, dim_n, num_W_in_dataset,
        context_examples_per_W, test_ratio, as_torch=as_torch,
        savez_fname=savez_fname, verbose=verbose, rng=rng
    )

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


def data_train_test_split_clusters(
        context_len=100,
        dim_n=128,
        num_W_in_dataset=100,
        context_examples_per_W=1,
        samples_per_context_example=1,
        test_ratio=0.2,
        verbose=True,
        seed=None,
        sigma2_corruption=DATAGEN_GLOBALS[1]['sigma2_corruption'],
        style_subspace_dimensions=DATAGEN_GLOBALS[1]['style_subspace_dimensions'],  # unused currently
        style_origin_subspace=DATAGEN_GLOBALS[1]['style_origin_subspace'],          # unused currently
        style_corruption_orthog=DATAGEN_GLOBALS[1]['style_corruption_orthog'],      # unused currently
        style_cluster_mus=DATAGEN_GLOBALS[1]['style_cluster_mus'],                  # unused currently (fix to 'unit_norm')
        style_cluster_vars=DATAGEN_GLOBALS[1]['style_cluster_vars'],                # unused currently (fix to 'isotropic')
        num_cluster=DATAGEN_GLOBALS[1]['num_cluster'],
        cluster_var=DATAGEN_GLOBALS[1]['cluster_var'],
        savez_fname=None,
        as_torch=True):
    """
    Args:
    - test_ratio: float      - 0 <= x <= 1
    - seed: None or int      - static randomization for the order of examples in test and train set

    Specific to clustering case
     - STYLE_NUM_CLUSTER
     - STYLE_FIXED_MU_NORMS = True  # if True, then the cluster centers lie on a (unit) sphere in R^d
     - STYLE_CLUSTER_MU = 'unit_norm'
     - STYLE_CLUSTER_VAR = 'isotropic'
          assert STYLE_CLUSTER_VAR in ['isotropic']
     - CLUSTER_VARS = 1.0  # controls radius of ball of pure of samples (default: 2.0); could be a list in Clustering case

    From setup in main.ipynb:

    Vectors x_i live in R^n

    Training sequences are of the form {x_1, ..., x_L, x_query} -> x_{L+1}
     - x_1, ..., x_L sampled from cluster_num gaussians ~ N( mu_k, Sigma_k )
     - the final input x_query lies outside W, and the task is to de-corrupt it by projecting onto W
    - corruption procedure: iid gaussian var = kick_size centered

    See also data_train_test_split_linear() for more details
    """
    assert style_corruption_orthog is False
    assert style_origin_subspace is True
    assert style_subspace_dimensions in ['full']
    assert style_cluster_mus in ['unit_norm']     # if unit_norm, then the cluster centers lie on a (unit) sphere in R^d
    assert style_cluster_vars in ['isotropic']
    assert num_cluster in ['random'] or isinstance(num_cluster, int)

    rng = np.random.default_rng(seed)  # Create a single RNG instance at the start

    # assert cluster_var is a scalar
    assert isinstance(cluster_var, (float, np.float32))

    print('data_train_test_split_clusters() args...')
    print('\tcontext_len:', context_len)
    print('\tdim_n:', dim_n)
    print('\tnum_W_in_dataset:', num_W_in_dataset)
    print('\tcontext_examples_per_W:', context_examples_per_W)
    print('\tsamples_per_context_example:', samples_per_context_example)
    print('\tsigma_corruption:',   sigma2_corruption)
    print('\t[unused] style_subspace_dimensions:', style_subspace_dimensions)
    print('\t[unused] style_origin_subspace:', style_origin_subspace)
    print('\t[unused] style_corruption_orthog:', style_corruption_orthog)
    print('\t[unused] style_cluster_mus:', style_cluster_mus)
    print('\t[unused] style_cluster_vars:', style_cluster_vars)
    print('\tnum_cluster:', num_cluster)
    print('\tcluster_var:', cluster_var)
    print('\tseed:', seed)
    print('\tsavez_fname:', savez_fname)

    assert samples_per_context_example == 1

    # generate (K) GMMs each involving num_cluster gaussians in R^n
    # - K = num_W_in_dataset
    if isinstance(num_cluster, (np.integer, int)):
        assert 1 <= num_cluster <= context_len//5  # 5 is arbitrary, we want each gaussian to have some samples
        arr_of_ncluster_sizes = num_cluster * np.ones(num_W_in_dataset, dtype=int)  # all subspace same size
    else:
        assert num_cluster == 'random'
        arr_of_ncluster_sizes = np.random.randint(1, context_len//5, size=num_W_in_dataset)
    print('data_train_test_split_clusters(...)')
    print('\tstyle_subspace_dimensions=%s' % style_subspace_dimensions)
    print('\tarr_of_ncluster_sizes min/max:', arr_of_ncluster_sizes.min(), arr_of_ncluster_sizes.max())
    assert context_len > max(arr_of_ncluster_sizes)

    # sanity check
    train_plus_test_size = num_W_in_dataset * context_examples_per_W * samples_per_context_example
    print('data_train_test_split_clusters(...)')
    print('\ttrain_plus_test_size (%d) = num_W_in_dataset (%d) x context_examples_per_W (%d) x samples_per_context_example (%d)' %
          (train_plus_test_size, num_W_in_dataset, context_examples_per_W, samples_per_context_example))

    # create X, y training blob
    x_total = []
    y_total = []
    data_subspace_dict = {j: {} for j in range(train_plus_test_size)}

    j = 0
    # for each dimensionality in the list above, generate a GMM (num_cluster x list of mu, sigma) and use them to sample > d_max points
    for k, num_cluster in enumerate(arr_of_ncluster_sizes):

        # Step 1: sample the cluster means mu_k for k = 1, ..., num_cluster
        if style_cluster_mus == 'unit_norm':
            sampled_cluster_mus = gen_gaussian_data(np.zeros(dim_n), np.eye(dim_n), num_cluster, rng=rng)  # num_cluster x dim_n
            row_norms = np.linalg.norm(sampled_cluster_mus, axis=1)
            sampled_cluster_mus = sampled_cluster_mus / row_norms[:, np.newaxis]  # num_cluster x dim_n, each row unit norm
            sampled_cluster_mus = sampled_cluster_mus.T  # dim_n x num_cluster
        else:
            assert style_cluster_mus in ['unit_norm']
            sampled_cluster_mus = gen_gaussian_data(np.zeros(dim_n), np.eye(dim_n), num_cluster, rng=rng)
            sampled_cluster_mus = sampled_cluster_mus.T  # dim_n x num_cluster

        # Step 2: sample the cluster variances Sigma_k for k=1, ..., num_cluster
        assert style_cluster_vars in ['isotropic']
        sampled_cluster_vars = cluster_var * np.eye(dim_n)  # isotropic case

        # Step 3: sample corruption kicks (we do not use orthogonal corruption in this case)
        corruption_mean = np.zeros(dim_n)
        corruption_cov = sigma2_corruption * np.eye(dim_n)
        corruption_kicks = gen_gaussian_data(corruption_mean, corruption_cov, context_examples_per_W, rng=rng)  # shape samples x dim_n
        corruption_kicks = corruption_kicks.T  # shape dim_n x context_examples_per_W

        for sample_idx in range(context_examples_per_W):
            # Step 4: generate samples from the K = num_cluster gaussians
            # - mean = sampled_cluster_mus[k, :]
            # - cov = sampled_cluster_vars         (all same, isotropic case)
            X_sequence, indices = sample_from_gmm_isotropic(sampled_cluster_mus, sampled_cluster_vars, context_len, rng=rng)

            # corruption of last column
            y_target          = np.copy(X_sequence[:, -1])
            X_sequence[:, -1] = y_target + corruption_kicks[:, sample_idx]
            x_total.append(X_sequence)
            y_total.append(y_target)

            # consider storing sample ids (cluster membership) as well
            data_subspace_dict[j]['num_cluster']        = num_cluster
            data_subspace_dict[j]['sampled_cluster_id'] = indices
            data_subspace_dict[j]['sampled_mus']        = sampled_cluster_mus
            data_subspace_dict[j]['sigma_isotropic']    = sampled_cluster_vars
            j += 1

    x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = data_train_test_split_util(
        x_total, y_total, data_subspace_dict, context_len, dim_n, num_W_in_dataset,
        context_examples_per_W, test_ratio, as_torch=as_torch,
        savez_fname=savez_fname, verbose=verbose, rng=rng
    )

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


def data_train_test_split_manifold(
        context_len=100,
        dim_n=128,
        num_W_in_dataset=100,
        context_examples_per_W=1,
        samples_per_context_example=1,
        test_ratio=0.2,
        verbose=True,
        seed=None,
        sigma2_corruption=DATAGEN_GLOBALS[2]['sigma2_corruption'],
        radius_sphere=DATAGEN_GLOBALS[2]['radius_sphere'],
        style_subspace_dimensions=DATAGEN_GLOBALS[2]['style_subspace_dimensions'],
        style_origin_subspace=DATAGEN_GLOBALS[2]['style_origin_subspace'],          # unused currently (force True)
        style_corruption_orthog=DATAGEN_GLOBALS[2]['style_corruption_orthog'],      # unused currently (force False)
        savez_fname=None,
        as_torch=True):
    """
    Args:
    - test_ratio: float      - 0 <= x <= 1
    - seed: None or int      - static randomization for the order of examples in test and train set

    Specific to manifold (sphere) case
     - radius_sphere=SIGMA2_PURE_CONTEXT,  # controls radius of sphere (d-dim manifold S in R^n)

    From setup in main.ipynb:

    Vectors x_i live in R^n

    Training sequences are of the form {x_1, ..., x_L, x_query} -> x_{L+1}
     - x_1, ..., x_L sampled from d-dim sphere in R^n
     - the final input x_query lies outside W, and the task is to de-corrupt it by projecting onto W
     - corruption procedure: iid gaussian var = kick_size centered

    See also data_train_test_split_linear() for more details
    """
    assert style_corruption_orthog is False
    assert style_origin_subspace is True
    assert isinstance(radius_sphere, (float, np.float32))

    print('data_train_test_split_manifold() args...')
    print('\tcontext_len:', context_len)
    print('\tdim_n:', dim_n)
    print('\tnum_W_in_dataset:', num_W_in_dataset)
    print('\tcontext_examples_per_W:', context_examples_per_W)
    print('\tsamples_per_context_example:', samples_per_context_example)
    print('\tsigma_corruption:',   sigma2_corruption)
    print('\tstyle_subspace_dimensions:', style_subspace_dimensions)
    print('\t[unused] style_origin_subspace:', style_origin_subspace)
    print('\t[unused] style_corruption_orthog:', style_corruption_orthog)
    print('\tradius_sphere:', radius_sphere)
    print('\tseed:', seed)
    print('\tsavez_fname:', savez_fname)

    assert samples_per_context_example == 1

    # generate potentially many k = 1, ..., K affine subspaces W of different dimension d << dim_n
    # - K = num_W_in_dataset

    rng = np.random.default_rng(seed)  # Create a single RNG instance at the start
    # np.random.seed(seed)

    # dim W should be at least 2? so dim_n >= 3?
    # dim W should be at most a fraction of the context_len --OR-- dim_n - 1
    if isinstance(style_subspace_dimensions, (np.integer, int)):
        assert 1 <= style_subspace_dimensions <= min(dim_n,  context_len//2)
        dim_d_k = style_subspace_dimensions * np.ones(num_W_in_dataset, dtype=int)  # all subspace same size
    else:
        assert style_subspace_dimensions == 'random'
        dim_d_k = np.random.randint(1, min(dim_n,  context_len//2), size=num_W_in_dataset)
    print('data_train_test_split_manifold(...)')
    print('\tstyle_subspace_dimensions=%s' % style_subspace_dimensions)
    print('\tdim_d_k min/max:', dim_d_k.min(), dim_d_k.max())

    assert context_len > max(dim_d_k)

    # sanity check
    train_plus_test_size = num_W_in_dataset * context_examples_per_W * samples_per_context_example
    print('data_train_test_split_manifold(...)')
    print('\ttrain_plus_test_size (%d) = num_W_in_dataset (%d) x context_examples_per_W (%d) x samples_per_context_example (%d)' %
          (train_plus_test_size, num_W_in_dataset, context_examples_per_W, samples_per_context_example))

    # create X, y training blob
    x_total = []
    y_total = []
    data_subspace_dict = {j: {} for j in range(train_plus_test_size)}

    j = 0
    # for each dimensionality in the list above, generate a d-dim sphere in R^n (...) and use them to sample > d_max points
    for k, dim_d in enumerate(dim_d_k):

        # Step 1: sample a rotation matrix P_k in R^n
        sampled_rot_matrix = special_ortho_group.rvs(dim_n, random_state=rng)

        # Step 2: sample corruption kicks (we do not use orthogonal corruption in this case)
        corruption_mean = np.zeros(dim_n)
        corruption_cov = sigma2_corruption * np.eye(dim_n)
        corruption_kicks = gen_gaussian_data(corruption_mean, corruption_cov, context_examples_per_W, rng=rng)  # shape samples x dim_n
        corruption_kicks = corruption_kicks.T  # shape dim_n x context_examples_per_W

        for sample_idx in range(context_examples_per_W):
            # Step 3: generate samples from S - the d-dim sphere in R^n
            X_sequence = sample_from_subsphere(dim_d, radius_sphere, sampled_rot_matrix, context_len, rng=rng)

            # corruption of last column
            y_target = np.copy(X_sequence[:, -1])
            X_sequence[:, -1] = y_target + corruption_kicks[:, sample_idx]
            x_total.append(X_sequence)
            y_total.append(y_target)

            data_subspace_dict[j]['dim_sphere'] = dim_d
            data_subspace_dict[j]['rotation'] = sampled_rot_matrix
            j += 1

    x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = data_train_test_split_util(
        x_total, y_total, data_subspace_dict, context_len, dim_n, num_W_in_dataset,
        context_examples_per_W, test_ratio, as_torch=as_torch,
        savez_fname=savez_fname, verbose=verbose, rng=rng
    )

    return x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces


class DatasetWrapper(Dataset):
    """
    (relic): currently, there is a "remainder" batch at the end, with size smaller than batch_size -- could discard it
    """
    def __init__(self, X, Y):
        self.x = X
        self.y = Y
        self.dim_n = self.x.size()[1]
        self.context_length = self.x.size()[2]

    # Mandatory: Get input pair for training
    def __getitem__(self, idx):
        return self.x[idx, :, :], self.y[idx, :]

    # Mandatory: Number of elements in dataset (i.e. size of batch dimension 0)
    def __len__(self):
        X_len = self.x.size()[0]
        return X_len

    # This function is not needed
    def plot(self):
        print('not implemented')
        return


def plot_one_prediction_batch(net, criterion, data_loader, max_example_from_batch=5, as_barplot=True,
                              show_corrupted_input=True, save=False, show=False, dir_out=DIR_OUT):
    """
    plot example predictions vs target
    - the desired vector lies on some affine subspace W subset R^n of dim d < n
    - the model spits out a vector in R^n (ideally a de-corrupted) version of the input

    Version 1 plot:
        - plot output vs target as n-dim vector
    """
    dataiter = iter(data_loader)

    with torch.no_grad():
        for idx in range(1):
            samples, targets = next(iter(dataiter))
            outputs = net(samples)
            batchsz = samples.size()[0]
            dim_n = samples.size()[1]

            for k in range(min(max_example_from_batch, batchsz)):
                plt.figure(figsize=(20, 2))  # 20, 2?

                x_target = targets[k, :]
                x_pred = outputs[k, :]
                input_final_vec = samples[k, :, -1]

                xarr = np.arange(dim_n)
                if as_barplot:
                    if show_corrupted_input:
                        bw = 0.3  # 0.4
                        # plt.bar(xarr - 0.75 * bw,        x_target, width=bw, linewidth=0.5, edgecolor='k', label='target',        color='darkgrey')
                        # plt.bar(xarr,                      x_pred, width=bw, linewidth=0.5, edgecolor='k', label='NN pred',       color='blue', alpha=1)
                        # plt.bar(xarr + 0.75 * bw, input_final_vec, width=bw, linewidth=0.5, edgecolor='k', label=r'$\tilde x_L$', color='purple', alpha=1, zorder=11)

                        plt.bar(xarr - 0.9 * bw, x_target, width=bw, linewidth=0.5, edgecolor='k', label='target',
                                color=COLOR_TARGET)
                        plt.bar(xarr, x_pred, width=bw, linewidth=0.5, edgecolor='k', label='NN pred', color=COLOR_PRED,
                                alpha=1)
                        plt.bar(xarr + 0.9 * bw, input_final_vec, width=bw, linewidth=0.5, edgecolor='k',
                                label=r'$\tilde x_L$', color=COLOR_INPUT, alpha=1, zorder=11)

                        # these could replace bar for input_final_vec
                        # plt.plot(xarr, input_final_vec, linewidth=1, label=r'$\tilde x_L$', color='k', alpha=1, zorder=11)
                        # plt.plot(xarr, input_final_vec, linewidth=2, color='white', alpha=1, zorder=10)  # for "oreo" line style
                    else:
                        bw = 0.4  # 0.4
                        plt.bar(xarr - 0.5 * bw, x_target, width=bw, label='target', color=COLOR_TARGET)
                        plt.bar(xarr + 0.5 * bw, x_pred, width=bw, label='NN pred', color=COLOR_PRED, alpha=0.5)

                else:
                    bw = 0.3  # 0.4
                    plt.plot(x_target, label='target', color=COLOR_TARGET)
                    plt.plot(x_pred, '--', label='model.forward()', color=COLOR_PRED, alpha=0.5)
                    if show_corrupted_input:
                        plt.bar(xarr + 0.0 * bw, input_final_vec, width=bw, label=r'$\tilde x_L$', color=COLOR_INPUT,
                                alpha=0.6, zorder=11)

                # https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
                loss_k = criterion(x_pred, x_target).item()
                explicit_dist_k = torch.sum(torch.pow(x_target - x_pred, 2), dim=0) / dim_n

                plt.title(r'Compare true vs. de-corrupted components of single example $\tilde x \in \mathbb{R}^n$'
                          + '\n' + r'loss: %.2e  |  dist $\frac{1}{n}\Vert x_{target} - x_{pred} \Vert^2$: %.2e' % (
                              loss_k, explicit_dist_k)
                          )
                plt.xlabel('dimension $i$ (dim $n=%d$)' % dim_n)
                plt.ylabel(r'value $y_i$')
                plt.legend()
                plt.tight_layout()
                if save:
                    plt.savefig(dir_out + os.sep + 'plot_prediction_example_%d.pdf' % k)
                if show:
                    plt.show()


def plot_batch_predictions(net, criterion, data_loader, nbatch=1, save=False, show=False, dir_out=DIR_OUT):
    """
    makes one plot per batch (max nbatch) in the data_loader
    """
    with torch.no_grad():
        count = 0
        # for idx in range(1):  # use only 1 batch
        for batch_idx, (samples, targets) in enumerate(data_loader):

            count += 1
            if count > nbatch:
                break

            # samples, targets = next(iter(dataiter))
            outputs = net(samples)
            batchsz = samples.size()[0]
            dim_n = samples.size()[1]

            losses_model = np.zeros(batchsz)
            losses_model_null = np.zeros(batchsz)

            for k in range(batchsz):  # could do a "max per batch" here...
                x_target = targets[k, :]
                x_pred = outputs[k, :]
                input_final_vec = samples[k, :, -1]

                # https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
                loss_k = criterion(x_target, x_pred).item()
                loss_k_null = criterion(x_target, input_final_vec).item()

                losses_model[k] = loss_k
                losses_model_null[k] = loss_k_null

            plt.figure(figsize=(20, 2))
            bw = 0.3
            xarr = np.arange(batchsz)
            plt.bar(xarr - 0.5 * bw, losses_model, width=bw, linewidth=0.5, edgecolor='k', label='model',
                    color=COLOR_PRED)
            plt.bar(xarr + 0.5 * bw, losses_model_null, width=bw, linewidth=0.5, edgecolor='k', label='null',
                    color=COLOR_INPUT)
            # plt.bar(xarr + 0.5 * bw, 0.01, width=bw, linewidth=0.5, edgecolor='k', label='null', color=COLOR_INPUT)

            plt.title(r'Compare error in model vs "guess last token" for one batch (%d examples)' % batchsz
                      + '\n' + r'MSE/n model: %.2e  |  MSE/n null: %.2e' % (
                          np.mean(losses_model), np.mean(losses_model_null))
                      )
            plt.axhline(np.mean(losses_model_null), linestyle='--', color=COLOR_INPUT)

            plt.xlabel('sample index $i$ (one batch)')
            plt.ylabel(r'$L(x_{target}, x_{pred})$')
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(dir_out + os.sep + 'plot_many_prediction_batch_batch%d_sz%d.pdf' % (batch_idx, batchsz))
            if show:
                plt.show()


if __name__ == '__main__':
    # demonstration of train/test split
    context_len = 100
    dim_n = 32
    test_ratio = 0.2

    datagen_choice = 2  # {0, 1, 2} -> {linear, clusters, manifold}

    base_kwargs = dict(
        context_len=context_len,
        dim_n=dim_n,
        num_W_in_dataset=1000,
        context_examples_per_W=1,
        samples_per_context_example=1,
        test_ratio=test_ratio,
        verbose=True,
        as_torch=True,
        savez_fname=DIR_DATA + os.sep + 'dataset_example.npz',
        seed=0,
        style_subspace_dimensions=DATAGEN_GLOBALS[datagen_choice]['style_subspace_dimensions'],
        style_origin_subspace=DATAGEN_GLOBALS[datagen_choice]['style_origin_subspace'],
        style_corruption_orthog=DATAGEN_GLOBALS[datagen_choice]['style_corruption_orthog'],
        sigma2_corruption=DATAGEN_GLOBALS[datagen_choice]['sigma2_corruption'],
    )

    print('='*20)
    ################################################################################
    # Build or load data
    ################################################################################
    if datagen_choice == 0:
        x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = data_train_test_split_linear(
            **base_kwargs,
            sigma2_pure_context=DATAGEN_GLOBALS[datagen_choice]['sigma2_pure_context'],
        )
    elif datagen_choice == 1:
        x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = data_train_test_split_clusters(
            **base_kwargs,
            style_cluster_mus=DATAGEN_GLOBALS[datagen_choice]['style_cluster_mus'],
            style_cluster_vars=DATAGEN_GLOBALS[datagen_choice]['style_cluster_vars'],
            num_cluster=DATAGEN_GLOBALS[datagen_choice]['num_cluster'],
            cluster_var=DATAGEN_GLOBALS[datagen_choice]['cluster_var'],
        )
    else:
        assert datagen_choice == 2
        x_train, y_train, x_test, y_test, train_data_subspaces, test_data_subspaces = data_train_test_split_manifold(
            **base_kwargs,
            radius_sphere=DATAGEN_GLOBALS[datagen_choice]['radius_sphere'],
        )

    print('x_train.shape', x_train.shape)

    # specify training and testing datasets using DatasetWrapper class (to give to torch DataLoader)
    print('\nNow check DatasetWrapper instance and give it to torch DataLoader')
    train_dataset = DatasetWrapper(x_train, y_train)
    test_dataset =  DatasetWrapper(x_test,  y_test)
    print('\ttrain_dataset.context_length', train_dataset.context_length)
    print('\ttrain_dataset.dim_n', train_dataset.dim_n)

    nwork = 0
    train_batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=nwork)

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data  # get the inputs; data is a list of [inputs, labels]
        print('\tbatch %d size (inputs --> targets):' % i, inputs.size(), '-->', labels.size())  # -- currently the last batch can be smaller than the rest (remainder)


    # load dataset from file
    dataset_fpath = DIR_DATA + os.sep + 'dataset_example.npz'
    loaded_x_train, loaded_y_train, loaded_x_test, loaded_y_test = load_dataset(dataset_fpath, as_torch=True)

    # using loaded data - specify training and testing datasets using DatasetWrapper class (to give to torch DataLoader)
    print('\n(Loaded data) - Now check DatasetWrapper instance and give it to torch DataLoader')
    loaded_train_dataset = DatasetWrapper(loaded_x_train, loaded_y_train)
    loaded_test_dataset =  DatasetWrapper(loaded_x_test,  loaded_y_test)
    print('\ttrain_dataset.context_length', loaded_train_dataset.context_length)
    print('\ttrain_dataset.dim_n', loaded_train_dataset.dim_n)
