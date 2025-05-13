import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax as scipy_softmax

from data_tools import proj_affine_subspace_estimator, bessel_ratio_subhalf_sub3half
from torch.utils.data.sampler import SequentialSampler


def loss_if_predict_zero(criterion, dataloader, data_label):
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data
            outputs = torch.zeros(targets.size())

            mse_loss_total += criterion(outputs, targets).item()
            count += 1

    mse_loss = mse_loss_total / count
    print('\t%s data loss if predicting context mean: %.3e' % (data_label, mse_loss))
    return mse_loss


def loss_if_predict_average(criterion, dataloader, data_label):
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data
            outputs = torch.mean(samples[:, :, :], -1)

            mse_loss_total += criterion(outputs, targets).item()
            count += 1

    mse_loss = mse_loss_total / count
    print('\t%s data loss if only predicting 0s: %.3e' % (data_label, mse_loss))
    return mse_loss


def loss_if_predict_mostrecent(criterion, dataloader, data_label):
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            outputs = samples[:, :, -1]

            mse_loss_total += criterion(outputs, targets).item()
            count += 1  # this +1 and divide by count assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    print('\t%s data loss if predicting (k-1)th element: %.3e' % (data_label, mse_loss))
    return mse_loss


def loss_if_predict_linalg(criterion, dataloader, data_label, print_val=True):
    # Note this loop is currently quite slow...
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            nbatch, ndim, ncontext = samples.size()
            for b in range(nbatch):
                X_seq =       samples[b, :, :-1].numpy()
                x_L_corrupt = samples[b, :, -1].numpy()

                x_L_corrupt = np.expand_dims(x_L_corrupt, axis=1)  # this is needed for linalg shape inference in fn
                project_xL_onto_W_np = proj_affine_subspace_estimator(X_seq, x_L_corrupt)
                project_xL_onto_W_np = np.squeeze(project_xL_onto_W_np)
                project_xL_onto_W = torch.from_numpy(project_xL_onto_W_np)
                mse_loss_total += criterion(project_xL_onto_W, targets[b, :]).item()

                count += 1  # this +1 and divide by count assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss if linalg. projection: %.3e (count=%d)' % (data_label, mse_loss, count))
    return mse_loss


def loss_if_predict_linalg_shrunken(criterion, dataloader, data_label, sig2_z, sig2_corrupt,
                                    style_origin_subspace=True, style_corruption_orthog=False,
                                    print_val=True):
    assert style_origin_subspace        # we assume proper subspace through origin
    assert not style_corruption_orthog  # we assume it is iid gaussian ball corruption (not orthogonal to W)

    # Note this loop is currently quite slow...
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            nbatch, ndim, ncontext = samples.size()
            for b in range(nbatch):
                X_seq =       samples[b, :, :-1].numpy()
                x_L_corrupt = samples[b, :,  -1].numpy()

                x_L_corrupt = np.expand_dims(x_L_corrupt, axis=1)  # this is needed for linalg shape inference in fn
                project_xL_onto_W_np = proj_affine_subspace_estimator(X_seq, x_L_corrupt)
                project_xL_onto_W_np = np.squeeze(project_xL_onto_W_np)

                # perform shrink step (should be doing inside proj_affine_subspace_estimator but this is fine with asserts above)
                shrink_factor = sig2_z / (sig2_z + sig2_corrupt)
                project_xL_onto_W_np = shrink_factor * project_xL_onto_W_np

                project_xL_onto_W = torch.from_numpy(project_xL_onto_W_np)
                mse_loss_total += criterion(project_xL_onto_W, targets[b, :]).item()

                count += 1  # this +1 and divide by count (below) assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss if linalg. projection (shrunk): %.3e (count=%d)' % (data_label, mse_loss, count))
    return mse_loss


def loss_if_predict_subsphere_baseline(criterion, dataloader, data_label, sphere_radius, sig2_corrupt,
                                       style_origin_subspace=True, style_corruption_orthog=False,
                                       plot_some=True, print_val=True, shrunken=True, sphere2_force_wolfram=False):
    assert style_origin_subspace        # we assume proper subspace through origin
    assert not style_corruption_orthog  # we assume it is iid gaussian ball corruption (not orthogonal to W)

    # Note this loop is currently quite slow...
    mse_loss_total = 0
    count = 0.0
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
                project_xL_onto_W_np, est_basis = proj_affine_subspace_estimator(X_seq, x_L_corrupt, return_basis=True)
                project_xL_onto_W_np = np.squeeze(project_xL_onto_W_np)

                # 1) get norm of x_L_corrupt
                # ===================================================================================
                x_L_corrupt_projected_norm = np.linalg.norm(project_xL_onto_W_np)

                # 2) estimate sphere radius from the sequence
                # ===================================================================================
                radius_est = np.mean(np.linalg.norm(X_seq[:, 0], axis=0))  # for now, we assert style_origin_subspace

                # 3) estimate sphere dimension from the sequence
                # ===================================================================================
                d_dim_infer = est_basis.shape[1]  # for a circle in 2d, this would be "2"; for the bessel ratio fn we then pass d-1

                # 4) perform shrink step (should be doing inside proj_affine_subspace_estimator but this is fine with asserts above)
                # ===================================================================================
                beta_val      = radius_est * x_L_corrupt_projected_norm / sig2_corrupt
                shrink_factor = bessel_ratio_subhalf_sub3half(d_dim_infer - 1, beta_val)  # note we pass d-1; a circle is 2d and we pass 1
                prediction_on_subsphere = radius_est * project_xL_onto_W_np / x_L_corrupt_projected_norm

                if shrunken:
                    if sphere2_force_wolfram:
                        shrink_factor_wolfram = 1/beta_val * (beta_val / np.tanh(beta_val) - 1)
                        assert d_dim_infer == 3. # i.e. points lie in 3d, on a 2-sphere
                        print('shrink_factor_wolfram', shrink_factor_wolfram)
                        baseline_vec = shrink_factor_wolfram * prediction_on_subsphere
                    else:
                        baseline_vec = shrink_factor * prediction_on_subsphere
                else:
                    baseline_vec = prediction_on_subsphere

                baseline_vec = torch.from_numpy(baseline_vec)
                mse_loss_total += criterion(baseline_vec, targets[b, :]).item()

                count += 1  # this +1 and divide by count (below) assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss if projection onto subsphere: %.3e (count=%d)' % (data_label, mse_loss, count))
    return mse_loss


def loss_if_predict_clustering_baseline(criterion, dataloader, data_label, sig2_corrupt, data_subspace_dict,
                                        style_origin_subspace=True, style_corruption_orthog=False,
                                        force_zero_cluster_var=False,
                                        print_val=True):
    """
    Assume the cluster center matrix [mu_1 ... mu_p] is provided
    Assume the cluster weights w_1 ... w_p are provided (don't need to be equal)
    Assume the cluster variances are identical and isotropic sigma_alpha^2 = sigma_0^2

    These will need to be recovered from the data generation info object - "data_subspace_dict"
        data_subspace_dict[j]['num_cluster']        = num_cluster
        data_subspace_dict[j]['sampled_cluster_id'] = indices
        data_subspace_dict[j]['sampled_mus']        = sampled_cluster_mus
        data_subspace_dict[j]['sigma_isotropic']    = sampled_cluster_vars

    The estimator we need to compute given a prompt is

        x_predict = a_res * x_corrupt + a_cluster *  M softmax(M.T @ x_corrupt + ln w)

    where
        a_res     =       sigma_0^2 / (sigma_0^2 + sigma_corrupt^2) is the residual factor
        a_cluster = sigma_corrupt^2 / (sigma_0^2 + sigma_corrupt^2) is the cluster factor

    Note: critical that dataloader is not shuffled (since we need to pass the cluster indices for each example) and bsz is 1
    """
    assert style_origin_subspace        # we assume proper subspace through origin
    assert not style_corruption_orthog  # we assume it is iid gaussian ball corruption (not orthogonal to W)

    assert isinstance(dataloader.sampler, SequentialSampler), "assert: dataLoader can't be shuffled (preserve ex IDs)"
    assert dataloader.batch_sampler.batch_size == 1,          "assert: batch size must be 1 (preserve example IDs)"

    mse_loss_total = 0
    count = 0.0

    # Assume same for whole dataset (el;se we cna move into the loop)
    gmm_sig2_0_matrix = data_subspace_dict[0]['sigma_isotropic']  # this is matrix (n x n)
    gmm_sig2_0        = gmm_sig2_0_matrix[0, 0]                   # this is a scalar (we assume isotropic)

    # data structure sanity check for one example
    example_gmm_mu_matrix = data_subspace_dict[0]['sampled_mus']
    dim_n, num_cluster = example_gmm_mu_matrix.shape
    assert data_subspace_dict[0]['num_cluster'] == num_cluster

    # 1) calculate scalars (fixed for dataset)
    a_res      =   gmm_sig2_0 / (gmm_sig2_0 + sig2_corrupt)
    a_cluster  = sig2_corrupt / (gmm_sig2_0 + sig2_corrupt)
    beta_scale = 1 / (gmm_sig2_0 + sig2_corrupt)

    # simpler baseline for plotting/analysis
    if force_zero_cluster_var:
        a_res = 0.0
        a_cluster = 1.0
        beta_scale = 1 / sig2_corrupt

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            nbatch, ndim, ncontext = samples.size()
            for b in range(nbatch):
                X_seq =       samples[b, :, :-1].numpy()
                x_L_corrupt = samples[b, :,  -1].numpy()

                # 2) extract subspace/submanifold/cluster parameters for this example
                # ===================================================================================
                gmm_mu_matrix = data_subspace_dict[i]['sampled_mus']

                # from the indices, infer the sampled cluster weights
                # (ALTERNATIVELY: could assume 1/p for all weights)
                cluster_indices = data_subspace_dict[i]['sampled_cluster_id']
                gmm_w_vec = np.zeros(num_cluster)
                # this caveman loop is probably not efficient
                for k in range(num_cluster):
                    gmm_w_vec[k] = np.sum(cluster_indices == k)
                gmm_w_vec = gmm_w_vec / np.sum(gmm_w_vec)
                # ===================================================================================

                # 3) compute residual term (this goes to zero when cluster var goes to zero)
                x_term_residual = a_res * x_L_corrupt

                # 4) compute cluster term - can do this as a sum or use softmax (same thing)

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
                x_term_cluster = a_cluster * gmm_mu_matrix @ scipy_softmax(beta_scale * gmm_mu_matrix.T @ x_L_corrupt + np.log(gmm_w_vec))

                # 5) combine
                x_predict = x_term_residual + x_term_cluster

                # 6) compute loss using estimator - x_predict
                baseline_vec = torch.from_numpy(x_predict)
                mse_loss_total += criterion(baseline_vec, targets[b, :]).item()

                count += 1  # this +1 and divide by count (below) assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss if projection onto subsphere: %.3e (count=%d)' % (data_label, mse_loss, count))
    return mse_loss


def theory_linear_expected_error(dim_n, W_dim_d, sigma2_corruption, sigma2_pure_context):
    """
    in paper we don't scale by 1/dim_n (ambient # features), but we do here since torch does same automatically
    """
    return (W_dim_d / dim_n) * sigma2_corruption * sigma2_pure_context / (sigma2_pure_context + sigma2_corruption)


def report_dataset_loss(net, criterion, dataloader, data_label, print_val=False, plot_some=False):
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data
            outputs = net(samples)

            if plot_some:
                # visualization and debugging
                if count < 5:
                    plt.plot(outputs, label='pred')
                    plt.plot(targets, label='true')
                    plt.legend()
                    plt.show()

            mse_loss_total += criterion(outputs, targets).item()  # add the *mean* MSE for the batch to mse_loss_total


            # Note: we could do count += bsz, but we assume batch-mean inside criterion call
            # e.g. count += samples.size(0)
            #  - this would resolve edge issue when dataset (e.g. test set) not divisible by bsz
            count += 1  # this +1 and divide by count assumes batch-mean inside criterion call, as is done above

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss: %.3e (batches=%d)' % (data_label, mse_loss, count))
    return mse_loss
