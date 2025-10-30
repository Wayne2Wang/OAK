import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import torch
from sklearn.utils import check_random_state
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs


def display_format_results(*results):
    return 'All {:.4f} | Old {:.4f} | New {:.4f}'.format(*results)


def log_accs_from_preds(y_true, y_pred, mask, eval_funcs, save_name, T, writer=None, logger=None, return_ind_map=False):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :param logger: Python logger
    :return:
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        acc = acc_f(y_true, y_pred, mask)
        all_acc, old_acc, new_acc = acc[:3]
        log_name = f'{save_name}_{f_name}'

        if writer is not None:
            writer.add_scalars(log_name,
                               {'Old': old_acc, 'New': new_acc,
                                'All': all_acc}, T)

        if f_name == 'v2':
            to_return = (all_acc, old_acc, new_acc)
            if len(acc) > 3 and return_ind_map:
                to_return = (*to_return, acc[3])

        if logger is not None:
            results = (all_acc, old_acc, new_acc)
            print_str = 'Epoch {}, {}: {}'.format(T, log_name, display_format_results(*results))
            logger.info(print_str)

    return to_return



"""
https://github.com/sgvaze/generalized-category-discovery/blob/831a645c3d09a68ec4633a45741025765bacf7e0/project_utils/cluster_and_log_utils.py#L29
"""

def split_cluster_acc_v2(y_true, y_pred, unlabeled_known_mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w) # (row_indices, col_indices) which is (pred_indices, gt_indices)
    ind = np.vstack(ind).T # each row is a pair of indices (pred_idx, gt_idx)

    ind_map = {j: i for i, j in ind} # gt_idx -> pred_idx
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    if sum(unlabeled_known_mask) == 0:
        return total_acc, -1.0, total_acc, ind_map
    else:
        old_classes_gt = set(y_true[unlabeled_known_mask])
        old_acc = 0
        total_old_instances = 0
        for i in old_classes_gt:
            old_acc += w[ind_map[i], i]
            total_old_instances += sum(w[:, i])
        old_acc /= total_old_instances

    if sum(~unlabeled_known_mask) == 0:
        return total_acc, old_acc, -1.0, ind_map
    else:
        new_classes_gt = set(y_true[~unlabeled_known_mask])
        new_acc = 0
        total_new_instances = 0
        for i in new_classes_gt:
            new_acc += w[ind_map[i], i]
            total_new_instances += sum(w[:, i])
        new_acc /= total_new_instances

    return total_acc, old_acc, new_acc, ind_map


EVAL_FUNCS = {
    'v2': split_cluster_acc_v2,
}



"""
https://github.com/sgvaze/generalized-category-discovery/blob/831a645c3d09a68ec4633a45741025765bacf7e0/methods/clustering/faster_mix_k_means_pytorch.py
"""

def pairwise_distance(data1, data2, batch_size=None):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)

    if batch_size == None:
        dis = (A-B)**2
        #return N*N matrix for pairwise distance
        dis = dis.sum(dim=-1)
        #  torch.cuda.empty_cache()
    else:
        i = 0
        dis = torch.zeros(data1.shape[0], data2.shape[0])
        while i < data1.shape[0]:
            if(i+batch_size < data1.shape[0]):
                dis_batch = (A[i:i+batch_size]-B)**2
                dis_batch = dis_batch.sum(dim=-1)
                dis[i:i+batch_size] = dis_batch
                i = i+batch_size
                #  torch.cuda.empty_cache()
            elif(i+batch_size >= data1.shape[0]):
                dis_final = (A[i:] - B)**2
                dis_final = dis_final.sum(dim=-1)
                dis[i:] = dis_final
                #  torch.cuda.empty_cache()
                break
    #  torch.cuda.empty_cache()
    return dis


class GCDSemiKMeans:

    def __init__(self, k=3, tolerance=1e-4, max_iterations=100, init='k-means++',
                 n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=None, mode=None):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.pairwise_batch_size = pairwise_batch_size
        self.mode = mode

    def split_for_val(self, l_feats, l_targets, val_prop=0.2):

        np.random.seed(0)

        # Reserve some labelled examples for validation
        num_val_instances = int(val_prop * len(l_targets))
        val_idxs = np.random.choice(range(len(l_targets)), size=(num_val_instances), replace=False)
        val_idxs.sort()
        remaining_idxs = list(set(range(len(l_targets))) - set(val_idxs.tolist()))
        remaining_idxs.sort()
        remaining_idxs = np.array(remaining_idxs)

        val_l_targets = l_targets[val_idxs]
        val_l_feats = l_feats[val_idxs]

        remaining_l_targets = l_targets[remaining_idxs]
        remaining_l_feats = l_feats[remaining_idxs]

        return remaining_l_feats, remaining_l_targets, val_l_feats, val_l_targets


    def kpp(self, X, pre_centers=None, k=10, random_state=None):
        random_state = check_random_state(random_state)

        if pre_centers is not None:

            C = pre_centers

        else:

            C = X[random_state.randint(0, len(X))]

        C = C.view(-1, X.shape[1])

        while C.shape[0] < k:

            dist = pairwise_distance(X, C, self.pairwise_batch_size)
            dist = dist.view(-1, C.shape[0])
            d2, _ = torch.min(dist, dim=1)
            prob = d2/d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = random_state.rand()

            if len((cum_prob >= r).nonzero()) == 0:
                debug = 0
            else:
                ind = (cum_prob >= r).nonzero()[0][0]
            C = torch.cat((C, X[ind].view(1, -1)), dim=0)

        return C


    def fit_once(self, X, random_state):

        centers = torch.zeros(self.k, X.shape[1]).type_as(X)
        labels = -torch.ones(len(X))
        #initialize the centers, the first 'k' elements in the dataset will be our initial centers

        if self.init == 'k-means++':
            centers = self.kpp(X, k=self.k, random_state=random_state)

        elif self.init == 'random':

            random_state = check_random_state(self.random_state)
            idx = random_state.choice(len(X), self.k, replace=False)
            for i in range(self.k):
                centers[i] = X[idx[i]]

        else:
            for i in range(self.k):
                centers[i] = X[i]

        #begin iterations

        best_labels, best_inertia, best_centers = None, None, None
        for i in range(self.max_iterations):

            centers_old = centers.clone()
            dist = pairwise_distance(X, centers, self.pairwise_batch_size)
            mindist, labels = torch.min(dist, dim=1)
            inertia = mindist.sum()

            for idx in range(self.k):
                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(X, 0, selected)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))
            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1


    def fit_mix_once(self, u_feats, l_feats, l_targets, random_state):

        def supp_idxs(c):
            return l_targets.eq(c).nonzero().squeeze(1)

        l_classes = torch.unique(l_targets)
        support_idxs = list(map(supp_idxs, l_classes))
        l_centers = torch.stack([l_feats[idx_list].mean(0) for idx_list in support_idxs])
        cat_feats = torch.cat((l_feats, u_feats))

        centers = torch.zeros([self.k, cat_feats.shape[1]]).type_as(cat_feats)
        centers[:len(l_classes)] = l_centers

        labels = -torch.ones(len(cat_feats)).type_as(cat_feats).long()

        l_classes = l_classes.cpu().long().numpy()
        l_targets = l_targets.cpu().long().numpy()
        l_num = len(l_targets)
        cid2ncid = {cid:ncid for ncid, cid in enumerate(l_classes)}  # Create the mapping table for New cid (ncid)
        for i in range(l_num):
            labels[i] = cid2ncid[l_targets[i]]

        #initialize the centers, the first 'k' elements in the dataset will be our initial centers
        centers = self.kpp(u_feats, l_centers, k=self.k, random_state=random_state)

        # Begin iterations
        best_labels, best_inertia, best_centers = None, None, None
        for it in range(self.max_iterations):
            centers_old = centers.clone()

            dist = pairwise_distance(u_feats, centers, self.pairwise_batch_size)
            u_mindist, u_labels = torch.min(dist, dim=1)
            u_inertia = u_mindist.sum()
            l_mindist = torch.sum((l_feats - centers[labels[:l_num]])**2, dim=1)
            l_inertia = l_mindist.sum()
            inertia = u_inertia + l_inertia
            labels[l_num:] = u_labels

            for idx in range(self.k):

                selected = torch.nonzero(labels == idx).squeeze()
                selected = torch.index_select(cat_feats, 0, selected)
                centers[idx] = selected.mean(dim=0)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia

            center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_old) ** 2, dim=1)))

            if center_shift ** 2 < self.tolerance:
                #break out of the main loop if the results are optimal, ie. the centers don't change their positions much(more than our tolerance)
                break

        return best_labels, best_inertia, best_centers, i + 1


    def fit(self, X):
        random_state = check_random_state(self.random_state)
        best_inertia = None
        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):
                labels, inertia, centers, n_iters = self.fit_once(X, random_state)
                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters
        else:
            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(self.fit_once)(X, seed) for seed in seeds)
            # Get results with the lowest inertia
            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]


    def fit_mix(self, u_feats, l_feats, l_targets):

        random_state = check_random_state(self.random_state)
        best_inertia = None
        fit_func = self.fit_mix_once

        if effective_n_jobs(self.n_jobs) == 1:
            for it in range(self.n_init):

                labels, inertia, centers, n_iters = fit_func(u_feats, l_feats, l_targets, random_state)

                if best_inertia is None or inertia < best_inertia:
                    self.labels_ = labels.clone()
                    self.cluster_centers_ = centers.clone()
                    best_inertia = inertia
                    self.inertia_ = inertia
                    self.n_iter_ = n_iters

        else:

            # parallelisation of k-means runs
            seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(delayed(fit_func)(u_feats, l_feats, l_targets, seed)
                                                              for seed in seeds)
            # Get results with the lowest inertia

            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            self.labels_ = labels[best]
            self.inertia_ = inertia[best]
            self.cluster_centers_ = centers[best]
            self.n_iter_ = n_iters[best]


