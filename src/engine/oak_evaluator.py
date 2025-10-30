import time

import numpy as np
import torch
from sklearn.metrics import silhouette_score

from .cluster_eval import log_accs_from_preds, GCDSemiKMeans
from src.utils.configs import namespace_to_dict, convert_keys_to_lowercase


class OAKEvaluator():
            
    def __init__(self, model, merged_loader, train_class_names, unlabeled_class_names, cluster_methods, eval_funcs, device, logger=None):
        self.model = model
        self.merged_loader = merged_loader
        self.train_class_names = train_class_names
        self.unlabeled_class_names = unlabeled_class_names
        self.all_class_names = train_class_names + unlabeled_class_names
        self.cluster_methods = cluster_methods
        self.eval_funcs = eval_funcs
        self.device = device
        self.logger = logger

        self.data_processed = False
        self.features = None
        self.targets = None
        self.labeled_mask = None
        self.known_mask = None


    @torch.no_grad()
    def process_data(self):
        self.model.eval()
        self.model.to(self.device)

        self.features = []
        self.targets = []
        self.labeled_mask = []
        self.known_mask = []
        self.uq_idxs = []

        start_time = time.time()
        for images, class_labels, uq_idx, mask_lab  in self.merged_loader:

            images = images.to(self.device)
            feats = self.model(images)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            self.features.append(feats)
            self.targets.append(class_labels)
            self.labeled_mask.append(mask_lab.squeeze())
            self.known_mask.append(torch.tensor([True if x.item() in range(len(self.train_class_names)) else False for x in class_labels]))
            self.uq_idxs.append(uq_idx)

        self.features = torch.concatenate(self.features, dim=0).to(self.device)
        self.targets = torch.concatenate(self.targets, dim=0).to(self.device)
        self.target_names = [self.all_class_names[x] for x in self.targets.tolist()]
        self.labeled_mask = torch.concatenate(self.labeled_mask, dim=0).to(self.device).bool()
        self.known_mask = torch.concatenate(self.known_mask, dim=0).to(self.device).bool()
        self.uq_idxs = torch.cat(self.uq_idxs, dim=0).to(self.device)
        
        if self.logger is not None:
            self.logger.info('Data processing took {:.2f} seconds'.format(time.time() - start_time))

        self.data_processed = True


    def sskmeans(features, targets, labeled_mask, **overide_kwargs):
        global_mean = np.mean(features.cpu().numpy(), axis=0)
        total_variance = np.sum((features.cpu().numpy() - global_mean) ** 2)
        n_clusters = len(set(targets.tolist()))
        cluster = GCDSemiKMeans(k=n_clusters, **overide_kwargs)
        cluster.fit_mix(features[~labeled_mask], features[labeled_mask], targets[labeled_mask])
        labeled_unlabeled_preds, inertia = cluster.labels_, cluster.inertia_ # (labeled_preds, unlabeled_preds)
        sil_score = silhouette_score(features.cpu().numpy(), labeled_unlabeled_preds.cpu().numpy())
        preds = torch.zeros_like(targets)
        preds[labeled_mask] = labeled_unlabeled_preds[:labeled_mask.sum()]
        preds[~labeled_mask] = labeled_unlabeled_preds[labeled_mask.sum():]
        normalized_inertia = inertia / total_variance
        return preds.cpu().numpy(), normalized_inertia, sil_score


    def evaluate(self, epoch=-1, save_name='Test', writer=None, overide_cluster_kwargs=None):
        if not self.data_processed:
            self.process_data()

        all_results = {}
        for method in self.cluster_methods:
            start_time = time.time()
            if overide_cluster_kwargs is not None:
                overide_cluster_kwargs = convert_keys_to_lowercase(namespace_to_dict(overide_cluster_kwargs))
                kwargs = overide_cluster_kwargs[method]
            else:
                kwargs = {}
            preds, inertia, sil_score = OAKEvaluator.__dict__[method](self.features, self.targets, self.labeled_mask, **kwargs)
            
            if self.logger is not None:
                self.logger.info('{} took {:.2f} seconds'.format(method, time.time() - start_time))
                self.logger.info('Inertia: {:.6f}'.format(inertia))
                self.logger.info('Silhouette Score: {:.6f}'.format(sil_score))
            
            # only unlabeled examples are evaluated although both are used for clustering
            all_results[method] = log_accs_from_preds(
                y_true=self.targets[~self.labeled_mask].cpu().numpy(), 
                y_pred=preds[(~self.labeled_mask).cpu().numpy()], 
                mask=self.known_mask[~self.labeled_mask].cpu().numpy(),
                T=epoch, eval_funcs=self.eval_funcs, save_name=save_name+"_"+method, 
                writer=writer, logger=self.logger, return_ind_map=True
            )

            # find predicted class names7 constrained text inversion
            gt_to_pred_map = all_results[method][3]
            pred_to_gt_map = {v: k for k, v in gt_to_pred_map.items()}
            pred_class_names = [self.all_class_names[pred_to_gt_map[x]] for x in preds]
            all_results[method] += ({i.item():j for i, j in zip(self.uq_idxs, pred_class_names)},)
            all_results[method] += ({i.item():j for i, j in zip(self.uq_idxs, self.target_names)}, inertia, sil_score)
        
        return all_results # acc_all, acc_old, acc_new, gt_to_pred_map, uq_idx_to_pred_class_name, uq_idx_to_gt_class_name, inertia, sil_score