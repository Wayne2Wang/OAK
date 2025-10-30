from collections import Counter

import numpy as np
from torch.utils.data import Dataset


def subsample_instances(dataset, prop_indices_to_subsample, class_wise, seed, logger):

    np.random.seed(seed)
    
    if not class_wise:

        logger.debug('Subsampling instances WITHOUT class balance, prop_indices_to_subsample ({}) is applied to the whole dataset. Seed is {}.'.format(prop_indices_to_subsample, seed))
        size = int(prop_indices_to_subsample * len(dataset)) if prop_indices_to_subsample < 1 else int(prop_indices_to_subsample)
        subsample_indices = np.random.choice(range(len(dataset)), replace=False, size=(size,))
    
    else:

        logger.debug('Subsampling instances WITH class balance, prop_indices_to_subsample ({}) is applied to each class. Seed is {}'.format(prop_indices_to_subsample, seed))
        subsample_indices = []
        class_counts = dict(Counter(dataset.targets))
        for cls in class_counts.keys():
            cls_idxs = np.where(np.array(dataset.targets) == cls)[0]
            size = int(prop_indices_to_subsample * len(cls_idxs)) if prop_indices_to_subsample < 1 else int(prop_indices_to_subsample)
            subsample_indices.extend(np.random.choice(cls_idxs, replace=False, size=(size,)))

    return subsample_indices


class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
