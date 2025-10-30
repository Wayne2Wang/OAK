import os
import json
from copy import deepcopy

import numpy as np
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from .utils import subsample_instances
from .transforms import get_transforms
from .METADATA import METADATA
METADATA = METADATA['clevr4']


class CLEVR4(Dataset):

    all_taxonomies = METADATA['all_taxonomies']
    train_classes = METADATA['train_classes']

    def __init__(self, root, taxonomy, train=True, transform=None, target_transform=None, loader=default_loader, description=""):

        self.root = os.path.expanduser(root)
        self.image_root = os.path.join(self.root, 'images')
        self.taxonomy = taxonomy
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.split = 'train' if train else 'val'

        # load annotations
        annot_path = os.path.join(root, 'clevr_4_annots.json')
        with open(annot_path, 'r') as f:
            self.annotations = json.load(f)
        self.class_name_to_label = {name: idx for idx, name in enumerate(self.all_taxonomies[taxonomy])}

        # list files
        self.filenames = sorted([fname for fname, meta in self.annotations.items() if meta["split"] == self.split])

        self.targets = self.target_ids()
        self.uq_idxs = np.array(range(len(self)))

        # load class name descriptions
        if description:
            self.taxonomy = '{}_{}'.format(self.taxonomy, description)


    def __len__(self):
        return len(self.filenames)
    

    def __getitem__(self, index):
        
        # load image
        fname = self.filenames[index]
        img_path = os.path.join(self.image_root, f"{fname}.png")
        image = self.loader(img_path)

        # load target
        class_name = str(self.annotations[fname][self.taxonomy])
        target = self.class_name_to_label[class_name]

        # transformations
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target, self.uq_idxs[index]
    

    def get_by_uq_idx(self, uq_idx):
        idx  = np.where(self.uq_idxs == uq_idx)[0].item()
        if idx is None:
            return None
        return self.__getitem__(idx)
    

    def get_classes(self):
        return CLEVR4.all_taxonomies[self.taxonomy]
    
    
    def _get_all_taxonomy_targets(self, taxonomy, return_class_names=False):
        
        class_name_to_label = {
            name: idx for idx, name in enumerate(self.all_taxonomies[taxonomy])
        }

        all_targets = []
        for fname in self.filenames:
            class_name = str(self.annotations[fname][taxonomy])
            if return_class_names:
                target = class_name
            else:
                target = class_name_to_label[class_name]
            all_targets.append(target)
        
        return all_targets

    def target_ids(self):
        return self._get_all_taxonomy_targets(self.taxonomy)


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.filenames = [dataset.filenames[i] for i in range(len(mask)) if mask[i]]
    dataset.targets = [dataset.targets[i] for i in range(len(mask)) if mask[i]]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=list(range(5))):

    cls_idxs = [x for x, t in enumerate(dataset.target_ids()) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset



def get_clevr4_datasets(taxonomy, args):

    train_transform, test_transform = get_transforms(args)
    root = args.DATA.ROOT
    description = args.DATA.DESCRIPTION
    prop_train_labels = args.DATA.PROP_TRAIN
    class_wise = args.DATA.PROP_TRAIN_CLASS_WISE
    data_random_seed = args.DATA.RANDOM_SEED
    
    # Init entire training set
    whole_training_set = CLEVR4(root, taxonomy, transform=train_transform, train=True, description=description)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_classes = list(CLEVR4.train_classes[taxonomy])
    unlabeled_classes = list(set(range(len(whole_training_set.get_classes()))) - set(train_classes))
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels, class_wise=class_wise, seed=data_random_seed, logger=args.logger)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Get unlabelled data
    unlabelled_uq_idxs = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    unlabelled_indices = [np.where((whole_training_set.uq_idxs==i))[0].item() for i in unlabelled_uq_idxs]
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Log classes
    args.train_classes = train_classes
    args.unlabeled_classes = unlabeled_classes
    args.logger.debug('Training classes and unlabeled classes (class ids starting from 0) added to args')
    args.train_class_names = [whole_training_set.get_classes()[i] for i in train_classes]
    args.unlabeled_class_names = [whole_training_set.get_classes()[i] for i in unlabeled_classes]
    args.logger.debug('Training class names and unlabeled class names (str) added to args')
    args.logger.debug('Training classes: {}'.format(args.train_class_names))
    args.logger.debug('Unlabelled classes: {}'.format(args.unlabeled_class_names))
    args.logger.debug('Number of training samples: {}'.format(len(train_dataset_labelled)))
    args.logger.debug('Number of unlabelled samples: {}'.format(len(train_dataset_unlabelled)))

    # Set test transform for evaluation (following GCD convention: evaluation is done on unlabeled training data)
    test_dataset_labelled = deepcopy(train_dataset_labelled)
    test_dataset_labelled.transform = test_transform
    test_dataset_unlabelled = deepcopy(train_dataset_unlabelled)
    test_dataset_unlabelled.transform = test_transform

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'test_labelled': test_dataset_labelled,
        'test_unlabelled': test_dataset_unlabelled,
    }

    return all_datasets


def get_clevr4_texture_datasets(args):
    return get_clevr4_datasets('texture', args)

def get_clevr4_color_datasets(args):
    return get_clevr4_datasets('color', args)

def get_clevr4_count_datasets(args):
    return get_clevr4_datasets('count', args)

def get_clevr4_shape_datasets(args):
    return get_clevr4_datasets('shape', args)