import os
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from .utils import subsample_instances
from .transforms import get_transforms
from .METADATA import METADATA
METADATA = METADATA['stanford_alm']


def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'JPEGImages', image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file, classes):

    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:])[:-1]) # remove newline character
    f.close()

    # index class names
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] if c in class_to_idx else c for c in targets] # possibly contains -1 classes

    return (image_ids, targets, classes, class_to_idx)


class StanfordALM(Dataset):

    all_taxonomies = METADATA['all_taxonomies']
    train_classes = METADATA['train_classes']

    def __init__(self, root, taxonomy, train=True, transform=None, target_transform=None, loader=default_loader, description=""):

        self.root = os.path.expanduser(root)
        self.taxonomy = taxonomy
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.classes_file = os.path.join(root, '{}.txt'.format(taxonomy))
        
        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file, self.get_classes())
        samples = make_dataset(self.root, image_ids, targets)

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.targets = self.target_ids()
        self.uq_idxs = np.array(range(len(self)))

        # remove classes not recognized by the taxonomy
        keep_idx = [i for i, target in enumerate(self.targets) if isinstance(target, int)]
        subsample_dataset(self, keep_idx)

        # load class name descriptions
        if description:
            self.taxonomy = '{}_{}'.format(self.taxonomy, description)


    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.uq_idxs[index]
    

    def get_by_uq_idx(self, uq_idx):
        idx  = np.where(self.uq_idxs == uq_idx)[0].item()
        if idx is None:
            return None
        return self.__getitem__(idx)
    

    def get_classes(self):
        return StanfordALM.all_taxonomies[self.taxonomy]


    def __len__(self):
        return len(self.samples)
    

    def target_ids(self):
        return [idx for _, idx in self.samples]
    

def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = [dataset.samples[i] for i in range(len(mask)) if mask[i]]
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


def get_stanford_alm_datasets(taxonomy, args):

    train_transform, test_transform = get_transforms(args)
    root = args.DATA.ROOT
    description = args.DATA.DESCRIPTION
    prop_train_labels = args.DATA.PROP_TRAIN
    class_wise = args.DATA.PROP_TRAIN_CLASS_WISE
    data_random_seed = args.DATA.RANDOM_SEED

    # Init entire training set
    whole_training_set = StanfordALM(root, taxonomy, transform=train_transform, train=True, description=description)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_classes = list(StanfordALM.train_classes[taxonomy])
    unlabeled_classes = list(set(range(len(whole_training_set.get_classes()))) - set(train_classes))
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels, class_wise=class_wise, seed=data_random_seed, logger=args.logger)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Get unlabelled data
    unlabelled_uq_idxs = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    unlabelled_indices = [np.where((whole_training_set.uq_idxs==i))[0].item() for i in unlabelled_uq_idxs]
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(unlabelled_indices))

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


def get_stanford_action_datasets(*args):
    return get_stanford_alm_datasets('action', *args)

def get_stanford_location_datasets(*args):
    return get_stanford_alm_datasets('location', *args)

def get_stanford_mood_datasets(*args):
    return get_stanford_alm_datasets('mood', *args)

def get_stanford_location_v2_datasets(*args):
    return get_stanford_alm_datasets('location_v2', *args)

def get_stanford_mood_v2_datasets(*args):
    return get_stanford_alm_datasets('mood_v2', *args)