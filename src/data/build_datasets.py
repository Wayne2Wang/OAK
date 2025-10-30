from copy import deepcopy

from .utils import MergedDataset
from .stanford_alm import get_stanford_action_datasets, get_stanford_location_datasets, get_stanford_mood_datasets
from .clevr4 import get_clevr4_texture_datasets, get_clevr4_color_datasets, get_clevr4_count_datasets, get_clevr4_shape_datasets

GET_DATASET_FUNCS = {
    'stanford_action': get_stanford_action_datasets,
    'stanford_location': get_stanford_location_datasets,
    'stanford_mood': get_stanford_mood_datasets,
    'clevr4_texture': get_clevr4_texture_datasets,
    'clevr4_color': get_clevr4_color_datasets,
    'clevr4_count': get_clevr4_count_datasets,
    'clevr4_shape': get_clevr4_shape_datasets,
}


def build_datasets(args):

    dataset_name = args.DATA.NAME

    if dataset_name not in GET_DATASET_FUNCS:
        raise ValueError('Dataset {} not supported'.format(dataset_name))
    
    datasets = GET_DATASET_FUNCS[dataset_name](args)

    # set target transforms
    target_transform_dict = {}
    for i, cls in enumerate(args.train_classes + args.unlabeled_classes):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # train split (labeled and unlabeled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']), unlabelled_dataset=deepcopy(datasets['train_unlabelled']))
    test_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['test_labelled']), unlabelled_dataset=deepcopy(datasets['test_unlabelled']))
    merged_datasets = {
        'train_merged': train_dataset,
        'test_merged': test_dataset,
    }

    return merged_datasets