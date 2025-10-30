import argparse
import sys
import time
from os.path import basename, join

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.utils.general import seed_torch
from src.utils.configs import load_args
from src.data.build_datasets import build_datasets
from src.models import build_model

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")



@torch.no_grad()
def extract_features(model, dataloader, train_class_names, unlabeled_class_names, device):
    model.eval()
    model.to(device)

    features = []
    targets = []
    labeled_mask = []
    known_mask = []
    uq_idxs = []

    start_time = time.time()
    all_class_names = train_class_names + unlabeled_class_names
    for images, class_labels, uq_idx, mask_lab  in tqdm(dataloader):

        images = images.to(device)
        feats = model(images)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        features.append(feats)
        targets.append(class_labels)
        labeled_mask.append(mask_lab.squeeze())
        known_mask.append(torch.tensor([True if x.item() in range(len(train_class_names)) else False for x in class_labels]))
        uq_idxs.append(uq_idx)

    features = torch.concatenate(features, dim=0).to(device)
    targets = torch.concatenate(targets, dim=0).to(device)
    target_names = [all_class_names[x] for x in targets.tolist()]
    labeled_mask = torch.concatenate(labeled_mask, dim=0).to(device).bool()
    known_mask = torch.concatenate(known_mask, dim=0).to(device).bool()
    uq_idxs = torch.cat(uq_idxs, dim=0).to(device)
    
    if logger is not None:
        logger.info('Data processing took {:.2f} seconds'.format(time.time() - start_time))
    
    return features, targets, target_names, labeled_mask, known_mask, uq_idxs



def plot_tsne(features, class_labels, known_mask, labeled_mask, output_path, tsne_random_state=0, override_class_to_color=None, flip=False):
    """
    Plots a t-SNE scatter plot where points are colored by class labels,
    known points are star-shaped, unknown points are circles, and
    labeled points have a thicker outer line.

    Args:
    - features (numpy.ndarray): A tensor of shape (N, d) containing the extracted features.
    - class_labels (numpy.ndarray): A tensor of shape (N,) containing class labels for each point.
    - known_mask (numpy.ndarray): A tensor of shape (N,) containing binary values (0 or 1) indicating if the point is known.
    - labeled_mask (numpy.ndarray): A tensor of shape (N,) containing binary values (0 or 1) indicating if the point is labeled.
    """

    # check if all labeled points are known
    known_mask = np.array(known_mask).astype(bool)
    labeled_mask = np.array(labeled_mask).astype(bool)
    assert np.all(known_mask[labeled_mask]), "Some labeled points are unknown!"

    # compute t-SNE embedding
    if features.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=tsne_random_state)
        features_2d = tsne.fit_transform(features)
    else:
        features_2d = features

    # get unique classes and assign colors
    unique_classes = np.unique(class_labels)
    num_classes = len(unique_classes)
    cmap = plt.get_cmap('tab20', num_classes)
    class_to_color = {cls_name: cmap(i) for i, cls_name in enumerate(unique_classes)}
    if not override_class_to_color is None:
        class_to_color = override_class_to_color

    # prepare data
    x = features_2d[:, 0] if not flip else features_2d[:, 0] * -1
    y = features_2d[:, 1] if not flip else features_2d[:, 0] * -1
    colors = np.array([class_to_color[cls_name] for cls_name in class_labels])

    # plot data points without labels to prevent them from appearing in legends
    plt.figure(figsize=(10, 10))
    for known in [True, False]:
        for labeled in [True, False]:
            mask = (known_mask == known) & (labeled_mask == labeled)
            if np.any(mask):
                # Define marker style
                marker = '*' if known else '.'
                edgecolors = 'red' if labeled else 'none'
                linewidths = 1 if labeled else 0
                zorder = 1 if known else 0
                zorder += 1 if labeled else 0
                plt.scatter(x[mask],y[mask],c=colors[mask],marker=marker,edgecolors=edgecolors,linewidths=linewidths,s=100,label='_nolegend_', zorder=zorder)

    # Create custom legend entries for markers and edge styles using dummy scatter plots
    marker_handles = []
    scatter_known_labeled = plt.scatter([], [], marker='*', color='black', edgecolor='red', linewidths=1, s=100, label='Known Labeled')
    marker_handles.append(scatter_known_labeled)
    scatter_known_unlabeled = plt.scatter([], [], marker='*', color='black', edgecolor='none', linewidths=0, s=100, label='Known Unlabeled')
    marker_handles.append(scatter_known_unlabeled)
    scatter_unknown_unlabeled = plt.scatter([], [], marker='.', color='black', edgecolor='none', linewidths=0, s=100, label='Novel')
    marker_handles.append(scatter_unknown_unlabeled)
    marker_legend = plt.legend(handles=marker_handles, title='Markers', bbox_to_anchor=(0.8, 0.98), loc='upper left', borderaxespad=0, fontsize='small')
    plt.gca().add_artist(marker_legend)

    # Create custom legend entries for classes using dummy scatter plots
    known_classes = set([class_labels[i] for i in np.where(known_mask)[0]])
    class_handles = []
    for cls_name in unique_classes:
        marker = '*' if cls_name in known_classes else '.'
        handle = plt.scatter([], [], color=class_to_color[cls_name], marker=marker, edgecolors='none', linewidths=0, s=100, label=str(cls_name))
        class_handles.append(handle)
    ncol = 1 if num_classes <= 40 else int(np.ceil(num_classes / 40))
    plt.legend(handles=class_handles, title='Classes', bbox_to_anchor=(1, 0.98), loc='upper left', borderaxespad=0, ncol=ncol, fontsize='small')

    plt.grid(False)
    plt.gca().set_axis_off()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close('all')

    return features_2d, class_to_color



def main():
    parser = argparse.ArgumentParser()
    # key args
    parser.add_argument("config", type=str, help="Config file")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--eval_path", type=str, default="", help="Path to evaluation data")
    parser.add_argument("--output", type=str, default="", help="Path to output t-SNE plot")
    # override config file
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    cli_args = parser.parse_args()
    args = load_args(cli_args.config, cli_opts=cli_args.opts)
    args.logger = logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.SEED = int(cli_args.seed)
    if args.SEED >= 0:
        seed_torch(args.SEED)

    # build dataset and model
    dataset = build_datasets(args)['test_merged']
    dataloader = DataLoader(dataset, num_workers=args.DATA.NUM_WORKERS, batch_size=args.DATA.BATCH_SIZE, shuffle=False)
    model = build_model(args)
    ckpt = torch.load(cli_args.eval_path, map_location="cpu")
    msg = model.load_state_dict(ckpt, strict=True)
    args.logger.info('Loaded model from {}'.format(cli_args.eval_path))
    args.logger.info(msg)

    # extract features
    features, targets, target_names, labeled_mask, known_mask, uq_idxs = extract_features(model, dataloader, args.train_class_names, args.unlabeled_class_names, device)

    # t-SNE visualization
    output_path = cli_args.output if cli_args.output else join('tsne_{}.png'.format(basename(cli_args.eval_path).split('.')[0]))
    plot_tsne(features.cpu().numpy(), target_names, known_mask.cpu().numpy(), labeled_mask.cpu().numpy(), output_path)
    logger.info('t-SNE plot saved to {}'.format(output_path))

if __name__ == "__main__":
    main()


