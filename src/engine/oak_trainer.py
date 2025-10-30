import os
from os.path import join
import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models import build_model, DINOHead
from src.models.clip import get_zeroshot_weights
from src.data import build_datasets, ContrastiveLearningViewGenerator, CHATGPT_PROPOSALS
from src.utils.general import AverageMeter, get_mean_lr
from .losses import SupConLoss, info_nce_logits
from .oak_evaluator import OAKEvaluator
from .cluster_eval import display_format_results



class OAKTrainer():

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.logger = args.logger
        self.epoch = -1
        self.epochs = int(args.TRAINER.EPOCHS)
        self.eval_funcs = args.EVALUATOR.EVAL_FUNCS
        self.log_every_n = int(args.TRAINER.LOG_EVERY_N)
        self.n_views = int(args.TRAINER.N_VIEWS)
        self.sup_con_weight = float(self.args.TRAINER.SUP_CON_WEIGHT)
        self.text_reg_weight = float(self.args.TRAINER.TEXT_REG_WEIGHT)
        self.text_inv_weight = float(self.args.TRAINER.TEXT_INV_WEIGHT)

        # initialize model
        self.model_variant = args.MODEL.VARIANT
        self.model = build_model(args).to(device)

        # initialize projection head
        proj_head_kwargs = {
            'in_dim': int(args.MODEL.FEAT_DIM), 
            'out_dim': int(args.TRAINER.MLP_PROJ.OUT_DIM),
            'nlayers': int(args.TRAINER.MLP_PROJ.NUM_LAYERS)
        }
        self.projection_head = DINOHead(**proj_head_kwargs).to(device)

        # initialize dataset
        merged_datasets = build_datasets(args)
        self.train_dataset = merged_datasets['train_merged']
        self.test_dataset = merged_datasets['test_merged']

        # initialize text embeddings for regularization and inversion
        if self.text_reg_weight > 0:
            self.text_reg = get_zeroshot_weights(
                self.model_variant, 
                self.args.train_class_names, 
                templates='imagenet', 
                device=self.device
            ).float()
        if args.TRAINER.TEXT_INV_VOCAB == 'ChatGPT':
            self.text_inv_vocab = CHATGPT_PROPOSALS[args.DATA.NAME]
        else:
            raise ValueError('Unexpected text inversion vocabulary!')
        self.text_inv = get_zeroshot_weights(
            self.model_variant,
            self.text_inv_vocab,
            templates='imagenet',
            device=self.device
        ).float()

        # change to contrastive learning view
        self.train_dataset.labelled_dataset.transform = ContrastiveLearningViewGenerator(
            base_transform=self.train_dataset.labelled_dataset.transform, 
            n_views=self.n_views
        )
        self.train_dataset.unlabelled_dataset.transform = ContrastiveLearningViewGenerator(
            base_transform=self.train_dataset.unlabelled_dataset.transform,
            n_views=self.n_views
        )

        # sampler
        label_len = len(self.train_dataset.labelled_dataset)
        unlabelled_len = len(self.train_dataset.unlabelled_dataset)
        sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(self.train_dataset))]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(self.train_dataset))

        # dataloader
        num_workers = args.DATA.NUM_WORKERS
        self.batch_size = args.DATA.BATCH_SIZE
        self.train_loader = DataLoader(
            self.train_dataset, num_workers=num_workers, batch_size=self.batch_size,
            shuffle=False, sampler=sampler, drop_last=True
        )
        self.test_loader = DataLoader(self.test_dataset, num_workers=num_workers, batch_size=self.batch_size, shuffle=False)

        # initialize optimizer
        optimizer_kwargs = {
            'lr': float(args.TRAINER.SOLVER.LR),
            'momentum': float(args.TRAINER.SOLVER.MOMENTUM),
            'weight_decay': float(args.TRAINER.SOLVER.WEIGHT_DECAY)
        }
        self.optimizer = optim.__dict__[args.TRAINER.SOLVER.NAME](
            list(self.projection_head.parameters()) + list(self.model.parameters()),
            **optimizer_kwargs
        )

        # initialize lr scheduler
        lr_scheduler_kwargs = {
            'T_max': int(args.TRAINER.EPOCHS),
            'eta_min': float(args.TRAINER.SOLVER.LR) * 1e-3
        }
        self.lr_scheduler = optim.lr_scheduler.__dict__[args.TRAINER.SCHEDULER.NAME](
            self.optimizer, 
            **lr_scheduler_kwargs
        )

        # initialize loss function
        self.sup_con_loss = SupConLoss()

        # initialize writer
        self.writer = SummaryWriter(join(args.OUTPUT_DIR, 'tensorboard'))


    def train(self):

        self.epoch = -1        
        _, _, _, _ , text_inv_map, _, _, _ = self.evaluate(override_unlabeled_class_names=self.text_inv_vocab)['sskmeans']
        best_sil_score = float('-inf')
        self.logger.info('Starting training...\n')

        for epoch in range(self.epochs):
            self.epoch = epoch

            meters = {
                'total_loss': AverageMeter(),
                'sup_con_loss': AverageMeter(),
                'contrastive_loss': AverageMeter(),
                'contrastive_acc': AverageMeter(),
                'text_reg_loss': AverageMeter(),
                'text_inv_loss': AverageMeter()
            }

            self.projection_head.train()
            self.model.train()

            for batch_idx, batch in enumerate(self.train_loader):

                images, class_labels, uq_idxs, mask_lab = batch
                mask_lab = mask_lab[:, 0]

                class_labels, mask_lab = class_labels.to(self.device), mask_lab.to(self.device).bool()
                images = torch.cat(images, dim=0).to(self.device)

                # Extract features with base model
                model_features = self.model(images)

                # Pass features through projection head
                features = self.projection_head(model_features)

                # L2-normalize features
                features = torch.nn.functional.normalize(features, dim=-1)

                # Choose which instances to run the contrastive loss on
                if self.args.TRAINER.CONTRAST_UNLABEL_ONLY:
                    # Contrastive loss only on unlabelled instances
                    f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                    con_feats = torch.cat([f1, f2], dim=0)
                else:
                    # Contrastive loss for all examples
                    con_feats = features

                contrastive_logits, contrastive_labels = info_nce_logits(
                    features=con_feats, 
                    n_views=self.n_views,
                    temperature=float(self.args.TRAINER.TEMPERATURE), 
                    device=self.device
                )
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # Supervised contrastive loss
                f1, f2 = [f[mask_lab] for f in features.chunk(2)]
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = self.sup_con_loss(sup_con_feats, labels=sup_con_labels)

                # labeled text embedding regularization
                if self.text_reg_weight > 0:
                    model_features = torch.nn.functional.normalize(model_features, dim=-1)
                    mf1, mf2 = [f[mask_lab] for f in model_features.chunk(2)]
                    logit_scale = torch.ones([]) * np.log(1 / 0.07)
                    logits_per_image1 = logit_scale * mf1 @ self.text_reg
                    logits_per_image2 = logit_scale * mf2 @ self.text_reg
                    text_reg_loss = (torch.nn.CrossEntropyLoss()(logits_per_image1, sup_con_labels) + torch.nn.CrossEntropyLoss()(logits_per_image2, sup_con_labels)) / 2
                else:
                    text_reg_loss = sup_con_loss.new_zeros(1)

                # unlabeled text inversion regularization
                if self.text_inv_weight > 0:
                    model_features = torch.nn.functional.normalize(model_features, dim=-1)
                    mf1, mf2 = [f[~mask_lab] for f in model_features.chunk(2)]
                    text_inv_result = [text_inv_map[i.item()] for i in uq_idxs[~mask_lab.cpu()]]
                    text_inv_classes = list(set(text_inv_result))
                    text_inv_labels = torch.tensor([text_inv_classes.index(i) for i in text_inv_result]).to(self.device)
                    text_inv = get_zeroshot_weights(self.model_variant, text_inv_classes, templates='imagenet', device=self.device).float()
                    logit_scale = torch.ones([]) * np.log(1 / 0.07)
                    logits_per_image1 = logit_scale * mf1 @ text_inv
                    logits_per_image2 = logit_scale * mf2 @ text_inv
                    text_inv_loss = (torch.nn.CrossEntropyLoss()(logits_per_image1, text_inv_labels) + torch.nn.CrossEntropyLoss()(logits_per_image2, text_inv_labels)) / 2
                else:
                    text_inv_loss = sup_con_loss.new_zeros(1)

                # Total loss
                total_loss = contrastive_loss + self.sup_con_weight * sup_con_loss + self.text_reg_weight * text_reg_loss + self.text_inv_weight * text_inv_loss

                # backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # contrastive acc
                _, contrastive_pred = contrastive_logits.max(1)
                contrastive_acc = (contrastive_pred == contrastive_labels).float().mean()

                # update meters
                for key, meter in meters.items():
                    meter.update(locals()[key].item(), class_labels.size(0))

                # per-batch logging
                if batch_idx % self.log_every_n == 0:
                    self.logger.info('Train Epoch {} [{}/{}]: {}'.format(epoch, batch_idx * self.batch_size, len(self.train_loader.dataset), ' | '.join(['{}: {:.4f}'.format(k, v.val) for k, v in meters.items()])))

            # Step schedule
            self.lr_scheduler.step()

            # per-epoch logging
            self.logger.info('Train Epoch {} average: {} '.format(epoch, ' | '.join(['{}: {:.4f}'.format(k, v.avg) for k, v in meters.items()])))
            for k, v in meters.items():
                self.writer.add_scalar(k, v.avg, epoch)
            self.writer.add_scalar('LR', get_mean_lr(self.optimizer), epoch)

            # save model
            model_path = join(self.args.OUTPUT_DIR, 'model.pt')
            torch.save(self.model.state_dict(), model_path)
            self.logger.info("Model saved to {}.".format(model_path))
            proj_head_path = join(self.args.OUTPUT_DIR, 'proj_head.pt')
            torch.save(self.projection_head.state_dict(), proj_head_path)
            self.logger.info("Projection head saved to {}.".format(proj_head_path))

            # Evaluation
            self.logger.info('Running on the whole dataset ...')
            logger = self.logger if epoch == self.epochs - 1 else None
            override_unlabeled_class_names = None if epoch == self.epochs - 1 else self.text_inv_vocab
            eval_result = self.evaluate(override_unlabeled_class_names=override_unlabeled_class_names, logger=logger)['sskmeans']
            _, _, _, _, text_inv_map, _, _, sil_score  = eval_result
            results_path = join(self.args.OUTPUT_DIR, 'results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(eval_result, f)


            if sil_score > best_sil_score:

                with open(results_path[:-4]+'_best.pkl', 'wb') as f:
                    pickle.dump(eval_result, f)
                self.logger.info("New best sil_score! Results saved to {}.".format(results_path[:-4]+'_best.pkl'))

                torch.save(self.model.state_dict(), model_path[:-3] + '_best.pt')
                self.logger.info("New best sil_score! Model saved to {}.".format(model_path[:-3] + '_best.pt'))

                torch.save(self.projection_head.state_dict(), proj_head_path[:-3] + '_best.pt')
                self.logger.info("New best sil_score! Projection head saved to {}.".format(proj_head_path[:-3] + '_best.pt'))

                self.logger.info('New best sil_score: {}'.format(sil_score))

                best_sil_score = sil_score

            self.logger.info('End of epoch {}\n'.format(epoch))

    
    def evaluate(self, override_unlabeled_class_names=None, logger=None):
        unlabeled_class_name = self.args.unlabeled_class_names if override_unlabeled_class_names is None else override_unlabeled_class_names
        evaluator = OAKEvaluator(
            self.model, self.test_loader, self.args.train_class_names, unlabeled_class_name,
            self.args.EVALUATOR.CLUSTER_METHODS, self.args.EVALUATOR.EVAL_FUNCS, self.device, logger=logger
        )
        results = evaluator.evaluate(
            epoch=self.epoch, writer=self.writer, overide_cluster_kwargs=self.args.EVALUATOR
        )
        return results