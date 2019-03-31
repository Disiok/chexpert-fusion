#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import time
import torch
import random
import shutil
import numpy as np
import tensorboardX

import models
import criteria

from data.utils import make_dataloader

from lib.utils import to_numpy
from lib.utils import to_device

from metrics.pr_meter import PRMeter
from metrics.auc_meter import AUCMeter

from config.chexpert import CHEXPERT_CLASSES, PAPER_TRAINING_CLASSES


__all__ = [

]


class Trainer(object):

    def __init__(self, config):
        """



        """
        super(Trainer, self).__init__()
        torch.set_default_tensor_type('torch.FloatTensor')
        self._set_seed(config['general']['seed'])

        self.config = config
        self.device = torch.device('cpu')
        self.use_cuda = self.config['general']['use_cuda']
        self.cuda_benchmark = self.config['general']['cuda_benchmark']

        # TODO(suo): Migrate this to use registry
        self.classes = {
            'default': CHEXPERT_CLASSES,
            'paper': PAPER_TRAINING_CLASSES,
        }[config['general']['classes']]
        self.num_epochs = self.config['general']['num_epochs']

        self.train_epoch = 0
        self.val_epoch = 0
        self.global_train_step = 0
        self.global_val_step = 0
        self.current_best_metric = -float('inf')

        self._configure_io()
        self._configure_cuda()
        self._configure_metrics()

        self._configure_model()
        self._configure_criterion()
        self._configure_optimizer()
        self._configure_dataloaders()

        if self.config['general']['checkpoint'] is not None:
            self._load_checkpoint(self.config['general']['checkpoint'])

        self._save_code()

    def _set_seed(self, seed):
        """


        """
        if seed is None:
            return

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _save_code(self):
        """


        """
        if not self.code_path:
            return

        # NOTE(suo): Trick to get n-th parent directory
        uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
        ROOT_DIR = uppath(__file__, 4)

        destination = os.path.join(self.code_path, os.path.basename(ROOT_DIR))
        if os.path.exists(destination):
            shutil.rmtree(destination)

        shutil.copytree(ROOT_DIR, destination)

    def _configure_io(self):
        """


        """
        outdir = os.path.abspath(self.config['io']['outdir'])
        self.code_path = os.path.join(outdir, 'code')
        self.logs_path = os.path.join(outdir, 'tensorboard')
        self.models_path = os.path.join(outdir, 'models')
        self.predictions_path = os.path.join(outdir, 'predictions')

        self.log_frequency = self.config['io']['log_frequency']
        self.val_frequency = self.config['io']['val_frequency']
        self.save_predictions = self.config['io']['save_predictions']

        if not os.path.exists(self.code_path):
            os.makedirs(self.code_path)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        if not os.path.exists(self.predictions_path):
            os.makedirs(self.predictions_path)

        self.summary_writer = tensorboardX.SummaryWriter(log_dir=self.logs_path)

    def _configure_cuda(self):
        """


        """
        if not self.use_cuda:
            return

        assert(torch.cuda.is_available())
        self.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = self.cuda_benchmark

    def _configure_metrics(self):
        """


        """
        self.pr_meter = PRMeter(self.classes)
        self.auc_meter = AUCMeter(self.classes)

    def _configure_model(self):
        """


        """
        self.model = models.registry.MODELS[self.config['model']['class']](self.config)
        self.model = self.model.to(self.device)

    def _configure_criterion(self):
        """


        """
        self.criterion = criteria.registry.CRITERIA[self.config['criterion']['class']](self.config)
        self.criterion = self.criterion.to(self.device)

    def _configure_optimizer(self):
        """


        """
        self.optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=self.config['optimizer']['learning_rate'],
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            self.config['optimizer']['milestones'],
            self.config['optimizer']['lr_decay'],
        )

    def _configure_dataloaders(self):
        """


        """
        self.train_dataloader = make_dataloader(self.config, mode='train')
        self.val_dataloader = make_dataloader(self.config, mode='valid')

    def _load_checkpoint(self, checkpoint_fn):
        """


        """
        checkpoint_fn = os.path.abspath(checkpoint_fn)
        checkpoint = torch.load(checkpoint_fn)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        self.val_epoch = checkpoint['val_epoch']
        self.train_epoch = checkpoint['train_epoch']
        self.global_val_step = checkpoint['global_val_step']
        self.global_train_step = checkpoint['global_train_step']
        self.current_best_metric = checkpoint['current_best_metric']

    def _save_checkpoint(self, current_metric=-float('inf')):
        """


        """
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        scheduler_state = self.scheduler.state_dict()

        fn = 'model_{}.pth.tar'.format(self.global_train_step)
        fn = os.path.join(self.models_path, fn)
        print('Saving checkpoint {}.'.format(fn))

        is_best = self.current_best_metric <= current_metric
        self.current_best_metric = max(self.current_best_metric, current_metric)

        torch.save({
            'model_state'        : model_state,
            'optimizer_state'    : optimizer_state,
            'scheduler_state'    : scheduler_state,
            'train_epoch'        : self.train_epoch,
            'val_epoch'          : self.val_epoch,
            'global_train_step'  : self.global_train_step,
            'global_val_step'    : self.global_val_step,
            'current_best_metric': self.current_best_metric,
        }, fn)

        if is_best:
            print('Saving best checkpoint {}.'.format(fn))
            best_fn = os.path.join(self.models_path, 'model_best.pth.tar')
            shutil.copyfile(fn, best_fn)

    def train(self):
        """


        """
        while self.train_epoch < self.num_epochs:
            self.scheduler.step()
            self.train_single_epoch()
            self.val_single_epoch()

    def evaluate(self):
        """


        """
        self.val_single_epoch()

    def train_single_epoch(self):
        """



        """
        for (step, batch) in enumerate(self.train_dataloader, 1):
            if batch is None:
                print('Encountered empty batch.')
                continue

            try:
                metrics = self._train_single_step(batch)
                self._log_train_step(step, metrics)
                self.global_train_step += 1
            except RuntimeError as e:
                print('WARNING: encountered exception {}'.format(str(e)))
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue

            if self.global_train_step % self.val_frequency == 0:
                self.val_single_epoch()

        self.train_epoch += 1

    def _train_single_step(self, batch):
        """



        """
        start_step = time.time()

        self.model.train()
        self.optimizer.zero_grad()

        frontal = to_device(batch['frontal'], self.device)
        lateral = to_device(batch['lateral'], self.device)
        labels = to_device(batch['labels'], self.device)
        mask = to_device(batch['mask'], self.device)

        start_forward = time.time()
        logits = self.model(frontal, lateral)
        loss = self.criterion(logits, labels, mask)
        end_forward = time.time()

        start_backward = time.time()
        loss.backward()
        self.optimizer.step()
        end_backward = time.time()

        end_step = time.time()

        meta = {}
        meta['train_time'] = end_step - start_step
        meta['forward_time'] = end_forward - start_forward
        meta['backward_time'] = end_backward - start_backward
        meta['learning_rate'] = self.optimizer.param_groups[-1]['lr']
        meta['memory_allocated'] = torch.cuda.memory_allocated()

        metrics = {}
        metrics['loss'] = loss.item()
        return {'meta': meta, 'metrics': metrics}

    def _log_train_step(self, step, metrics):
        """



        """
        total_steps = len(self.train_dataloader)
        log = 'Train Epoch {} [{}/{}]: '.format(self.train_epoch, step, total_steps)
        log += 'Loss - {:.4f} '.format(metrics['metrics']['loss'])
        log += 'Train Time - {:.4f} '.format(metrics['meta']['train_time'])
        log += 'Forward Time - {:.4f} '.format(metrics['meta']['forward_time'])
        log += 'Backward Time - {:.4f} '.format(metrics['meta']['backward_time'])
        log += 'Learning Rate - {:.4f} '.format(metrics['meta']['learning_rate'])
        log += 'Memory - {} GB '.format(metrics['meta']['memory_allocated'] / 1e9)
        print(log)

        for (key, metric) in metrics['metrics'].items():
            name = 'train-metrics/{}'.format(key)
            self.summary_writer.add_scalar(name, metric, self.global_train_step)

        for (key, metric) in metrics['meta'].items():
            name = 'train-meta/{}'.format(key)
            self.summary_writer.add_scalar(name, metric, self.global_train_step)

    def val_single_epoch(self):
        """



        """
        self.pr_meter.reset()
        self.auc_meter.reset()

        for (step, batch) in enumerate(self.val_dataloader, 1):
            if batch is None:
                print('Encountered empty batch.')
                continue

            try:
                with torch.no_grad():
                    metrics = self._val_single_step(batch)
                    self._log_val_step(step, metrics)
                    self.global_val_step += 1
            except RuntimeError as e:
                print('WARNING: encountered exception {}'.format(str(e)))
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        self.val_epoch += 1
        self._log_val_epoch()
        self._save_checkpoint(self.auc_meter.values()['mean'])

    def _val_single_step(self, batch):
        """


        """
        start_step = time.time()

        self.model.eval()

        frontal = to_device(batch['frontal'], self.device)
        lateral = to_device(batch['lateral'], self.device)
        labels = to_device(batch['labels'], self.device)
        mask = to_device(batch['mask'], self.device)

        start_forward = time.time()
        logits = self.model(frontal, lateral)
        loss = self.criterion(logits, labels, mask)
        end_forward = time.time()

        mask = to_numpy(batch['mask'])
        labels = to_numpy(batch['labels'].long())
        scores = to_numpy(torch.sigmoid(logits))

        self.pr_meter.add_predictions(mask, scores, labels)
        self.auc_meter.add_predictions(mask, scores, labels)

        end_step = time.time()

        meta = {}
        meta['val_time'] = end_step - start_step
        meta['forward_time'] = end_forward - start_forward
        meta['learning_rate'] = self.optimizer.param_groups[-1]['lr']
        meta['memory_allocated'] = torch.cuda.memory_allocated()

        metrics = {}
        metrics['loss'] = loss.item()
        return {'meta': meta, 'metrics': metrics}

    def _log_val_step(self, step, metrics):
        """



        """
        total_steps = len(self.val_dataloader)
        log = 'Val Epoch {} [{}/{}]: '.format(self.val_epoch, step, total_steps)
        log += 'Loss - {:.4f} '.format(metrics['metrics']['loss'])
        log += 'Val Time - {:.4f} '.format(metrics['meta']['val_time'])
        log += 'Forward Time - {:.4f} '.format(metrics['meta']['forward_time'])
        log += 'Learning Rate - {:.4f} '.format(metrics['meta']['learning_rate'])
        log += 'Memory - {} GB '.format(metrics['meta']['memory_allocated'] / 1e9)
        log += 'AUC - {:.4f} '.format(self.auc_meter.values()['mean'])
        log += 'AP - {:.4f} '.format(self.pr_meter.values()['mean'])
        print(log)

        for (key, metric) in metrics['metrics'].items():
            name = 'val-metrics/{}'.format(key)
            self.summary_writer.add_scalar(name, metric, self.global_val_step)

        for (key, metric) in metrics['meta'].items():
            name = 'val-meta/{}'.format(key)
            self.summary_writer.add_scalar(name, metric, self.global_val_step)

    def _log_val_epoch(self):
        """



        """
        auc_metrics = self.auc_meter.values()
        for (key, metric) in auc_metrics.items():
            name = 'val-metrics/auc-metrics-{}'.format(key)
            self.summary_writer.add_scalar(name, metric, self.val_epoch)

        ap_metrics = self.pr_meter.values()
        for (key, metric) in ap_metrics.items():
            name = 'val-metrics/ap-metrics-{}'.format(key)
            self.summary_writer.add_scalar(name, metric, self.val_epoch)

        for class_id, class_name in enumerate(self.classes):
            scores = self.pr_meter.get_scores(class_id)
            targets = self.pr_meter.get_targets(class_id)
            name = 'val-metrics/pr-curve-{}'.format(class_name)
            self.summary_writer.add_pr_curve(name, targets, scores, self.val_epoch)

