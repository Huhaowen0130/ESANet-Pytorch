# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric, MetricsLambda
import numbers

import torch
import torch.nn as nn

def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file,
                                    map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        epoch = checkpoint['epoch']
        if 'best_miou' in checkpoint:
            best_miou = checkpoint['best_miou']
            print('Best mIoU:', best_miou)
        else:
            best_miou = 0

        if 'best_miou_epoch' in checkpoint:
            best_miou_epoch = checkpoint['best_miou_epoch']
            print('Best mIoU epoch:', best_miou_epoch)
        else:
            best_miou_epoch = 0
        return epoch, best_miou, best_miou_epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        sys.exit(1)

def save_ckpt(ckpt_dir, model, optimizer, epoch):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{}.pth".format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))

def save_ckpt_every_epoch(ckpt_dir, model, optimizer, epoch, best_miou,
                          best_miou_epoch):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_miou': best_miou,
        'best_miou_epoch': best_miou_epoch
    }
    # ckpt_model_filename = "ckpt_latest.pth".format(epoch)
    ckpt_model_filename = "ckpt_latest.pth"
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, device, weight):
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction='none',
            ignore_index=-1
        )
        self.ce_loss.to(device)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            # mask = targets > 0
            targets_m = targets.clone()
            targets_m -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())

            number_of_pixels_per_class = \
                torch.bincount(targets.flatten().type(self.dtype),
                               minlength=self.num_classes)
            divisor_weighted_pixel_sum = \
                torch.sum(number_of_pixels_per_class[1:] * self.weight)   # without void
            losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
            # losses.append(torch.sum(loss_all) / torch.sum(mask.float()))

        return losses
        
class CrossEntropyLoss2dForValidData:
    def __init__(self, device, weight, weighted_pixel_sum):
        super(CrossEntropyLoss2dForValidData, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction='sum',
            ignore_index=-1
        )
        self.ce_loss.to(device)
        self.weighted_pixel_sum = weighted_pixel_sum
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss

    def compute_whole_loss(self):
        return self.total_loss.cpu().numpy().item() / self.weighted_pixel_sum.item()

    def reset_loss(self):
        self.total_loss = 0
        
class ConfusionMatrixPytorch(Metric):
    def __init__(self,
                 num_classes,
                 average=None,
                 output_transform=lambda x: x):
        if average is not None and average not in ("samples", "recall",
                                                   "precision"):
            raise ValueError("Argument average can None or one of "
                             "['samples', 'recall', 'precision']")

        self.num_classes = num_classes
        if self.num_classes < np.sqrt(2**8):
            self.dtype = torch.uint8
        elif self.num_classes < np.sqrt(2**16 / 2):
            self.dtype = torch.int16
        elif self.num_classes < np.sqrt(2**32 / 2):
            self.dtype = torch.int32
        else:
            self.dtype = torch.int64
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None
        super(ConfusionMatrixPytorch, self).__init__(
            output_transform=output_transform
        )

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes,
                                            self.num_classes,
                                            dtype=torch.int64,
                                            device='cpu')
        self._num_examples = 0

    def update(self, y, y_pred, num_examples=1):
        assert len(y) == len(y_pred), ('label and prediction need to have the'
                                       ' same size')
        self._num_examples += num_examples

        y = y.type(self.dtype)
        y_pred = y_pred.type(self.dtype)

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices,
                           minlength=self.num_classes ** 2)
        m = m.reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one '
                                     'example before it can be computed.')
        if self.average:
            self.confusion_matrix = self.confusion_matrix.float()
            if self.average == "samples":
                return self.confusion_matrix / self._num_examples
            elif self.average == "recall":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=1) + 1e-15)
            elif self.average == "precision":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=0) + 1e-15)
        return self.confusion_matrix

def iou_pytorch(cm, ignore_index=None):
    if not isinstance(cm, ConfusionMatrixPytorch):
        raise TypeError("Argument cm should be instance of ConfusionMatrix, "
                        "but given {}".format(type(cm)))

    if ignore_index is not None:
        if (not (isinstance(ignore_index, numbers.Integral)
                 and 0 <= ignore_index < cm.num_classes)):
            raise ValueError("ignore_index should be non-negative integer, "
                             "but given {}".format(ignore_index))

    # Increase floating point precision and pass to CPU
    cm = cm.type(torch.DoubleTensor)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    if ignore_index is not None:

        def ignore_index_fn(iou_vector):
            if ignore_index >= len(iou_vector):
                raise ValueError("ignore_index {} is larger than the length "
                                 "of IoU vector {}"
                                 .format(ignore_index, len(iou_vector)))
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_index)
            return iou_vector[indices]

        return MetricsLambda(ignore_index_fn, iou)
    else:
        return iou

def miou_pytorch(cm, ignore_index=None):
    return iou_pytorch(cm=cm, ignore_index=ignore_index).mean()

# def print_log(epoch, local_count, count_inter, dataset_size, loss, time_inter,
#               learning_rates):
def print_log(epoch, local_count, count_inter, dataset_size, loss):
    print_string = 'Train Epoch: {:>3} [{:>4}/{:>4} ({: 5.1f}%)]'.format(
        epoch, local_count, dataset_size,
        100. * local_count / dataset_size)
    print_string += '   Loss: {:0.6f}'.format(loss.item())
    print(print_string, flush=True)