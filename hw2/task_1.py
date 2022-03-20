import argparse
import os
import shutil
import time
import sys
import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


from AlexNet import *
from voc_dataset import *
from utils import *

import wandb
USE_WANDB = True # use flags, wandb is not convenient for debugging


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=256,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_true',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0

def rescale(hm):
    hm -= torch.min(hm)
    hm /= torch.max(hm)
    return hm

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO:
    # define loss function (criterion) and optimizer
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    # you can also use PlateauLR scheduler, which usually works well

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, threshold=1e-3)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # Data loading code
    
    #TODO: Create Datasets and Dataloaders using VOCDataset - Ensure that the sizes are as required
    # Also ensure that data directories are correct - the ones use for testing by TAs might be different
    # Resize the images to 512x512
    train_dataset = VOCDataset('trainval', image_size=512)
    val_dataset = VOCDataset('test', image_size=512)

    torch.manual_seed(0)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # shuffle=(train_sampler is None),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    
    
    # TODO: Create loggers for wandb - ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                m1, m2 = validate(val_loader, model, criterion, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

            scheduler.step(m1)

    if USE_WANDB:
        rand_indices = [10, 20, 30]
        for i in range(3):
            data = val_dataset.__getitem__(rand_indices[i])
            images = data['image']
            labels = data['label']
            wgts = data['wgt']
            imoutput = torch.sigmoid(model.forward(images.unsqueeze(0)))

            im = wandb.Image(tensor_to_PIL(images))
            tgt = torch.argmax(labels)
            hm = wandb.Image(transforms.ToPILImage(mode='L')(transforms.Resize((512, 512))(rescale(imoutput[0, tgt:tgt+1]))))

            wandb.log({
                "done/im" : im,
                "done/hm" : hm,
                })



#TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO: Get inputs from the data dict
        images = data['image']
        labels = data['label']
        wgts = data['wgt']

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output such as clamping
        # TODO: Compute loss using ``criterion``
        imoutput = torch.sigmoid(model.forward(images))
        output = torch.amax(imoutput, dim=(2, 3))
        loss = nn.functional.binary_cross_entropy(output, labels.cuda(), weight=wgts.cuda())


        # measure metrics and record loss
        m1 = metric1(output, labels, wgts)
        m2 = metric2(output, labels, wgts)
        losses.update(loss.item(), images.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)


        # TODO:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

    #TODO: Visualize/log things as mentioned in handout
    #TODO: Visualize at appropriate intervals

    if USE_WANDB:
        i1 = epoch % args.batch_size
        i2 = (epoch + args.batch_size // 2) % args.batch_size
        im1 = wandb.Image(tensor_to_PIL(images[i1]))
        im2 = wandb.Image(tensor_to_PIL(images[i2]))
        tgt1 = torch.argmax(labels[i1])
        tgt2 = torch.argmax(labels[i2])
        hm1 = wandb.Image(transforms.ToPILImage(mode='L')(transforms.Resize((512, 512))(rescale(imoutput[i1, tgt1:tgt1+1]))))
        hm2 = wandb.Image(transforms.ToPILImage(mode='L')(transforms.Resize((512, 512))(rescale(imoutput[i2, tgt2:tgt2+1]))))

        wandb.log({
            "train/loss" : losses.avg,
            "train/metric1" : avg_m1.avg,
            "train/metric2" : avg_m2.avg,
            "epoch" : epoch,
            "train/im1" : im1,
            "train/im2" : im2,
            "train/hm1" : hm1,
            "train/hm2" : hm2,
            })


    # End of train()


def validate(val_loader, model, criterion, epoch = 0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO: Get inputs from the data dict
        images = data['image']
        labels = data['label']
        wgts = data['wgt']

        # TODO: Get output from model
        # TODO: Perform any necessary functions on the output
        # TODO: Compute loss using ``criterion``
        imoutput = torch.sigmoid(model.forward(images))
        output = torch.amax(imoutput, dim=(2, 3))
        loss = nn.functional.binary_cross_entropy(output, labels.cuda(), weight=wgts.cuda())

        # measure metrics and record loss
        m1 = metric1(output, labels, wgts)
        m2 = metric2(output, labels, wgts)
        losses.update(loss.item(), images.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

    #TODO: Visualize things as mentioned in handout
    #TODO: Visualize at appropriate intervals

    if USE_WANDB:
        i1 = epoch % args.batch_size
        i2 = (epoch + args.batch_size // 2) % args.batch_size
        im1 = wandb.Image(tensor_to_PIL(images[i1]))
        im2 = wandb.Image(tensor_to_PIL(images[i2]))
        tgt1 = torch.argmax(labels[i1])
        tgt2 = torch.argmax(labels[i2])
        hm1 = wandb.Image(transforms.ToPILImage(mode='L')(transforms.Resize((512, 512))(rescale(imoutput[i1, tgt1:tgt1+1]))))
        hm2 = wandb.Image(transforms.ToPILImage(mode='L')(transforms.Resize((512, 512))(rescale(imoutput[i2, tgt2:tgt2+1]))))

        wandb.log({
            "test/loss" : losses.avg,
            "test/metric1" : avg_m1.avg,
            "test/metric2" : avg_m2.avg,
            "epoch" : epoch,
            "test/im1" : im1,
            "test/im2" : im2,
            "test/hm1" : hm1,
            "test/hm2" : hm2,
            })


    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def metric1(output, target, weights):
    #TODO: Ignore for now - proceed till instructed
    AP = []
    valid = weights.cpu().detach().numpy()
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    for cid in range(20):
        gt_cls = target[:, cid][valid[:, cid] > 0].astype('float32')
        if np.sum(gt_cls) == 0:
            continue
        pred_cls = output[:, cid][valid[:, cid] > 0].astype('float32')
        ap = sklearn.metrics.average_precision_score(gt_cls, pred_cls)
        AP.append(ap)
    return np.mean(AP)

THRESHOLD = 0.5
def metric2(output, target, weights):
    #TODO: Ignore for now - proceed till instructed
    valid = weights.cpu().detach().numpy().flatten()
    output = output.cpu().detach().numpy().flatten()
    target = target.cpu().detach().numpy().flatten()

    gt = target[valid > 0].astype('float32')
    pred = output[valid > 0].astype('float32')
    return sklearn.metrics.recall_score(gt, pred > THRESHOLD, average='binary')


if __name__ == '__main__':
    main()
