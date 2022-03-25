from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL, get_box_data, get_box_data_scores, iou
from PIL import Image, ImageDraw


# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = {150000}
lr_decay = 1. / 10
rand_seed = 1024

lr = 0.00005
momentum = 0.9
weight_decay = 0.0005
# ------------

USE_WANDB = True

if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

if USE_WANDB:
    wandb.init(project="vlr2q2")

# load datasets and create dataloaders

train_dataset = VOCDataset('trainval', image_size=512)
val_dataset = VOCDataset('test', image_size=512)

CLASS_ID_TO_LABEL = dict(enumerate(val_dataset.CLASS_NAMES))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,   # batchsize is one for this implementation
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    sampler=None,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True)


# Create network and initialize
net = WSDDN(classes=train_dataset.CLASS_NAMES)
print(net)

if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()

for name, param in pret_net.items():
    print(name)
    if name not in own_state:
        print('Did not find {}'.format(name))
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue

# Move model to GPU and set train mode
net.load_state_dict(own_state)
net.cuda()
net.train()

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)

opt_params = []
for name, param in net.named_parameters():
    if 'features.0' not in name:
        opt_params.append(param)
optimizer = torch.optim.SGD(opt_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, threshold=1e-3)

output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

# training
train_loss = AverageMeter()
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
disp_interval = 10
#val_interval = 1000
#val_interval = 50
val_interval = 2000

def test_net(model, val_loader=None, thresh=0.05):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """

    correct = [[] for i in range(20)]
    confidence = [[] for i in range(20)]
    total = [0 for i in range(20)]

    

    for iter, data in enumerate(val_loader):
#        if iter == 100:
#            break
        # one batch = data for one image
        image           = data['image']
        target          = data['label']
        wgt             = data['wgt']
        rois            = data['rois']
        gt_boxes        = torch.tensor(data['gt_boxes']).numpy()
        gt_class_list   = torch.tensor(data['gt_classes']).numpy()

        #TODO: perform forward pass, compute cls_probs
        image = image.cuda()
        rois = rois[0].to(torch.float32).cuda()
        target = target.cuda()

        cls_probs = model.forward(image, rois=rois)

        all_boxes = []
        all_labels = []
        all_scores = []

        for label in gt_class_list:
            total[label] += 1

        # TODO: Iterate over each class (follow comments)
        for class_num in range(20):            
            # get valid rois and cls_scores based on thresh
            confidence_score = cls_probs[:, class_num]
            
            # use NMS to get boxes and scores
            boxes, scores = nms(rois, confidence_score)
            all_boxes.append(boxes.detach().cpu().numpy())
            all_labels.append([class_num for i in range(boxes.shape[0])])
            all_scores.append(scores.detach().cpu().numpy())

            for i in range(boxes.shape[0]):
                box = boxes[i]
                score = scores[i]
                confidence[class_num].append(score)
                box_correct = False
                if class_num in gt_class_list:
                    for j in range(gt_class_list.shape[0]):
                        if gt_class_list[j] == class_num and iou(gt_boxes[j], box.detach().cpu().numpy()) > 0.5:
                            box_correct = True

                correct[class_num].append(box_correct)

        all_boxes = np.concatenate(all_boxes)
        all_labels = np.concatenate(all_labels).astype(int)
        all_scores = np.concatenate(all_scores)

        #TODO: visualize bounding box predictions when required


    #TODO: Calculate mAP on test set
    APs = []
    for class_num in range(20):
        class_correct = np.array(correct[class_num])
        class_scores = np.array(confidence[class_num])

        class_total = total[class_num]

        indices = np.argsort(-1 * np.array(class_scores))
        class_correct = class_correct[indices]
        class_scores = class_scores[indices]
        precisions = []
        recalls = []
        tp = 0
        fp = 0
        for i in range(len(class_correct)):
            if class_correct[i]:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / class_total)

        AP = 0
        r1 = 0
        for i in range(1, len(precisions)):
            r2 = recalls[i]
            if r1 < r2:
                AP += (r2 - r1) * precisions[i]
                r1 = r2
        APs.append(AP)
    return APs

for epoch in range(6):
    for iter, data in enumerate(train_loader):
        print(iter)

        #TODO: get one batch and perform forward pass
        # one batch = data for one image
        image           = data['image']
        target          = data['label']
        wgt             = data['wgt']
        rois            = data['rois']
        gt_boxes        = data['gt_boxes']
        gt_class_list   = data['gt_classes']
        

        #TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
        # also convert inputs to cuda if training on GPU
        image = image.cuda()
        rois = rois[0].to(torch.float32).cuda()
        target = target.cuda()

        net.forward(image, rois=rois, gt_vec=target)

        # backward pass and update
        loss = net.loss    
        train_loss.update(loss.item())
        step_cnt += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #TODO: evaluate the model every N iterations (N defined in handout)
        
        if iter%val_interval == 0:
            net.eval()
            with torch.no_grad():
                ap = test_net(net, val_loader)
            print("AP ", ap)
            mAP = np.mean(ap)
            scheduler.step(mAP)
            print("mAP ", mAP)
            print("train_loss ", train_loss.avg)
            if USE_WANDB:
                ap_dict = {"test/" + train_dataset.CLASS_NAMES[i] + "_ap" : ap[i] for i in range(20)}
                wandb.log({**ap_dict, "test/mAP" : mAP})
            net.train()

        if iter%500 == 0:
            wandb.log({"train/loss" : train_loss.avg})
            train_loss.reset()


        #TODO: Perform all visualizations here
        #The intervals for different things are defined in the handout
        if USE_WANDB and iter >= len(train_loader) - 11:
            net.eval()
            image           = data['image']
            target          = data['label']
            wgt             = data['wgt']
            rois            = data['rois']
            gt_boxes        = torch.tensor(data['gt_boxes']).numpy()
            gt_class_list   = torch.tensor(data['gt_classes']).numpy()

            #TODO: perform forward pass, compute cls_probs
            image = image.cuda()
            rois = rois[0].to(torch.float32).cuda()
            target = target.cuda()

            cls_probs = net.forward(image, rois=rois)

            all_boxes = []
            all_labels = []
            all_scores = []

            # TODO: Iterate over each class (follow comments)
            for class_num in range(20):            
                # get valid rois and cls_scores based on thresh
                confidence_score = cls_probs[:, class_num]
                
                # use NMS to get boxes and scores
                boxes, scores = nms(rois, confidence_score)
                all_boxes.append(boxes.detach().cpu().numpy())
                all_labels.append([class_num for i in range(boxes.shape[0])])
                all_scores.append(scores.detach().cpu().numpy())

            all_boxes = np.concatenate(all_boxes)
            all_labels = np.concatenate(all_labels).astype(int)
            all_scores = np.concatenate(all_scores)
            pil_image = tensor_to_PIL(image[0].cpu())
            predictions_image = wandb.Image(pil_image, boxes={
                "predictions" : {
                    "box_data" : get_box_data_scores(all_labels, all_boxes, all_scores),
                    "class_labels" : CLASS_ID_TO_LABEL,
                    },
                })
            gt_image = wandb.Image(pil_image, boxes={
                "predictions" : {
                "box_data" : get_box_data(gt_class_list, gt_boxes),
                "class_labels" : CLASS_ID_TO_LABEL,
                },
            })
            wandb.log({'gt_boxes' : gt_image, 'pred_boxes' : predictions_image})
            net.train()

