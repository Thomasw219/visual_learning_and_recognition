import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import numpy as np

from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        #TODO: Define the WSDDN model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        self.roi_pool   = roi_pool
        self.roi_pool_output_shape = (15, 15)
        self.p = 0.5
        self.classifier = nn.Sequential(
                nn.Linear(15 * 15 * 256, 4096), 
                nn.ReLU(), 
                nn.Linear(4096, 4096), 
                nn.ReLU())


        self.score_fc   = nn.Sequential(
                nn.Linear(4096, 20),
                nn.Softmax(dim=1))
        self.bbox_fc    = nn.Sequential(
                nn.Linear(4096, 20),
                nn.Softmax(dim=0))

        
        # loss
        self.cross_entropy = None


    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):
        

        #TODO: Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        feats = self.features(image)
        boxes = torch.cat([torch.zeros(rois.shape[0], 1).cuda(), rois * 512], dim=-1)
        pooled_feats = roi_pool(feats, boxes=boxes, output_size=self.roi_pool_output_shape, spatial_scale=31/512)
        flattened_feats = self.classifier(pooled_feats.view(pooled_feats.shape[0], -1))

        classification_scores = self.score_fc(flattened_feats)
        detection_scores = self.bbox_fc(flattened_feats)

        
        cls_prob = classification_scores * detection_scores

        if self.training:
            label_vec = gt_vec.view(self.n_classes)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        
        return cls_prob

    
    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        #TODO: Compute the appropriate loss using the cls_prob that is the
        #output of forward()
        #Checkout forward() to see how it is called

        return F.binary_cross_entropy(torch.clamp(torch.sum(cls_prob, axis=0), min=0, max=1), label_vec, reduction='sum')
