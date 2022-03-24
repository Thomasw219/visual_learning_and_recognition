import os
import random
import time
import copy

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np


#TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """

    bounding_boxes = bounding_boxes[confidence_score > threshold]
    scores = confidence_score[confidence_score > threshold]
    N = bounding_boxes.shape[0]

    diffs = bounding_boxes[:,2].reshape(N, 1) - bounding_boxes[:,0].reshape(1, N)
    i_width = torch.minimum(diffs, torch.transpose(diffs, 0, 1))
    diffs = bounding_boxes[:,3].reshape(N, 1) - bounding_boxes[:,1].reshape(1, N)
    i_height = torch.minimum(diffs, torch.transpose(diffs, 0, 1))
    i_area = torch.maximum(i_width * i_height, torch.zeros_like(i_width))

    boxes_area = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (bounding_boxes[:, 3] - bounding_boxes[:, 1])
    u_area = boxes_area.reshape(N, 1) + boxes_area.reshape(1, N) - i_area
    iou = i_area / u_area - torch.eye(N, device=bounding_boxes.device)
    non_maximal = torch.any(torch.logical_and(iou > 0.3, scores.reshape(N, 1) > scores.reshape(1, N)), dim=0)
    maximal = torch.logical_not(non_maximal)

    return bounding_boxes[maximal], scores[maximal]


#TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """

    i_w = np.minimum(box1[2] - box2[0], box2[2] - box1[0])
    i_h = np.minimum(box1[3] - box2[1], box2[3] - box1[1])
    i_a = np.maximum(i_w * i_h, 0)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    u_a = a1 + a2 - i_a

    return i_a / u_a


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id" : int(classes[i]),
        } for i in range(len(classes))
        ]

    return box_list

def get_box_data_scores(classes, bbox_coordinates, scores):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": float(bbox_coordinates[i][0]),
                "minY": float(bbox_coordinates[i][1]),
                "maxX": float(bbox_coordinates[i][2]),
                "maxY": float(bbox_coordinates[i][3]),
            },
            "class_id" : int(classes[i]),
            "scores" : {"confidence" : float(scores[i])}
        } for i in range(len(classes))
        ]

    return box_list

