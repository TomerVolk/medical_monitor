from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def get_id_from_name(name):
    return int(name[2:4])


def suffixing(name, mod=None):
    ind = get_id_from_name(name)
    if ind < 22:
        video_normal = os.path.join(name + ".wm1")
        video_zoom = os.path.join(name + ".wmv")
    else:
        video_normal = os.path.join(name + ".wmv")
        video_zoom = os.path.join(name + ".wm1")

    if mod == ("zoom" or "Zoom"):
        return video_zoom
    else:
        return video_normal


def k_majority(k_list):
    """
    :param k_list: list of k events
    :return: output: the event that appears maximal number of times in the list.
    """
    for i in range(len(k_list)):
        k_list[i] = str(k_list[i])
    categories_list=[]
    categories_scores=[]
    string=""
    for entry in k_list:
        if type(entry) != type(string):
            entry=""
        for index,cat in enumerate(categories_list):
            if cat == entry:
                categories_scores[index]+=1
                break
        categories_list.append(entry)
        categories_scores.append(1)
    categories_scores = np.array(categories_scores)
    max_index= -1
    if len(categories_scores) != 0:
        max_index = np.argmax(categories_scores)
    output = 'Error'
    if max_index != -1:
        output =categories_list[max_index]
    return output


# def update_filters_wondows(filter_window,new_val, filtration_factor):
#
#     current_length = len(filter_window)
#
#     if current_length < filtration_factor:
#         filter_window.append(new_val)
#     else:
#         del filter_window[0]
#         filter_window.append(new_val)
#
#     return filter_window


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    if type(boxes)== type(torch.tensor([])):
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x
        # Rescale bounding boxes to dimension of original image
        boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
        boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xyxy2xywh(y):
    x = y.new(y.shape)
    x[..., 0] = y[...,0] + (( y[...,2] - y[...,0] ) / 2)
    x[...,1] = y[...,1] + ((y[...,3] - y[...,1])/2)
    x[..., 2] = y[...,2] - y[...,0]
    x[..., 3] =y[...,3] - y[...,1]
    return x


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_batch_statistics(outputs, targets, iou_threshold,paths=""):
    """ Compute true positives, predicted scores and predicted labels per sample """

    batch_metrics = []

    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        x = ((2 in pred_labels) and (2 in target_labels))
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:

                    continue
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)

                if iou >= iou_threshold and int(box_index) not in detected_boxes and\
                        int(pred_label) == int(target_labels[box_index]):
                    true_positives[pred_i] = 1
                    detected_boxes += [int(box_index)]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics



def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def bbox_relationship_parameters_calculator(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    return inter_area, b1_area, b2_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    inter_area, b1_area, b2_area = bbox_relationship_parameters_calculator(box1, box2, x1y1x2y2)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def bbox_overlap(box1, box2, x1y1x2y2=True):
    """The overlap coefficient or Szymkiewicz–Simpson coefficien"""
    inter_area, b1_area, b2_area = bbox_relationship_parameters_calculator(box1, box2, x1y1x2y2=True)
    overlap = inter_area / torch.min((b1_area + 1e-16),(b2_area + 1e-16))
    return overlap


def box_quarters_calculator(bbox,x1y1x2y2=True):
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = bbox[:, 0] - bbox[:, 2] / 2, bbox[:, 0] + bbox[:, 2] / 2
        b1_y1, b1_y2 = bbox[:, 1] - bbox[:, 3] / 2, bbox[:, 1] + bbox[:, 3] / 2

    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    q_l_x1 = b1_x1
    q_1_x2 = b1_x1 + 0.5*(b1_x2-b1_x1)
    q_1_y1 = b1_y1 + 0.5*(b1_y2-b1_y1)
    q_1_y2 = b1_y2

    q_r_x1 = b1_x1 + 0.5*(b1_x2-b1_x1)
    q_r_x2 = b1_x2
    q_r_y1 = b1_y1 + 0.5*(b1_y2-b1_y1)
    q_r_y2 = b1_y2

    # if keep_boxes:
    #     output[image_i] = torch.stack(keep_boxes)

    q_l = [q_l_x1,q_1_y1,q_1_x2,q_1_y2]
    q_l = torch.stack(q_l,dim=1)
    q_r = [q_r_x1,q_r_y1,q_r_x2,q_r_y2]
    q_r = torch.stack(q_r,dim=1)

    return q_l, q_r

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])


    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []

        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            invalid[0] = 1
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]

        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output



def non_max_suppression_post_process(prediction,  nms_thres=0.4,conf_thres=0.01):
    output = []
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        if type(image_pred) == type(torch.tensor([])):
            image_pred = image_pred[image_pred[:, 4] >= conf_thres]


        if image_pred is None:
            return output
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Perform non-maximum suppression
        keep_boxes = []
        removel_set=set()

        for object_idx in range(image_pred.size(0)):
            # print(image_pred)
            is_right_hand = 0 == image_pred[:, 6]
            is_left_hand = 1 == image_pred[:, 6]
            is_a_hand = (is_right_hand + is_left_hand) == 1
            is_not_hand = (is_right_hand + is_left_hand) == 0
            i_am_a_hand =  1 - is_not_hand[object_idx]
            if i_am_a_hand:
                if object_idx not in removel_set:
                    large_overlap = bbox_overlap(image_pred[object_idx, :4].unsqueeze(0), image_pred[:, :4]) > nms_thres
                    # only the one with the maximum confidence survive from the suspected_list
                    suspected_list = large_overlap & is_a_hand
                    weights = image_pred[:, 4]
                    relevant_weights = (weights * suspected_list.to(dtype=torch.float))
                    if torch.max(relevant_weights) == relevant_weights[object_idx]:
                        keep_boxes += [image_pred[object_idx]]
                        suspected_list_np = suspected_list.numpy()
                        indexes_to_remove = set(np.nonzero(suspected_list_np)[0])
                        indexes_to_remove.remove(object_idx)
                        removel_set = removel_set.union(indexes_to_remove)
            else:
                if object_idx not in removel_set:
                    large_overlap = bbox_overlap(image_pred[object_idx, :4].unsqueeze(0), image_pred[:, :4]) > nms_thres
                    # only the one with the maximum confidence survive from the suspected_list
                    suspected_list = large_overlap & is_not_hand
                    weights = image_pred[:, 4]
                    relevant_weights = (weights * suspected_list.to(dtype=torch.float))
                    if torch.max(relevant_weights) == relevant_weights[object_idx]:
                        keep_boxes += [image_pred[object_idx]]
                        suspected_list_np = suspected_list.numpy()
                        indexes_to_remove = set(np.nonzero(suspected_list_np)[0])
                        indexes_to_remove.remove(object_idx)
                        removel_set=removel_set.union(indexes_to_remove)
        if keep_boxes:
            output.append(torch.stack(keep_boxes))
    return output



def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):


            noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0


    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
