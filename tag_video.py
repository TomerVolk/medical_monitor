from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
import cv2
from utils.parse_config import *
from terminaltables import AsciiTable
import os
import sys
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import cv2


def create_tagged_video(model, conf_thres, nms_thres, img_size: int, class_list, video_path: str):
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colors_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (127, 0, 255), (255, 255, 0), (0, 128, 255)]

    cap = cv2.VideoCapture(video_path)
    out_path = video_path.replace(".mp4", " tagged.mp4")
    out_vid = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 360))
    i = 0
    while cap.isOpened():
        i += 1
        if i % 100 == 0:
            print(f'finished {i} images')
        ret, image = cap.read()
        if ret == False:
            break

        ds_test = SingleImage(image, img_size)
        dl_test = DataLoader(ds_test)
        imgs = dl_test.__iter__().__next__()
        with torch.no_grad():
            imgs_cuda = Variable(imgs.to(device))
            outputs = model(imgs_cuda)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        if outputs[0] is None:
            continue

        output = outputs[0]
        pred_boxes = output[:, :4]
        pred_boxes = rescale_boxes(pred_boxes, 416, (360, 640))
        pred_labels = output[:, -1]
        for pred_idx in range(len(pred_labels)):
            pred_box = pred_boxes[pred_idx]
            pred = int(pred_labels[pred_idx].item())
            label = class_list[pred]
            x1 = (pred_box[0], pred_box[1])
            x2 = (pred_box[2], pred_box[3])
            image = cv2.rectangle(image, x1, x2, thickness=2, color=colors_list[pred])
            image = cv2.putText(image, label,
                                (x1[0], x1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        out_vid.write(image)
    cap.release()


def tag_video():
    np.random.seed(seed=42)
    torch.manual_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg",
                        help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="if specified starts from checkpoint model")
    parser.add_argument("--video_path", type=str, default="data/custom/archive/Simulated Patient Monitor.mp4",
                        help="The path to the video file")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Get data configuration
    video_path = opt.video_path
    class_names = load_classes("data/custom/classes.names")
    print(class_names)

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    print(model.hyperparams)
    model.apply(weights_init_normal)
    # If specified w start from checkpoint
    if opt.pretrained_weights is not None:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
            print("pth was loaded")
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    create_tagged_video(
        model=model,
        conf_thres=0.96,
        nms_thres=0.3,
        img_size=416,
        class_list=class_names,
        video_path=video_path
    )

"""
def draw_image_labeled():
    img_path = "data/custom/images/frame0.jpg"
    label_path = "data/custom/labels/frame0.txt"
    label_list_path = "data/custom/classes.names"
    colors_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (127, 0, 255), (255, 255, 0), (0, 128, 255)]
    with open(label_list_path) as f:
        label_list = f.readlines()
    label_list = [x.strip() for x in label_list]
    with open(label_path) as f:
        cur_labels = f.readlines()
    cur_labels = [x.strip().split(" ") for x in cur_labels]
    img = cv2.imread(img_path)
    for bbox in cur_labels:
        label = label_list[int(bbox[0])]
        x1 = int((float(bbox[1]) - float(bbox[3])/2) * 640)
        x2 = int((float(bbox[1]) + float(bbox[3])/2) * 640)
        y1 = int((float(bbox[2]) - float(bbox[4])/2) * 360)
        y2 = int((float(bbox[2]) + float(bbox[4])/2) * 360)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), thickness=2, color=colors_list[int(bbox[0])])
        img = cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    print(cur_labels)
"""

if __name__ == "__main__":
    tag_video()
