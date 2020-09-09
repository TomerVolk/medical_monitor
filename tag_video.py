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


def create_video_folder():
    video_path = "data/custom/archive/Simulated Patient Monitor.mp4"
    out_folder_path = "data/custom/video_photos"
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(out_folder_path + '/frame' + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def test_epoch(model, dataloader: DataLoader, conf_thres, nms_thres, img_size: int):
    model.eval()

    losses_list = []
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=__print)):
    num_batches = len(dataloader)

    with tqdm.tqdm(desc=f'{" evaluation"}', total=num_batches) as pbar:

        for batch_i, (_, imgs, _) in enumerate(dataloader):
            # print(targets)
            # print("###")

            with torch.no_grad():
                imgs_cuda = Variable(imgs.to(device))

                outputs = model(imgs_cuda)
                outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
            pbar.update()
            for sample_i in range(len(outputs)):
                if outputs[sample_i] is None:
                    continue

                output = outputs[sample_i]
                pred_boxes = output[:, :4]
                pred_scores = output[:, 4]
                pred_labels = output[:, -1]
                print(pred_labels)
                print(pred_scores)
                print(pred_boxes)
                exit(98)


def tag_video():
    np.random.seed(seed=42)
    torch.manual_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg",
                        help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Get data configuration
    video_path = "data/custom/video_photos.txt"
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

    ds_test = ListDataset(
        video_path,
        img_size=416,
        augment=False,
        multiscale=False,
    )

    dataloader_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
        collate_fn=ds_test.collate_fn
    )
    test_epoch(
        model=model,
        dataloader=dataloader_test,
        conf_thres=0.5,
        nms_thres=0.5,
        img_size=416,
    )


def create_vid_names():
    out_path = "data/custom/video_photos"
    out_str = ""
    in_path = "data/custom/labels/frame0.txt"
    with open(in_path, "r") as f:
        out_str = f.read()
    for i in range(3701):
        cur_out_path = out_path + "/frame" + str(i) + ".txt"
        with open(cur_out_path, "w") as f:
            f.write(out_str)


if __name__ == "__main__":
    tag_video()
