from __future__ import division

from utils.utils import *
from utils.datasets import *
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2
from os import listdir


def create_tagged_video(class_list, img_path: str):

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    colors_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (127, 0, 255), (255, 255, 0), (0, 128, 255)]

    out_path = "ground_truth_tagged.mp4"
    out_vid = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 360))
    with open(img_path) as f:
        images = f.readlines()
    for i, img in enumerate(images):
        i += 1
        if i % 100 == 0:
            print(f'finished {i} images')
        image = img.strip()
        labels_path = image.replace("images", "labels").replace(".jpg", ".txt")
        with open(labels_path) as f:
            labels = f.readlines()
        image = cv2.imread(image)
        for pred_idx in range(len(labels)):
            tag = labels[pred_idx].split(" ")
            pred_box = [float(x) for x in tag[1:]]
            pred_box[0] *= 640
            pred_box[1] *= 360
            pred_box[2] *= 640
            pred_box[3] *= 360

            pred = int(tag[0])
            label = class_list[pred]
            x1 = (int(pred_box[0] - pred_box[2]/2), int(pred_box[1] - pred_box[3]/2))
            x2 = (int(pred_box[0] + pred_box[2]/2), int(pred_box[1] + pred_box[3]/2))
            image = cv2.rectangle(image, x1, x2, color=colors_list[pred], thickness=2)
            image = cv2.putText(image, label,
                                (x1[0], x1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        out_vid.write(image)


def tag_video(create_file=False):
    np.random.seed(seed=42)
    torch.manual_seed(seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Get data configuration
    class_names = load_classes("data/custom/classes.names")
    print(class_names)
    if create_file:
        img_path = "data/custom/images"
        file_list = listdir(img_path)
        file_list = [(int(x.strip().replace(".jpg", "").replace("frame", "")), x) for x in file_list]
        file_list = sorted(file_list)
        file_list = [img_path + "/" + x[1] for x in file_list]
        out_str = str(file_list).replace("[", "").replace("]", "")
        out_str = out_str.replace("\'", "").replace(", ", "\n").strip()
        out_path = "data/custom/images_list.txt"

        with open(out_path, "w") as f:
            f.write(out_str)
    img_path = "data/custom/images_list.txt"
    create_tagged_video(
        class_list=class_names,
        img_path=img_path
    )


if __name__ == "__main__":
    tag_video()
