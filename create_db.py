from os.path import join
import json
from os import listdir
import random


def create_img_labels():
    labels_loc = "data/custom/archive/images labeled.json"
    out_folder = "data/custom/labels"
    labels_file = "data/custom/classes.names"
    with open(labels_loc, "r") as f:
        data = json.load(f)
    labels_list = data["labels"]
    with open(labels_file, "w") as f:
        for label in labels_list:
            f.write(label + "\n")
    labels_dict = {}
    for idx, label in enumerate(labels_list):
        labels_dict[label] = idx
    for img in data["images"]:
        name = img["image_name"].replace(".jpg", ".txt")
        cur_str = ""
        for idx, label in enumerate(img["labels"].keys()):
            cur_str += str(labels_dict[label]) + " "
            bbox = img["labels"][label]
            new_bbox = [float(bbox[0] + bbox[2]) / 2, float(bbox[1] + bbox[3] / 2),
                        float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]
            new_bbox = str(new_bbox).replace("[", "").replace("]", "").replace(",", "")
            if idx == len(img["labels"]):
                cur_str += new_bbox
            else:
                cur_str += new_bbox + "\n"
        cur_img_path = join(out_folder, name)
        with open(cur_img_path, "w") as f:
            f.write(cur_str)


def normalize_bbox():
    img_height = 360
    img_width = 640
    folder_path = "data/custom/labels"
    for file in listdir(folder_path):
        file_path = folder_path + "/" + file
        out_str = ""
        with open(file_path, "r") as f:
            for row in f:
                row = row.strip()
                labels = row.split(" ")
                labels[1] = str(float(labels[1]) / img_width)
                labels[2] = str(float(labels[2]) / img_height)
                labels[3] = str(float(labels[3]) / img_width)
                labels[4] = str(float(labels[4]) / img_height)
                out_str += " ".join(labels) + "\n"
        out_str = out_str.strip()
        with open(file_path, "w") as f:
            f.write(out_str)


def train_valid_split(train_per=0.7):
    img_loc = "data/custom/images"
    images = listdir(img_loc)
    train = random.sample(images, int(len(images) * train_per))
    print("We have " + str(len(train)) + " images in the train data")
    validation = [x for x in images if x not in train]
    print("We have " + str(len(validation)) + " images in the validation data")
    train_file = "data/custom/train.txt"
    valid_path = "data/custom/valid.txt"
    img_loc += "/"
    with open(train_file, "w") as f:
        for img in train:
            f.write(img_loc + img + "\n")
    with open(valid_path, "w") as f:
        for img in validation:
            f.write(img_loc + img + "\n")


if __name__ == "__main__":
    normalize_bbox()
