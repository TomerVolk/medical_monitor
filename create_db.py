from os.path import join
import json
from os import listdir
import random
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


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
        cur_str = cur_str.strip()
        cur_img_path = join(out_folder, name)
        with open(cur_img_path, "w") as f:
            f.write(cur_str)


def show_img(img_path, bbox):
    img_pil = Image.open(img_path)
    im_width, im_height = img_pil.size
    w = bbox[2] * im_width
    h = bbox[3] * im_height
    x = bbox[0] * im_width - w/2
    y = bbox[1] * im_height - h/2
    rect = Rectangle((int(x), int(y)), int(w), int(h), linewidth=1, edgecolor='r', facecolor='none')
    fig, ax = plt.subplots(1)
    ax.imshow(img_pil)

    ax.add_patch(rect)
    plt.show()


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


def check_boxes():
    img = "data/custom/images/frame336.jpg"
    label = "data/custom/labels/frame336.txt"
    types = "data/custom/classes.names"
    type_list = []
    with open(types) as f:
        for row in f:
            type_list.append(row.strip())
    print(list(enumerate(type_list)))
    img_pil = Image.open(img)
    im_width, im_height = img_pil.size
    classes = []
    with open(label) as f:
        for row in f:
            classes.append(row.split(" "))
    print(type_list[int(classes[0][0])])
    fig, ax = plt.subplots(1)
    ax.imshow(img_pil)
    x, y, w, h = classes[0][1:]
    w = float(w) * im_width
    h = float(h) * im_height
    x = float(x) * im_width - w/2
    y = float(y) * im_height + h/2
    rect = Rectangle((int(x), int(y)), int(w), int(h), linewidth=1, edgecolor='r', facecolor='none')

    ax.add_patch(rect)
    plt.show()


def to_one_class():
    folder_path = "data/custom/labels"
    for file in listdir(folder_path):
        file_path = folder_path + "/" + file
        out_str = ""
        with open(file_path, "r") as f:
            for row in f:
                row = row.strip()
                labels = row.split(" ")
                labels[0] = "0"
                out_str += " ".join(labels) + "\n"
        out_str = out_str.strip()
        with open(file_path, "w") as f:
            f.write(out_str)


if __name__ == "__main__":
    to_one_class()
    exit(90)
    label_path = "data/sample/train.txt"
    img_path = "data/sample/train.jpg"
    with open(label_path) as f:
        for row in f:
            break
    bbox_str = row.strip().split(" ")[1:]
    bbox = [float(x) for x in bbox_str]
    show_img(img_path, bbox)

    label_path = "data/custom/labels/frame126.txt"
    img_path = "data/custom/images/frame126.jpg"
    with open(label_path) as f:
        for row in f:
            break
    bbox_str = row.strip().split(" ")[1:]
    bbox = [float(x) for x in bbox_str]
    show_img(img_path, bbox)

    # create_img_labels()
