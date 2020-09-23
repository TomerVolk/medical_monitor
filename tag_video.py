from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2
from digit_ocr import Model


def preprocess_image(img, label, low_border: int):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    h, w = img.shape
    if label == "ABP":
        img = img[:, 30:w]
    if label != "ABP" and label != "PAP":
        kernel = np.ones((3, 3), np.uint8)
    else:
        kernel = np.ones((2, 1))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.copyMakeBorder(img, low_border, low_border, low_border, low_border, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contours[i]) for i in range(len(contours))]
    return img, rects


def get_number_small_labels(tags_bbox: list):
    tags_bbox = sorted(tags_bbox)
    min_y = tags_bbox[0][0]
    max_y = tags_bbox[-1][0]
    upper_row = []
    lower_row = []
    for tag in tags_bbox:
        if abs(tag[0] - min_y) < abs(max_y - tag[0]):
            upper_row.append((tag[1], tag[2]))
        else:
            lower_row.append((tag[1], tag[2]))
    lower_row = sorted(lower_row)
    upper_row = sorted(upper_row)
    lower_row = [x[1] for x in lower_row]
    upper_row = [x[1] for x in upper_row]
    upper_row = "".join(upper_row)
    lower_row = "".join(lower_row)
    ans = upper_row + " " + lower_row
    return ans


def find_numbers(image, output, class_list, is_ground=False, buffer=(2, 5), model=None):
    big_labels = {"Heart rate", "Pulse", "SpO2", "awRR", "etCO2"}
    small_labels = {"ABP", "PAP"}
    colors_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (127, 0, 255), (255, 255, 0), (0, 128, 255)]
    pred_boxes = output[:, :4]
    if not is_ground:
        pred_boxes = rescale_boxes(pred_boxes, 416, (360, 640))
    pred_labels = output[:, -1]
    drawn_img = image.copy()
    for pred_idx in range(len(pred_labels)):
        pred = int(pred_labels[pred_idx].item())
        label = class_list[pred]

        pred_box = pred_boxes[pred_idx]
        cur_rect = image.copy()
        crop_rect = (int(pred_box[1]) - buffer[0], int(pred_box[3]) + buffer[0],
                     int(pred_box[0]) - buffer[1], int(pred_box[2]) + buffer[1])
        # Y is first in crop_rect because this is how cv2 crop works
        cur_rect = cur_rect[crop_rect[0]: crop_rect[1], crop_rect[2]: crop_rect[3]]
        h, w, _ = cur_rect.shape
        low_border = 30
        cur_rect, rects = preprocess_image(cur_rect, label, low_border)
        cur_tags = []
        for i, bbox in enumerate(rects):
            digit = cur_rect.copy()
            digit = digit[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
            # digit = cv2.resize(digit, (20, 20))
            digit = cv2.copyMakeBorder(digit, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
            h, w = digit.shape
            if h * w < 200:
                continue
            tag = model.predict(digit)
            if tag == "junk":
                continue
            tag = str(tag[0])
            x1_digit = (bbox[0], bbox[1])
            cur_tags.append((x1_digit[1], x1_digit[0], tag))
        cur_tags = sorted(cur_tags)
        number = ""
        if label in big_labels:
            cur_tags = [(x[1], x[2]) for x in cur_tags]
            cur_tags = sorted(cur_tags)
            cur_tags = [x[1] for x in cur_tags]
            number = "".join(cur_tags)
        if label in small_labels:
            number = get_number_small_labels(cur_tags)
        x1 = (pred_box[0], pred_box[1])
        x2 = (pred_box[2], pred_box[3])
        drawn_img = cv2.rectangle(drawn_img, x1, x2, thickness=2, color=colors_list[pred])
        drawn_img = cv2.putText(drawn_img, label + ": " + number,
                                (x1[0], x1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12), 2)

    # cv2.imshow("image", drawn_img)
    # cv2.waitKey(0)
    return drawn_img


def create_tagged_video(model, conf_thres, nms_thres, img_size: int, class_list, video_path: str, ground=None):
    model.eval()
    ocr_model = Model()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(video_path)
    out_path = video_path.replace(".mp4", " tagged numbers.mp4")
    out_vid = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (640, 360))
    i = 0
    while cap.isOpened():
        i += 1
        if i % 100 == 0:
            print(f'finished {i} images')
        ret, image = cap.read()
        if ret == False:
            break
        if ground is None:
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
            image = find_numbers(image, output, class_list, model=ocr_model)
        else:
            image = find_numbers(image, ground, class_list, model=ocr_model, is_ground=True)

        out_vid.write(image)
    cap.release()


def tag_video(ground=None):
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
        video_path=video_path,
        ground=ground
    )


if __name__ == "__main__":
    # image = cv2.imread(r"data/custom/images/frame0.jpg")

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # exit(45)

    class_names = load_classes("data/custom/classes.names")
    print(class_names)
    """
    labels_path = "data/custom/labels/frame70.txt"
    h, w, _ = image.shape
    output = None
    with open(labels_path, "r") as file:
        for line in file.readlines():
            output_arr = line.split(" ")
            output_arr = [float(x) for x in output_arr]
            output_arr = output_arr[1:] + [output_arr[0]]
            output_arr[0] *= w
            output_arr[1] *= h
            output_arr[2] *= w
            output_arr[3] *= h
            output_arr = [output_arr[0] - output_arr[2]/2, output_arr[1] - output_arr[3]/2,
                          output_arr[0] + output_arr[2]/2, output_arr[1] + output_arr[3]/2, output_arr[-1]]
            output_arr = torch.Tensor(output_arr).unsqueeze(0)
            if output is None:
                output = output_arr
            else:
                output = torch.cat([output, output_arr])
    """
    tag_video()
