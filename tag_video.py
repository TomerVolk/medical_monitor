from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2
import pytesseract


def preprocess_image(img, label):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if label != "ABP" and label != "PAP":
        kernel = np.ones((3, 3), np.uint8)
    else:
        kernel = np.ones((2, 1))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # img = cv2.Canny(img, 100, 200)
    return img


def find_numbers(image, output, class_list, is_ground=False, buffer=(2, 5)):
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
        cur_rect = preprocess_image(cur_rect, label)

        # cur_rect = cv2.cvtColor(cur_rect, cv2.COLOR_BGR2GRAY)
        h, w = cur_rect.shape
        boxes = pytesseract.image_to_boxes(cur_rect)
        print(label)
        print(boxes)
        print(" ************** ")
        print()
        cv2.imshow("image", cur_rect)
        cv2.waitKey(0)
        for b in boxes.splitlines():
            b = b.split(' ')
            text = b[0]
            try:
                b[0] = float(b[0])
            except ValueError:
                continue
            cur_x1 = (int(b[1]), h - int(b[2]))
            full_x1 = (cur_x1[0] + crop_rect[2], cur_x1[1] + crop_rect[0])
            cur_x2 = (int(b[3]), h - int(b[4]))
            full_x2 = (cur_x2[0] + crop_rect[2], cur_x2[1] + crop_rect[0])
            drawn_img = cv2.rectangle(drawn_img, full_x1, full_x2, thickness=2, color=colors_list[pred])
            drawn_img = cv2.putText(drawn_img, text,
                                    (full_x1[0], full_x2[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12), 2)
        # cv2.imshow("rect", cur_rect)
        # cv2.waitKey(0)
        # x1 = (pred_box[0], pred_box[1])
        # x2 = (pred_box[2], pred_box[3])
        x1 = (crop_rect[2], crop_rect[0])
        x2 = (crop_rect[3], crop_rect[1])
        drawn_img = cv2.rectangle(drawn_img, x1, x2, thickness=2, color=colors_list[pred])
        drawn_img = cv2.putText(drawn_img, label,
                                (x1[0] - 15, x1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    cv2.imshow("image", drawn_img)
    cv2.waitKey(0)
    return drawn_img


def create_tagged_video(model, conf_thres, nms_thres, img_size: int, class_list, video_path: str):
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        image = find_numbers(image, output, class_list)

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


if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR/tesseract.exe'
    image = cv2.imread("data\\custom\\images\\frame1008.jpg")

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # exit(45)

    class_names = load_classes("data/custom/classes.names")
    print(class_names)
    labels_path = "data\\custom\\labels\\frame70.txt"
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
    print(output)
    find_numbers(image, output, class_names, True)
    # tag_video()
