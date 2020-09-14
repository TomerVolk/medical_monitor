import cv2
import json
from data.custom.archive.image_obj import ImageObj
import numpy as np
from os import listdir


def create_db(path, inter):
    cap = cv2.VideoCapture(path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % inter == 0:
            cv2.imwrite('data/img/frame' + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def reformat_bbox(bbox_lst: list):
    bbox_lst = bbox_lst[0]
    ans = [0] * 4
    ans[0] = (bbox_lst[0] + bbox_lst[2]) / 2
    ans[1] = (bbox_lst[1] + bbox_lst[3]) / 2
    ans[2] = (bbox_lst[2] - bbox_lst[0])
    ans[3] = bbox_lst[3] - bbox_lst[1]
    ans[0] = ans[0] / 640
    ans[1] = ans[1] / 360
    ans[2] = ans[2] / 640
    ans[3] = ans[3] / 360
    return ans


def create_dict(img_obj_lst: list):
    ans_dict = {}

    hr_boxes = [x.hr_box for x in img_obj_lst]
    bbox_hr = reformat_bbox(hr_boxes)
    ans_dict["Heart rate"] = bbox_hr

    pulse_boxes = [x.pulse_box for x in img_obj_lst]
    bbox_pulse = reformat_bbox(pulse_boxes)
    ans_dict["Pulse"] = bbox_pulse

    abp_boxes = [x.abp_box for x in img_obj_lst]
    bbox_abp = reformat_bbox(abp_boxes)
    ans_dict["ABP"] = bbox_abp

    spo2_boxes = [x.spo2_box for x in img_obj_lst]
    bbox_spo2 = reformat_bbox(spo2_boxes)
    ans_dict["SpO2"] = bbox_spo2

    pap_boxes = [x.pap_box for x in img_obj_lst]
    bbox_pap = reformat_bbox(pap_boxes)
    ans_dict["PAP"] = bbox_pap

    et_boxes = [x.et_co2_box for x in img_obj_lst]
    bbox_pap = reformat_bbox(et_boxes)
    ans_dict["etCO2"] = bbox_pap

    aw_boxes = [x.aw_rr_box for x in img_obj_lst]
    bbox_aw = reformat_bbox(aw_boxes)
    ans_dict["awRR"] = bbox_aw

    return ans_dict


def get_labels_dict():
    ans_dict = {}
    labels_path = "C:\\Users\\tomer\\Documents\\medical_monitor\\data\\custom\\classes.names"
    with open(labels_path) as f:
        labels_list = f.readlines()
    labels_list = [x.strip() for x in labels_list]
    for k, v in enumerate(labels_list):
        ans_dict[v] = k
    return ans_dict


def create_out_str(images):
    images_loc = "C:\\Users\\tomer\\Documents\\medical_monitor\\data\\custom\\images"
    img_dict = create_dict(images)
    labels_dict = get_labels_dict()
    out_str = ""
    for label in labels_dict.keys():
        bbox = img_dict[label]
        bbox = [str(x) for x in bbox]
        out_str += str(labels_dict[label]) + " " + " ".join(bbox) + "\n"
    out_str = out_str.strip()
    for file in listdir(images_loc):
        full_path = images_loc + "\\" + file
        full_path = full_path.replace("images", "labels").replace(".jpg", ".txt")
        with open(full_path, "w") as f:
            f.write(out_str)


if __name__ == '__main__':
    file_path = "images labeled hasti.json"
    with open(file_path) as f:
        data = json.load(f)
    classes = data["label_classes"]
    classes = [x["class_name"] for x in classes]
    imgs = data["images"]
    img_obj = [ImageObj(x) for x in imgs]
    label_dict = create_dict(img_obj)
    create_out_str(img_obj)

