import cv2
import json
from image_obj import ImageObj
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


def get_max_bbox(bbox_lst: list):
    arr = np.array(bbox_lst)
    bbox_1 = arr.min(axis=0)[0:2]
    bbox_2 = arr.max(axis=0)[2:]
    bbox = np.concatenate((bbox_1, bbox_2)).tolist()
    return bbox


def create_dict(img_obj_lst: list):
    ans_dict = {}

    hr_boxes = [x.hr_box for x in img_obj_lst]
    bbox_hr = get_max_bbox(hr_boxes)
    ans_dict["Heart rate"] = bbox_hr

    pulse_boxes = [x.pulse_box for x in img_obj_lst]
    bbox_pulse = get_max_bbox(pulse_boxes)
    ans_dict["Pulse"] = bbox_pulse

    abp_boxes = [x.abp_box for x in img_obj_lst]
    bbox_abp = get_max_bbox(abp_boxes)
    ans_dict["ABP"] = bbox_abp

    spo2_boxes = [x.spo2_box for x in img_obj_lst]
    bbox_spo2 = get_max_bbox(spo2_boxes)
    ans_dict["SpO2"] = bbox_spo2

    pap_boxes = [x.pap_box for x in img_obj_lst]
    bbox_pap = get_max_bbox(pap_boxes)
    ans_dict["PAP"] = bbox_pap

    et_boxes = [x.et_co2_box for x in img_obj_lst]
    bbox_pap = get_max_bbox(et_boxes)
    ans_dict["etCO2"] = bbox_pap

    aw_boxes = [x.aw_rr_box for x in img_obj_lst]
    bbox_aw = get_max_bbox(aw_boxes)
    ans_dict["awRR"] = bbox_aw

    return ans_dict


def create_out_str(classes, images):
    out_dict = {"labels": classes}
    images_loc = "data/img"
    img_list = []
    img_dict = create_dict(images)
    for file in listdir(images_loc):
        cur_dict = {"image_name": file, "labels": img_dict}
        img_list.append(cur_dict)
    out_dict["images"] = img_list
    return out_dict


if __name__ == '__main__':
    file_path = "data/images labeled hasti.json"
    with open(file_path) as f:
        data = json.load(f)
    classes = data["label_classes"]
    classes = [x["class_name"] for x in classes]
    imgs = data["images"]
    img_obj = [ImageObj(x) for x in imgs]
    label_dict = create_dict(img_obj)
    out_dict = create_out_str(classes, img_obj)
    # out_json = json.dumps(out_dict)
    with open("data/images labeled.json", "w") as f:
        json.dump(out_dict, f)

