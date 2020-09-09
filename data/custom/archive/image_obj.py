class ImageObj:

    def __init__(self, img_dict: dict):
        self.data = img_dict
        self.name = img_dict["image_name"]
        self.hr_box = []
        self.pulse_box = []
        self.abp_box = []
        self.spo2_box = []
        self.pap_box = []
        self.et_co2_box = []
        self.aw_rr_box = []
        self.find_boxes()

    def find_boxes(self):
        for label in self.data["labels"]:
            if label["class_name"] == "Heart rate":
                self.hr_box = label["bbox"]
            if label["class_name"] == "Pulse":
                self.pulse_box = label["bbox"]
            if label["class_name"] == "etCO2":
                self.et_co2_box = label["bbox"]
            if label["class_name"] == "awRR":
                self.aw_rr_box = label["bbox"]
            if label["class_name"] == "ABP":
                self.abp_box = label["bbox"]
            if label["class_name"] == "PAP":
                self.pap_box = label["bbox"]
            if label["class_name"] == "SpO2":
                self.spo2_box = label["bbox"]
