import json
from sklearn.neighbors import KNeighborsClassifier
import cv2


class Model:

    def __init__(self):
        path = r"digits labeled.json"
        img_set = set()
        with open(path, "r") as f:
            data = json.load(f)
        data = data["images"]
        self.targets = []
        self.train_img = []
        for x in data:
            for t in x["tags"]:
                self.targets.append(t)
        for img in data:
            img_path = r"data/custom/digits/"+img["image_name"]
            frame_name = img["image_name"].split("_")[0]
            img_set.add(frame_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (30, 30))
            image = image.flatten()
            self.train_img += [image]
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.model.fit(self.train_img, self.targets)

    def predict(self, image):
        image = image.copy()
        crop = cv2.resize(image, (30, 30))
        crop = crop.flatten().reshape(1, -1)
        tag = self.model.predict(crop)
        return tag


if __name__ == "__main__":
    model = Model()
