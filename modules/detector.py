import torch
import cv2
import numpy as np
from tracker import *

tracker = EuclideanDistTracker()
k_tracker = cv2.TrackerKCF_create()


class Detector:
    def __init__(self, model_name, stream):

        self.model_path = "./utils/"
        self.stream = stream
        self.model_name = model_name
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=self.model_path + self.model_name
        )
        self.model.classes = 2

        self.detect_kcf()

    def detect(self):
        cap = cv2.VideoCapture(self.stream)
        while True:
            save = []
            ret, frame = cap.read()

            results = self.model(frame)
            _results = np.squeeze(results.render())

            detections = results.pandas().xyxy[0].values.tolist()
            for detec in detections:
                save.append(detec)

            boxes_ids = tracker.update(save)
            for box_id in boxes_ids:
                x, y, xm, ym, id = box_id
                cv2.putText(
                    _results,
                    str(id),
                    (int(x), int(y) - 15),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

            cv2.imshow("stream", _results)
            key = cv2.waitKey(30)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_kcf(self):
        cap = cv2.VideoCapture(self.stream)
        ret, frame = cap.read()
        results = self.model(frame)
        _results = np.squeeze(results.render())
        detections = results.pandas().xyxy[0].values.tolist()
        xi, yi, xm, ym, conf, cls, nm = detections[0]

        bbox = np.array([int(xi), int(yi), int(xm - xi), int(ym - yi)])
        k_tracker.init(frame, bbox)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            (success, box) = k_tracker.update(frame)
            if success:
                (x, y, w, h) = [int(a) for a in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("stream", frame)

            key = cv2.waitKey(30)
            if key == 27:
                break


if __name__ == "__main__":
    Detector("yolov5n6.pt", "./data/traffic.mp4")
