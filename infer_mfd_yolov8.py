import logging
from ultralytics import YOLO

logging.disable(logging.CRITICAL)


class ModelMFD:  # 数学公式检测
    def __init__(self, weight=None):
        if weight is None:
            weight = "weights/mfd_weight_yolov8.pt"
        self.model = YOLO(weight)
        print("infer_mfd_yolov8.py, 公式检测模型初始化成功")
        # 0:isolated_formula:行内公式
        # 1:embedding_formula:独立公式
        # self.img_size = 1888
        # self.conf_thres = 0.25
        # self.iou_thres = 0.45
        # self.pdf_dpi = 200

    def predict(self, image):
        mfd_result = self.model.predict(image, imgsz=1888, conf=0.25, iou=0.45, verbose=True)[0]
        xyxys = mfd_result.boxes.xyxy.cpu()
        confs = mfd_result.boxes.conf.cpu()
        clas = mfd_result.boxes.cls.cpu()

        if len(xyxys) == 0:
            return []
        mfd_list_position_score_labels = []

        for xyxy, conf, cla in zip(xyxys, confs, clas):
            ltx, lty, rbx, rby = [int(p.item()) for p in xyxy]
            score = round(float(conf.item()), 2)
            if score < 0.5:
                continue
            cla = int(cla.item())
            if cla == 0:
                cla = "embedding"
            else:
                cla = "isolated"
            mfd_list_position_score_labels.append([ltx, lty, rbx, rby, score, cla])
        return mfd_list_position_score_labels


if __name__ == '__main__':
    import cv2
    import time
    import glob

    model = ModelMFD()

    folder = "/home/wangzhisheng/code/OCR项目/测试数据/yolov9/images"

    list_suffix = ["jpg", "png", "jpeg", "PNG"]
    paths = []
    for suffix in list_suffix:
        paths.extend(glob.glob(f"{folder}/*.{suffix}"))
    paths = list(sorted(paths, key=lambda item: int(item.split("/")[-1].split(".")[0])))

    for path in paths:
        file = path.split("/")[-1]
        t0 = time.time()
        image = cv2.imread(path)
        mfd_list_position_score_labels = model.predict(image)
        print(path, image.shape, round(time.time() - t0, 5))
        for position_score_label in mfd_list_position_score_labels:
            print(position_score_label)
            ltx, lty, rbx, rby, score, label = position_score_label
            if label == "embedding_formula":
                continue
            thickness = 1
            lineType = cv2.LINE_8
            color = (255, 0, 0)
            cv2.rectangle(image, (ltx, lty), (rbx, rby), color, thickness, lineType)
            cv2.putText(image, label, (ltx + 10, lty + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(f"show_mfd/{file}", image)
        print()
