import math
import threading
import time
import logging

logging.disable(logging.CRITICAL)

import cv2
import glob
from doclayout_yolo import YOLOv10


class InferenceLayout:
    def __init__(self, weight="weights/doclayout_yolov10.pt"):
        self.lock = threading.Lock()
        self.model = YOLOv10(weight)  # 最长边为1024
        self.mapping = ["title", "text", "abandon", "figure", "figure_caption", "table", "table_caption",
                        "table_footnote", "isolate_formula", "formula_caption"]
        self.dict_mapping = {"title": "标题",
                             # "plain_text": "文本",
                             "text": "文本",
                             "abandon": "页眉、页脚、页码、注释",
                             "figure": "图片",
                             "figure_caption": "图片描述",
                             "table": "表格",
                             "table_caption": "表格描述",
                             "table_footnote": "表格注释",
                             "isolate_formula": "行间公式",
                             "formula_caption": "行间公式标号"}
        print("infer_doclayout_yolov10.py, 布局检测模型初始化成功")

    def forward(self, image):
        with self.lock:  # 确保线程块是安全的
            h, w = image.shape[0:2]
            list_position_score_labels = []
            det_res = self.model.predict(image, imgsz=1024, conf=0.2, device="cuda:0")[0]
            boxes = det_res.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()

            for n, layout in enumerate(xyxy):
                ltx = int(layout[0])
                lty = int(layout[1])
                rbx = math.ceil(layout[2])
                rby = math.ceil(layout[3])
                label = int(cls[n])
                score = round(float(conf[n]), 5)
                english_label = self.mapping[label]
                list_position_score_labels.append([ltx, lty, rbx, rby, score, english_label])

                # 去除框内框
            # list_position_score_labels = self.remove_inside_box(list_position_score_labels)
            # list_position_score_labels = self.remove_inside_box(list_position_score_labels)

            # 重叠的框直接去掉,就当检测不到,交给ocr去检测识别
            list_position_score_labels = self.remove_overlap(list_position_score_labels)

            for n, layout in enumerate(list_position_score_labels):
                ltx, lty, rbx, rby, score, label = layout
                img = image[lty:rby, ltx:rbx]
                if label in ["figure", "table"]:
                    continue
                num_line = self.count_lines_by_projection(img)
                if num_line == 1:
                    ltx = max(0, ltx - 8)
                    rbx = min(w, rbx + 8)
                    ltx, lty, rbx, rby = self.get_single_line_precise_coord(ltx, lty, rbx, rby, image)
                    list_position_score_labels[n][0:4] = [ltx, lty, rbx, rby]

            return list_position_score_labels

    def iou(self, pos1, pos2):
        ltx = max(pos1[0], pos2[0])
        lty = max(pos1[1], pos2[1])
        rbx = min(pos1[2], pos2[2])
        rby = min(pos1[3], pos2[3])
        area_inter = max(0, rbx - ltx + 1) * max(0, rby - lty + 1)
        # iou_ = area_inter / ((pos1[2] - pos1[0] + 1) * (pos1[3] - pos1[1] + 1) + (pos2[2] - pos2[0] + 1) * (
        #         pos2[3] - pos2[1] + 1) - area_inter)
        # 相交面积/小框面积
        area_box1 = (pos1[2] - pos1[0] + 1) * (pos1[3] - pos1[1] + 1)
        area_box2 = (pos2[2] - pos2[0] + 1) * (pos2[3] - pos2[1] + 1)
        iou_ = area_inter / min(area_box1, area_box2)
        return iou_

    def remove_inside_box(self, list_position_score_labels):
        list_position_score_labels = list(
            sorted(list_position_score_labels, key=lambda item: (item[2] - item[0]) * (item[3] - item[1])))

        num = len(list_position_score_labels)
        for i in range(num):
            flag = True
            now_pos = list_position_score_labels.pop(0)
            for j in range(num - i - 1):
                pos = list_position_score_labels[j]
                if self.iou(now_pos[0:4], pos[0:4]) > 0.05:
                    ltx = min(now_pos[0], pos[0])
                    lty = min(now_pos[1], pos[1])
                    rbx = max(now_pos[2], pos[2])
                    rby = max(now_pos[3], pos[3])
                    list_position_score_labels[j][0:4] = [ltx, lty, rbx, rby]
                    flag = False
                    break
            if flag:
                list_position_score_labels.append(now_pos)
        return list_position_score_labels

    def get_single_line_precise_coord(self, ori_ltx, ori_lty, ori_rbx, ori_rby, image):
        # ori_ltx, ori_lty, ori_rbx, ori_rby是在image上的坐标,左闭右开,上闭下开
        padding_size = 20
        height, width = image.shape[0:2]
        # 对小图进行灰度图并二值化
        new_ltx = max(0, ori_ltx - padding_size)
        new_lty = max(0, ori_lty - padding_size)
        new_rbx = max(width, ori_rbx + padding_size)
        new_rby = max(height, ori_rby + padding_size)
        img = image[new_lty:new_rby, new_ltx:new_rbx]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ltx = ori_ltx - new_ltx
        lty = ori_lty - new_lty
        rbx = ori_rbx - new_ltx
        rby = ori_rby - new_lty

        h, w = binary.shape[0:2]
        padding_size = 80

        # 左：
        _ = binary[lty:rby, ltx: ltx + 1]
        contain_char = (_ == 0).any()
        if contain_char:  # 有字，左移
            for i in range(1, padding_size):
                if ltx - i >= 0:
                    _ = binary[lty:rby, ltx - i:ltx - i + 1]
                    contain_char = (_ == 0).any()
                    if contain_char:
                        continue
                    else:
                        ltx = ltx - i + 1
                        break
                else:
                    break
        else:  # 无字，右移
            for i in range(1, padding_size):
                if ltx + i < w:
                    _ = binary[lty:rby, ltx + i:ltx + i + 1]
                    contain_char = (_ == 0).any()
                    if contain_char:  # 有字，停止
                        ltx = ltx + i
                        break
                    else:
                        continue
                else:
                    break

        # 右：
        _ = binary[lty:rby, rbx - 1: rbx]
        contain_char = (_ == 0).any()
        if contain_char:  # 最右边有字，右移
            for i in range(padding_size):
                if rbx + i < w:
                    _ = binary[lty:rby, rbx + i:rbx + i + 1]
                    contain_char = (_ == 0).any()
                    if contain_char:  # 有字，右移
                        continue
                    else:
                        rbx = rbx + i
                        break
                else:
                    break
        else:  # 最右边无字，左移
            for i in range(padding_size):
                if rbx - i >= 0:
                    _ = binary[lty:rby, rbx - i:rbx - i + 1]
                    contain_char = (_ == 0).any()
                    if contain_char:  # 有字，停止
                        rbx = rbx - i + 1
                        break
                    else:
                        continue
                else:
                    break

        # 上：
        _ = binary[lty:lty + 1, ltx:rbx]
        contain_char = (_ == 0).any()
        if contain_char:  # 有字，上移
            for i in range(1, padding_size):
                if lty - i <= 0:
                    _ = binary[lty - i:lty - i + 1, ltx:rbx]
                    contain_char = (_ == 0).any()
                    if contain_char:  # 有字，继续上移
                        continue
                    else:
                        lty = lty - i + 1
                        break
                else:
                    break
        else:  # 无字，下移
            for i in range(1, padding_size):
                if lty + i < h:
                    _ = binary[lty + i:lty + i + 1, ltx:rbx]
                    contain_char = (_ == 0).any()
                    if contain_char:  # 有字，停止
                        lty = lty + i
                        break
                    else:
                        continue
                else:
                    break

        # 下：
        _ = binary[rby - 1:rby, ltx:rbx]
        contain_char = (_ == 0).any()
        if contain_char:  # 有字，下移
            for i in range(padding_size):
                if rby + i < h:
                    _ = binary[rby + i:rby + i + 1, ltx:rbx]
                    contain_char = (_ == 0).any()
                    if contain_char:  # 有字，继续下移
                        continue
                    else:
                        rby = rby + i
                        break
                else:
                    break
        else:  # 无字，上移
            for i in range(padding_size):
                if rby - i >= 0:
                    _ = binary[rby - i:rby - i + 1, ltx:rbx]
                    contain_char = (_ == 0).any()
                    if contain_char:  # 无字，停止
                        rby = rby - i + 1
                        break
                    else:
                        continue
                else:
                    break

        return ltx + new_ltx, lty + new_lty, rbx + new_ltx, rby + new_lty

    def count_lines_by_projection(self, img):  # 判断图中段落有几行
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h, w = binary.shape[0:2]
            if h > w:
                return 2

            begin = h // 2
            up_black = False  # 上空白
            down_black = False
            for i in range(1, begin):
                up = begin - i
                down = begin + i
                _ = binary[up:up + 1]
                contain_char = (_ == 0).any()
                if contain_char:
                    if up_black:  # 经过了空白又出现字
                        return 2
                else:
                    up_black = True

                _ = binary[down:down + 1]
                contain_char = (_ == 0).any()
                if contain_char:
                    if down_black:  # 经过了空白又出现字
                        return 2
                else:
                    down_black = True
            return 1
        except Exception:
            return 2

    def remove_overlap(self, list_position_score_labels):
        overlap_index = set()
        for i, (ltx_i, lty_i, rbx_i, rby_i, score_i, label_i) in enumerate(list_position_score_labels):
            for j, (ltx_j, lty_j, rbx_j, rby_j, score_j, label_j) in enumerate(list_position_score_labels):
                if j > i:
                    inter = self.iou((ltx_i, lty_i, rbx_i, rby_i), (ltx_j, lty_j, rbx_j, rby_j))
                    if inter > 0.1 and inter < 0.95:
                        overlap_index.add(i)
                        overlap_index.add(j)
                    elif inter > 0.95:
                        if score_i > score_j:
                            overlap_index.add(j)
                        else:
                            overlap_index.add(i)

        new_list_position_score_labels = []
        for i in range(len(list_position_score_labels)):
            if i not in overlap_index:
                new_list_position_score_labels.append(list_position_score_labels[i])
        return new_list_position_score_labels


if __name__ == '__main__':
    model = InferenceLayout()
    folder = "../show_0"
    list_suffix = ["jpg", "png", "jpeg", "PNG"]
    paths = []
    for suffix in list_suffix:
        paths.extend(glob.glob(f"{folder}/*.{suffix}"))
    paths = list(sorted(paths, key=lambda item: int(item.split("/")[-1].split(".")[0])))
    for path in paths:
        t0 = time.time()
        file = path.split("/")[-1]
        image = cv2.imread(path, -1)
        print(path, image.shape)
        list_position_score_labels = model.forward(image)
        print(list_position_score_labels, "\n", time.time() - t0, "\n")

        for n, layout in enumerate(list_position_score_labels):
            ltx = layout[0]
            lty = layout[1]
            rbx = layout[2]
            rby = layout[3]
            score = layout[4]
            label = layout[5]

            thickness = 1
            lineType = cv2.LINE_8
            color = (255, 0, 0)
            cv2.rectangle(image, (ltx, lty), (rbx, rby), color, thickness, lineType)
            cv2.putText(image, label, (ltx + 10, lty + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(f"../show_layout/{file}", image)

