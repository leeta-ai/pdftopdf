import collections
import copy
import json
import time
import unicodedata

import cv2
import math
import numpy as np


# from spire.doc import *
# from spire.doc.common import *


def sorted_boxes(dt_boxes):
    # 对一张图像上所有的检测框进行排序
    dt_boxes = np.array(dt_boxes)
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    # 将图像按文本框坐标仿射变换并裁切
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height),
                                  borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def draw_quadrangle(image, four_point_coordinates, color=(0, 255, 0)):
    # 画四边形
    thickness = 2
    lineType = cv2.LINE_8
    # 在图像上画线
    pt0 = [int(four_point_coordinates[0][0]), int(four_point_coordinates[0][1])]
    pt1 = [int(four_point_coordinates[1][0]), int(four_point_coordinates[1][1])]
    pt2 = [int(four_point_coordinates[2][0]), int(four_point_coordinates[2][1])]
    pt3 = [int(four_point_coordinates[3][0]), int(four_point_coordinates[3][1])]
    cv2.line(image, pt0, pt1, color, thickness, lineType)
    cv2.line(image, pt1, pt2, color, thickness, lineType)
    cv2.line(image, pt2, pt3, color, thickness, lineType)
    cv2.line(image, pt3, pt0, color, thickness, lineType)


def draw_rectangle(image, four_point_coordinates):
    # 画矩形
    color = (0, 255, 0)
    thickness = 2
    lineType = cv2.LINE_8
    # 在图像上画线
    ltx = min(four_point_coordinates[0][0], four_point_coordinates[3][0])
    lty = min(four_point_coordinates[0][1], four_point_coordinates[1][1])
    rbx = max(four_point_coordinates[1][0], four_point_coordinates[2][0])
    rby = max(four_point_coordinates[2][1], four_point_coordinates[3][1])
    cv2.rectangle(image, (ltx, lty), (rbx, rby), color, thickness, lineType)


def coordinate_cluster(coordinates):
    # 对一页图中的检测框进行聚类
    # coordinates:list:[np,np,...]
    import numpy as np
    from sklearn.cluster import DBSCAN

    points = []
    for coo in coordinates:
        points.append([int((coo[0][0] + coo[1][0] + coo[2][0] + coo[3][0]) / 4),
                       int((coo[0][1] + coo[1][1] + coo[2][1] + coo[3][1]) / 4)])

    dbscan = DBSCAN(eps=400, min_samples=3)  # 设置半径和最小样本数

    # 进行聚类
    labels = dbscan.fit_predict(points)  # 和points包含一样多的元素个数，元素是类别
    return labels


def sert(section, text_w, text_h, ltx, lty, text, font_size=8):
    # 在当前页面添加文本行
    paragraph = section.AddParagraph()
    tb = paragraph.AppendTextBox(text_w, text_h)  # 宽高
    # /设置文本框相对页边距的位置
    tb.Format.HorizontalOrigin = HorizontalOrigin.Margin
    tb.Format.HorizontalPosition = ltx  # 离左边0距离
    tb.Format.VerticalOrigin = VerticalOrigin.Margin
    tb.Format.VerticalPosition = lty  # 离上边50距离

    # //设置文本框填充色、边框颜色及样式
    # tb.Format.LineColor = Color.get_White()  # 文本框颜色
    tb.Format.LineColor = Color.get_Black()  # 文本框颜色
    tb.Format.LineStyle = TextBoxLineStyle.Simple
    tb.Format.FillColor = Color.get_LightGreen()  # 文字背景颜色

    # //在文本框中添加段落及文字
    para = tb.Body.AddParagraph()
    para.Format.HorizontalAlignment = HorizontalAlignment.Distribute  # 两端对齐
    tr = para.AppendText(text)

    # //设置文字格式
    tr.CharacterFormat.FontName = "simsun"
    # tr.CharacterFormat.FontSize = font_size
    tr.CharacterFormat.Bold = True
    tr.CharacterFormat.TextColor = Color.get_Black()  # 文字颜色


def make_new_section():
    # 创建Document对象并加载文档
    document = Document()
    section = document.AddSection()
    section.PageSetup.Margins.All = 0

    paragraph = section.AddParagraph()
    tb = paragraph.AppendTextBox(594, 842)  # 宽高

    # /设置文本框相对页边距的位置
    tb.Format.HorizontalOrigin = HorizontalOrigin.Margin
    tb.Format.HorizontalPosition = 0  # 离左边0距离
    tb.Format.VerticalOrigin = VerticalOrigin.Margin
    tb.Format.VerticalPosition = 0  # 离上边50距离

    # //设置文本框填充色、边框颜色及样式
    tb.Format.LineColor = Color.get_Black()  # 文本框颜色
    tb.Format.LineStyle = TextBoxLineStyle.Simple
    # tb.Format.FillColor = Color.get_LightGreen()  # 文字背景颜色

    return document, section


def make_next_section(document):
    section = document.AddSection()
    return section


def align_coordinates(coo, image_h, image_w):
    # 对齐坐标,将其变成矩形，方便在word中矩形插入,coo：numpy[4,2]
    # word里面字宽高，594, 842
    page_w = 594
    page_h = 842

    ltx = int((coo[0][0] + coo[3][0]) / 2)
    lty = int((coo[0][1] + coo[1][1]) / 2)
    rbx = int((coo[1][0] + coo[2][0]) / 2)
    rby = int((coo[2][1] + coo[3][1]) / 2)
    text_w = rbx - ltx
    text_h = rby - lty

    text_w = int(text_w / image_w * page_w)
    text_h = int(text_h / image_h * page_h)
    ltx = int(ltx / image_w * page_w)
    lty = int(lty / image_h * page_h)

    # lty = int(lty-text_h/2)
    text_h = int(text_h * 1.8)
    if text_w < 0.12 * page_w:
        text_w = int(text_w * 1.1)
    return text_w, text_h, ltx, lty


def save_document(document, save_word="Textbox.docx", save_pdf="PDF1.pdf"):
    # 保存文件
    document.SaveToFile(save_word, FileFormat.Docx)
    document.SaveToFile(save_pdf, FileFormat.PDF)

    # 关闭并释放资源
    document.Close()
    document.Dispose()


def padding(img, rate=0.1):
    h, w = img.shape[0:2]
    new_h = h + int(h * rate) * 2
    new_w = w + int(w * rate) * 2
    img_padding = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
    img_padding[int(rate * h):int(rate * h) + h, int(rate * w):int(rate * w) + w] = img
    return img_padding


def padding_size(img, padding=25):
    h, w = img.shape[0:2]
    new_h = h + padding * 2
    new_w = w + padding * 2
    img_padding = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
    img_padding[padding:padding + h, padding:padding + w] = img
    return img_padding


def cut_image(image, ltx, lty, rbx, rby, padding=10):
    img = image[lty:rby + 1, ltx:rbx + 1]
    h, w = img.shape[0:2]
    new_h = h + 2 * padding
    new_w = w + 2 * padding
    new_image = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
    new_image[padding:padding + h, padding:padding + w] = img
    return new_image


def restore_location(np_positions, ltx, lty, padding=20):
    np_positions[:, 0] = ltx - padding + np_positions[:, 0]
    np_positions[:, 1] = lty - padding + np_positions[:, 1]
    np_positions[:, 2] = ltx - padding + np_positions[:, 2]
    np_positions[:, 3] = lty - padding + np_positions[:, 3]
    return np_positions


def layout_drawing(image, layout_res, save_path="ii/image_布局图.png"):
    color = (0, 255, 0)
    thickness = 2
    lineType = cv2.LINE_8
    new_image = copy.deepcopy(image)
    for n, layout_re in enumerate(layout_res):
        ltx = math.ceil(layout_re["bbox"][0])
        lty = math.ceil(layout_re["bbox"][1])
        rbx = math.ceil(layout_re["bbox"][2])
        rby = math.ceil(layout_re["bbox"][3])
        label = layout_re["label"]
        cv2.rectangle(new_image, (ltx, lty), (rbx, rby), color, thickness, lineType)
        cv2.putText(new_image, label, (ltx, lty + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.imwrite(save_path, new_image)


def analyze_a_layout(image, layout_re):
    pass


def analyze_paragraphs_and_positions_(list_position_text_prob):
    list_positions = []
    list_text = []
    for position_text_prob in list_position_text_prob:
        list_positions.append(position_text_prob[0])
        list_text.append(position_text_prob[1][0])

    para_begin = [0]
    np_positions = np.array(list_positions)  # -> (n,4,2)

    min_ltx = min(np_positions[:, 0, 0])
    for n in range(len(np_positions) - 1):
        if np_positions[n + 1][0][0] - min_ltx > 50 and abs(np_positions[n + 1][0][1] - np_positions[n][0][1]) > 15:
            para_begin.append(n + 1)

    text = []  # 多段文字
    para_coordinate = []  # 多段的坐标位置元素

    for n in range(len(para_begin) - 1):  # 当只有一段时不执行
        _ = list_text[para_begin[n]:para_begin[n + 1]]
        text.append("".join(_))

        _ = np_positions[para_begin[n]:para_begin[n + 1]]
        ltx = int(np.min(_[:, :, 0]))
        lty = int(np.min(_[:, :, 1]))
        rbx = int(np.max(_[:, :, 0]))
        rby = int(np.max(_[:, :, 1]))
        para_coordinate.append([ltx, lty, rbx, rby])

    _ = list_text[para_begin[-1]:]
    text.append("".join(_))  # 一行一行的变成一段一段的

    _ = np_positions[para_begin[-1]:]
    ltx = int(np.min(_[:, :, 0]))
    lty = int(np.min(_[:, :, 1]))
    rbx = int(np.max(_[:, :, 0]))
    rby = int(np.max(_[:, :, 1]))
    para_coordinate.append([ltx, lty, rbx, rby])

    return np.array(para_coordinate), text


def analyze_paragraphs_and_positions__(list_position_text_prob):
    list_positions = []
    list_text = []
    for position_text_prob in list_position_text_prob:
        if position_text_prob[1][1] < 0.66:
            continue
        list_positions.append(position_text_prob[0])
        list_text.append(position_text_prob[1][0])

    para_begin = [0]
    np_positions = np.array(list_positions)  # -> (n,4,2)

    if len(np_positions) > 1:
        for n in range(1, len(np_positions)):
            up_x = np_positions[n - 1][0][0]
            up_y = np_positions[n - 1][0][1]
            current_x = np_positions[n][0][0]
            current_y = np_positions[n][0][1]
            if n == len(np_positions) - 1:  # 最后一个了
                if (current_x - up_x > 10) and (abs(current_y - up_y) > 5):
                    para_begin.append(n)
                elif (abs(current_x - up_x) < 3) and (abs(current_y - up_y) > 5) and para_begin[-1] == n - 1:
                    para_begin.append(n)
            else:
                down_x = np_positions[n + 1][0][0]
                down_y = np_positions[n + 1][0][1]
                if abs(current_y - up_y) < 3:  # 同一行
                    continue

                if ((current_x - up_x > 10) and (abs(current_y - up_y) > 5)) or (
                        (current_x - down_x > 10) and (abs(down_y - current_y) > 5)):
                    para_begin.append(n)

    text = []  # 多段文字
    para_coordinate = []  # 多段的坐标位置元素

    for n in range(len(para_begin) - 1):  # 当只有一段时不执行
        _ = list_text[para_begin[n]:para_begin[n + 1]]
        text.append("".join(_))

        _ = np_positions[para_begin[n]:para_begin[n + 1]]
        ltx = int(np.min(_[:, :, 0]))
        lty = int(np.min(_[:, :, 1]))
        rbx = int(np.max(_[:, :, 0]))
        rby = int(np.max(_[:, :, 1]))
        para_coordinate.append([ltx, lty, rbx, rby])

    _ = list_text[para_begin[-1]:]
    text.append("".join(_))  # 一行一行的变成一段一段的

    _ = np_positions[para_begin[-1]:]
    ltx = int(np.min(_[:, :, 0]))
    lty = int(np.min(_[:, :, 1]))
    rbx = int(np.max(_[:, :, 0]))
    rby = int(np.max(_[:, :, 1]))
    para_coordinate.append([ltx, lty, rbx, rby])

    return np.array(para_coordinate), text


def analyze_paragraphs_and_positions(list_position_text_prob):
    list_positions = []
    list_text = []
    for n in range(len(list_position_text_prob)):
        if list_position_text_prob[n][1][1] < 0.2:
            continue
        # 一行有多个框，需要合并
        if n == 0:
            list_positions.append(list_position_text_prob[n][0])
            list_text.append(list_position_text_prob[n][1][0])
        else:
            center_h_ = (list_positions[-1][3][1] + list_positions[-1][0][1]) / 2
            center_h = (list_position_text_prob[n][0][3][1] + list_position_text_prob[n][0][0][1]) / 2
            if abs(center_h - center_h_) < 5:  # 一行，需要合并
                list_positions[-1][1] = list_position_text_prob[n][0][1]
                list_positions[-1][2] = list_position_text_prob[n][0][2]
            else:
                list_positions.append(list_position_text_prob[n][0])
                list_text.append(list_position_text_prob[n][1][0])
    # print(360,list_positions)
    # print(360,list_text)

    para_begin = [0]
    np_positions = np.array(list_positions)  # -> (n,4,2)

    if len(np_positions) > 1:
        for n in range(1, len(np_positions)):
            up_x = np_positions[n - 1][0][0]
            up_y = np_positions[n - 1][0][1]
            current_x = np_positions[n][0][0]
            current_y = np_positions[n][0][1]
            if n == len(np_positions) - 1:  # 最后一个了
                if (current_x - up_x > 10) and (abs(current_y - up_y) > 5):
                    para_begin.append(n)
                elif (abs(current_x - up_x) < 3) and (abs(current_y - up_y) > 5) and para_begin[-1] == n - 1:
                    para_begin.append(n)
            else:
                down_x = np_positions[n + 1][0][0]
                down_y = np_positions[n + 1][0][1]
                if abs(current_y - up_y) < 3:  # 同一行
                    continue

                if ((current_x - up_x > 10) and (abs(current_y - up_y) > 5)) or (
                        (current_x - down_x > 10) and (abs(down_y - current_y) > 5)):
                    para_begin.append(n)

    text = []  # 多段文字
    para_coordinate = []  # 多段的坐标位置元素

    for n in range(len(para_begin) - 1):  # 当只有一段时不执行
        _ = list_text[para_begin[n]:para_begin[n + 1]]
        text.append("".join(_))

        _ = np_positions[para_begin[n]:para_begin[n + 1]]
        ltx = int(np.min(_[:, :, 0]))
        lty = int(np.min(_[:, :, 1]))
        rbx = int(np.max(_[:, :, 0]))
        rby = int(np.max(_[:, :, 1]))
        para_coordinate.append([ltx, lty, rbx, rby])

    _ = list_text[para_begin[-1]:]
    text.append("".join(_))  # 一行一行的变成一段一段的

    _ = np_positions[para_begin[-1]:]
    ltx = int(np.min(_[:, :, 0]))
    lty = int(np.min(_[:, :, 1]))
    rbx = int(np.max(_[:, :, 0]))
    rby = int(np.max(_[:, :, 1]))
    para_coordinate.append([ltx, lty, rbx, rby])

    return np.array(para_coordinate), text


def analyze_paragraphs_and_positions_title(list_position_text_prob):
    list_positions = []
    list_text = []
    for position_text_prob in list_position_text_prob:
        list_positions.append(position_text_prob[0])
        list_text.append(position_text_prob[1][0])
    np_positions = np.array(list_positions)  # -> (n,4,2)

    para_coordinate = []  # 多段的坐标位置元素
    text = []  # 多段文字

    ltx = int(np.min(np_positions[:, :, 0]))
    lty = int(np.min(np_positions[:, :, 1]))
    rbx = int(np.max(np_positions[:, :, 0]))
    rby = int(np.max(np_positions[:, :, 1]))
    para_coordinate.append([ltx, lty, rbx, rby])
    text.append("".join(list_text))  # 一行一行的变成一段一段的

    return np.array(para_coordinate), text


def adjust_single_line_position(list_position_text_prob):
    list_positions = []
    list_text = []
    for position_text_prob in list_position_text_prob:
        if is_punctuation(position_text_prob[1][0]) or position_text_prob[1][1] < 0.2:
            continue
        list_single_line_position = position_text_prob[0]  # -> list
        np_single_line_position = np.array(list_single_line_position)  # -> np:(4,2)
        ltx = int(np.min(np_single_line_position[:, 0]))
        lty = int(np.min(np_single_line_position[:, 1]))
        rbx = int(np.max(np_single_line_position[:, 0]))
        rby = int(np.max(np_single_line_position[:, 1]))
        list_positions.append([ltx, lty, rbx, rby])
        list_text.append(position_text_prob[1][0])
    return np.array(list_positions), list_text


def drawing_rectangle(image, position, label=None, color=(0, 255, 0)):
    thickness = 3 # 线的粗细占一个像素
    lineType = cv2.LINE_8
    ltx, lty, rbx, rby = position
    cv2.rectangle(image, (ltx, lty), (rbx, rby), color, thickness, lineType)
    # label = None
    if label is not None:
        cv2.putText(image, label, (ltx, lty + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


def write_txt(label, position, text, file_text="data.json", font_size="", end=False):
    ltx, lty, rbx, rby = position
    ltx = int(ltx)
    lty = int(lty)
    rbx = int(rbx)
    rby = int(rby)
    data = {"label": label, "position": f"[{ltx}, {lty}, {rbx}, {rby}]", "text": text, "font_size": font_size}
    # 打开文件并写入JSON数据
    with open(file_text, "a+", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        if not end:
            file.write(",\n")


def get_write_content(label, position, text, font_size="", indent=False):
    ltx, lty, rbx, rby = position  # 四个int
    data = {"label": label, "position": f"[{ltx}, {lty}, {rbx}, {rby}]", "text": text, "font_size": font_size,
            "indent": indent}
    return data


accuracy = 0
num_all_char = 0


def write_json(image, width, height, list_write_contents, file_json, ocr, gpt):
    global accuracy
    global num_all_char
    data = {"width": width, "height": height}
    with open(file_json, "w", encoding="utf-8") as file:
        for n, content in enumerate(list_write_contents):
            # content:dict={'label': 'figure', 'position': '[0, 0, 1695, 2435]', 'text': '', 'font_size': ''}
            if (content["label"] == "figure") or (content["label"] == "table"):
                data[f"content_{n + 1}"] = content
            elif len(list_write_contents) == 1 and content["label"] == "title":
                # text_old = content["text"]
                # content["text"] = gpt.correct(content["text"])
                # text_new = content["text"]
                # for char_old, char_new in zip(text_old, text_new):
                #     num_all_char = num_all_char + 1
                #     if char_old == char_new:
                #         accuracy = accuracy + 1

                content["font_size"] = 22
                data[f"content_{n + 1}"] = content
            else:
                # text_old = content["text"]
                # content["text"] = gpt.correct(content["text"])
                # text_new = content["text"]
                # for char_old, char_new in zip(text_old, text_new):
                #     num_all_char = num_all_char + 1
                #     if char_old == char_new:
                #         accuracy = accuracy + 1

                ltx, lty, rbx, rby = eval(content["position"])  # 一个布局的坐标位置
                img = image[lty:rby + 1, ltx:rbx + 1]  # 一个布局小图
                char_size = font_size(img)  # 求当前布局小图中字的像素高度
                if char_size >= 20 and char_size <= 29:
                    size = 8
                elif char_size >= 30 and char_size <= 39:
                    size = 10
                elif char_size >= 40 and char_size <= 49:
                    size = 12  # 12号字体
                elif char_size >= 50 and char_size <= 59:
                    size = 14
                elif char_size >= 60 and char_size <= 69:
                    size = 16
                elif char_size >= 70 and char_size <= 79:
                    size = 18
                elif char_size >= 80 and char_size <= 89:
                    size = 20
                elif char_size >= 70 and char_size <= 99:
                    size = 22
                else:
                    size = 24
                content["font_size"] = size

                if content["indent"] == None:
                    if_indent = analysis_indent(img, ocr)  # 是否缩进
                    content["indent"] = if_indent

                data[f"content_{n + 1}"] = content
            print(f"内容:{n + 1}", data[f"content_{n + 1}"])
        json.dump(data, file, indent=4, ensure_ascii=False)
    # if num_all_char == 0:
    #     print("ocr准确率:1.0")
    # else:
    #     print("ocr准确率:", accuracy, num_all_char, accuracy / num_all_char)


def get_dict_result(image, width, height, list_write_contents, ocr):
    data = {"width": width, "height": height}
    for n, content in enumerate(list_write_contents):
        # content:dict={'label': 'figure', 'position': '[0, 0, 1695, 2435]', 'text': '', 'font_size': ''}
        if (content["label"] == "figure") or (content["label"] == "table"):
            data[f"content_{n + 1}"] = content
        elif len(list_write_contents) == 1 and content["label"] == "title":
            content["font_size"] = 22
            data[f"content_{n + 1}"] = content
        else:
            ltx, lty, rbx, rby = eval(content["position"])  # 一个布局的坐标位置
            img = image[lty:rby + 1, ltx:rbx + 1]  # 一个布局小图
            char_size = font_size(img)  # 求当前布局小图中字的像素高度
            if char_size >= 20 and char_size <= 29:
                size = 8
            elif char_size >= 30 and char_size <= 39:
                size = 10
            elif char_size >= 40 and char_size <= 49:
                size = 12  # 12号字体
            elif char_size >= 50 and char_size <= 59:
                size = 14
            elif char_size >= 60 and char_size <= 69:
                size = 16
            elif char_size >= 70 and char_size <= 79:
                size = 18
            elif char_size >= 80 and char_size <= 89:
                size = 20
            elif char_size >= 70 and char_size <= 99:
                size = 22
            else:
                size = 24
            content["font_size"] = size

            if content["indent"] == None:
                if_indent = analysis_indent(img, ocr)  # 是否缩进
                content["indent"] = if_indent

            data[f"content_{n + 1}"] = content
    return data


def detect_box_font(ocr, image):
    list_positions = ocr.ocr(image, det=True, rec=False, cls=False, bin=False, inv=False)[0]  # -> list
    list_h = []
    list_box_font_sizes = []
    for n, position in enumerate(list_positions):  # position:list
        ltx = min(position[0][0], position[3][0])
        lty = min(position[0][1], position[1][1])
        rbx = max(position[1][0], position[2][0])
        rby = max(position[2][1], position[3][1])
        h = int(rby - lty + 1)
        list_box_font_sizes.append([ltx, lty, rbx, rby, h])
        list_h.append(h)
    list_h_count = list(collections.Counter(list_h))
    dict_h_count = dict(collections.Counter(list_h))
    print(354354, list_h_count)
    print(354354, dict_h_count)
    h_12 = list_h_count[0]  # 当前页面中12号字体的高度


def detect_box_font_text(ocr, image):
    list_position_text_scores = ocr.ocr(image, det=True, rec=True, cls=False, bin=False, inv=False)[0]  # -> list
    list_h = []
    list_box_text_font_sizes = []
    for n, position_text_score in enumerate(list_position_text_scores):  # position:list
        position = position_text_score[0]
        text, score = position_text_score[1]
        ltx = min(position[0][0], position[3][0])
        lty = min(position[0][1], position[1][1])
        rbx = max(position[1][0], position[2][0])
        rby = max(position[2][1], position[3][1])
        h = int(rby - lty + 1)
        list_box_text_font_sizes.append([ltx, lty, rbx, rby, text, h])
        list_h.append(h)
    dict_h_count = dict(collections.Counter(list_h))
    print(3765376, dict_h_count)
    list_h_count = list(sorted(dict_h_count.items(), key=lambda item: int(item[1]), reverse=True))
    print(37777, list_h_count)
    h_12 = list_h_count[0][0]  # 当前页面中12号字体的高度

    for n in range(len(list_box_text_font_sizes)):
        print(380, h_12, list_box_text_font_sizes[n])
        h = list_box_text_font_sizes[n][5]
        size = int((h - h_12) / 3)
        print(384384, h, h_12, size)
        list_box_text_font_sizes[n][5] = 12 + size
        print(384, h_12, list_box_text_font_sizes[n])


def font_size(image):
    height, width = image.shape[0:2]
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)[1]
    image_binary = cv2.bitwise_not(image_binary)

    kernel = np.ones((3, 3), dtype=np.uint8)
    image_binary = cv2.dilate(image_binary, kernel, iterations=2)  # 膨胀
    # image_binary = cv2.erode(image_binary, kernel, iterations=1)  # 腐蚀

    # image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel)
    # 闭运算 = 先膨胀运算，再腐蚀运算（看上去将两个细微连接的图块封闭在一起）

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_binary, connectivity=8)

    sizes = []
    for num in range(1, num_labels):
        # ii = np.where(labels == num, 255, 0)
        aa = np.where(labels == num)  # -> tuple:(np, np)
        row = aa[0]
        column = aa[1]
        h = max(row) - min(row) + 1
        w = max(column) - min(column) + 1
        sizes.append(max(h, w))
    dict_size_count = collections.Counter(sizes)
    list_size_count = list(sorted(dict_size_count.items(), key=lambda item: int(item[1]), reverse=True))
    if len(list_size_count) == 0:
        return 25
    return list_size_count[0][0]


import string


def is_punctuation(char="."):
    english_punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""  # 英文标点
    chinese_punctuation = r"""！？｡。 ●■·￥＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."""
    flag = (char in english_punctuation) or (char in chinese_punctuation)
    return flag


def analysis_indent(img, ocr):
    img = padding_size(img)
    # cv2.imwrite("1.png",img)
    # time.sleep(10)
    list_positions = ocr.ocr(img, det=True, rec=False, cls=False, bin=False, inv=False)[0]
    if (list_positions is None) or (len(list_positions) <= 1):
        return False
    # ltx_first = int(min(list_positions[0][0][0], list_positions[0][3][0]))
    ltx_first = int(np.max(np.array(list_positions)[:, 0, 0]))
    ltx_min = int(np.min(np.array(list_positions)[:, 0, 0]))
    ltx_first_index = int(np.argmax(np.array(list_positions)[:, 0, 0]))
    ltx_min_index = int(np.argmin(np.array(list_positions)[:, 0, 0]))
    lty_first = np.array(list_positions)[ltx_first_index, 0, 1]
    lty_min = np.array(list_positions)[ltx_min_index, 0, 1]
    # print(608,ltx_first , ltx_min ,lty_first, lty_min)

    # print(602, len(list_positions), ltx_first, ltx_min)
    if (ltx_first - ltx_min > 25) and (abs(lty_first - lty_min) > 5):
        # cv2.imwrite("1.png", img)
        # print(ltx_first, ltx_min, len(list_positions))
        # for i in list_positions:
        #     print(i)
        # time.sleep(10000000)
        return True
    else:
        return False
    # h, w = img.shape[0:2]
    # if h < 10 or w < 15:
    #     return False
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]
    # # cv2.imwrite("1.png", img_binary)
    # # time.sleep(1)
    # img = img_binary[0:10, 0:15]
    # print(597, dict(collections.Counter(img.flatten())))
    # count = dict(collections.Counter(img.flatten()))[255]
    # if count / (10 * 15) > 0.1:
    #     return True
    # else:
    #     return False


if __name__ == '__main__':
    chars = " "
    for c in chars:
        print(c, is_punctuation(c))
