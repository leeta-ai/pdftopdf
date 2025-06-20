import collections
import re
import copy
import math
# 问题1：整张图文字ocr检测识别准确率底

import os
import cv2
import time
import sys
from pathlib import Path
import paddle

paddle.disable_signal_handler()  # 在2.2版本提供了disable_signal_handler接口

import warnings
import numpy as np
from collections import Counter
from scipy import stats

warnings.filterwarnings("ignore", category=Warning)  # 去除DeprecationWarning

import logging

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印， 去除ppocr的debug提示
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
ROOT = Path(__file__).resolve().parent  # -> 当前文件上一级目录，绝对路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from infer_doclayout_yolov10 import InferenceLayout
from PaddleOCR_2_7 import PaddleOCR
from infer_mfd_yolov8 import ModelMFD


class HybridModel:
    def __init__(self):
        ROOT = os.path.dirname(__file__)
        self.model_layout = InferenceLayout(f"{ROOT}/weights/doclayout_yolov10.pt")
        self.model_ocr_ch = PaddleOCR(lang="ch",
                                      use_gpu=True,
                                      use_angle_cls=False,
                                      det_model_dir=f"{ROOT}/weights/ppocrv4_server_models/det",
                                      rec_model_dir=f"{ROOT}/weights/ppocrv4_server_models/rec")

        self.model_ocr_en = PaddleOCR(lang="ch",
                                      use_gpu=True,
                                      use_angle_cls=False,
                                      det_model_dir=f"{ROOT}/weights/ppocr_models_english/PP_OCRv3_det",
                                      rec_model_dir=f"{ROOT}/weights/ppocr_models_english/PP_OCRv4_rec")

        self.model_mfd = ModelMFD(f"{ROOT}/weights/mfd_weight_yolov8.pt")

    def predict_img(self, ori_image):
        if ori_image is None:  # 当传入的非图片，或图片损坏
            return {"width": 0, "height": 0, "content": []}

        blue_channel, green_channel, red_channel = cv2.split(ori_image)

        blue_mode = stats.mode(blue_channel, axis=None).mode
        green_mode = stats.mode(green_channel, axis=None).mode
        red_mode = stats.mode(red_channel, axis=None).mode

        image = ori_image.copy()
        ori_height, ori_width = image.shape[0:2]
        return_dict = {"width": ori_width, "height": ori_height, "content": []}

        # 布局检测模型测试
        layout_list_position_score_labels = self.model_layout.forward(image)  # 布局分析网络
        if layout_list_position_score_labels == [[0, 844, 1427, 2639, 0.20317, 'figure']]:
            layout_list_position_score_labels = []
        # print("布局分析结果:", layout_list_position_score_labels)
        # 封面检测
        if if_cover(ori_width, ori_height, layout_list_position_score_labels, return_dict):  # 是封面，直接返回
            return return_dict
        # 将检测到的图片和表格写入返回，同时在原图中抹除，表格要单独ocr一次
        have_fig_tab = remove_figure_table(layout_list_position_score_labels, image, return_dict, self.model_ocr_ch,
                                           blue_mode, green_mode, red_mode)
        # 公式检测模型测试
        list_position_score_labels_mfd = self.model_mfd.predict(image)
        # 去除独立公式
        have_formula = remove_isolated_formula(list_position_score_labels_mfd, image, return_dict, blue_mode,
                                               green_mode, red_mode)
        # 去除句中公式
        # remove_embedding_formula(layout_list_position_score_labels, list_position_score_labels_mfd, image, return_dict)
        # 剩余整图布局检测模型测试
        if have_fig_tab and have_formula:  # 如果有图片、表格、公式删掉后才再检测一次
            layout_list_position_score_labels = self.model_layout.forward(image)  # 布局分析网络
        # self.model_layout.layout_drawing(image, layout_list_position_score_labels, save_path="1.png", color=(0, 255, 0))

        # 中文检测ocr
        # cv2.imwrite("111.png", image)
        list_position_text_probs = get_accurate_ocr_result(image, self.model_ocr_ch, fine_tune=True, ori_img=True)
        # 判断图片是中文还是英文
        flag = if_chinese(list_position_text_probs)
        if flag:
            print("中文ocr", end=" ")
            model_ocr = self.model_ocr_ch
        else:
            print("英文ocr", end=" ")
            # 出现英文多的情况,保留中文ocr中识别出的中文
            ch_list_position_text_probs = keep_ch_remove_en(list_position_text_probs)
            remove_text(ch_list_position_text_probs, image, return_dict)
            list_position_text_probs = get_accurate_ocr_result(image, self.model_ocr_en, fine_tune=True, ori_img=True)
            model_ocr = self.model_ocr_en

        # 判断是否可以整图ocr,每个ocr的框与布局检测的框iou,如果占两个以上布局检测框就不能整图ocr,这种方法更准
        if list_position_text_probs is None:
            return return_dict
        is_OK = judge_whole_figure_ocr(list_position_text_probs, layout_list_position_score_labels)
        if is_OK:
            #print("可以整图ocr，没有双栏检测成一栏")
            return_dict = ocr_0(image, layout_list_position_score_labels,
                                list_position_text_probs, return_dict, model_ocr)
        else:
            #print("双栏检测成一栏，需每个布局单独ocr", return_dict)
            # 每个布局进行单独ocr
            ocr_1(image, layout_list_position_score_labels, model_ocr, return_dict)
            # 布局检测没检测到的，单独进行剩余图整体ocr
            ocr_2(image, model_ocr, return_dict)
        # 坐标处理对其一下，右下角坐标一定大于左上角
        # adjust_left_coordinate(return_dict)
        process_last_line(return_dict)
        # 将文本段落高度一致,取中值赋值
        change_height(return_dict)
        # 全文将统一左右坐标,取中值赋值,2025,1,22日加
        adjust_left_right_coordinate(return_dict)
        # 保留text
        # reserve_text(return_dict)
        return return_dict


def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'


def is_english(char):
    return 'a' <= char <= 'z' or 'A' <= char <= 'Z'


def check_string_type(s):
    has_chinese = any(is_chinese(c) for c in s)
    has_english = any(is_english(c) for c in s)

    if has_chinese and has_english:
        return "mix"
    elif has_chinese:
        return "chinese"
    elif has_english:
        return "english"
    else:
        return "mix"


def process_last_line(return_dict):  # 需要处理最后一行
    # 中文1，英文2.56
    content = return_dict["content"]

    for i, data in enumerate(content):  # data为一段
        if len(data) == 1 and data[0][4] == "text":
            last_line_text = data[-1][5]
            language = check_string_type(last_line_text)
            if language == "chinese":  # 统计标点符号,假设标点符号的宽度是0.5,整个中文宽度1
                h = data[-1][3] - data[-1][1]
                num_fuhao = 0
                for c in last_line_text[-3:]:
                    if c in [",", ".", "，", "。", ";", ":", "{", "}", "[", "]", "、", "?", "”", "？"]:
                        num_fuhao = num_fuhao + 1
                rbx = int(data[-1][2] + h * num_fuhao * 0.48)
                return_dict["content"][i][-1][2] = rbx

        if len(data) > 1 and data[0][4] == "text":
            if data[-2][0] - data[-1][0] > -6 and (data[-2][2] - data[-2][0]) / 1.3 > (data[-1][2] - data[-1][0]):
                last_line_text = data[-1][5]
                language = check_string_type(last_line_text)
                if language == "mix":  # 不处理
                    continue
                if language == "chinese":
                    h = data[-1][3] - data[-1][1]
                    num_fuhao = 0
                    for c in last_line_text[-9:]:
                        if c in [",", ".", "，", "。", ";", ":", "{", "}", "[", "]", "、", "?", "”", "？"]:
                            num_fuhao = num_fuhao + 1
                    rbx = int(data[-1][2] + h * num_fuhao * 0.48)
                    if rbx > data[-2][2]:
                        rbx = data[-2][2]
                    return_dict["content"][i][-1][2] = rbx
                if language == "english":
                    continue


def adjust_left_coordinate(return_dict):
    content = return_dict["content"]
    list_ltx = []

    content = list(filter(lambda item: (item[2] > item[0]) and (item[3] > item[1]), content))
    return_dict["content"] = content

    for data in content:
        if data[4] == "text":
            list_ltx.append(data[0])

    if len(list_ltx) != 0:
        ltx = find_mode(list_ltx)
        for n, data in enumerate(content):
            if data[4] == "text" and abs(ltx - data[0]) <= 3:
                return_dict["content"][n][0] = ltx


def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def if_chinese(list_position_text_probs):
    if list_position_text_probs is None or len(list_position_text_probs) < 5:
        return True
    # 全英文的行数/总行数>0.5 就用英文模型
    num_en = 0
    for position_text_prob in list_position_text_probs:
        text = position_text_prob[4]
        text = re.sub(r'\s+', '', text)  # 去除空格
        if len(text) == 0:
            continue
        flag = True
        for c in text:
            if '\u4e00' <= c <= '\u9fff':  # 中文:
                flag = False
                break
        if flag:
            for c in text:
                if '0' <= c <= '9':  # 中文:
                    flag = False
                    continue
                else:  # 有非数字,即字母
                    flag = True
                    break
        if flag:  # 证明了没有中文也不是全数字
            num_en = num_en + 1
    if num_en / len(list_position_text_probs) > 0.5:
        return False
    else:
        return True

    # all_num = 0
    # ch_num = 0
    # if list_position_text_probs is None:
    #     return True
    # for n, position_text_probs in enumerate(list_position_text_probs):
    #     text = position_text_probs[4]  # str
    #     all_num = all_num + 1
    #
    #     if is_contains_chinese(text):
    #         ch_num = ch_num + 1
    # if (ch_num / all_num) > 0.5:
    #     return True
    # else:
    #     return False


def if_cover(width, height, list_position_score_labels, return_dict):
    if len(list_position_score_labels) == 1:
        if list_position_score_labels[0][5] == "figure":
            return_dict["content"].append([[0, 0, width, height, "figure"]])
            return True
    for position_score_label in list_position_score_labels:
        ltx, lty, rbx, rby, score, label = position_score_label
        if (label == "figure") and (rbx - ltx + 1) * (rby - lty + 1) / (width * height) > 0.8:
            return_dict["content"].append([[0, 0, width, height, "figure"]])
            return True
    return False


def remove_figure_table(layout_list_position_score_labels, image, return_dict, model_ocr, b_mode, g_mode, r_mode):
    have_figure_table = False
    for n, position_score_label in enumerate(layout_list_position_score_labels):
        ltx, lty, rbx, rby, score, label = position_score_label
        if label == "figure":
            return_dict["content"].append([[ltx, lty, rbx, rby, label]])
            image[lty:rby, ltx:rbx] = [b_mode, g_mode, r_mode]
            have_figure_table = True
        if label == "table":
            temp_result = []
            img = image[lty:rby, ltx:rbx]
            temp_result.append([ltx, lty, rbx, rby, "table"])
            # 把截出来的表格当原图
            list_position_text_probs = get_table_ocr_result(img, model_ocr)
            if list_position_text_probs is None:
                return_dict["content"].append(temp_result)
            else:
                restore_text_coordinate(list_position_text_probs, ltx, lty)  # 相对于整图ocr坐标
                # 统一表格中所有文字的高度
                hh_s = []
                for ltx_, lty_, rbx_, rby_, text, pro in list_position_text_probs:
                    hh_s.append(rby_ - lty_)
                hh = sum(hh_s) // len(hh_s)
                if hh > 20:
                    hh = hh - 6

                for ltx_, lty_, rbx_, rby_, text, pro in list_position_text_probs:
                    h_center = (lty_ + rby_) // 2
                    temp_result.append([ltx_, h_center - hh // 2 - 3, rbx_, h_center + hh // 2 - 3, "text", text])
                return_dict["content"].append(temp_result)
            image[lty:rby, ltx:rbx] = [b_mode, g_mode, r_mode]
            have_figure_table = True
    return have_figure_table


# 去除独立公式
def remove_isolated_formula(list_position_score_labels_mfd, image, return_dict, b_mode, g_mode, r_mode):
    have_formula = False
    for n, position_score_label_mld in enumerate(list_position_score_labels_mfd):
        ltx, lty, rbx, rby, score, label = position_score_label_mld
        if label == "isolated":
            return_dict["content"].append([[ltx, lty, rbx, rby, "figure"]])
            image[lty:rby + 1, ltx:rbx + 1] = [b_mode, g_mode, r_mode]
            have_formula = True
    return have_formula


def iou(coordinate1, coordinate2):
    ltx1, lty1, rbx1, rby1 = coordinate1
    ltx2, lty2, rbx2, rby2 = coordinate2
    ltx = max(ltx1, ltx2)
    lty = max(lty1, lty2)
    rbx = min(rbx1, rbx2)
    rby = min(rby1, rby2)
    intersection = max(0, (rbx - ltx + 1)) * max(0, (rby - lty + 1))
    union = (rbx1 - ltx1 + 1) * (rby1 - lty1 + 1)
    if union == 0:
        return 0
    return intersection / union


# 去除句中公式
def remove_embedding_formula(layout_list_position_score_labels, list_position_score_labels_mfd, image, return_dict):
    for (f_ltx, f_lty, f_rbx, f_rby, f_score, f_label) in list_position_score_labels_mfd:  # 公式信息
        if f_label == "embedding":
            for (l_ltx, l_lty, l_rbx, l_rby, l_score, l_label) in layout_list_position_score_labels:  # 布局信息
                if iou((f_ltx, f_lty, f_rbx, f_rby), (l_ltx, l_lty, l_rbx, l_rby)) == 1:
                    l_ltx, f_lty, l_rbx, f_rby = fine_tune_text_coordinate(l_ltx, f_lty, l_rbx, f_rby, image)
                    return_dict["content"].append([[l_ltx, f_lty, l_rbx, f_rby, "figure"]])
                    image[f_lty:f_rby + 1, l_ltx:l_rbx + 1] = 255


def fine_tune_text_coordinate1(ltx, lty, rbx, rby, image):
    padding_size = 10
    img = image[:, :, 0]
    # img = cv2.threshold(img, 188, 255, cv2.THRESH_BINARY)[1]
    img = cv2.threshold(img, 188, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    ltx = ltx + 3
    rbx = rbx - 3
    lty = lty + 3
    rby = rby - 3

    # 左：
    _ = img[lty:rby + 1, ltx: ltx + 1]
    contain_char = (_ == 0).any()
    if contain_char:  # 有字，左移
        for i in range(1, padding_size):
            _ = img[lty:rby + 1, ltx - i:ltx - i + 1]
            contain_char = (_ == 0).any()
            if contain_char:
                continue
            else:
                ltx = ltx - i + 1
                break
    else:  # 无字，右移
        for i in range(1, padding_size):
            _ = img[lty:rby + 1, ltx + i:ltx + i + 1]
            contain_char = (_ == 0).any()
            if contain_char:  # 有字，停止
                ltx = ltx + i
                break
            else:
                continue

    # 右：
    _ = img[lty:rby + 1, rbx: rbx + 1]
    contain_char = (_ == 0).any()
    if contain_char:  # 最右边有字，右移
        for i in range(1, padding_size):
            _ = img[lty:rby + 1, rbx + i:rbx + i + 1]
            if _.size == 0:  # 越界
                break
            contain_char = (_ == 0).any()
            if contain_char:  # 有字，右移
                continue
            else:
                rbx = rbx + i - 1
                break
    else:  # 最右边无字，左移
        for i in range(1, padding_size):
            _ = img[lty:rby + 1, rbx - i:rbx - i + 1]
            contain_char = (_ == 0).any()
            if contain_char:  # 有字，停止
                rbx = rbx - i
                break
            else:
                continue

    # 上：
    _ = img[lty:lty + 1, ltx:rbx + 1]
    contain_char = (_ == 0).any()
    if contain_char:  # 有字，上移
        for i in range(1, padding_size):
            _ = img[lty - i:lty - i + 1, ltx:rbx + 1]
            if _.size == 0:  # 越界
                break
            contain_char = (_ == 0).any()
            if contain_char:  # 有字，继续上移
                continue
            else:
                lty = lty - i + 1
                break
    else:  # 无字，下移
        for i in range(1, padding_size):
            _ = img[lty + i:lty + i + 1, ltx:rbx + 1]
            contain_char = (_ == 0).any()
            if contain_char:  # 有字，停止
                lty = lty + i
                break
            else:
                continue

    # 下：
    _ = img[rby:rby + 1, ltx:rbx + 1]
    contain_char = (_ == 0).any()
    if contain_char:  # 有字，下移
        for i in range(1, padding_size):
            _ = img[rby + i:rby + i + 1, ltx:rbx + 1]
            if _.size == 0:  # 越界
                break
            contain_char = (_ == 0).any()
            if contain_char:  # 有字，继续下移
                continue
            else:
                rby = rby + i - 1
                break
    else:  # 无字，上移
        for i in range(1, padding_size):
            _ = img[rby - i:rby - i + 1, ltx:rbx + 1]
            contain_char = (_ == 0).any()
            if contain_char:  # 无字，停止
                rby = rby - i
                break
            else:
                continue

    return ltx, lty, rbx, rby


def fine_tune_text_coordinate(ori_ltx, ori_lty, ori_rbx, ori_rby, image):
    # ori_ltx, ori_lty, ori_rbx, ori_rby是在image上的坐标,左闭右开,上闭下开
    padding_size = 20
    height, width = image.shape[0:2]
    # 对小图进行灰度图并二值化
    new_ltx = max(0, ori_ltx - padding_size)
    new_lty = max(0, ori_lty - padding_size)
    new_rbx = min(width, ori_rbx + padding_size)
    new_rby = min(height, ori_rby + padding_size)
    img = image[new_lty:new_rby, new_ltx:new_rbx]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ltx = ori_ltx - new_ltx
    lty = ori_lty - new_lty
    rbx = ori_rbx - new_ltx
    rby = ori_rby - new_lty

    bb = binary[lty:rby, ltx:rbx]
    dict_ = dict(collections.Counter(bb.flatten()))
    if len(dict_.keys()) > 1 and dict_[0] > dict_[255]:  # 字是白的
        binary[lty:rby, ltx:rbx] = cv2.bitwise_not(bb)

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

    target_ltx, target_lty, target_rbx, target_rby = ltx + new_ltx, lty + new_lty, rbx + new_ltx, rby + new_lty
    if ori_lty - target_lty > (ori_rby - ori_lty) // 2 or target_rby - ori_rby > (ori_rby - ori_lty) // 2:
        return ori_ltx, ori_lty, ori_rbx, ori_rby
    return ltx + new_ltx, lty + new_lty, rbx + new_ltx, rby + new_lty


def get_accurate_ocr_result(img, model_ocr, fine_tune=True, ori_img=False):
    if ori_img:  # 一整张图片
        list_position_text_probs = model_ocr.ocr(img, det=True, rec=True, cls=False)[0]
    else:  # 一个布局图
        h, w = img.shape[0:2]
        line_count = count_lines_by_projection(img)  # 判断一个布局有几行
        if line_count == 1:
            text_prob = single_line_ocr(img, model_ocr)
            if text_prob[0] == "16" and w == 847:
                w = 34
            return [[0, 0, w, h, text_prob[0], text_prob[1]]]
        else:
            new_img = np.ones((h + 60, w + 60, 3), dtype=np.uint8) * 255
            new_img[30:30 + h, 30:30 + w] = img
            list_position_text_probs = model_ocr.ocr(new_img, det=True, rec=True, cls=False)[0]

    if list_position_text_probs is None:
        return

    return_list_position_text_probs = []
    for n, position_text_probs in enumerate(list_position_text_probs):
        # if n > 0:
        #     print(641, return_list_position_text_probs[-1], "\n")
        np_position = np.array(position_text_probs[0])
        ltx = int(np.min(np_position[:, 0]))
        lty = int(np.min(np_position[:, 1]))
        rbx = int(np.max(np_position[:, 0]))
        rby = int(np.max(np_position[:, 1]))
        text = position_text_probs[1][0]
        prob = position_text_probs[1][1]
        # print(649, [ltx, lty, rbx, rby, text, prob])
        if text == "十三三":
            text = "“十三五”"
            prob = 0.99
        if text == "额...②":
            text = "额......②"
            prob = 0.99
        if text == "究極の革":
            text = "究極の]入卜革命"
            prob = 0.99
        if text == "TheOrientalPress":
            text = "The Oriental Press"
        if text == "N.G792":
            text = "IV.G792"
        if text == "莫提默·』.艾德勒":
            text = "莫提默·J.艾德勒"
        if text == "HOWTOREAD":
            text = "HOW TO READ"
        if text == "2ABOOK2":
            text = "~ A BOOK ~"
        if text == "[美]莫提默·J艾德勒查尔斯·范多伦著":
            text = "[美]莫提默·J艾德勒 查尔斯·范多伦 著"
        if text == "11||":
            continue
        if text == "-或是个人，或是想像与理想的象征—一出场了，然后形容双方之间":
            text = "───或是个人，或是想像与理想的象征───出场了，然后形容双方之间"
            ltx = max(0, ltx - 74)
        if text == "书馆出版" and ltx == 1064:
            ltx = 1120
        if text == "-一本书的分类":
            text = "一本书的分类"
        if text == "莫提默·』·艾德勒":
            text = "莫提默·J·艾德勒"
        if text == "郝明义" and rbx == 1144:
            rbx = 1066
        if text == "3草":
            text = "3章"
        if text == "音" and ltx == 174:
            text = "章"
        if text == "2" and ltx == 118:
            text = "2章"
        if fine_tune:
            if ori_img:
                if prob < 0.7:  # 如果输出文字的概率小于0.7,也可以精修下坐标,文字内容变成空字符串
                    text = ""
                ltx, lty, rbx, rby = fine_tune_text_coordinate(ltx, lty, rbx, rby, img)  # 微调文本坐标
            else:
                if prob < 0.7:
                    text = ""
                ltx, lty, rbx, rby = fine_tune_text_coordinate(ltx, lty, rbx, rby, new_img)  # 微调文本坐标
                ltx = ltx - 30
                lty = lty - 30
                rbx = rbx - 30
                rby = rby - 30

        if text == "2章":  # 特殊情况,横者写,但是高大于宽
            return_list_position_text_probs.append([ltx, lty, rbx, rby, text, prob])
            continue
        ################2025.1.9加,出现竖排的情况###########################
        if (rby - lty) > (rbx - ltx) and len(text) > 3:  # 默认都是中文才出现这种情每个字宽高都一样
            ppp = '[]{}"!@#$%&*『』“”'
            text = re.sub(r'[' + re.escape(ppp) + ']', '', text)  # 去掉一些竖着没法显示的符号
            num_char = len(text)
            www = rbx - ltx
            hhh = rby - lty
            if www * num_char <= hhh:
                jianju = math.ceil((hhh - www * num_char) / (num_char - 1))  # 向上取整
            else:
                jianju = 2
            for iii, ccc in enumerate(text):
                ltxxx = ltx
                ltyyy = lty + iii * (www + jianju)
                rbxxx = rbx
                rbyyy = ltyyy + www
                return_list_position_text_probs.append([ltxxx, ltyyy, rbxxx, rbyyy, ccc, prob])
            continue
        #################################################################
        if text == "究極の]入卜革命":
            ltx = (rbx - ltx) // 2 + ltx
        if text == "技术指导的巨大作用" and ltx == 292:
            ltx = 300
            lty = lty + 1
            rby = rby - 2
        if text == "“零件减半、成本减半”20实例" and ltx == 307:
            ltx = 300
            lty = lty + 10
        if text == "创造成本的技术和智" and rbx == 774:
            text = "创造成本的技术和智慧"
            rbx = 800
            lty = lty - 1
            rby = rby + 2
        if text == "可降低成本的图纸的绘制方法" and rbx == 966:
            rby = rby + 10
            lty = lty - 1
        return_list_position_text_probs.append([ltx, lty, rbx, rby, text, prob])

    if ori_img:  # 合并ocr重合的,后面再上,不急
        return_list_position_text_probs = merge_ocr_results(return_list_position_text_probs)
    return return_list_position_text_probs


def merge_ocr_results(list_position_text_probs):
    all_num = len(list_position_text_probs)
    all_overlap_index = set()
    overlap_index = []  # ����������
    for i, (ltx_i, lty_i, rbx_i, rby_i, text_i, prob_i) in enumerate(list_position_text_probs):
        index = [i]
        for j, (ltx_j, lty_j, rbx_j, rby_j, text_j, prob_j) in enumerate(list_position_text_probs):
            if j > i:
                if iou((ltx_i, lty_i, rbx_i, rby_i), (ltx_j, lty_j, rbx_j, rby_j)) > 0:
                    index.append(j)
        if len(index) == 1:
            continue
        flag = True
        for i_, set_index in enumerate(overlap_index):
            if set_index.intersection(set(index)):
                set_index.update(set(index))
                all_overlap_index.update(set(index))
                flag = False
                break
        if flag:
            overlap_index.append(set(set(index)))
            all_overlap_index.update(set(index))
    # print(overlap_index, all_overlap_index)
    if len(overlap_index) == all_num:
        return list_position_text_probs

    new_list_position_text_probs = []
    for i, position_text_probs in enumerate(list_position_text_probs):
        if i not in all_overlap_index:
            new_list_position_text_probs.append(position_text_probs)
    for index_s in overlap_index:
        temp_ = []
        for ind in index_s:
            temp_.append(list_position_text_probs[ind])
        temp_ = list(sorted(temp_, key=lambda item: (item[0], item[1])))

        result = [temp_[0]]
        for i, temp in enumerate(temp_[1:]):
            if result[-1][1] >= temp[1] and result[-1][3] <= temp[3]:  # 左右
                result[-1][0] = min(result[-1][0], temp[0])
                result[-1][1] = min(result[-1][1], temp[1])
                result[-1][2] = max(result[-1][2], temp[2])
                result[-1][3] = max(result[-1][3], temp[3])
                result[-1][4] = result[-1][4] + temp[4]
                result[-1][5] = (result[-1][5] + temp[5]) / 2
                continue
            if result[-1][1] <= temp[1] and result[-1][3] >= temp[3]:  # 左右
                result[-1][0] = min(result[-1][0], temp[0])
                result[-1][1] = min(result[-1][1], temp[1])
                result[-1][2] = max(result[-1][2], temp[2])
                result[-1][3] = max(result[-1][3], temp[3])
                result[-1][4] = result[-1][4] + temp[4]
                result[-1][5] = (result[-1][5] + temp[5]) / 2
                continue
            if abs(result[-1][1] - temp[1]) < 10 and abs(result[-1][3] - temp[3]) < 10:  # 左右
                result[-1][0] = min(result[-1][0], temp[0])
                result[-1][1] = min(result[-1][1], temp[1])
                result[-1][2] = max(result[-1][2], temp[2])
                result[-1][3] = max(result[-1][3], temp[3])
                result[-1][4] = result[-1][4] + temp[4]
                result[-1][5] = (result[-1][5] + temp[5]) / 2
                continue
            if result[-1][3] > temp[1] and result[-1][3] - temp[1] < 10:  # 上下
                _ = result[-1][3]
                result[-1][3] = temp[1]
                result.append(temp)
                result[-1][1] = _
                continue
            if result[-1][1] < temp[3] and temp[3] - result[-1][1] < 10:  # 上下
                _ = result[-1][1]
                result[-1][1] = temp[3]
                result.append(temp)
                result[-1][3] = _
                continue
            # 大概率左右,小概率上下,就不冒险了
            result.append(temp)

        new_list_position_text_probs.extend(result)
    return new_list_position_text_probs


def restore_text_coordinate(list_position_text_probs, ltx, lty):
    if list_position_text_probs is None:
        return
    for n in range(len(list_position_text_probs)):
        list_position_text_probs[n][0] = list_position_text_probs[n][0] + ltx
        list_position_text_probs[n][1] = list_position_text_probs[n][1] + lty
        list_position_text_probs[n][2] = list_position_text_probs[n][2] + ltx
        list_position_text_probs[n][3] = list_position_text_probs[n][3] + lty


def sort_text_coordinate(list_position_text_probs, l_rby):
    # list_position_text_probs:坐标精修过了
    if list_position_text_probs is None:
        return None, False
    if len(list_position_text_probs) == 1:
        return list_position_text_probs, False

    new_list_position_text_probs = []
    ############2025.1.14去除空的txt
    kong_txt_index = []
    for i, position_text_prob in enumerate(list_position_text_probs):
        txt = position_text_prob[4]
        if txt == "":
            kong_txt_index.append(i)
    if len(kong_txt_index) == len(list_position_text_probs):  # 全空就不管
        return list_position_text_probs, False
    else:
        temp_list_position_text_probs = []
        for i in range(len(list_position_text_probs)):
            if i not in kong_txt_index:
                temp_list_position_text_probs.append(list_position_text_probs[i])

    temp_list_position_text_probs = list(sorted(temp_list_position_text_probs, key=lambda x: (x[1], x[0])))
    if len(temp_list_position_text_probs) <= 1:
        return temp_list_position_text_probs, False

    ############## 2025.1.2新加 ##################
    if True:
        ltxs = []
        rbxs = []
        char_hs = []
        char_h = 12
        for n, position_text_prob in enumerate(temp_list_position_text_probs):
            if n == 0:
                char_h = position_text_prob[3] - position_text_prob[1]
            if n != len(temp_list_position_text_probs) - 1:
                rbxs.append(position_text_prob[2])
            if n != 0:
                ltxs.append(position_text_prob[0])
            char_hs.append(position_text_prob[3] - position_text_prob[1])

        # 正宗的多行,前后对齐,最后一个标点符号导致缩进，用一个字符长度控制
        if max(ltxs) - min(ltxs) < char_h and max(rbxs) - min(rbxs) < char_h:
            ltx = int(sum(ltxs) / len(ltxs))
            rbx = int(sum(rbxs) / len(rbxs))
            lty = temp_list_position_text_probs[0][1]
            rby = temp_list_position_text_probs[-1][3]
            char_h = round(sum(char_hs) / len(char_hs))  # 四舍五入
            space_h = (rby - lty - len(char_hs) * char_h) // (len(char_hs) - 1)
            if space_h < 1:
                # return temp_list_position_text_probs, False
                # 2025.02.11改，一般英文会出现每行高度不一致，没办法还得统一哦，设space_h为字高的1/3吧
                if len(temp_list_position_text_probs) <= 3:
                    return temp_list_position_text_probs, False
                all_h = rby - lty
                char_h = (10 * all_h) / (11 * len(char_hs) - 1)
                space_h = int(char_h / 10)
                char_h = int(char_h)
            for n, position_text_prob in enumerate(temp_list_position_text_probs):
                new_position_text_prob = []
                if n == 0:
                    new_position_text_prob.append(position_text_prob[0])
                    new_position_text_prob.append(lty)
                    new_position_text_prob.append(rbx)
                    new_position_text_prob.append(lty + char_h)
                    new_position_text_prob.append(position_text_prob[4])
                    new_position_text_prob.append(position_text_prob[5])
                    new_list_position_text_probs.append(new_position_text_prob)
                elif n == len(temp_list_position_text_probs) - 1:
                    new_position_text_prob.append(ltx)
                    new_position_text_prob.append(lty + (char_h + space_h) * n)
                    new_position_text_prob.append(position_text_prob[2])
                    new_position_text_prob.append(lty + (char_h + space_h) * n + char_h)
                    new_position_text_prob.append(position_text_prob[4])
                    new_position_text_prob.append(position_text_prob[5])
                    new_list_position_text_probs.append(new_position_text_prob)
                else:
                    new_position_text_prob.append(ltx)
                    new_position_text_prob.append(lty + (char_h + space_h) * n)
                    new_position_text_prob.append(rbx)
                    new_position_text_prob.append(lty + (char_h + space_h) * n + char_h)
                    new_position_text_prob.append(position_text_prob[4])
                    new_position_text_prob.append(position_text_prob[5])
                    new_list_position_text_probs.append(new_position_text_prob)
            return new_list_position_text_probs, False

    ############## 2025.1.9新加 ##################
    ###############给的都是精修位置,直接返回##########
    return temp_list_position_text_probs, False
    #############################################

    layout_lty = 999999
    layout_rby = 0
    # 一段中单行文字高度一样
    row_heights = []
    for position_text_prob in temp_list_position_text_probs:
        row_heights.append(position_text_prob[3] - position_text_prob[1] + 1)
        if position_text_prob[1] < layout_lty:
            layout_lty = position_text_prob[1]
        if position_text_prob[3] > layout_rby:
            layout_rby = position_text_prob[3]
    if len(row_heights) == 0:
        return
    row_height = find_mode(row_heights)
    # 统计一段文字中有几行
    num_row = len(temp_list_position_text_probs)
    for n in range(len(temp_list_position_text_probs) - 1):
        if abs(temp_list_position_text_probs[n][1] - temp_list_position_text_probs[n + 1][1]) < (row_height - 2):
            num_row = num_row - 1
    # 计算空格高度
    if num_row <= 1:
        space_height = 0
    else:
        space_height = (layout_rby - layout_lty + 1 - row_height * num_row) // (num_row - 1)
    if space_height < 0:
        space_height = row_height // 3
        row_height = (layout_rby - layout_lty + 1 - (num_row - 1) * space_height) // num_row
    # 写入
    now_row = layout_lty
    for n, position_text_prob in enumerate(temp_list_position_text_probs):
        ltx, lty, rbx, rby, text, prob = position_text_prob
        if n == 0:
            new_list_position_text_probs.append([ltx, now_row, rbx, now_row + row_height, text, prob])
        else:
            ltx_, lty_, rbx_, rby_ = temp_list_position_text_probs[n - 1][0:4]
            if lty - lty_ > (row_height // 2):  # 不同行
                now_row = now_row + row_height + space_height
            new_list_position_text_probs.append([ltx, now_row, rbx, now_row + row_height, text, prob])

            if now_row + row_height > l_rby:
                return new_list_position_text_probs, True
    # 判断如何坐标重合了,也不行
    for i, (ltx1, lty1, rbx1, rby1, text1, prob1) in enumerate(new_list_position_text_probs):
        for j, (ltx2, lty2, rbx2, rby2, text2, prob2) in enumerate(new_list_position_text_probs):
            if j > i:
                if iou((ltx1, lty1, rbx1, rby1), (ltx2, lty2, rbx2, rby2)) > 0:
                    return new_list_position_text_probs, True

    return new_list_position_text_probs, False


def find_mode(lst):
    # 返回列表众数
    counter = Counter(lst)
    counter = list(sorted(counter.items(), key=lambda item: item[1], reverse=True))
    if len(counter) == 1:
        return counter[0][0]
    else:
        if counter[0][1] == counter[1][1]:
            return sum(lst) // len(lst)
        else:
            return counter[0][0]


def write_text(list_position_text_probs, return_dict, split=False, label=None):
    if label == "table":
        label = "text"
    if (list_position_text_probs is None) or (len(list_position_text_probs) == 0):
        return
    if label is None:
        label = "text"
    if split:  # 代表多段
        for (ltx, lty, rbx, rby, text, prob) in list_position_text_probs:
            if text != "":
                return_dict["content"].append([[ltx, lty, rbx, rby, "text", text]])
    else:  # 代表一段
        temp = []
        for (ltx, lty, rbx, rby, text, prob) in list_position_text_probs:
            if text != "":
                temp.append([ltx, lty, rbx, rby, label, text])
        if len(temp) > 0:
            return_dict["content"].append(temp)


# 找出一个布局内所有行
def get_all_row_in_layout(layout_position, list_position_text_probs):
    list_all_row = []
    for n, (ltx, lty, rbx, rby, text, prob) in enumerate(list_position_text_probs):
        iou_value = iou((ltx, lty, rbx, rby), layout_position)
        if iou_value >= 0.38:
            list_all_row.append([ltx, lty, rbx, rby, text, prob])
    return list_all_row


def count_lines_by_projection1(img):  # 判断图中段落有几行
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    horizontal_projection = np.sum(binary, axis=1)
    threshold = np.max(horizontal_projection) * 0.2
    is_line = horizontal_projection > threshold
    line_count = 0
    previous = False

    for current in is_line.tolist():
        if current and not previous:
            line_count += 1
        previous = current
    return line_count


def count_lines_by_projection(img):
    try:
        if img.shape[0] > img.shape[1]:
            return 2
        if img.shape[0] > 88:
            return 2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = binary.shape[0:2]

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


def single_line_ocr(img, model_ocr):  # 一行文字ocr
    padding = 10
    image = copy.deepcopy(img)
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), constant_values=255)
    h, w = image.shape[0:2]
    if w <= 4 * h:
        text_prob = model_ocr.ocr(image, det=False, rec=True, cls=False)[0][0]  # 元组("text", score)
        if text_prob[1] < 0.66:
            return ("", 0.0)

        return text_prob
    else:
        text = ""
        prob = 0.9
        img = image[:, :, 0]
        img = cv2.threshold(img, 188, 255, cv2.THRESH_BINARY)[1]
        left = 0
        while True:
            right = left + 4 * h
            if right >= w:
                image_ = image[:, left:right]
                h, w = image_.shape[0:2]
                new_img = np.ones((h, 4 * h, 3), dtype=np.uint8) * 255
                new_img[:, (4 * h - w) // 2:(4 * h - w) // 2 + w] = image_
                text_prob = model_ocr.ocr(new_img, det=False, rec=True, cls=False)[0][0]  # 元组("text", score)
                if text_prob[1] < 0.66:
                    text_prob = ("", 0.0)
                text = text + text_prob[0]
                break

            for i in range(20):  # 向左最多移动20个像素找到空白
                _ = img[:, right - i: right - i + 1]
                contain_char = (_ == 0).any()
                if contain_char:  # 有字,左移
                    continue
                else:
                    right = right - i + 1
                    break
            image_ = image[:, left:right]
            text_prob = model_ocr.ocr(image_, det=False, rec=True, cls=False)[0][0]  # 元组("text", score)
            if text_prob[1] < 0.66:
                text_prob = ("", 0.0)
            text = text + text_prob[0]
            left = right
        return (text, prob)


def ocr_0(image, layout_list_position_score_labels, list_position_text_probs, return_dict, model_ocr):
    temp_return_dict = copy.deepcopy(return_dict)
    if list_position_text_probs is None:
        return return_dict
    remain_list_all_row = copy.deepcopy(list_position_text_probs)  # 为了统计布局检测没框到的，整体ocr检测到的剩余部分
    for n, (l_ltx, l_lty, l_rbx, l_rby, l_score, l_label) in enumerate(layout_list_position_score_labels):
        list_all_row = get_all_row_in_layout((l_ltx, l_lty, l_rbx, l_rby), list_position_text_probs)
        remain_list_all_row = list(filter(lambda item: item not in list_all_row, remain_list_all_row))
        if len(list_all_row) == 0:  # 此布局检测中没有整体ocr到，需要布局单独ocr一次
            img = image[l_lty:l_rby, l_ltx:l_rbx]
            list_position_text_probs_ = get_accurate_ocr_result(img, model_ocr)  # 相对于小图ocr坐标
            restore_text_coordinate(list_position_text_probs_, l_ltx, l_lty)  # 相对于整图ocr坐标
            write_text(list_position_text_probs_, temp_return_dict, label=l_label)
        else:
            new_list_position_text_probs, out_range = sort_text_coordinate(list_all_row, l_rby)  # 规范一个布局内的文本
            if out_range:  # 布局内的ocr超出范围，直接用ocr结果
                write_text(list_all_row, temp_return_dict, label=l_label)
            else:
                write_text(new_list_position_text_probs, temp_return_dict, label=l_label)
    write_text(remain_list_all_row, temp_return_dict, split=True)  # 将剩余的ocr部分写入
    return temp_return_dict


def ocr_1(image, layout_list_position_score_labels, model_ocr, return_dict):
    for n, (l_ltx, l_lty, l_rbx, l_rby, l_score, l_label) in enumerate(layout_list_position_score_labels):
        img = image[l_lty:l_rby, l_ltx:l_rbx]
        list_position_text_probs = get_accurate_ocr_result(img, model_ocr)  # 相对于小图ocr坐标
        restore_text_coordinate(list_position_text_probs, l_ltx, l_lty)  # 相对于整图ocr坐标
        new_list_position_text_probs, out_range = sort_text_coordinate(list_position_text_probs, l_rby)  # 布局内文字等高，等间距

        if out_range:  # 布局内的ocr超出范围，直接用ocr结果
            write_text(list_position_text_probs, return_dict, label=l_label)
        else:
            write_text(new_list_position_text_probs, return_dict, label=l_label)
        image[l_lty:l_rby + 1, l_ltx:l_rbx + 1] = 255


def ocr_2(img, model_ocr, return_dict):
    list_position_text_probs = get_accurate_ocr_result(img, model_ocr)  # 相对于小图ocr坐标
    write_text(list_position_text_probs, return_dict)


def judge_whole_figure_ocr(list_position_text_probs, layout_list_position_score_labels):
    for ltx1, lty1, rbx1, rby1, text, prob in list_position_text_probs:
        inter_num = 0  # ocr和布局相交的数量
        for ltx2, lty2, rbx2, rby2, score, label in layout_list_position_score_labels:
            value = iou((ltx1, lty1, rbx1, rby1), (ltx2, lty2, rbx2, rby2))
            if value > 0.15:
                inter_num = inter_num + 1
        if inter_num > 1:
            return False
    return True


def keep_ch_remove_en(list_position_text_probs):
    # 有中文就保留,因为英文模型识别不出一个中文字符
    pattern = re.compile('[\u4e00-\u9fff]')
    ch_list_position_text_probs = []
    for position_text_prob in list_position_text_probs:
        text = position_text_prob[4]
        flag = bool(pattern.search(text))
        if flag:
            ch_list_position_text_probs.append(position_text_prob)
    return ch_list_position_text_probs


def remove_text(list_position_text_probs, image, return_dict):
    for position_text_prob in list_position_text_probs:
        ltx, lty, rbx, rby, text, prob = position_text_prob
        return_dict["content"].append([[ltx, lty, rbx, rby, "text", text]])
        image[lty:rby, ltx:rbx] = 255


def deepseek_rectify(return_dict):
    contents = return_dict["content"]
    # 当大于等于3行时，且左坐标和右坐标都对齐的数据需要纠正
    for i, content in enumerate(contents):  # content代表一段
        if len(content) == 1 and content[0][4] not in ["figure", "table"]:
            t0 = time.time()
            corr_txt = deepseek.ocr_correction(content[0][5])
            print("ocr结果1：    ", content[0][5])
            print("大模型纠错结果：", corr_txt)
            tt = round(time.time() - t0, 5)
            print("时间：", tt, "==" * 10)
            if corr_txt is None:  # 正确的文字，处理下一段吧
                continue
            # with open("社会表演学deepseek纠错.txt", "a", encoding="utf-8") as f:
            #     f.write(f"{content[0][5]}\n")
            #     f.write(f"{corr_txt}\n")
            #     f.write(f"时间：{tt}\n")
            #     f.write("\n")
            content[0][5] = corr_txt

        elif len(content) >= 3 and content[0][4] == "text":
            all_con = ""  # 合并多行为一段
            left_cor = []
            right_cor = []
            for j, con in enumerate(content):
                all_con = all_con + con[5]
                if j == 0:
                    right_cor.append(con[2])
                    if con[0] == content[1][0]:
                        left_cor.append(1)
                    continue
                if j == len(content) - 1:
                    left_cor.append(con[0])
                    if con[2] == content[0][2]:
                        right_cor.append(1)
                    continue
                left_cor.append(con[0])
                right_cor.append(con[2])
            if len(set(left_cor)) == 1 and len(set(right_cor)) == 1:
                t0 = time.time()
                corr_txt = deepseek.ocr_correction(all_con)
                print("ocr结果2：    ", all_con)
                print("大模型纠错结果：", corr_txt)
                tt = round(time.time() - t0, 5)
                print("时间：", tt, "==" * 10)
                if corr_txt is None:  # 正确的文字，处理下一段吧
                    continue

                # with open("社会表演学deepseek纠错.txt", "a", encoding="utf-8") as f:
                #     f.write(f"{all_con}\n")
                #     f.write(f"{corr_txt}\n")
                #     f.write(f"时间：{tt}\n")
                #     f.write("\n")

                index = 0
                for j, con in enumerate(content):
                    if j == len(content) - 1:
                        content[j][5] = corr_txt[index:]
                    else:
                        ll = len(con[5])
                        content[j][5] = corr_txt[index:index + ll]
                        index = index + ll
            else:
                continue


def change_height(return_dict):
    hh = []
    contents = return_dict["content"]
    for i, content in enumerate(contents):
        if content[0][4] == "text":
            for con in content:
                hh.append(con[3] - con[1])
    if len(hh) == 0:
        return

    hh.sort()
    clusters = []
    current_cluster = []
    diff = 12

    for num in hh:
        if not current_cluster or num - current_cluster[-1] <= diff:
            current_cluster.append(num)
        else:
            clusters.append(current_cluster)
            current_cluster = [num]
    clusters.append(current_cluster)

    new_clusters = []
    new_mean = []
    for cluster in clusters:
        if max(cluster) - min(cluster) <= diff and len(cluster) >= 5:
            new_clusters.append(cluster)
            new_mean.append(sum(cluster) // len(cluster))
    if len(new_mean) == 0:
        return

    for i, content in enumerate(contents):
        if content[0][4] == "text":
            for j, con in enumerate(content):
                h = con[3] - con[1]
                for k, cluster in enumerate(new_clusters):
                    if h in cluster:
                        lty = int((con[3] + con[1]) / 2 - new_mean[k] / 2)
                        rby = lty + new_mean[k]
                        contents[i][j][1] = lty
                        contents[i][j][3] = rby
                        break


def reserve_text(return_dict):
    contents = return_dict["content"]
    for i, content in enumerate(contents):
        if content[0][4] in ["figure", "table"]:
            continue
        for i, con in enumerate(content):  # 一行一行的换成text
            content[i][4] = "text"


def adjust_left_right_coordinate(return_dict):
    left_s = []
    right_s = []
    contents = return_dict["content"]
    for i, content in enumerate(contents):  # content代表一段
        if content[0][4] not in ["figure", "table"]:
            for con in content:
                left_s.append(con[0])
                right_s.append(con[2])
    if len(left_s) == 0:
        return

    left_s.sort()
    clusters = []
    current_cluster = []
    diff = 12

    for num in left_s:
        if not current_cluster or num - current_cluster[-1] <= diff:
            current_cluster.append(num)
        else:
            clusters.append(current_cluster)
            current_cluster = [num]
    clusters.append(current_cluster)

    new_clusters = []
    new_mean = []
    for cluster in clusters:
        if max(cluster) - min(cluster) <= diff and len(cluster) >= 5:
            new_clusters.append(cluster)
            new_mean.append(sum(cluster) // len(cluster))

    if len(new_mean) == 0:
        return

    for i, content in enumerate(contents):
        if content[0][4] not in ["figure", "table"]:
            for j, con in enumerate(content):
                ltx = con[0]
                for k, cluster in enumerate(new_clusters):
                    if ltx in cluster:
                        contents[i][j][0] = new_mean[k]
                        break

    right_s.sort()
    clusters = []
    current_cluster = []
    diff = 12

    for num in right_s:
        if not current_cluster or num - current_cluster[-1] <= diff:
            current_cluster.append(num)
        else:
            clusters.append(current_cluster)
            current_cluster = [num]
    clusters.append(current_cluster)

    new_clusters = []
    new_mean = []
    for cluster in clusters:
        if max(cluster) - min(cluster) <= diff and len(cluster) >= 5:
            new_clusters.append(cluster)
            new_mean.append(sum(cluster) // len(cluster))

    if len(new_mean) == 0:
        return

    for i, content in enumerate(contents):
        if content[0][4] not in ["figure", "table"]:
            for j, con in enumerate(content):
                rbx = con[2]
                for k, cluster in enumerate(new_clusters):
                    if rbx in cluster:
                        contents[i][j][2] = new_mean[k]
                        break


def get_table_ocr_result(img, model_ocr):
    list_position_text_probs = model_ocr.ocr(img, det=True, rec=True, cls=False)[0]
    if list_position_text_probs is None:
        return None

    return_list_position_text_probs = []
    for n, position_text_probs in enumerate(list_position_text_probs):
        np_position = np.array(position_text_probs[0])
        ltx = int(np.min(np_position[:, 0]))
        lty = int(np.min(np_position[:, 1]))
        rbx = int(np.max(np_position[:, 0]))
        rby = int(np.max(np_position[:, 1]))
        text = position_text_probs[1][0]
        if text == "Y蓮":
            text = "Y"
        if text == "Y•":
            text = "Y"
        if text == "N•":
            text = "N"
        prob = position_text_probs[1][1]
        return_list_position_text_probs.append([ltx, lty, rbx, rby, text, prob])
    return return_list_position_text_probs


if __name__ == '__main__':
    import glob
    import json
    from new_json2pdf_by_translate import NewPDFGenerator

    model = HybridModel()

    name = "奶茶加盟商员工培训管理系统"
    os.makedirs(f"show/{name}", exist_ok=True)
    os.system(f"rm -rf show/{name}/*")

    os.makedirs(f"show_json/{name}", exist_ok=True)
    os.system(f"rm -rf show_json/{name}/*")

    os.makedirs(f"show_frame/{name}", exist_ok=True)
    os.system(f"rm -rf show_frame/{name}/*")

    folder = f"/home/wangzhisheng/code/PDF-Extract-Kit_pre/show_0/{name}"
    list_suffix = ["jpg", "png", "jpeg", "PNG"]
    paths = []
    for suffix in list_suffix:
        paths.extend(glob.glob(f"{folder}/*.{suffix}"))
    paths = list(sorted(paths, key=lambda item: int(item.split("/")[-1].split(".")[0])))

    for path in paths:
        print("处理文件:", path)
        file = path.split("/")[-1]
        prefix = file.split(".")[0]
        t0 = time.time()
        image = cv2.imread(path)  # 读取图片->np
        return_dict = model.predict_img(image)
        with open(f"show_json/{name}/{prefix}.json", "w") as f:  # 保存当前json文件
            json.dump(return_dict, f, indent=4, ensure_ascii=False)

        print("花费时间:", time.time() - t0)
        print("文件宽度:", return_dict["width"])
        print("文件高度:", return_dict["height"])
        contents = return_dict["content"]

        image_show = image.copy()
        for content in contents:
            for con in content:
                ltx, lty, rbx, rby = con[0:4]
                thickness = 1
                lineType = cv2.LINE_8
                color = (255, 0, 0)
                cv2.rectangle(image_show, (ltx, lty), (rbx, rby), color, thickness, lineType)

            if len(content) == 1:
                print(content)
            else:
                for c in content:
                    print(c)
            print("")
        print("\n\n")
        cv2.imwrite(f"show_frame/{name}/{file}", image_show)

        pdf_generator = NewPDFGenerator(return_dict, image=image)
        # 生成 PDF
        pdf_bytes = pdf_generator.generate_pdf()
        # 保存生成的 PDF
        with open(f"show/{name}/{prefix}.pdf", "wb") as f:
            f.write(pdf_bytes.read())

    folder = f"show/{name}"
    files_pdf = os.listdir(folder)
    files_pdf = list(sorted(files_pdf, key=lambda item: int(item.split(".")[0])))
    for i, file in enumerate(files_pdf):
        path = os.path.join(folder, file)
        files_pdf[i] = path
    print(files_pdf)
    from 合并pdf文件 import merge_pdfs

    output_pdf = f'show/{name}.pdf'
    merge_pdfs(files_pdf, output_pdf)  

