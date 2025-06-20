# _*_ coding: utf-8 _*_
"""
wiki:
Time:     2024/5/29 20:51
Author:   Mendax.Zhang
Version:  v0.1
Describe: Write at shareworks,
"""
# -*- coding: utf-8 -*-
import io
import json
import os

import cv2
import numpy as np

from PIL import Image
from reportlab.lib.colors import red, black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont

from reportlab.pdfgen import canvas

pdfmetrics.registerFont(TTFont('Noto Serif SC', '/usr/share/fonts/NotoSerifSC.ttf'))


# pdfmetrics.registerFont(TTFont('Noto Sans SC', '/usr/share/fonts/NotoSansSC-VF.ttf'))
# pdfmetrics.registerFont(TTFont('Noto Sans CJK', '/usr/share/fonts/noto/NotoSansCJK-Regular.otf'))
# pdfmetrics.registerFont(UnicodeCIDFont('STHeiti-Regular'))


class NewPDFGenerator:
    def __init__(self, json_data, image=None, original_image_stream=None, table_map=None, multiple=1):
        self.data = json.loads(json_data) if isinstance(json_data, str) else json_data
        self.image = image
        self.original_image_stream = original_image_stream
        self.table_map = table_map
        self.buffer = io.BytesIO()
        self.pdf_width, self.pdf_height = self.data['width'] * multiple, self.data['height'] * multiple
        self.c = canvas.Canvas(self.buffer, pagesize=(self.pdf_width, self.pdf_height))
        self.font_name = "Times-Roman"
        self.multiple = multiple

    def get_watermark_path(self):
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "watermark.png")

    def replace_chinese_punctuation(self, text):
        # 定义中英文标点对应关系
        punctuation_map = {
            '，': ',',  # 中文逗号
            '。': '.',  # 中文句号
            '！': '!',  # 中文感叹号
            '？': '?',  # 中文问号
            '：': ':',  # 中文冒号
            '；': ';',  # 中文分号
            '“': '"',  # 中文双引号开
            '”': '"',  # 中文双引号闭
            '‘': "'",  # 中文单引号开
            '’': "'",  # 中文单引号闭
            '（': '(',  # 中文左括号
            '）': ')',  # 中文右括号
            '【': '[',  # 中文左方括号
            '】': ']',  # 中文右方括号
            '《': '<',  # 中文书名号开
            '》': '>',  # 中文书名号闭
            '——': '--',  # 中文破折号
            '、': ',',  # 中文顿号
            '·': '•',  # 中文间隔号
            '•': '•',  # 中文圆圈记号
            '……': '...',  # 中文省略号
            '—': '-',  # 中文破折号
            '━': '-',  # 中文长破折号
            '〈': '<',  # 中文左尖括号
            '〉': '>',  # 中文右尖括号
            '「': '‘',  # 中文左单引号
            '」': '’',  # 中文右单引号
            '『': '“',  # 中文左双引号
            '』': '”',  # 中文右双引号
            '〔': '[',  # 中文左方头括号
            '〕': ']',  # 中文右方头括号
            '〖': '[',  # 中文左六角括号
            '〗': ']',  # 中文右六角括号
            '〘': '[',  # 中文左白括号
            '〙': ']',  # 中文右白括号
            '〚': '[',  # 中文左黑括号
            '〛': ']',  # 中文右黑括号
            '〜': '~',  # 中文波浪号
            '〝': '“',  # 中文左双角引号
            '〞': '”',  # 中文右双角引号
            # 可以继续添加其他中文标点符号
        }

        # 对文本中的每个中文标点符号进行替换
        for chinese, english in punctuation_map.items():
            text = text.replace(chinese, english)
        return text

    def select_font(self, text):
        if self.contains_chinese(text):
            return 'Noto Serif SC', text, True  # 中文字体
        else:
            return 'Times-Roman', self.replace_chinese_punctuation(text), False  # 英文字体

    def draw_watermark(self):
        """在PDF的右下角添加水印"""
        watermark_image_path = self.get_watermark_path()
        if os.path.exists(watermark_image_path):
            with Image.open(watermark_image_path) as watermark_image:
                watermark_image = watermark_image.convert('RGBA')
                # 获取水印图片的尺寸
                wm_width, wm_height = watermark_image.size

                # 设置缩放比例,可以根据需要调整
                scale_factor = 0.4  # 将水印图片缩小到原来的 40%
                new_width = int(wm_width * scale_factor)
                new_height = int(wm_height * scale_factor)

                # 调整水印图片的大小
                watermark_image = watermark_image.resize((new_width, new_height), resample=Image.LANCZOS)

                # 计算水印的位置
                x_position = self.pdf_width - new_width - 50  # 右下角，留出50像素的边距
                y_position = 50  # 从底部50像素的位置开始

                # 绘制水印图片
                self.c.saveState()  # 保存当前画布状态
                self.c.translate(x_position, y_position)  # 移动坐标原点到水印位置
                self.c.drawImage(watermark_image_path, 0, 0, width=new_width, height=new_height, mask='auto')  # 绘制水印
                self.c.restoreState()  # 恢复画布状态

    def contains_chinese(self, text):
        # 简单的检查是否包含中文字符的函数
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def draw_text_box(self, content, font_size_map):
        position = json.loads(content['position'])
        text = content['text']
        self.font_name, text, is_contains_chinese = self.select_font(text)

        x1, y1, x2, y2 = [int(coord) for coord in position]
        box_width = x2 - x1
        box_height = y2 - y1

        # # 尝试将字体大小设置为与框的高度相等
        font_size = box_height
        #
        # # 计算文本在尝试的字体大小下的宽度
        text_width = pdfmetrics.stringWidth(text, self.font_name, font_size)
        #
        # # 如果文本宽度超出框宽，则减小字体大小
        while text_width > box_width and font_size > 1:
            font_size -= 1
            text_width = pdfmetrics.stringWidth(text, self.font_name, font_size)

        # 计算合适的字间距使得文本正好填充整个框的宽度
        num_of_chars = len(text) - 1  # 字间距的数量
        if num_of_chars > 0:
            char_space = (box_width - text_width) / num_of_chars
        else:
            char_space = 0
        if char_space < 0:
            char_space = 0.5
            text_x = x1
        else:
            # 计算文本的起始x坐标，使文本水平居中
            text_x = x1 + (box_width - (text_width + char_space * num_of_chars)) / 2

        map_key = (x1, box_height)
        max_size = font_size_map.get(map_key, font_size)
        if max_size - font_size >= 5 and is_contains_chinese:
            font_size = max_size
        font_size_map[map_key] = max(font_size, font_size_map.get(map_key, 0))
        # 设置字体和大小
        self.c.setFont(self.font_name, font_size)
        # 绘制文本，逐个字符
        for char in text:
            self.c.drawString(text_x, self.pdf_height - y2, char)
            # 更新x坐标，包括字符宽度和字间距
            text_x += pdfmetrics.stringWidth(char, self.font_name, font_size) + char_space

    def draw_table_image(self, content, table_path, save_path):
        """从原图中截取表格图片，并绘制到新的 PDF 上"""
        position = json.loads(content['position'])
        x1, y1, x2, y2 = [int(coord) for coord in position]
        table_width = x2 - x1
        table_height = y2 - y1
        print("table_path:", table_path)
        with open(table_path, 'rb') as table_file:
            table = io.BytesIO(table_file.read())
        if table and table_width > 0 and table_height > 0:
            with Image.open(table) as original_image:
                # original_image = original_image.convert('RGB')
                # crop_box = (x1, y1, x2, y2)  # 使用y1和y2直接作为裁剪框的上下边界
                # table_image = original_image.crop(crop_box)
                pdf_y_pos = self.pdf_height - y2  # 计算PDF中的y坐标
                self.c.drawInlineImage(original_image, x1, pdf_y_pos, width=table_width, height=table_height)

                # 保存表格图像到文件
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                original_image.save(save_path)

    def draw_translation_table_image(self, content, table_path):
        """从指定路径的表格图片,并绘制到新的 PDF 上"""
        position = json.loads(content['position'])
        x1, y1, x2, y2 = [int(coord) for coord in position]
        table_width = x2 - x1
        table_height = y2 - y1

        print("table_path:", table_path)
        if table_path and table_width > 0 and table_height > 0:
            with Image.open(table_path) as table_image:
                pdf_y_pos = self.pdf_height - y2  # 计算PDF中的y坐标
                self.c.drawInlineImage(table_image, x1, pdf_y_pos, width=table_width, height=table_height)

    def draw_image(self, content, stream, save_path):
        """Extract useful information from the original image and draw it on a new PDF."""
        position = json.loads(content['position'])
        x1, y1, x2, y2 = [int(coord) for coord in position]
        table_width = x2 - x1
        table_height = y2 - y1
        if table_width > 0 and table_height > 0:

            # Convert stream to a NumPy array
            #image_data = np.frombuffer(stream.getvalue(), np.uint8)
            #original_image = Image.open(io.BytesIO(image_data))
            original_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

            # Convert image to 'RGB' if necessary
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')

            crop_box = (x1, y1, x2, y2)
            table_image = original_image.crop(crop_box)

            # Save the cropped image to a temporary file
            #if not os.path.exists(os.path.dirname(save_path)):
            #    os.makedirs(os.path.dirname(save_path))

            # Save the image with compression
            table_image.save(save_path, format='JPEG', quality=85)

            pdf_y_pos = self.pdf_height - y2
            self.c.drawImage(save_path, x1, pdf_y_pos, width=table_width, height=table_height)


    def draw_translation_image(self, image_path):
        """将指定图片路径的图像绘制到新的 PDF 上"""
        if image_path:
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                pdf_y_pos = self.pdf_height - image.height  # 计算PDF中的y坐标
                self.c.drawInlineImage(image, 0, pdf_y_pos, width=image.width, height=image.height)

    def draw_colored_border(self, position, color=red, border_width=1):
        """绘制一个带有颜色的边框"""
        x1, y1, x2, y2 = [int(coord) for coord in position]
        self.c.setStrokeColor(color)  # 设置边框颜色
        self.c.setLineWidth(border_width)  # 设置边框宽度
        self.c.rect(x1, self.pdf_height - y2, x2 - x1, y2 - y1, fill=0)  # 绘制边框
        self.c.setStrokeColor(black)  # 重置边框颜色为黑色，以备后用
        self.c.setLineWidth(1)  # 重置边框宽度为默认值

    def generate_pdf(self):
        font_size_map = {}
        """根据新的 JSON 数据结构生成 PDF"""
        for paragraph in self.data['content']:
            for content_item in paragraph:
                if content_item[4] not in  ["figure", "table"]:
                    x1, y1, x2, y2, label, text = content_item
                    image_path = ""
                else:
                    # x1, y1, x2, y2, label, image_path = content_item
                    x1, y1, x2, y2, label = content_item
                    # img = self.image[y1:y2, x1:x2]
                    # cv2.imwrite("temp.png", img)
                    image_path = "temp.png"
                    # image_path = None
                    text = ""

                x1, y1, x2, y2 = (x * self.multiple for x in (x1, y1, x2, y2))

                content = {
                    'label': label,
                    'text': text.replace("\n", ""),
                    'position': json.dumps([x1, y1, x2, y2]),
                }

                # 检查 table_path 是否为 None
                if label == 'table' and self.table_map:
                    table_path = self.table_map.get(tuple(content_item[:5]))
                    if table_path is None:
                        self.draw_image(content, self.original_image_stream, image_path)
                    else:
                        self.draw_table_image(content, table_path, image_path)
                elif label == 'figure':
                    self.draw_image(content, self.original_image_stream, image_path)
                else:
                    self.draw_text_box(content, font_size_map)
        self.draw_watermark()

        self.c.save()
        self.buffer.seek(0)
        return self.buffer

    def generate_translation_pdf(self):
        font_size_map = {}
        """根据新的 JSON 数据结构生成 PDF"""
        # for content_item in self.data['content']:
        for paragraph in self.data['content']:
            for content_item in paragraph:
                if content_item[4] == "text":
                    x1, y1, x2, y2, label, text = content_item
                    image_path = ""
                else:
                    print(content_item)
                    print(len(content_item))
                    x1, y1, x2, y2, label, image_path = content_item
                    text = ""

                x1, y1, x2, y2 = (x * self.multiple for x in (x1, y1, x2, y2))

                content = {
                    'label': label,
                    'text': text.replace("\n", ""),
                    'position': json.dumps([x1, y1, x2, y2]),  # 新结构中的坐标已经是正确的格式，这里直接转换为 JSON 字符串
                }
                # 绘制颜色边框
                # self.draw_colored_border([x1, y1, x2, y2])
                # 根据标签判断绘制类型
                if label == 'figure' and self.original_image_stream:
                    self.draw_translation_image(image_path)
                if label == 'table' and self.table_map:
                    self.draw_translation_table_image(content, image_path)
                else:
                    self.draw_text_box(content, font_size_map)
        self.draw_watermark()

        self.c.save()
        self.buffer.seek(0)
        return self.buffer


if __name__ == '__main__':
    current_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    "watermark.png")

    print(current_dir_path)
