pdftopdf项目
输入印刷版pdf，输出可编辑版pdf
方便用户复制粘贴文本信息

conda create -n py39 python=3.9
pip install -r requirement.txt


weights
    -- doclayout_yolov10.pt   布局检测模型
    -- mfd_weight_yolov8.pt   公式检测模型
    -- ppocr_models_english   英文ocr模型
    -- ppocrv4_server_models  中文ocr模型
获取链接：https://www.modelscope.cn/models/leetaai/pdftopdf/summary
将下载的权重文件放入weights文件夹中

执行方式:
    from inference import pdf2pdf
    input_pdf = "test.pdf"
    output_pdf = "out.pdf"
    pdf2pdf(input_pdf, output_pdf)
~                                    
