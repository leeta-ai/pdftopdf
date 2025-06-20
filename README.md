pdf2pdf项目
输入印刷版pdf，输出可编辑版pdf，方便用户复制粘贴文字

conda create -n py39 python=3.9
pip install -r requirement.txt


weights
    -- doclayout_yolov10.pt
    -- mfd-v20240618.onnx
    -- mfd_weight_yolov8.pt
    -- ppocr_models_english
    -- ppocrv4_server_models


infer:
    from inference import pdf2pdf
    pdf2pdf(input_pdf, output_pdf) 
