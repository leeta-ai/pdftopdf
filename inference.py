import os
import glob
import cv2
import time
import json
from split_merge_pdf import pdf_parsing, merge_pdfs
from main import HybridModel
from new_json2pdf_by_translate import NewPDFGenerator

model = HybridModel()

pdf_parsing("奶茶加盟商员工培训管理系统.pdf", "temp_img")
os.makedirs("temp_json", exist_ok=True)
os.system(f"rm -rf temp_json/*")

os.makedirs("temp_pdf", exist_ok=True)
os.system(f"rm -rf temp_pdf/*")

save_pdf_path = "result.pdf"

folder = "temp_img"
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
    for key, value in return_dict.items():
        if key=="content":
            for v in value:
                print(v)
    with open(f"temp_json/{prefix}.json", "w") as f:  # 保存当前json文件
        json.dump(return_dict, f, indent=4, ensure_ascii=False)

    pdf_generator = NewPDFGenerator(return_dict, image)
    # 生成 PDF
    pdf_bytes = pdf_generator.generate_pdf()
    # 保存生成的 PDF
    with open(f"temp_pdf/{prefix}.pdf", "wb") as f:
        f.write(pdf_bytes.read())

merge_pdfs("temp_pdf", save_pdf_path)
