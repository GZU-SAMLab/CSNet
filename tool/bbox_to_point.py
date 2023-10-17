import json
import numpy as np
import scipy.io as sio
import os

path = "\\gwhd_2021\\val"       #数据集的图片路径
files = os.listdir(path)     #返回文件夹下包含的文件的名字列表
fp = open("gwhd_2021\\annotations\\" + "val.json", "r")      #读取文件
json_data = json.load(fp)
image_id = 0

bbox = []
for name in files:
    points = []
    for image in json_data["images"]:
        # print(image)
        if image["file_name"] == name:
            image_id = image["id"]
            # print(image_id)
            break
    for annotation in json_data["annotations"]:
        if annotation["image_id"] == image_id:
            bbox = annotation["bbox"]
            point = []
            point.append(bbox[0] + bbox[2]//2)
            point.append(bbox[1] + bbox[3]//2)
            points.append(point)

    data_inner = {"location":points, "number":len(points)}
    print(len(points))
    image_info = np.zeros((1,), dtype=object)
    image_info[0] = data_inner

    mat_name = name.split(".")[0] + '.mat'
    sio.savemat(os.path.join("gwhd_2021\\annotations", "val_density", mat_name), {'image_info': image_info})
print("完成")

