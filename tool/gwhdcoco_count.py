import json
import os
import csv

#装载json文件
def load_json(json_path):
    with open(json_path) as f:
        file_json = json.load(f)
    return file_json

#把框标记数据集转换为计数数据集
def transform_count(file_path):
    files = next(os.walk(file_path))[2]
    #循环每一个json文件
    for i in range(len(files)):
        json_path = file_path + str(files[i])
        json_file = load_json(json_path)
        images_len = len(json_file['images'])
        ann_len = len(json_file['annotations'])

        #循环每一个图片
        for j in range(images_len):
            image_name = json_file['images'][j]['file_name']
            image_id = json_file['images'][j]['id']
            # 创建csv文件
            f = open('\\annotation\\train\\{}.csv'.format(image_name.split('.')[0]), 'w', encoding='UTF-8', newline='')
            writer = csv.writer(f)
            dot = ("x", "y")
            writer.writerow(dot)
            for k in range(ann_len):
                if json_file['annotations'][k]['image_id'] == image_id:
                    spot = json_file['annotations'][k]['bbox']
                    dot = (spot[0]+0.5*spot[2], spot[1]+0.5*spot[3])
                    writer.writerow(dot)
                    # print(dot)
                    # spots.append(dot)
                    # count = count + 1
            f.close()
            print('成功输入第{}张图片'.format(j+1))
        print('成功完成{}文件'.format(files[i]))
    print('全部完成!')


file_path = "annotation\\"
if __name__ == "__main__":
    transform_count(file_path)




