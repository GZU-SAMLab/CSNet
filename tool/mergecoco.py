import os
import json


def load_json(filenamejson):
    with open(filenamejson) as f:
        raw_data = json.load(f)
    return raw_data


def merge_coco_json(file_path):
    file_count = 0
    files = next(os.walk(file_path))[2]
    for x in range(len(files)):
        # file_suffix = str(files[x]).split(".")[1]
        # file_name = str(files[x]).split(".")[0]

        # 计数
        file_count = file_count + 1
        # 组合文件路径
        filenamejson = file_path + str(files[x])
        # 读取文件
        if x == 0:
            # 第一个文件作为root
            root_data = load_json(filenamejson)
        else:
            raw_data = load_json(filenamejson)
            # 追加images的数据
            ##root_data['images'].append(raw_data['images'][0])

            ###追加images
            root_images_len = len(root_data['images'])
            raw_images_len = len(raw_data['images'])
            for i in range(raw_images_len):
                raw_data['images'][i]['id'] = int(raw_data['images'][i]['id']) + int(root_images_len)
            root_data['images'].extend(raw_data['images'])

            ###追加annotations
            root_annotations_len = len(root_data['annotations'])
            raw_annotations_len = len(raw_data['annotations'])
            for j in range(raw_annotations_len):
                raw_data['annotations'][j]['id'] = int(raw_data['annotations'][j]['id']) + int(root_annotations_len)
                raw_data['annotations'][j]['image_id'] = int(raw_data['annotations'][j]['image_id']) + int(
                    root_images_len)
            root_data['annotations'].extend(raw_data['annotations'])

    temp = []
    for m in root_data["categories"]:
        if m not in temp:
            temp.append(m)
    root_data["categories"] = temp
    print("共处理 {0} 个json文件".format(file_count))
    print("共找到 {0} 个类别".format(str(root_data["categories"]).count('name', 0, len(str(root_data["categories"])))))

    json_str = json.dumps(root_data)
    with open('merge.json', 'w') as json_file:
        json_file.write(json_str)
    # 写出合并文件

    print("Done!")


file_path = "C:\\Users\liyaoxi\\Desktop\\data\\"   #待合并的路径
if __name__ == "__main__":
    merge_coco_json(file_path)