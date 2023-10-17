import os
import random
import copy
import csv
import shutil


def subset(alist, idxs):
    '''
        用法：根据下标idxs取出列表alist的子集
        alist: list
        idxs: list
    '''
    sub_list = []
    for idx in idxs:
        sub_list.append(alist[idx])

    return sub_list


def split_list(alist, rate=0.2 ,shuffle=True):
    '''
        用法：将alist切分成两个列表
        shuffle: 表示是否要随机切分列表，默认为True
        rate：表示划分比率
    '''

    index = list(range(len(alist)))  # 保留下标

    # 是否打乱列表
    if shuffle:
        random.shuffle(index)

    elem_num = int(len(alist) * 0.2)  # 划分列表所含有的元素数量
    sub_lists = []


    for i in range(elem_num):
        sub_lists.append(copy.deepcopy(alist[index[i]]))

    return sub_lists


def SplitDataset(path):
    file_names = []
    #读取文件夹下所有文件的名称
    for file_name in os.listdir(path):
        file_names.append(file_name)
    #随机划分文件名称
    sub_list = split_list(file_names)
    #更改数据集标注
    all_reader = csv.reader(file_all)
    train_writer = csv.writer(file_train)
    test_writer = csv.writer(file_test)

    i = 0
    for line in all_reader:
        i = i + 1
        print(line)
        if line[0] == "image_name":
            train_writer.writerow(line)
            test_writer.writerow(line)
        else:
            if line[0] in sub_list:
                test_writer.writerow(line)
                shutil.copyfile(os.path.join(path, line[0]), os.path.join(path_test, line[0]))
            else:
                train_writer.writerow(line)
                shutil.copyfile(os.path.join(path, line[0]), os.path.join(path_train, line[0]))
        print("成功{}".format(i))
    # print(all_reader)
    file_all.close()
    file_train.close()
    file_test.close()
    print("全部成功")




path = '..\gwhd' 
path_train = '..\gwhd\\train'
path_test = '..\gwhd\\test'
file_path = "..\gwhd\\"
file_all = open( file_path + "all.csv", "r", encoding="utf-8-sig")
file_train = open( file_path + "train.csv", "w", newline="")
file_test = open( file_path + "test.csv", "w", newline="")
if __name__ == "__main__":
    SplitDataset(path)
