import argparse
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import dataset.dataset
from  model.model_vgg16 import *

def main():
    print("start testing")
    parser = argparse.ArgumentParser('Set parameters for test ', add_help=False)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--data_root", default='/home/liyaoxi/mmdetection/data/gwhd/', type=str)
    args = parser.parse_args()

    # 定义训练参数
    device = args.device
    img_size = args.img_size

    data_root = args.data_root
    test_root = data_root + 'test/'
    test_ann = data_root + 'annotations/test.csv'

    test_dataset = dataset.Countgwhd(img_path=test_root, ann_path=test_ann, resize_shape=img_size)

    #创建网络模型
    model = Multi_Granularity(device=device)
    model.to(device)
    test(test_dataset,model, device)

def test(test_dataset, model, device):
    #加载数据集
    val_dataloader = DataLoader(test_dataset, batch_size=1)
    model.eval()

    mae = 0.0
    mse = 0.0
    i = 0
    for data in val_dataloader:
        i = i + 1
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            output = model(imgs)
            count = torch.sum(output).item()

        gt_count = torch.sum(targets).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        print("真实数量：{}     \t 预测数量：{}".format(gt_count, count))

    mae = mae * 1.0 / i
    mse = math.sqrt(mse / i)
    print("此次测试结果为：MAE：{}  \t MSE：{}".format(mae, mse))


if __name__ == '__main__':
    main()







