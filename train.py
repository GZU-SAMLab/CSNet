import argparse
import time

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import datase.dataset
from  model.model_vgg16 import *


workers = 32
save_file = "weight_best.pth"
save_sumary = "gwhd_2020"
def main():
    print("start training")
    parser = argparse.ArgumentParser('Set parameters for training ', add_help=False)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument("--epoch", default=1000, type=int)
    parser.add_argument("--img_size", default=512, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    args = parser.parse_args()

    # 定义训练参数
    device = args.device
    learning_rate = args.lr
    # 网络超参数
    batch_size = args.batch_size
    epoch = args.epoch
    img_size = args.img_size

    data_root = '/home/liyaoxi/data/gwhd/gwhd_2021/'
    train_root = data_root + 'train/'
    train_ann = data_root + 'annotations/train.csv'
    val_root = data_root + 'val/'
    val_ann = data_root + 'annotations/val.csv'
    test_root = data_root + 'test/'
    test_ann = data_root + 'annotations/test.csv'

    
    # 准备数据集
    train_dataset = dataset.Countgwhd(img_path=train_root, ann_path=train_ann, resize_shape=img_size)
    val_dataset = dataset.Countgwhd(img_path=val_root, ann_path=val_ann, resize_shape=img_size)
    test_dataset = dataset.Countgwhd(img_path=test_root, ann_path=test_ann, resize_shape=img_size)

    #创建网络模型
    model = Multi_Granularity(device=device)
    model.to(device)

    #创建损失函数
    loss_fn = nn.L1Loss(reduction="sum")
    loss_fn = loss_fn.to(device)

    #优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800], gamma=0.1)

    torch.set_num_threads(workers)
    writer = SummaryWriter("log/"+ save_sumary)
    best = 10
    val_epoch = 0
    #训练
    for i in range(epoch):
        start1 = time.time()
        print("----------epoch: {}, lr: {}----------".format(i + 1, optimizer.param_groups[0]['lr']))
        loss_one_epoch = train(train_dataset, model, loss_fn, optimizer, lr_scheduler, args.batch_size, device, mask)
        writer.add_scalar("train_loss", loss_one_epoch, i)
        end1 = time.time()
        print("这轮所用时间为：{}min \n\n".format((end1-start1)/60))

        if (i+1) % 5 == 0 and i>=9 :
            val_epoch = val_epoch + 1
            start2 = time.time()
            print("----------开始验证----------")
            prec = val(val_dataset, model, device)
            if prec < best:
                best = prec
                torch.save(model.state_dict(), save_file)
            end2 = time.time()
            print("测试所用时间为：{}min".format((end2-start2)/60))
            print("当前最好的mae为：{}".format(best))
            writer.add_scalar("val_mae", prec, val_epoch)

    print("\n----------开始测试----------")
    test(test_dataset, model, device, best)
    writer.close()

def train(train_dataset, model, loss_fn, optimizer, lr_scheduler, batch_size, device, mask):
    # 加载数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=workers)
    model.train()
    loss_ave = 0
    print_freq = 20
    data_num = 0
    for data in train_dataloader:

        data_num = data_num + 1
        imgs, targets = data


        imgs = imgs.to(device)

        targets = targets.float().to(device)

        outputs = model(imgs)
        outputs = torch.reshape(outputs, [-1])

        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_ave = loss_ave + loss.item()
        if data_num % print_freq == 0:
            print("---loss: {}---".format(loss.item()))
    print("----------本轮的平均loss为：{}  ----------\n".format(loss_ave / data_num))
    lr_scheduler.step()
    return loss_ave / data_num

def val(val_dataset, model, device):
    #加载数据集
    val_dataloader = DataLoader(val_dataset, batch_size=1)
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

        if i % 15 == 0:
            print("真实数量：{}     \t 预测数量：{}".format(gt_count, count))

    mae = mae * 1.0 / i
    mse = math.sqrt(mse / i)
    print("此次测试结果为：MAE：{}  \t MSE：{}".format(mae, mse))

    return mae

def test(test_dataset, model, device, best):
    #加载数据集
    if best < 10:
        model.load_state_dict(torch.load(save_file))
    else:
        pass
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







