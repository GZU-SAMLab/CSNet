import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image

from ..model.model_vgg16 import *




def main():
    # 定义训练设备
    device = torch.device("cuda")
    # 创建网络模型
    model = Multi_Granularity(device=device)
    model.load_state_dict(torch.load("\\best_weight\\CSNet_best_weight.pth", ))
    model.to(device)
    target_layers = [model.backbone.features[-2]]

    data_transform = transforms.Compose([transforms.ToTensor()])
    Resize = transforms.Resize([512, 512])
    # load image
    img_path = "/count_w/MLP/test/figure1_7.png"
    img = Image.open(img_path).convert('RGB')
    img = Resize(img)

    # [N, C, H, W]
    img_tensor = data_transform(img)*255    # expand batch dimension
    input_tensor = torch.reshape(img_tensor, (1, 3, 512, 512))

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target = torch.reshape(torch.tensor([4.]),(1, 1))    # 填入真实数量
    # target = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target=target)
    img = np.array(img, dtype=np.uint8)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32)/255. ,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.axis("off")
    # plt.savefig("C:\\Users\liyaoxi\Desktop\gwhd\\figure1_7_cam.png", bbox_inches="tight", pad_inches=0)
    plt.show()



if __name__ == '__main__':
    main()
