from matplotlib import pyplot as plt
from torchvision import transforms

from PIL import Image
from torchsummary import summary

from  model_vgg16_ablation import *
from backbone_vgg16 import *


#定义训练设备
device = torch.device("cpu")

#创建网络模型
model = Multi_Granularity(device="cpu")
model.load_state_dict(torch.load("\\best_weight\CSNet_best_weight.pth", map_location="cpu"))
model.to(device)
model.eval()
img_path = "D:\python_project\practice\count_w\MLP\\test\\0addc041-a6b6-4643-8e10-8b9c51e932f1.png"
image = Image.open(img_path)
image = image.convert("RGB")
transform = transforms.ToTensor()
image = transform(image) * 255
Resize = transforms.Resize([512, 512])
image = Resize(image)
image = torch.reshape(image, (1, 3, 512, 512))

output = model(image)
print(output)








