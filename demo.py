from matplotlib import pyplot as plt
from torchvision import transforms

from PIL import Image
from torchsummary import summary

from  model.model_vgg16_ import *


#定义训练设备
device = torch.device("cuda")

#创建网络模型
model = Multi_Granularity(decive=device)
model.load_state_dict(torch.load("\\best_weight\CSNet_best_weight.pth"))   #权重路径
model.to(device)
model.eval()
img_path = "D:\python_project\practice\count_w\MLP\\test\\0addc041-a6b6-4643-8e10-8b9c51e932f1.png"  #图片路径
image = Image.open(img_path)
image = image.convert("RGB")
transform = transforms.ToTensor()
image = transform(image) * 255
Resize = transforms.Resize([512, 512])
image = Resize(image)
image = torch.reshape(image, (1, 3, 512, 512))

output = model(image)
print(output)








