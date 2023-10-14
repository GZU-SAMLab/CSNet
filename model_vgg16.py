
from backbone_vgg16 import *
from torch import nn
from fightingcv_attention.attention.CBAM import CBAMBlock


class mlp_block(nn.Module):
    def __init__(self, in_channels, mlp_dim, drop_ratio=0.):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop_ratio),
            nn.Linear(mlp_dim, in_channels),
            nn.Dropout(drop_ratio)
        )

    def forward(self, x):
            x = self.block(x)
            return x

class mlp_layer(nn.Module):
    def __init__(self, seq_length_s, hidden_size_c, dc, ds, drop=0.):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size_c)
        # 注意两个block分别作用于输入的行和列， 即SXC，所以in_channels不一样
        self.token_mixing = mlp_block(in_channels=seq_length_s, mlp_dim=int(dc * seq_length_s), drop_ratio=drop)
        self.channel_mixing = mlp_block(in_channels=hidden_size_c, mlp_dim=int(ds * hidden_size_c), drop_ratio=drop)

    def forward(self, x):
        x1 = self.ln(x)
        x2 = x1.transpose(1, 2)  # 转置矩阵
        x3 = self.token_mixing(x2)
        x4 = x3.transpose(1, 2)

        y1 = x + x4  # skip-connection
        y2 = self.ln(y1)
        y3 = self.channel_mixing(y2)
        y = y1 + y3

        return y

# 按照paper中的 Table 1 来配置参数
class mlp_mixer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 layer_num=4,
                 patch_size=32,
                 hidden_size_c=768,
                 seq_length_s=49,
                 dc=0.5,
                 ds=4,
                 drop=0.
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.layer_num = layer_num
        self.hidden_size_c = hidden_size_c
        self.seq_length_s = seq_length_s
        self.dc = dc
        self.ds = ds

        self.ln = nn.LayerNorm(self.hidden_size_c)

        # 图片切割并做映射embedding，通过一个卷积实现
        self.proj = nn.Conv2d(self.in_channels, self.hidden_size_c, kernel_size=self.patch_size,
                              stride=self.patch_size)

        # 添加多个mixer-layer
        self.mixer_layer = nn.ModuleList([])
        for _ in range(self.layer_num):
            self.mixer_layer.append(mlp_layer(seq_length_s, hidden_size_c, ds, dc, drop))


    # 定义正向传播过程
    def forward(self, x):
        
        # flatten: [B, C, H, W] -> [B, C, HW]  # 第二个维度上展平 刚好是高度维度
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        print(x.shape)
        for mixer_layer in self.mixer_layer:
            x = mixer_layer(x)
        x = self.ln(x)
        return x



class Multi_Granularity(nn.Module):
    def __init__(self,
                 layer_num = 4,
                 hidden_size_c = 256,
                 img_size = 32,
                 device = "cuda"
                 ):
        super().__init__()
        self.layer_num = layer_num
        self.hidden_size_c = hidden_size_c
        self.img_size = img_size
        self.device = device
        self.mlp_mixer = nn.ModuleList([])
        self.mlp_layer = nn.ModuleList([])
        self.patch = [16, 8, 4, 32, 64]
        self.seq = [16, 64, 256, 256]
        self.seq_all = 336
        # self.conv = nn.Conv2d(512, 128, 1)


        for i in range(3):
            self.mlp_mixer.append(mlp_mixer(in_channels=512, layer_num=self.layer_num, patch_size=self.patch[i],
                                            hidden_size_c=self.hidden_size_c, seq_length_s=self.seq[i]))

        for _ in range(self.layer_num):
            self.mlp_layer.append(mlp_layer(hidden_size_c=self.hidden_size_c, seq_length_s= self.seq_all, dc=0.5, ds=4))

        self.CounterHead = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(self.hidden_size_c * self.seq_all, 512),
            # nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
            # nn.Linear(10, 1),
            nn.AvgPool1d((10)),
            nn.ReLU()

        )
        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.backbone = Backbone_VGG16()
        self.backbone.to(device=self.device)
        self.backbone.load_state_dict(torch.load("VGG16_10.pth", map_location=self.device),strict=False)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.cbam = CBAMBlock(channel=512, reduction=16, kernel_size=7)
        self.cbam.to(device=self.device)




    def forward(self, x):
        #经过vgg16
        x1 = self.backbone(x)
        x1 = self.cbam(x1)
        # x1 = self.conv(x1)



        x_coarse = self.mlp_mixer[0](x1)
        x_middle = self.mlp_mixer[1](x1)
        x_fine = self.mlp_mixer[2](x1)
        print(x_coarse.shape)
        x_all = torch.cat([ x_coarse, x_middle, x_fine], 1)
        for mlp_layer in self.mlp_layer:
            x_all = mlp_layer(x_all)
        x_cout = self.CounterHead(x_all)
        return x_cout


if __name__ == '__main__':
    neck = Multi_Granularity(device="cpu")
    # neck.load_state_dict(torch.load("mobilenetv2_1.0-f2a8633.pth", map_location="cpu"))
    # summary(neck, (3, 512, 512))
    input = torch.ones((16, 3, 512, 512))
    imgs = torch.ones((16, 3, 512, 512))
    output = neck(input)
    # print(output)

