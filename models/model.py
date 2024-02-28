from layers import *
from tools.utils import *
import torch
from torch.autograd import grad


img_h, img_w = 120, 120


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         self.conv0 = nn.Sequential(
#             conv(5, 64, 7, 1, cal_conv_pad(img_h, img_w, 7, 1), 'kaiming'),
#             ResidualBlock(64)
#         )
#         self.conv1 = nn.Sequential(
#             conv(64, 64, 5, 2, cal_conv_pad(img_h, img_w/2, 5, 2)),
#             ResidualBlock(64)
#         )
#         self.conv2 = nn.Sequential(
#             conv(64, 128, 5, 2, cal_conv_pad(img_h/2, img_w/4, 5, 2)),
#             ResidualBlock(128)
#         )
#         self.conv3 = nn.Sequential(
#             conv(128, 256, 3, 2, cal_conv_pad(img_h/4, img_w/8, 3, 2)),
#             ResidualBlock(256)
#         )
#         # -> 512x8x8
#         self.conv4 = nn.Sequential(
#             conv(256, 512, 3, 2, cal_conv_pad(img_h/8, img_w/16, 3, 2)),
#             ResidualBlock(512)
#         )
#         # self.fc1 = nn.Linear(512*8*8, 512)  #
#         # self.maxout = nn.MaxPool1d(2)
#         # self.fc2 = nn.Linear(512, 64*8*8)
#         self.relu = nn.ReLU(inplace=True)
#
#         # 解码器
#         # self.dc0_1 = deconv(64, 32, 4, 2, cal_deconv_pad(img_h/16, img_w/4, 4, 2))
#         # self.dc0_2 = deconv(32, 16, 3, 2, cal_deconv_pad(img_h/4, img_w/2, 3, 2))
#         # self.dc0_3 = deconv(16, 8, 3, 2, cal_deconv_pad(img_h/2, img_w, 3, 2))
#
#         self.dc1 = nn.Sequential(
#             deconv(512, 256, 3, 2, cal_deconv_pad(img_h/16, img_w/8, 3, 2)),
#             ResidualBlock(256), ResidualBlock(256)
#         )
#         self.dc2 = nn.Sequential(
#             deconv(256, 128, 2, 2, cal_deconv_pad(img_h/8, img_w/4, 2, 2)),
#             ResidualBlock(128), ResidualBlock(128)
#         )
#         self.dc3 = nn.Sequential(
#             deconv(128, 64, 2, 2, cal_deconv_pad(img_h/4, img_w/2, 2, 2)),
#             ResidualBlock(64), ResidualBlock(64)
#         )
#         self.dc4 = nn.Sequential(
#             deconv(64, 32, 4, 2, cal_deconv_pad(img_h/2, img_w, 4, 2)),
#             ResidualBlock(32), ResidualBlock(32)
#         )
#         self.conv5 = conv(128, 3, 3, 1, cal_conv_pad(img_h/4, img_w/4, 3, 1))
#         self.conv6 = conv(64, 3, 3, 1, cal_conv_pad(img_h/2, img_w/2, 3, 1))
#         self.conv7 = conv(32, 16, 3, 1, cal_conv_pad(img_h, img_w, 3, 1))
#         self.conv8 = conv(16, 3, 3, 1, cal_conv_pad(img_h, img_w, 3, 1))
#
#     @staticmethod
#     def GLoss(Syn_F_GAN):
#         Syn_F = Syn_F_GAN[0] + Syn_F_GAN[1] + Syn_F_GAN[2] + Syn_F_GAN[3] + Syn_F_GAN[4]
#         loss = -Syn_F.mean() / 5
#
#         return loss
#
#     def forward(self, x, mark_real, mark_target):
#         x = torch.cat((x, mark_real, mark_target), dim=1)
#         c0 = self.conv0(x)
#         # print('c0', c0.shape)
#         c1 = self.conv1(c0)
#         # print('c1', c1.shape)
#         c2 = self.conv2(c1)
#         # print('c2', c2.shape)
#         c3 = self.conv3(c2)
#         # print('c3', c3.shape)
#         c4 = self.conv4(c3)  # -> 512x8x8
#         # print('c4', c4.shape)
#         dc1 = self.dc1(c4)
#         # print('dc1', dc1.shape)
#         dc2 = self.dc2(dc1)
#         # print('dc2', dc2.shape)
#         dc3 = self.dc3(dc2)
#         # print('dc3', dc3.shape)
#         dc4 = self.dc4(dc3)
#         # print('dc4', dc4.shape)
#         img_30 = self.conv5(dc2)
#         img_60 = self.conv6(dc3)
#         conv7 = self.conv7(dc4)
#         img_120 = self.conv8(conv7)
#
#         return img_120, img_60, img_30
class Generator(nn.Module):
    # 生成器网络结构，分为编码器与解码器两部分，编码器由几层卷积层和残差块堆叠而成，获得深层网络
    def __init__(self):
        super(Generator, self).__init__()

        # 编码器部分
        self.conv0 = nn.Sequential(
            conv(5, 64, ks=5, s=1, pad=2),  #
            ResidualBlock(64),
        )
        # 64x120x120 -> 64x60x60
        self.conv1 = nn.Sequential(
            conv(64, 64, 4, 2, 1),
            ResidualBlock(64),
        )
        # 64x60x60 -> 128x30x30
        self.conv2 = nn.Sequential(
            conv(64, 128, 4, 2, 1),
            ResidualBlock(128)
        )
        # 128x30x30 -> 256x15x15
        self.conv3 = nn.Sequential(
            conv(128, 256, 4, 2, 1),
            ResidualBlock(256)
        )
        # 256x15x15 -> 512x8x8
        self.conv4 = nn.Sequential(
            conv(256, 512, 3, 2, 1),
            ResidualBlock(512)
        )
        self.fc1 = nn.Linear(512*8*8, 512)  # 全连接层，提取完特征
        self.relu = nn.ReLU(inplace=True)
        self.maxout = nn.MaxPool1d(2)  # 步长为2，在最后一个维度将数据量减半
        self.fc2 = nn.Linear(256, 64*8*8)

        # 重点在于解码器部分，直接关系到生成图片的质量，泛化能力等。
        # 反卷积层
        self.dc0_0 = deconv(320, 64, 8, 1)  # BxCx1x1 -> BxCx8x8
        # 64x8x8 ->32x30x30
        self.dc0_1 = deconv(64, 32, 4, 4, 1)
        # 32x30x30 -> 16x60x60
        self.dc0_2 = deconv(32, 16, 2, 2, 0)
        # 16x60x60 ->8x120x120
        self.dc0_3 = deconv(16, 8, 2, 2, 0)

        # 多层跳跃连接
        # 8x8 -> 15x15
        self.dc1 = nn.Sequential(
            deconv(512+64, 512, 2, 2, 1, out_pad=1),
            ResidualBlock(512), ResidualBlock(512)
        )
        # 15x15 -> 30x30
        self.dc2 = nn.Sequential(
            deconv(512+256, 256, 2, 2, 0),
            ResidualBlock(256), ResidualBlock(256)
        )
        # 30x30 -> 60x60
        self.dc3 = nn.Sequential(
            deconv(256+128+3+32, 128, 2, 2, 0),
            ResidualBlock(128), ResidualBlock(128)
        )
        # 60x60 -> 120x120
        self.dc4 = nn.Sequential(
            deconv(128+64+3+16, 64, 2, 2, 0),
            ResidualBlock(64), ResidualBlock(64)
        )

        # 最后的卷积融合
        self.conv5 = conv(256, 3, 3, 1, 1)
        self.conv6 = conv(128, 3, 3, 1, 1)
        self.conv7 = conv(139, 64, 5, 1, 2)
        self.conv8 = conv(64, 32, 3, 1, 1)
        self.conv9 = conv(32, 3, 3, 1, 1)

    @staticmethod
    def flatten(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features  # 返回打平后的数据量

    def forward(self, picture, landmarks_real, landmarks_wanted, noise):
        x = torch.cat([picture, landmarks_real, landmarks_wanted], dim=1)
        c0 = self.conv0(x)
        # 64x120x120 -> 64x60x60
        c1 = self.conv1(c0)
        # 64x60x60 -> 128x30x30
        c2 = self.conv2(c1)
        # 128x30x30 -> 256x15x15
        c3 = self.conv3(c2)
        # 256x15x15 -> 512x8x8
        c4 = self.conv4(c3)
        # print('c4', c4.shape)
        tmp = Generator.flatten(c4)
        f1 = c4.view(x.size(0), tmp)  # 打平操作
        f1 = self.fc1(f1)  # 51200 —> 512
        f1 = self.relu(f1)  # Relu激活函数
        f1 = f1.unsqueeze(0)  # 增加一个维度，变成三维，使MaxPool1d正常进行
        maxout = self.maxout(f1)[0]  # B x 512 -> B x 256

        # 解码器
        # 1
        x = torch.cat([maxout, noise], dim=1).reshape(maxout.size(0), -1, 1, 1)  # B x 320 x 1 x 1
        # f2 = self.relu(self.fc2(maxout))
        # rsh = f2.reshape(x.size(0), 64, 8, 8)
        dc00 = self.dc0_0(x)  # Bx320x1x1 -> Bx64x8x8
        # print('rsh', rsh.shape)
        dc01 = self.dc0_1(dc00)
        # print('dc01', dc01.shape)
        dc02 = self.dc0_2(dc01)
        dc03 = self.dc0_3(dc02)

        # 2 多尺度融合
        # 576 x8x8 -> 512 x15x15
        dc1r = self.dc1(torch.cat([dc00, c4], dim=1))
        # print('dc1r', dc1r.shape)
        # 32的多尺度融合完成
        # 768x15x15 -> 256 x30x30
        dc2r = self.dc2(torch.cat([dc1r, c3], dim=1))
        # print('dc2r', dc2r.shape)
        # dc2r, p1 = self.attn1(dc2r)  # 30维加入注意力机制
        # 30和60的维度再加上原图pool之后的结果
        pic_div_2 = nn.MaxPool2d(2)(picture)  # 60x60
        pic_div_4 = nn.MaxPool2d(2)(pic_div_2)  # 30x30
        # 64的多尺度融合完成
        # (128 + 32 + 256 + 6) x 30 x 30 -> 128 x 60 x 60
        dc3r = self.dc3(torch.cat([dc2r, c2, dc01, pic_div_4], dim=1))
        # dc3r, _ = self.attn2(dc3r)  # 60维加入自注意力层

        # (64 + 16 + 6 + 128) x 60 x 60 -> 64 x 120 x 120
        dc4r = self.dc4(torch.cat([dc3r, c1, dc02, pic_div_2], dim=1))

        # 3 卷积操作
        # 32的结果图
        img30 = self.conv5(dc2r)
        # 64的结果图
        img60 = self.conv6(dc3r)
        syn = nn.functional.interpolate()
        # 120的多尺度融合完成
        # 多尺度融合，(3 + 64 + 8 + 64 + 3) x 120 x 120 -> 64 x 120 x 120
        #                           64     8    64    3       3
        c7 = self.conv7(torch.cat([dc4r, dc03, c0, picture], dim=1))
        c8 = self.conv8(c7)
        # 120最终结果图
        img120 = self.conv9(c8)

        return img120, img60, img30


class Discriminator_landmark(nn.Module):
    def __init__(self):
        super(Discriminator_landmark, self).__init__()
        batch_norm = False
        self.model = nn.Sequential(
            conv(4, 64, 4, 2, cal_conv_pad(img_h, img_w/2, 4, 2),
                 batch_norm=batch_norm, activate=nn.LeakyReLU(1e-2)),
            conv(64, 128, 4, 2, cal_conv_pad(img_h/2, img_w/4, 4, 2),
                 batch_norm=batch_norm, activate=nn.LeakyReLU(1e-2)),
            conv(128, 256, 4, 2, cal_conv_pad(img_h/4, img_w/8, 4, 2),
                 batch_norm=batch_norm, activate=nn.LeakyReLU(1e-2)),
            conv(256, 512, 4, 2, cal_conv_pad(img_h/8, img_w/16, 4, 2),
                 batch_norm=batch_norm, activate=nn.LeakyReLU(1e-2)),
            conv(512, 512, 4, 1, cal_conv_pad(img_h/16, 7, 4, 1),
                 batch_norm=batch_norm, activate=nn.LeakyReLU(1e-2)),
            conv(512, 1, 4, 1, cal_conv_pad(7, 6, 4, 1), activate=None)  # Bx6x6
        )

    def forward(self, x):
        return self.model(x)


# class Discriminator_Attention(nn.Module):
#     def __init__(self):
#         super(Discriminator_Attention, self).__init__()
#         batch_norm = False
#         self.model = nn.Sequential(
#             conv(6, 64, 4, 2, cal_conv_pad(img_h, img_w/2, 4, 2),
#                  batch_norm=batch_norm, activate=nn.LeakyReLU(1e-2)),
#             conv(64, 128, 4, 2, cal_conv_pad(img_h/2, img_w/4, 4, 2),
#                  batch_norm=batch_norm, activate=nn.LeakyReLU(1e-2)),
#             conv(128, 256, 4, 2, cal_conv_pad(img_h/4, img_w/8, 4, 2),
#                  batch_norm=batch_norm, activate=nn.LeakyReLU(1e-2)),
#             conv(256, 512, 4, 2, cal_conv_pad(img_h/8, img_w/16, 4, 2),
#                  batch_norm=batch_norm, activate=nn.LeakyReLU(1e-2)),
#             conv(512, 512, 4, 1, cal_conv_pad(img_h/16, 7, 4, 1),
#                  batch_norm=batch_norm, activate=nn.LeakyReLU(1e-2)),
#             conv(512, 1, 4, 1, cal_conv_pad(7, 6, 4, 1), activate=None)  # Bx6x6
#         )
#
#     def forward(self, x):
#         return self.model(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Discriminator_Attention(nn.Module):
    def __init__(self, args):
        super(Discriminator_Attention, self).__init__()
        self.features = []
        # torch.nn.Conv2d(in, out, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.lambda_gp = args.lambda_gp

        image_convLayers = [
            nn.Conv2d(3, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            nn.LayerNorm([30, 30]),  # 120
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            nn.LayerNorm([15, 15]),  # 120
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            nn.LayerNorm([8, 8]),  # 120
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),  # d_conv3
            nn.LayerNorm([4, 4]),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(4096, 1)
        ]
        eyes_convLayers = [
            #                k  s  p
            nn.Conv2d(3, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            # nn.LayerNorm([12, 34]),  # 224
            nn.LayerNorm([11, 26]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            # nn.LayerNorm([6, 17]),  # 224
            nn.LayerNorm([6, 13]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            # nn.LayerNorm([3, 9]),  # 224
            nn.LayerNorm([3, 7]),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(5376, 1)
        ]
        nose_convLayers = [
            nn.Conv2d(3, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            # nn.LayerNorm([19, 12]),  # 224
            nn.LayerNorm([9, 8]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            # nn.LayerNorm([10, 6]),  # 224
            nn.LayerNorm([5, 4]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            # nn.LayerNorm([5, 3]),  # 224
            nn.LayerNorm([3, 2]),
            nn.LeakyReLU(),
            Flatten(),
            # nn.Linear(3840, 1),
            nn.Linear(1536, 1)
        ]
        mouth_convLayers = [
            nn.Conv2d(3, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            # nn.LayerNorm([9, 16]),  # 224
            nn.LayerNorm([6, 11]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            # nn.LayerNorm([5, 8]),  # 224
            nn.LayerNorm([3, 6]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            # nn.LayerNorm([3, 4]),  # 224
            nn.LayerNorm([2, 3]),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(1536, 1)
        ]
        face_convLayers = [
            nn.Conv2d(3, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            nn.LayerNorm([22, 22]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            nn.LayerNorm([11, 11]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            nn.LayerNorm([6, 6]),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(9216, 1)
        ]

        self.eyes_convLayers = nn.Sequential(*eyes_convLayers)
        self.nose_convLayers = nn.Sequential(*nose_convLayers)
        self.mouth_convLayers = nn.Sequential(*mouth_convLayers)
        self.face_convLayers = nn.Sequential(*face_convLayers)
        self.image_connLayers = nn.Sequential(*image_convLayers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)

            if isinstance(m, nn.LayerNorm):
                m.weight.data.normal_(0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def Disc(self, img):
        # 分割图片每个器官的位置，得到感兴趣区域子图
        eyes_ROI, nose_ROI, mouth_ROI, face_ROI = segment(img)
        eyes = self.eyes_convLayers(eyes_ROI)
        nose = self.nose_convLayers(nose_ROI)
        mouth = self.mouth_convLayers(mouth_ROI)
        face = self.face_convLayers(face_ROI)
        entire_img = self.image_connLayers(img)

        return entire_img, eyes, nose, mouth, face  # 返回元组

    # 注意力判别器损失
    def forward(self, img):
        out = self.Disc(img)

        return out

    def CriticWithGP_Loss(self, Syn_F_Gan, Real_Gan, Interpolates):
        Syn_F = Syn_F_Gan[0] + Syn_F_Gan[1] + Syn_F_Gan[2] + Syn_F_Gan[3] + Syn_F_Gan[4]
        Real = Real_Gan[0] + Real_Gan[1] + Real_Gan[2] + Real_Gan[3] + Real_Gan[4]
        Wasserstein_Dis = (Syn_F - Real).mean() / 5  # 对抗损失，最小最大

        inter = self.Disc(Interpolates)
        gradinput = grad(outputs=inter[0].sum(),
                         inputs=Interpolates,
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)[0]
        gradeyes = grad(outputs=inter[1].sum(),
                        inputs=Interpolates,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]
        gradnose = grad(outputs=inter[2].sum(),
                        inputs=Interpolates,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]
        gradmouth = grad(outputs=inter[3].sum(),
                         inputs=Interpolates,
                         create_graph=True,
                         retain_graph=True,
                         only_inputs=True)[0]
        gradface = grad(outputs=inter[4].sum(),
                        inputs=Interpolates,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]
        gradients = gradinput + gradeyes + gradnose + gradmouth + gradface

        # calculate gradient penalty
        gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

        loss = Wasserstein_Dis + self.lambda_gp * gradient_penalty

        return loss, Wasserstein_Dis, gradient_penalty


if __name__ == '__main__':
    t1 = torch.rand(2, 3, 120, 120)
    m1 = torch.rand(2, 1, 120, 120)
    m2 = torch.rand(2, 1, 120, 120)
    # D = Discriminator_Attention()
    G = Generator()
    noise = torch.tensor(np.random.uniform(0, 1, (t1.size(0), 64)), dtype=torch.float32)
    out = G(t1, m1, m2, noise)
    # print('img120', out[0].shape)
    # print('img60', out[1].shape)
    # print('img30', out[2].shape)
    # D = Discriminator('c')
    # out = D(torch.cat([out[0], m1], dim=1))
    print(out[2].shape)
