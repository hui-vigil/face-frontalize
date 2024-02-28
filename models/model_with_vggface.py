from layers import *
from tools.utils import *
import torch
from torch.autograd import grad


img_h, img_w = 120, 120


class Generator_with_Vgg(nn.Module):
    def __init__(self):
        super(Generator_with_Vgg, self).__init__()

        self.conv0 = nn.Sequential(
            conv(5, 64, 7, 1, cal_conv_pad(img_h, img_w, 7, 1), 'kaiming'),
            ResidualBlock(64)
        )
        self.conv1 = nn.Sequential(
            conv(64, 64, 5, 2, cal_conv_pad(img_h, img_w/2, 5, 2)),
            ResidualBlock(64)
        )
        self.conv2 = nn.Sequential(
            conv(64, 128, 5, 2, cal_conv_pad(img_h/2, img_w/4, 5, 2)),
            ResidualBlock(128)
        )
        self.conv3 = nn.Sequential(
            conv(128, 256, 3, 2, cal_conv_pad(img_h/4, img_w/8, 3, 2)),
            ResidualBlock(256)
        )
        # -> 512x8x8
        self.conv4 = nn.Sequential(
            conv(256, 512, 3, 2, cal_conv_pad(img_h/8, 7, 3, 2)),
            ResidualBlock(512)
        )

        self.vgg_face2_reduce_map = nn.Conv2d(2048, 256, 3, 1, 1)  # 2048x7x7 -> 256 x7x7
        # self.fc1 = nn.Linear(512*8*8, 512)  #
        # self.maxout = nn.MaxPool1d(2)
        # self.fc2 = nn.Linear(512, 64*8*8)
        self.relu = nn.ReLU(inplace=True)

        # 解码器
        # self.dc0_1 = deconv(64, 32, 4, 2, cal_deconv_pad(img_h/16, img_w/4, 4, 2))
        # self.dc0_2 = deconv(32, 16, 3, 2, cal_deconv_pad(img_h/4, img_w/2, 3, 2))
        # self.dc0_3 = deconv(16, 8, 3, 2, cal_deconv_pad(img_h/2, img_w, 3, 2))

        self.dc1 = nn.Sequential(
            deconv(768, 256, 3, 2, cal_deconv_pad(7, img_w/8, 3, 2)),
            ResidualBlock(256), ResidualBlock(256)
        )
        self.dc2 = nn.Sequential(
            deconv(256, 128, 2, 2, cal_deconv_pad(img_h/8, img_w/4, 2, 2)),
            ResidualBlock(128), ResidualBlock(128)
        )
        self.dc3 = nn.Sequential(
            deconv(128, 64, 2, 2, cal_deconv_pad(img_h/4, img_w/2, 2, 2)),
            ResidualBlock(64), ResidualBlock(64)
        )
        self.dc4 = nn.Sequential(
            deconv(64, 32, 4, 2, cal_deconv_pad(img_h/2, img_w, 4, 2)),
            ResidualBlock(32), ResidualBlock(32)
        )
        self.conv5 = conv(128, 3, 3, 1, cal_conv_pad(img_h/4, img_w/4, 3, 1))
        self.conv6 = conv(64, 3, 3, 1, cal_conv_pad(img_h/2, img_w/2, 3, 1))
        self.conv7 = conv(32, 16, 3, 1, cal_conv_pad(img_h, img_w, 3, 1))
        self.conv8 = conv(16, 3, 3, 1, cal_conv_pad(img_h, img_w, 3, 1))

    @staticmethod
    def GLoss(Syn_F_GAN):
        Syn_F = Syn_F_GAN[0] + Syn_F_GAN[1] + Syn_F_GAN[2] + Syn_F_GAN[3] + Syn_F_GAN[4]
        loss = -Syn_F.mean() / 5

        return loss

    def forward(self, x, mark_real, mark_target, feature_map):
        x = torch.cat((x, mark_real, mark_target), dim=1)
        c0 = self.conv0(x)
        # print('c0', c0.shape)
        c1 = self.conv1(c0)
        # print('c1', c1.shape)
        c2 = self.conv2(c1)
        # print('c2', c2.shape)
        c3 = self.conv3(c2)
        # print('c3', c3.shape)
        c4 = self.conv4(c3)  # -> 512x7x7
        # feature_map = 2048x7x7 将其通道减少
        vgg_fea = self.vgg_face2_reduce_map(feature_map)  # 256x7x7
        c4_to_dec = torch.cat([c4, vgg_fea], dim=1)  # 768x7x7
        # print('c4', c4_to_dec.shape)
        dc1 = self.dc1(c4_to_dec)
        # print('dc1', dc1.shape)
        dc2 = self.dc2(dc1)
        # print('dc2', dc2.shape)
        dc3 = self.dc3(dc2)
        # print('dc3', dc3.shape)
        dc4 = self.dc4(dc3)
        # print('dc4', dc4.shape)
        img_30 = self.conv5(dc2)
        img_60 = self.conv6(dc3)
        conv7 = self.conv7(dc4)
        img_120 = self.conv8(conv7)

        return img_120, img_60, img_30


class Discriminator_landmark(nn.Module):
    def __init__(self):
        super(Discriminator_landmark, self).__init__()
        batch_norm = True
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
            nn.Conv2d(3, 32, 3, 2, 1),  # d_conv0
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # d_conv1
            # nn.LayerNorm([12, 34]),  # 224
            nn.LayerNorm([7, 18]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # d_conv2
            # nn.LayerNorm([6, 17]),  # 224
            nn.LayerNorm([4, 9]),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # d_conv3
            # nn.LayerNorm([3, 9]),  # 224
            nn.LayerNorm([2, 5]),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(2560, 1)
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
    fm = torch.rand(2, 2048, 7, 7)
    # D = Discriminator_Attention()
    G = Generator_with_Vgg()
    out = G(t1, m1, m2, fm)
    # print('img120', out[0].shape)
    # print('img60', out[1].shape)
    # print('img30', out[2].shape)
    # D = Discriminator('c')
    # out = D(torch.cat([out[0], m1], dim=1))
    print(out[0].shape)
