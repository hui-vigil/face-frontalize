from layers import *
from tools.utils import *
import torch
from torch.autograd import grad
from models.unet3plus import Unet3plus

img_h, img_w = 120, 120



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
    # D = Discriminator_Attention()
    G = Generator()
    out = G(t1, m1, m2)
    # print('img120', out[0].shape)
    # print('img60', out[1].shape)
    # print('img30', out[2].shape)
    # D = Discriminator('c')
    # out = D(torch.cat([out[0], m1], dim=1))
    print(out[0].shape)
