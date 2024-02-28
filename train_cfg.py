import argparse
import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.optim as optimizer
from torch.nn import L1Loss, CosineSimilarity
from models.model import Generator, Discriminator_landmark, Discriminator_Attention, grad
# from models.model_full_scale import Generator, Discriminator_landmark, Discriminator_Attention, grad
# from models.model_with_vggface import Generator_with_Vgg, Discriminator_landmark, Discriminator_Attention, grad
from dataset import ImageData
from tools.utils import resume_model, set_requires_grad
from tools.vggface2 import resnet50  # 额外的分支提取网络


parser = argparse.ArgumentParser(description='CAPG-GAN-with-SA')
# dataset set
parser.add_argument('--profile_train_set', type=str, default='')
parser.add_argument('--front_train_set', type=str, default='')
parser.add_argument('--profile_val_set', type=str, default='')
parser.add_argument('--front_val_set', type=str, default='')
parser.add_argument('--modelout', type=str, default='./modelout')
parser.add_argument('--imgout', type=str, default='./imgout')

# train set
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=int, default=1e-4)
# parser.add_argument('--b1', type=float, default=0.5)
# parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--resume_model', type=bool, default=False)

# 损失参数设置
parser.add_argument('--lambda_L1', type=float, default=0.1)
parser.add_argument('--adv1', type=float, default=0.1)
parser.add_argument('--adv2', type=float, default=0.1)
parser.add_argument('--lambda_ip', type=float, default=10)
parser.add_argument('--lambda_tv', type=float, default=1e-4)
parser.add_argument('--lambda_gp', type=float, default=10)

# 设备
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

# s随机种子
torch.manual_seed(1234)

# Create sample and checkpoint directories
os.makedirs(args.imgout, exist_ok=True)
os.makedirs(args.modelout, exist_ok=True)


# 特征提取网络
Extract_Model = torch.load('./tools/InceptionResnetV1.mdl').to(args.device)
set_requires_grad(Extract_Model, False)

# 额外的分支提取网络
# VggFace2 = resnet50('./tools/resnet50_ft_weight.pkl', num_classes=8631, include_top=False)
# set_requires_grad(resnet50, False)

# 网络设置
G = Generator().to(args.device)
# G_Vgg = Generator_with_Vgg().to(args.device)
D_attention = Discriminator_Attention(args).to(args.device)
D_landmark = Discriminator_landmark().to(args.device)

# 如果不从0开始
if args.resume_model:
    resume_model(G, 'G_param', args.modelout, args.start_epoch)
    resume_model(D_attention, 'D_attention_param', args.modelout, args.start_epoch)
    resume_model(D_landmark, 'D_landmark_param', args.modelout, args.start_epoch)

# 数据集设置
transform = transforms.Compose([
    transforms.CenterCrop((120, 120)),  # 图片中心裁剪成 120x120
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_dataset = ImageData(args.profile_train_set, args.front_train_set, transform=transform)
val_dataset = ImageData(args.profile_val_set, args.front_val_set, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# 优化器设置
optimizer_G = optimizer.Adam(G.parameters(), lr=args.lr)
optimizer_D_attention = optimizer.Adam(D_attention.parameters(), lr=args.lr)
optimizer_D_landmark = optimizer.Adam(D_landmark.parameters(), lr=args.lr)

# 损失函数设置
# 逐像素损失
L1 = L1Loss().to(args.device)
# 余弦相似度用于身份损失
cosine = CosineSimilarity(dim=1).to(args.device)


# landmark判别器wgan-gp损失
def landmark_gp(D_real_output, D_fake_output, Interpolated):
    loss_no_gp = -torch.mean(D_real_output) + torch.mean(D_fake_output)
    out = D_landmark(Interpolated)
    out_grad = grad(outputs=out, inputs=Interpolated,
                    grad_outputs=torch.ones(out.size()).to(args.device),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True)[0].view(out.size(0), -1)
    gradient_penalty = torch.mean((torch.norm(out_grad, p=2) - 1) ** 2)
    landmark_gp_loss = loss_no_gp + args.lambda_gp * gradient_penalty

    return landmark_gp_loss


# 全变分损失
def total_Var(gen_f):
    genf_tv = torch.mean(torch.abs(gen_f[:, :, :-1, :] - gen_f[:, :, 1:, :])) + torch.mean(
        torch.abs(gen_f[:, :, :, :-1] - gen_f[:, :, :, 1:]))

    return genf_tv


# 得到训练数据和生成数据图片
def sample_images(epoch):

    """Saves a generated sample from the test set"""
    val_imgs = next(iter(val_loader))  # 得到一批数据
    train_imgs = next(iter(train_loader))
    for key in val_imgs.keys():
        val_imgs[key] = val_imgs[key][:5].to(args.device)
    for key in train_imgs.keys():
        train_imgs[key] = train_imgs[key][:5].to(args.device)

    # G.eval()
    # test_img
    img120, img60, img30 = G(val_imgs['profile'], val_imgs['real_lm'], val_imgs['target_lm'])

    img_profile = vutils.make_grid(val_imgs['profile'], nrow=1, normalize=True)
    syn_front = vutils.make_grid(img120, nrow=1, normalize=True)
    true_front = vutils.make_grid(val_imgs['front'], nrow=1, normalize=True)
    image_grid = torch.cat([img_profile, syn_front, true_front], dim=2)
    vutils.save_image(image_grid, f'{args.imgout}/epoch_{epoch}_test.png', padding=4, pad_value=255)

    # train_img
    img120, img60, img30 = G(train_imgs['profile'], train_imgs['real_lm'], train_imgs['target_lm'])

    img_profile = vutils.make_grid(train_imgs['profile'], nrow=1, normalize=True)
    syn_front = vutils.make_grid(img120, nrow=1, normalize=True)
    real_front = vutils.make_grid(train_imgs['front'], nrow=1, normalize=True)
    image_grid = torch.cat([img_profile, syn_front, real_front], dim=1)
    vutils.save_image(image_grid, f'{args.imgout}/epoch_{epoch}_train.png', padding=4, pad_value=255)
