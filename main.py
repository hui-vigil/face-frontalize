from train_cfg import *
from tools.utils import save_model, set_requires_grad
import sys


def main():
    G.train()
    D_attention.train()
    D_landmark.train()
    Extract_Model.eval()

    step = 0

    for epoch in range(args.start_epoch+1, args.epochs+1):
        for idx, data in enumerate(train_loader):
            # 搬到cuda
            for key in data.keys():
                data[key] = data[key].to(args.device)

            #  train model and model_full_scale
            img120, img60, img30 = G(data['profile'], data['real_lm'], data['target_lm'])

            # # train model_with_vggface
            # _, feature_map = VggFace2(data['profile'])
            # img120, img60, img30 = G_Vgg(data['profile'], data['real_lm'], data['target_lm'], feature_map)

            # ===============================更新 D_Attention=======================
            set_requires_grad(D_attention, True)

            # syn_front = D_attention(img120.detach())
            # real_front = D_attention(data['front'])
            # gp_alpha = torch.rand(data['front'].size(0), 1, 1, 1)
            # gp_alpha = gp_alpha.expand_as(img120).clone().pin_memory().to(args.device)
            # t1 = torch.ones_like(gp_alpha)
            # interpolated = gp_alpha * img120 + (t1 - gp_alpha) * data['front']
            # interpolated = interpolated.to(args.device).requires_grad_()
            # D_A_Loss, Wdis, GP = D_attention.CriticWithGP_Loss(syn_front, real_front, interpolated)

            D_a_real_input = torch.cat([data['front'], data['profile']], dim=1)
            D_a_fake_input = torch.cat([img120.detach(), data['profile']], dim=1)
            D_real_output = D_attention(D_a_real_input)
            D_fake_output = D_attention(D_a_fake_input)
            gp_alpha = torch.rand(data['front'].size(0), 1, 1, 1)
            gp_alpha = gp_alpha.expand_as(D_a_fake_input).clone().pin_memory().to(args.device)
            t1 = torch.ones_like(gp_alpha)
            interpolated_1 = gp_alpha * D_a_fake_input + (t1 - gp_alpha) * D_a_real_input
            interpolated_1 = interpolated_1.to(args.device).requires_grad_()
            D_A_Loss = landmark_gp(D_real_output, D_fake_output, interpolated_1)

            optimizer_D_attention.zero_grad()
            D_A_Loss.backward()
            optimizer_D_attention.step()

            # ===========================更新 D_landmark=============================
            set_requires_grad(D_attention, False)
            set_requires_grad(D_landmark, True)
            D_lm_real_input = torch.cat([data['front'], data['target_lm']], dim=1)
            D_lm_fake_input = torch.cat([img120.detach(), data['target_lm']], dim=1)
            D_real_output = D_landmark(D_lm_real_input)
            D_fake_output = D_landmark(D_lm_fake_input)
            gp_alpha = torch.rand(data['front'].size(0), 1, 1, 1)
            gp_alpha = gp_alpha.expand_as(D_lm_fake_input).clone().pin_memory().to(args.device)
            t2 = torch.ones_like(gp_alpha)
            interpolated_2 = gp_alpha * D_lm_fake_input + (t2 - gp_alpha) * D_lm_real_input
            interpolated_2 = interpolated_2.to(args.device).requires_grad_()
            D_lm_Loss = landmark_gp(D_real_output, D_fake_output, interpolated_2)

            optimizer_D_landmark.zero_grad()
            D_lm_Loss.backward()
            optimizer_D_landmark.step()

            set_requires_grad(D_landmark, False)

            # ======================== 更新生成器 ===============================
            set_requires_grad(G, True)
            img120, img60, img30 = G(data['profile'], data['real_lm'], data['target_lm'])

            # 生成器对抗损失
            adv1_loss = G.GLoss(D_attention(img120))
            adv2_loss = -torch.mean(D_landmark(torch.cat([img120, data['target_lm']], dim=1)))

            # 逐像素损失
            pixel_120_loss = L1(img120, data['front'])
            pixel_60_loss = L1(img60, data['img_60'])
            pixel_30_loss = L1(img60, data['img_30'])
            pixel_loss = (pixel_30_loss + pixel_60_loss + pixel_120_loss) / 3

            # 身份感知损失，余弦距离衡量
            features_real = Extract_Model(data['front'])  # fetures_real 没有梯度
            features_fake = Extract_Model(img120)  # fetures_fake 是带有梯度的张量
            cos_sim = cosine(features_fake, features_real)
            cos_dis = torch.mean(1.0 - cos_sim)

            # 全变分损失
            total_var = total_Var(img120)

            total_G_loss = args.lambda_L1 * pixel_loss + args.lambda_adv1 * adv1_loss + \
                args.lambda_adv2 * adv2_loss + args.lambda_ip * cos_dis + args.lambda_tv * total_var

            optimizer_G.zero_grad()
            total_G_loss.backward()
            optimizer_G.step()

            step += 1
            sys.stdout.write(
                f'step:{step}, G_loss:{total_G_loss}, D_a_loss:{D_A_Loss}, D_lm_loss:{D_lm_Loss}\n'
            )
        if epoch % 2 == 0:
            sample_images(epoch)
            save_model(
                {'epoch': epoch,
                 'G_param': G.state_dict(),
                 'D_landmark_param': D_landmark.state_dict(),
                 'D_attention_param': D_attention.state_dict()},
                epoch,
                dirname=args.model_out
            )
