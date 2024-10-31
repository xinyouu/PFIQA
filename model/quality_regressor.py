import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange


class Pixel_Prediction(nn.Module):
    def __init__(self, inchannels=768 * 5, outchannels=256, d_hidn=1024):
        super().__init__()
        self.d_hidn = d_hidn
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size=1)
        self.down_channel1 = nn.Conv2d(3840, outchannels, kernel_size=1)

        self.fully_connected = nn.Linear(in_features=2, out_features=2)
        self.fully_connected_2 = nn.Linear(in_features=2, out_features=2)

        self.feat_smoothing = nn.Sequential(
            nn.Conv2d(in_channels=256 * 4 + 1, out_channels=self.d_hidn, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1)
        )
        self.feat_smoothing2_1 = nn.Sequential(
            nn.Conv2d(in_channels=256*2+1 , out_channels=self.d_hidn, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1)
        )
        self.feat_smoothing2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256*2+1 , out_channels=self.d_hidn, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1)
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_attent = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_f1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )
        self.conv_f2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 784),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def _upsample_add(self, x, y):
         _, _, H, W = y.size()
         return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, scale,f_dis, f_ref,cnn_dis_1,cnn_dis_2,cnn_dis_3,cnn_dis_4,cnn_ref_1,cnn_ref_2,cnn_ref_3,cnn_ref_4):
        cnn_dis = torch.cat((cnn_dis_1,cnn_dis_2,cnn_dis_3,cnn_dis_4), 1)

        cnn_ref = torch.cat((cnn_ref_1,cnn_ref_2,cnn_ref_3,cnn_ref_4), 1)
        cnn_dis = self.down_channel1(cnn_dis)
        cnn_ref = self.down_channel1(cnn_ref)

        cnn_diff = cnn_dis - cnn_ref

        f_scale = self.fc1(scale.unsqueeze(-1))
        if f_dis.shape[0] == 1:
            f_scale = f_scale.view(1, -1)
        f_s = rearrange(f_scale, 'b (c h w) -> b c h w', c=1, h=28, w=28)
        f_dis = self.down_channel(f_dis)
        f_ref = self.down_channel(f_ref)
        f_diff =f_dis  - f_ref

        f_cat = torch.cat((f_dis,cnn_dis,f_diff,cnn_diff,f_s), 1)

        F_dis = torch.cat((f_dis.unsqueeze(4),cnn_dis.unsqueeze(4)), dim=4)
        output_features = self.fully_connected(F_dis)
        output_features= output_features.view(4,512,28,28)

        relu = nn.ReLU()
        output_features_dis = relu(output_features)

        F_diff = torch.cat((f_diff.unsqueeze(4), cnn_diff.unsqueeze(4)), dim=4)
        output_features_2 = self.fully_connected_2(F_diff)
        output_features_2 = output_features_2.view(4, 512, 28, 28)
        relu_2 = nn.ReLU()
        output_features_diff = relu_2(output_features_2)

        feat_fused_cat = self.feat_smoothing(f_cat)
        feat_cat = self.conv1_1(feat_fused_cat)
        w = self.conv_attent(feat_cat)

        feat_fused_diff = self.feat_smoothing2_1(torch.cat((output_features_diff,f_s),1))
        feat_diff = self.conv1_2(feat_fused_diff)
        f1 = self.conv_f1(feat_diff)

        feat_fused_dis = self.feat_smoothing2_2(torch.cat((output_features_dis,f_s),1))
        feat_dis = self.conv1_3(feat_fused_dis)
        f2 = self.conv_f2(feat_dis)

        w1 = w[:, 0:1, :, :]
        w2 = w[:, 1:2, :, :]
        pred1 = ((f1 * w1).sum(dim=2).sum(dim=2)) / w1.sum(dim=2).sum(dim=2)
        pred2 = ((f2 * w2).sum(dim=2).sum(dim=2)) / w2.sum(dim=2).sum(dim=2)

        pred = pred1+pred2
        return pred
