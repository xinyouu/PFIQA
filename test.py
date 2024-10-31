from tqdm import tqdm 
import os
import torch
import logging
import timm
from timm.models.vision_transformer import Block
from timm.models.resnet import Bottleneck
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from utils.util import setup_seed,set_logging,SaveOutput
from script.extract_feature import get_single_resnet_feature,get_vit_feature
from options.test_options import TestOptions
from model.quality_regressor  import  Pixel_Prediction
from data.RealSRQ import RealSRQ

from utils.process import ToTensor, five_random_crop,nine_random_crop
from utils.process_image import five_point_crop
from torchvision import transforms
import numpy as np

class Test:
    def __init__(self, config):
        self.opt = config
        self.create_model()
        self.init_saveoutput()
        self.init_data()
        self.load_model()
        self.test()

    def create_model(self):
        self.resnet50 =  timm.create_model('resnet50',pretrained=True).cuda()
        if self.opt.patch_size == 8:
            self.vit = timm.create_model('vit_base_patch8_224',pretrained=True).cuda()
        else:
            self.vit = timm.create_model('vit_base_patch16_224',pretrained=True).cuda()
        self.regressor = Pixel_Prediction().cuda()

    def init_saveoutput(self):
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.resnet50.modules():
            if isinstance(layer, Bottleneck):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
    
    def init_data(self):
        # test_dataset = PIPAL(
        test_dataset = RealSRQ(
            ref_path=self.opt.test_ref_path,
            dis_path=self.opt.test_dis_path,
            txt_file_name=self.opt.test_list,
            resize=self.opt.resize,
            size=(self.opt.size,self.opt.size),
            flip=self.opt.flip,
            transform=ToTensor(),
        )
        logging.info('number of test scenes: {}'.format(len(test_dataset)))

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=False
        )

    def load_model(self):
        models_dir = self.opt.checkpoints_dir
        checkpoint = torch.load("/home/fofo/sanwei614b/lxy/AHIQ-main/AHIQ-main/checkpoints/ahiq_Waterloo_20240117_ViT_CNN_ver5/epoch_216.pth")
        self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])


    def test(self):
        f = open(os.path.join(self.opt.checkpoints_dir,self.opt.test_file_name), 'w')
        with torch.no_grad():
            losses = []
            self.regressor.eval()
            self.vit.eval()
            self.resnet50.eval()
            pred_epoch = []
            labels_epoch = []

            for data in tqdm(self.test_loader):
                pred = 0
                pred1 = 0
                for i in range(self.opt.num_avg_val):
                    d_img_org = data['d_img_org'].cuda()
                    r_img_org = data['r_img_org'].cuda()
                    d_img_name = data['d_img_name']
                    labels = data['score']
                    scale = data['scale']
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                    scale = torch.squeeze(scale.type(torch.FloatTensor)).cuda()

                    d_img_org, r_img_org = five_point_crop(i, d_img=d_img_org, r_img=r_img_org, config=self.opt)
                    _x = self.vit(d_img_org,scale/5)
                    vit_dis = get_vit_feature(self.save_output).repeat(4, 1, 1)
                    self.save_output.outputs.clear()

                    _y = self.vit(r_img_org,scale/5)
                    vit_ref = get_vit_feature(self.save_output).repeat(4, 1, 1)
                    self.save_output.outputs.clear()
                    B, N, C = vit_ref.shape
                    if self.opt.patch_size == 8:
                        H, W = 28, 28
                    else:
                        H, W = 14, 14
                    assert H * W == N
                    vit_ref = vit_ref.transpose(1, 2).view(B, C, H, W)
                    vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)
                    scale = scale.repeat(4)

                    _ = self.resnet50(d_img_org)
                    cnn_dis_1 = torch.nn.functional.interpolate(get_single_resnet_feature(self.save_output, 0), size=(28, 28),
                                              mode='bilinear', align_corners=False).repeat(4, 1, 1,1)
                    cnn_dis_2 = get_single_resnet_feature(self.save_output, 3).repeat(4, 1, 1,1)
                    cnn_dis_3 = torch.nn.functional.interpolate(get_single_resnet_feature(self.save_output, 7), size=(28, 28),
                                              mode='bilinear', align_corners=False).repeat(4, 1, 1,1)
                    cnn_dis_4 = torch.nn.functional.interpolate(get_single_resnet_feature(self.save_output, 13), size=(28, 28),
                                              mode='bilinear', align_corners=False).repeat(4, 1, 1,1)
                    self.save_output.outputs.clear()

                    _ = self.resnet50(r_img_org)
                    cnn_ref_1 = torch.nn.functional.interpolate(get_single_resnet_feature(self.save_output, 0), size=(28, 28),
                                              mode='bilinear', align_corners=False).repeat(4, 1, 1,1)
                    cnn_ref_2 = get_single_resnet_feature(self.save_output, 3).repeat(4, 1, 1,1)
                    cnn_ref_3 = torch.nn.functional.interpolate(get_single_resnet_feature(self.save_output, 7), size=(28, 28),
                                              mode='bilinear', align_corners=False).repeat(4, 1, 1,1)
                    cnn_ref_4 = torch.nn.functional.interpolate(get_single_resnet_feature(self.save_output, 13), size=(28, 28),
                                              mode='bilinear', align_corners=False).repeat(4, 1, 1,1)
                    self.save_output.outputs.clear()

                    pred1 += self.regressor(scale / 10, vit_dis, vit_ref, cnn_dis_1, cnn_dis_2, cnn_dis_3, cnn_dis_4,
                                           cnn_ref_1, cnn_ref_2, cnn_ref_3, cnn_ref_4)
                    pred += self.regressor(scale / 10, vit_dis, vit_ref, cnn_dis_1, cnn_dis_2, cnn_dis_3, cnn_dis_4,
                                          cnn_ref_1, cnn_ref_2, cnn_ref_3, cnn_ref_4).mean()

                pred /= self.opt.num_avg_val

                for i in range(len(d_img_name)):
                    line = "%s#%f\n" % (d_img_name[i], float(pred))
                    f.write(line)

                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = labels.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)

            # compute correlation coefficient
            rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            print(
                ' SRCC:{:.4} ===== PLCC:{:.4}'.format(rho_s, rho_p))
            logging.info(
                'SRCC:{:.4} ===== PLCC:{:.4}'.format(rho_s, rho_p))
            return rho_s, rho_p

        f.close()

                

if __name__ == '__main__':
    config = TestOptions().parse()
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
    setup_seed(config.seed)
    set_logging(config)
    Test(config)
