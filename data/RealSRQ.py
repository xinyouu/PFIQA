import os
import torch
import numpy as np
import cv2
import re


class RealSRQ(torch.utils.data.Dataset):
    def __init__(self, ref_path, dis_path, txt_file_name, transform, resize=False, size=None, flip=False):
        super(RealSRQ, self).__init__()
        self.ref_path = ref_path
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.flip = flip
        self.resize = resize
        self.size = size
        ref_files_data, dis_files_data, score_data, scale_data = [], [], [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                value = line.split('#')
                dis = value[0]
                value1 = value[0].split('_')
                scale = float(value1[2][-1])###realsrq
                # scale = float(value1[1][-1])###waterloo
                # scale = float(value1[1])###qads
                score = float(value[1])
                ref = value[2].strip()
                dis_files_data.append(dis)
                score_data.append(score)
                ref_files_data.append(ref)
                scale_data.append(scale)

        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.data_dict = {'r_img_list': ref_files_data, 'd_img_list': dis_files_data, 'score_list': score_data, 'scale_list': scale_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['r_img_list'])

    def __getitem__(self, idx):
        # r_img: H x W x C -> C x H x W

        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        if self.flip:
            d_img = np.fliplr(d_img).copy()
        if self.resize:
            d_img = cv2.resize(d_img, self.size)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = (d_img - 0.5) / 0.5
        d_img = np.transpose(d_img, (2, 0, 1))

        r_img_name = self.data_dict['r_img_list'][idx]
        r_img = cv2.imread(os.path.join(self.ref_path, r_img_name), cv2.IMREAD_COLOR)
        r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
        if self.flip:
            r_img = np.fliplr(r_img).copy()
        if self.resize:
            r_img = cv2.resize(r_img, self.size)
        r_img = np.array(r_img).astype('float32') / 255
        r_img = (r_img - 0.5) / 0.5
        r_img = cv2.resize(r_img, dsize=(d_img.shape[2], d_img.shape[1]), interpolation=cv2.INTER_LINEAR)
        r_img = np.transpose(r_img, (2, 0, 1))
        # if r_img.shape != d_img.shape:
        #     print('1111')
        score = self.data_dict['score_list'][idx]
        scale = self.data_dict['scale_list'][idx]

        sample = {
            'r_img_org': r_img,
            'd_img_org': d_img,
            'score': score,
            'd_img_name': d_img_name,
            'scale':scale
        }
        if self.transform:
            sample = self.transform(sample)
        return sample