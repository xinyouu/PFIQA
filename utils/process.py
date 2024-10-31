import torch
import numpy as np
from torchvision import transforms


def random_crop(d_img, config):
    b, c, h, w = d_img.shape
    top = np.random.randint(0, h - config.crop_size)
    left = np.random.randint(0, w - config.crop_size)
    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    return d_img_org


def crop_image(top, left, patch_size, img=None):
    tmp_img = img[:, :, top:top + patch_size, left:left + patch_size]
    return tmp_img


def five_point_crop(idx, d_img, r_img, config):
    new_h = config.crop_size
    new_w = config.crop_size
    b, c, h, w = d_img.shape
    b_lr, c_lr, h_lr, w_lr = r_img.shape
    if idx == 0:
        top = 0
        left = 0
        top_lr = 0
        left_lr = 0


    elif idx == 1:
        top = 0
        top_lr = 0
        left = w - new_w
        left_lr = w_lr - 224
    elif idx == 2:
        top = h - new_h
        # top_lr = 0
        top_lr = h_lr - 224

        left = 0
        left_lr = 0
    elif idx == 3:
        top = h - new_h
        top_lr = h_lr - 224
        left = w - new_w
        left_lr = w_lr - 224
    elif idx == 4:
        center_h = h // 2
        center_w = w // 2
        top = center_h - new_h // 2
        left = center_w - new_w // 2
        # -----------LR:
        center_h_lr = h_lr // 2
        center_w_lr = w_lr // 2
        top_lr = center_h_lr - 224 // 2
        left_lr = center_w_lr - 224 // 2

    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    r_img_org = crop_image(top_lr, left_lr, 224, img=r_img)
    return d_img_org, r_img_org

def five_random_crop(idx, d_img, r_img, config):
    new_h = config.crop_size
    new_w = config.crop_size
    b, c, h, w = d_img.shape
    b_lr, c_lr, h_lr, w_lr = r_img.shape
    if idx == 0:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left


    elif idx == 1:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left
    elif idx == 2:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left
    elif idx == 3:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left
    elif idx == 4:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left


    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    r_img_org = crop_image(top_lr, left_lr, 224, img=r_img)
    return d_img_org, r_img_org

def nine_random_crop(idx, d_img, r_img, config):
    new_h = config.crop_size
    new_w = config.crop_size
    b, c, h, w = d_img.shape
    b_lr, c_lr, h_lr, w_lr = r_img.shape
    if idx == 0:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left


    elif idx == 1:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left
    elif idx == 2:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left
    elif idx == 3:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left
    elif idx == 4:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left
    elif idx == 5:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left
    elif idx == 6:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left
    elif idx ==7:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left
    elif idx == 8:
        top = np.random.randint(0, h - config.crop_size)
        left = np.random.randint(0, w - config.crop_size)
        top_lr = top
        left_lr = left


    d_img_org = crop_image(top, left, config.crop_size, img=d_img)
    r_img_org = crop_image(top_lr, left_lr, 224, img=r_img)
    return d_img_org, r_img_org


# 自己加上去的。、
def imgCut_test(img, patch_size):
    b, c, imhigh, imwidth = img.shape
    range_y = np.arange(0, imhigh - patch_size, patch_size)
    range_x = np.arange(0, imwidth - patch_size, patch_size)
    if range_y[-1] != imhigh - patch_size:
        range_y = np.append(range_y, imhigh - patch_size)
    if range_x[-1] != imwidth - patch_size:
        range_x = np.append(range_x, imwidth - patch_size)

    sz = len(range_y) * len(range_x)

    res = torch.zeros(sz, 3, patch_size, patch_size)

    index = 0

    for y in range_y:
        for x in range_x:
            patch = img[..., y:y + patch_size, x:x + patch_size]
            res[index] = patch
            # res2.append(patch)
            index = index + 1
    return res, sz


class RandCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        d_img = sample['d_img_org']
        score = sample['score']
        scale = sample['scale']
        r_img = sample['r_img_org']
        # scale = sample['scale']  #####
        d_img_name = sample['d_img_name']
        c, h, w = d_img.shape
        new_h = self.patch_size
        new_w = self.patch_size
        ret_r_img = np.zeros((c, self.patch_size, self.patch_size))
        ret_d_img = np.zeros((c, self.patch_size, self.patch_size))  # 创建两个用零填充的数组，表示处理后的图像结果
        top = np.random.randint(0, h - new_h - 1)
        left = np.random.randint(0, w - new_w - 1)
        tmp_r_img = r_img[:, top: top + new_h, left: left + new_w]
        tmp_d_img = d_img[:, top: top + new_h, left: left + new_w]
        # ret_r_img += tmp_r_img
        # ret_d_img += tmp_d_img
        # ret_d_img = d_img[:, top: top + new_h, left: left + new_w]
        # r_img = r_img[:, top:top + new_h, left: left + new_w] ######lxy加，为了配准
        if tmp_r_img.shape != (3, 224, 224):
            tmp_r_img[:, :tmp_r_img.shape[1], :tmp_r_img.shape[2]] = tmp_r_img

        sample = {
            'd_img_org': tmp_d_img,
            'score': score,
            'scale': scale,
            'r_img_org': tmp_r_img,
            'd_img_name': d_img_name
            # 'scale': scale  ####
        }
        return sample


class RandCrop_LR(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        # r_img : C x H x W (numpy)
        r_img = sample['r_img_org']
        score = sample['score']
        scale = sample['scale']
        d_img = sample['d_img_org']
        d_img_name = sample['d_img_name']

        c, h, w = r_img.shape
        new_h = self.patch_size
        new_w = self.patch_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        ret_r_img = r_img[:, top: top + new_h, left: left + new_w]

        sample = {
            'd_img_org': d_img,
            'score': score,
            'scale': scale,
            'r_img_org': ret_r_img,
            'd_img_name': d_img_name
        }
        return sample


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, sample):
        # r_img: C x H x W (numpy)
        r_img, d_img = sample['r_img_org'], sample['d_img_org']
        d_img_name = sample['d_img_name']
        score = sample['score']
        scale = sample['scale']
        # r_img = (r_img - self.mean) / self.var
        d_img = (d_img - self.mean) / self.var

        sample = {'d_img_org': d_img, 'score': score,'scale': scale, 'r_img_org': r_img, 'd_img_name': d_img_name}
        return sample


class RandHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        r_img, d_img = sample['r_img_org'], sample['d_img_org']
        d_img_name = sample['d_img_name']
        score = sample['score']
        scale = sample['scale']
        prob_lr = np.random.random()
        # np.fliplr needs HxWxC
        if prob_lr > 0.5:
            d_img = np.fliplr(d_img).copy()
            r_img = np.fliplr(r_img).copy()
        sample = {
            'd_img_org': d_img,
            'score': score,
            'scale': scale,
            'r_img_org': r_img,
            'd_img_name': d_img_name
        }
        return sample


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        r_img, d_img = sample['r_img_org'], sample['d_img_org']
        score = sample['score']
        scale = sample['scale']
        d_img_name = sample['d_img_name']
        d_img = torch.from_numpy(d_img).type(torch.FloatTensor)
        r_img = torch.from_numpy(r_img).type(torch.FloatTensor)
        score = torch.from_numpy(score).type(torch.FloatTensor)
        sample = {
            'd_img_org': d_img,
            'score': score,
            'scale': scale,
            'r_img_org': r_img,
            'd_img_name': d_img_name
        }
        return sample

# if __name__ == '__main__':
#     x = torch.randn(1,3,500,500)
#
#     y,_ = imgCut_test(x,224)
#     print(y.shape)