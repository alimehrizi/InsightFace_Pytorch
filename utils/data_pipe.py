from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm

def de_preprocess(tensor):
    return tensor*0.5 + 0.5


def get_train_loader(conf,dataset_type="custom"):
    if conf.data_mode in ['ms1m', 'concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder/'imgs')
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'concat']:
        vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder/'imgs')
        print('vgg loader generated')        
    if conf.data_mode == 'vgg':
        ds = vgg_ds
        class_num = vgg_class_num
    elif conf.data_mode == 'ms1m': 
        ds = ms1m_ds
        class_num = ms1m_class_num
    elif conf.data_mode == 'concat':
        for i,(url,label) in enumerate(vgg_ds.imgs):
            vgg_ds.imgs[i] = (url, label + ms1m_class_num)
        ds = ConcatDataset([ms1m_ds,vgg_ds])
        class_num = vgg_class_num + ms1m_class_num
    elif conf.data_mode == 'emore':
        ds, class_num = get_train_dataset(conf.emore_folder/'imgs',dataset_type)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True,prefetch_factor=8,persistent_workers=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num 
    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite()
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img  =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img,mode="RGB")
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=100)

# class train_dataset(Dataset):
#     def __init__(self, imgs_bcolz, label_bcolz, h_flip=True):
#         self.imgs = bcolz.carray(rootdir = imgs_bcolz)
#         self.labels = bcolz.carray(rootdir = label_bcolz)
#         self.h_flip = h_flip
#         self.length = len(self.imgs) - 1
#         if h_flip:
#             self.transform = trans.Compose([
#                 trans.ToPILImage(),
#                 trans.RandomHorizontalFlip(),
#                 trans.ToTensor(),
#                 trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])
#         self.class_num = self.labels[-1] + 1
        
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, index):
#         img = torch.tensor(self.imgs[index+1], dtype=torch.float)
#         label = torch.tensor(self.labels[index+1], dtype=torch.long)
#         if self.h_flip:
#             img = de_preprocess(img)
#             img = self.transform(img)
#         return img, label

import logging
import random

import cv2
import math
import numpy as np
from numpy.random.mtrand import randint
from scipy.sparse.construct import rand




def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im_hsv[0,:,:] = np.random.uniform(0,1)*180
        im_hsv[1,:,:] = np.random.uniform(0,1)*255
        hue, sat, val = cv2.split(im_hsv)
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
    return im 


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def random_perspective(im, segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    # s = 1.0
    # if scale>0:
    #     s = random.uniform(1 - scale/2, 1 + scale)
    # else:
    #     s = random.uniform(1 + scale, 1 - scale/2)
    s = random.uniform(1 - scale, 1)
    #s = 2 ** random.uniform(-scale, 0)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped


    return im



def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates




def cutomizeAugmentations(img, p=0.5):
    config = {
            "sigma_list": [150],
            "G"         : 5.0,
            "b"         : 25.0,
            "alpha"     : 125.0,
            "beta"      : 46.0,
            "low_clip"  : 0.01,
            "high_clip" : 0.99
        }
    H, W, c = img.shape 
    K = 8
    w = np.random.randint(W//10,W//2)
    h = np.random.randint(H//10,H//2)
    xi = np.random.randint(0,W-w)
    yi = np.random.randint(0,H-h)
    color = img[yi,xi]#[np.random.choice(img[yc,:,0],1)[0],np.random.choice(img[yc,:,0],1)[0],np.random.choice(img[yc,:,0],1)[0]]
    if random.uniform(0,1)<p:
        img[yi:yi+h,xi:xi+w,:]= color 
    
    return img




def getAugmentations(p=0.5):
    try:
        import imgaug as ia
        import imgaug.augmenters as iaa
    except:
        ImportError("cant import imgaug")
    seq = None


    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(p, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.

    seq = iaa.Sequential(
        [
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((1, 3),
                [
                    # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(200, 255))), # convert images into their superpixel representation
                    # iaa.OneOf([
                    #     iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    #     iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    #     iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                    #     iaa.MotionBlur(k=5)
                    # ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    # iaa.UniformColorQuantization(),
                    iaa.SigmoidContrast(gain=(8,10),cutoff=(0.1,1.0)),
                    iaa.OneOf([
                        iaa.Cutout(nb_iterations=2),
                        iaa.Dropout((0.01, 0.3), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    #iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-50, 50), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.Multiply((0.1, 2.0), per_channel=0.5),
                    iaa.JpegCompression(compression=(70,100)),
                    
                    iaa.LinearContrast((0.5, 1.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.PiecewiseAffine(scale=(0,0.05)),
                    

                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq 

def randomPad(img,p=0.5):
    px = int(np.random.uniform(0,0.05)*img.shape[1]) 
    py = int(np.random.uniform(0,0.05)*img.shape[0])
    if np.random.uniform(0,1)<p:
        temp_img = np.full((img.shape[0]+py,img.shape[1]+px,3),img[:,:10].mean(),dtype=float)*np.random.uniform(0,1) 
        temp_img = temp_img.astype(np.uint8)
        temp_img[py//2:py//2+img.shape[0],px//2:px//2+img.shape[1],:] = img 
        img = temp_img.copy()
    else:
        img = img[py:img.shape[0]-py,px:img.shape[1]-px] 
    return img 


import os 
import glob 
from random import shuffle 
def getDirImages(dir):
    images_list = []
    
    for fp in glob.glob(dir+'/*'):
        fp_images = []
        if os.path.isdir(fp):
            fp_images = getDirImages(fp)
        images_list.extend(fp_images)
    images_list.extend(glob.glob(dir+'/*.jpg'))

    return images_list

class ImageDataset(Dataset):
    def __init__(self, root_path,img_size=(32,120), to_tensor=False, augment=False,return_name=False,cache_images=False):
        super(ImageDataset,self).__init__()
        self.to_tensor = to_tensor
        self.cache_images = cache_images
        self.augment = augment 
        self.root = root_path
        self.data_path = str(root_path)
        self.img_size = img_size
        self.return_name = return_name
        self.class_folder_index = -2
        self.augmnet_prob  = 0.5
        self.negative_prob = 0.1
        self.images_list = []
        self.hsv_h = 0.01  # image HSV-Hue augmentation (fraction)
        self.hsv_s = 0.01  # image HSV-Saturation augmentation (fraction)
        self.hsv_v = 0.3  # image HSV-Value augmentation (fraction)
        self.degrees =  5  # image rotation (+/- deg)
        self.translate =  0.0001  # image translation (+/- fraction)
        self.scale =  0.05  # image scale (+/- gain)
        self.shear =  0.01  # image shear (+/- deg)
        self.perspective =  0.0001
        self.random_pad = True 
        
        

        self.extra_augment = getAugmentations(1.0)
        self.agment_functions = [
            lambda img : random_perspective(img,
                                    degrees=self.degrees,
                                    translate=self.translate,
                                    scale=self.scale,
                                    shear=self.shear,
                                    perspective=self.perspective,border=(10,10)), 
                                    randomPad,
                                    # lambda img: cutomizeAugmentations(img,p=0.7),
                                    # lambda img: self.extra_augment(images=[img])[0],
                                    lambda img: cv2.flip(img,1),
                                    lambda img : augment_hsv(img, hgain=self.hsv_h, sgain=self.hsv_s, vgain=self.hsv_v),

                                    ]
        self.class_map = {}
        if os.path.isfile(self.data_path ):
            with open(self.data_path ,'r') as f:
                line = f.readline()
                while line:
                    ipath = line.strip()

                    if os.path.exists(ipath):
                    
                        self.images_list.append(ipath)
                    line = f.readline()
        else:
            self.images_list = getDirImages(self.data_path )

        self.images_cache = {}
        cached_size = 0
        print("Load and cache images ...")
        for index,ipath in enumerate(self.images_list):
            cls = ipath.split('/')[self.class_folder_index] 
            if cls not in self.class_map.keys():
                self.class_map[cls] = int(cls)
            if self.cache_images and cached_size<64.0:
                img = cv2.imread(ipath)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                ss = float(np.prod(img.shape))/(1024.0*1024.0*1024.0)
                cached_size += ss
                self.images_cache[ipath] = img 
                if cached_size%8<=ss:
                    print(f"Caching {index}:  {cached_size} GB")
        print(f"Cached {cached_size} GB")
        if self.augment:
            random.shuffle(self.images_list) 
        self.num_classes = len(self.class_map)
        
        if False and self.augment:  # over sampling for small classes
            self.samples_per_class = dict([(k, []) for k in range(self.num_classes)])
            for img_path in self.images_list:
                name = img_path.split('/')[-2]
                label = int(self.class_map[name])
                self.samples_per_class[label].append(img_path)
            num_balanced_samples = len(self.images_list)/self.num_classes  
            for k in self.samples_per_class.keys():
                n = len(self.samples_per_class[k]) 
                r = int(num_balanced_samples/n) 
                print(f"samples for class {k} = {n}")
                if r>1:
                    print(f"    Adding {(r-1)*n} new samples")
                    self.images_list += self.samples_per_class[k]*r


        
    def get_annot(self, img_path):

        name = img_path.split('/')[self.class_folder_index]
        label = int(self.class_map[name])
        annot = label
        return annot


    def __len__(self):
        data_len = len(self.images_list) 
        return data_len

    

    def __getitem__(self,index):
        shuffle(self.agment_functions)

        img = None 
        img_path = self.images_list[index]
        label = self.get_annot(img_path)
        if img_path in self.images_cache.keys():
            img = self.images_cache[img_path]
        else:   
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # try:
        if self.augment and random.uniform(0,1)<self.augmnet_prob:
            na = 0
            for t in range(len(self.agment_functions)):
                if random.uniform(0,1)<2.0/(len(self.agment_functions)):
                    img = self.agment_functions[t](img)
                    na += 1

        if img.shape[0] != self.img_size[0] or img.shape[1] != self.img_size[1]:
            img  = cv2.resize(img,self.img_size)

        frame = img#np.uint8(img)


        if self.to_tensor:
            frame = torch.from_numpy(frame)
            frame = frame.to(torch.float32)/255.0
            frame = (frame - torch.Tensor([0.5,0.5,0.5]).view(1,1,3))/torch.Tensor([0.5,0.5,0.5]).view(1,1,3)
            frame = frame.permute((2,0,1))
            #frame = torch.permute(frame,(2,0,1))
            

        # if self.to_tensor:
        #     label = torch.from_numpy(label)
        if self.return_name:
            return frame, label, img_path
        return frame, label
        


    
def get_train_dataset(imgs_folder,dataset="custom"):
    ds = None 
    class_num = 0
    if dataset=="torch":
        train_transform = trans.Compose([
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        ds = ImageFolder(imgs_folder, train_transform)
        class_num = ds[-1][1] + 1
    elif dataset=="custom":
        ds = ImageDataset(imgs_folder,(112,112),to_tensor=True,cache_images=False,augment=True)
        class_num = ds.num_classes
    else:
        raise ValueError("Unvalid dataset <"+dataset+">")
    return ds, class_num

