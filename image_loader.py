import glob
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as DATA
import random
import numpy
from PIL import ImageDraw
import torch
from config import args


def make_dataset(input_path, gt_path): 
    dataset = []
   
    #dev3
    original_img_rpath = input_path
    shadow_mask_img_rpath = gt_path

    for img_path, mask_path in zip(sorted(glob.glob(os.path.join(original_img_rpath, '*.png'))), sorted(glob.glob(os.path.join(shadow_mask_img_rpath, '*.png')))):
        basename_img = os.path.basename(img_path)
        basename_mask = os.path.basename(mask_path)
        original_img_path = os.path.join(original_img_rpath, basename_img)
        shadow_mask_path = os.path.join(shadow_mask_img_rpath, basename_mask)
        dataset.append([original_img_path, shadow_mask_path])

    return dataset

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

gt_transform = transforms.Compose([
    transforms.ToTensor()
])

def dataset_transform(original_img, shadow_mask_img):
    data_augmentation = ['same','horizontal', 'vertical', 'crop', 'rotation']
    width, height = original_img.size

    sel = random.randint(0,2)
    mode = data_augmentation[sel]

    if mode == 'same':
        original_img = img_transform(original_img.resize((416, 416)))
        shadow_mask_img = gt_transform(shadow_mask_img.resize((416, 416), 1))

        return original_img, shadow_mask_img

    elif mode == 'vertical':
        ver_transform = transforms.RandomVerticalFlip(p=1)
        original_img = ver_transform(original_img)
        shadow_mask_img = ver_transform(shadow_mask_img)
     
        original_img = img_transform(original_img.resize((416, 416)))
        shadow_mask_img = gt_transform(shadow_mask_img.resize((416, 416), 1))
     
        return original_img, shadow_mask_img

    elif mode == 'horizontal':
        hor_transform = transforms.RandomHorizontalFlip(p=1)
        original_img = hor_transform(original_img)
        shadow_mask_img = hor_transform(shadow_mask_img)
     
        original_img = img_transform(original_img.resize((416, 416)))
        shadow_mask_img = gt_transform(shadow_mask_img.resize((416, 416), 1))

        return original_img, shadow_mask_img
    


class shadow_train_triplets_loader(DATA.Dataset):
    def __init__(self):
        super(shadow_train_triplets_loader, self).__init__()

        self.train_set_path = make_dataset(args['input_path'], args['gt_path'])
        
    def __getitem__(self, item):
        original_img_path, shadow_mask_img_path = self.train_set_path[item]

        original_img = Image.open(original_img_path)
        shadow_mask_img = Image.open(shadow_mask_img_path)
        
        original_img, shadow_mask_img = dataset_transform(original_img, shadow_mask_img)
     
        return original_img, shadow_mask_img

    def __len__(self):
        return len(self.train_set_path)




