import glob
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as DATA
from config import args

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


gt_transform = transforms.Compose([
    transforms.ToTensor()
])

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

class shadow_test_triplets_loader(DATA.Dataset):
    def __init__(self):
        super(shadow_test_triplets_loader, self).__init__()
        self.train_set_path = make_dataset(args['test_input_path'], args['test_gt_path'])
      
    def __getitem__(self, item):
        original_img_path, shadow_mask_img_path = self.train_set_path[item]
        transform = transforms.ToTensor()

        original_img = Image.open(original_img_path)
        shadow_mask_img = Image.open(shadow_mask_img_path)

        original_img = img_transform(original_img.resize((416, 416)))
        shadow_mask_img = gt_transform(shadow_mask_img.resize((416, 416)))

        return original_img, shadow_mask_img

    def __len__(self):
        return len(self.train_set_path)


