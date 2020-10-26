from test_image_loader import *
from torch.autograd import Variable 
from models import *                
import torchvision
from torch import optim
import torch.utils.data as Data
import os
from torchvision.utils import save_image
import torch.nn as nn
import numpy as np
import numpy as npnvid
import cv2
import scipy.misc
import torch.nn.functional as F
import argparse
from PIL import Image
from torchvision import transforms
import torchvision.transforms as transforms
from config import args
to_pil = transforms.ToPILImage()


os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_num']

     
def gpu_test():
    dataset = shadow_test_triplets_loader()
    data_loader = Data.DataLoader(dataset, batch_size=args['test_batch_size'])

    model = AFFPNet(args['fb_num_steps'])
    model = nn.DataParallel(model)
    model = model.cuda()

    checkpoint = torch.load(args['ck_path'])
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    
    for i, data in enumerate(data_loader):
        original_image, shadow_mask_image = data
        
        original_image = original_image.cuda()
        shadow_mask_image = shadow_mask_image.cuda()

        model.eval()
        print(i)
        output_img, fb_imgs_x1, fb_imgs_x2, fb_imgs_x3 = model(original_image)
        
        batchsize_in = original_image.size(0)

        for img_idx in range(batchsize_in):
                save_image(output_img[img_idx], "./output/" + args['dataset'] + "/out_%d.png" %((img_idx+batchsize_in*i))) 
                
gpu_test()
    
