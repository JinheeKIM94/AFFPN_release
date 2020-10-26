from image_loader import *
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


def gpu_train():
    dataset = shadow_train_triplets_loader()
    data_loader = Data.DataLoader(dataset, batch_size=args['train_batch_size'], shuffle=True)
    
    model = AFFPNet(args['fb_num_steps'])
    model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])
    
    criterion = nn.BCEWithLogitsLoss().cuda()

    cr_iter = 0

    while(cr_iter<=args['iter_num']):
       
        for i, data in enumerate(data_loader):
            original_image, shadow_mask_image = data

            original_image = original_image.cuda()
            
            with torch.no_grad():
                shadow_mask_image = shadow_mask_image.cuda()

            
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(cr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(cr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
            
            optimizer.zero_grad()
           
            output_img, fb_imgs_x1, fb_imgs_x2, fb_imgs_x3 = model(original_image)
      
            loss0 = criterion(output_img, shadow_mask_image)
            loss1 = 0
            loss2 = 0
            loss3 = 0
            for idx in range(args['fb_num_steps']):
                loss1 += criterion(fb_imgs_x1[idx], shadow_mask_image)
                loss2 += criterion(fb_imgs_x2[idx], shadow_mask_image)
                loss3 += criterion(fb_imgs_x3[idx], shadow_mask_image)


            loss = loss0 + loss1/(float(args['fb_num_steps'])) + loss2/(float(args['fb_num_steps'])) + loss3/(float(args['fb_num_steps']))


            if ((i % 8 == 0) or (i % 8 == 3)):
                log = '[cr_iter %d], [train loss %.5f], [loss0 %.5f], [loss1 %.5f], [loss2 %.5f], [loss3 %.5f], [lr %.13f]' % \
                    (cr_iter , loss, loss0, loss1/(float(args['fb_num_steps'])), loss2/(float(args['fb_num_steps'])), loss3/(float(args['fb_num_steps'])), optimizer.param_groups[1]['lr'])

                print(log)

            loss.backward(retain_graph=True)
            optimizer.step()
                
        cr_iter += 1
        
        if ((cr_iter) % 500) == 0:
            checkpoint_path = os.path.join("./save_param/" + args['dataset'] +"/trained_%d.tar" % cr_iter)
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, checkpoint_path)

        
gpu_train()

