args = {
    'dataset' : 'ISTD', 
    'ck_path' : './trained/trained_ISTD.tar',
    'input_path' : './dataset/ISTD/train/train_A',
    'gt_path' : './dataset/ISTD/train/train_B',
    'test_input_path' :'./dataset/ISTD/test/test_A',
    'test_gt_path' : './dataset/ISTD/test/test_B',
    'fb_num_steps' : 3,
    'gpu_num': '0,1',
    'train_batch_size': 8,
    'test_batch_size': 1,
    'iter_num' : 2000,
    'lr': 5e-3,  
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9
}

resnext_101_32_path = 'resnext/resnext_101_32x4d.pth'
