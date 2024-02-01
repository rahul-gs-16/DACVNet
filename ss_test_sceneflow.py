# from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss_train_ss_apply_disparity
from utils import *
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser(description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='acvnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/data/sceneflow/", help='data path')
parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--loadckpt', default='./pretrained_model/sceneflow.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--save_path', default='./predictions',help='Path to save predictions and ground truth')
parser.add_argument('--save_left',action='store_true',default=False,help='To save left image')

# parse arguments, set seeds
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=16, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, False, False)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

def test():
    cnt = 0
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):    
        start_time = time.time()
        loss, scalar_outputs = test_sample(sample,cnt)
        cnt += 1
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        if(batch_idx+1)%500 == 0:
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx,
                                                                    len(TestImgLoader), loss,
                                                                    time.time() - start_time))
        
    avg_test_scalars = avg_test_scalars.mean()
    print("avg_test_scalars", avg_test_scalars)


# test one sample
@make_nograd_func
def test_sample(sample,cnt):
    model.eval()
    imgL, imgR = sample['left'], sample['right']

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    disp_ests = model(imgL, imgR)

    # For image save
    save_disp_est = disp_ests[0].detach().cpu().numpy()[0,:,:]

    fn = sample['left_filename'][0].split('/')[-1].replace('rec_left','disp').replace('tiff','png')
    
    # print('saving ',os.path.join(args.datapath,'left',sample['left_filename'][0]))
    if args.save_left:
        left_fn = sample['left_filename'][0].split('/')[-1].replace('rec_left','left').replace('tiff','png')
        cv2.imwrite(os.path.join(args.save_path,left_fn),cv2.imread(os.path.join(args.datapath,sample['left_filename'][0])))
    cv2.imwrite(os.path.join(args.save_path,'pred_'+fn),save_disp_est)

    # disp_ests[0] = disp_ests[0][:,:,:200]
    for i in range(len(disp_ests)):
        disp_ests[i] = disp_ests[i].unsqueeze(1)

    image_loss,disp_grad_loss,warped_left,warped_right = model_loss_train_ss_apply_disparity(imgL,imgR,disp_ests)

    scalar_outputs = {"loss": image_loss}
    scalar_outputs['grad_loss'] = disp_grad_loss
    # scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(image_loss), tensor2float(scalar_outputs)

if __name__ == '__main__':
    test()
