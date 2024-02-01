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
from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test
from utils import *
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2
import matplotlib.pyplot as plt
import csv
from skimage.metrics import structural_similarity as ssim

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
parser.add_argument('--csv_filename', default='./predictions/metrics.csv',help='Path to save metrics for each image')


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
viridis_colormap = plt.get_cmap('viridis')
def test():
    avg_test_scalars = AverageMeterDict()
    with open(args.csv_filename,'w',newline='') as csvfile:
        # Create a CSV writer
        csv_writer = csv.writer(csvfile)

        # Write the header (if needed)
        csv_writer.writerow(['File name', 'EPE', 'D1', 'Thresh1', 'Thresh2', 'Thresh3','ssim'])


        for batch_idx, sample in enumerate(TestImgLoader):    
            start_time = time.time()
            loss, scalar_outputs = test_sample(sample,viridis_colormap)
            fn = sample['left_filename'][0].split('/')[-1].split('.')[0][:-6]
            csv_writer.writerow([fn,scalar_outputs['EPE'],scalar_outputs['D1'],scalar_outputs['Thres1'],scalar_outputs['Thres2'],scalar_outputs['Thres3'],scalar_outputs['ssim']])
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            if(batch_idx+1)%500 == 0:
                print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx,
                                                                        len(TestImgLoader), loss,
                                                                        time.time() - start_time))

        avg_test_scalars = avg_test_scalars.mean()
        print("avg_test_scalars", avg_test_scalars)
        csv_writer.writerow(["Average",avg_test_scalars['EPE'],avg_test_scalars['D1'],avg_test_scalars['Thres1'],avg_test_scalars['Thres2'],avg_test_scalars['Thres3'],avg_test_scalars['ssim']])


# test one sample
@make_nograd_func
def test_sample(sample,cmap):
    model.eval()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    disp_ests = model(imgL, imgR)

    # print(f'disp_est shape = {disp_ests[0].shape}')
    save_disp_est = disp_ests[0].detach().cpu().numpy()[0,:,:]
    save_disp_gt  = disp_gt[0,0,:,:].detach().cpu().numpy()
    # print(f'save_disp_est shape = {save_disp_est.shape} save_disp_gt shape = {save_disp_gt.shape}')
    fn = sample['left_filename'][0].split('/')[-1]
    # print('saving ',os.path.join(args.save_path,'pred_'+fn))

    vc_est = cmap(save_disp_est.astype(np.uint8))
    vc_gt = cmap(save_disp_gt.astype(np.uint8))

    vc_est = (vc_est[:,:,:3]*255).astype(np.uint8)
    vc_gt = (vc_gt[:,:,:3]*255).astype(np.uint8)

    print('fn = ',os.path.join(args.save_path,'pred_'+fn))
    cv2.imwrite(os.path.join(args.save_path,'pred_'+fn),save_disp_est) #vc_est
    cv2.imwrite(os.path.join(args.save_path,'gt_'+fn),save_disp_gt) #vc_gt

    disp_gts = [disp_gt]
    loss = model_loss_test(disp_ests, disp_gt, mask)
    scalar_outputs = {"loss": loss}
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    scalar_outputs["ssim"] = ssim(save_disp_est.astype(np.uint8),save_disp_gt.astype(np.uint8))

    return tensor2float(loss), tensor2float(scalar_outputs)

if __name__ == '__main__':
    test()
