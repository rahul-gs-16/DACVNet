import torch.nn.functional as F
import torch


def model_loss_train_attn_only(disp_ests, disp_gt, mask):
    weights = [1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def model_loss_train_freeze_attn(disp_ests, disp_gt, mask):
    weights = [0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)
    
def model_loss_train(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0] 
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        # print(f'shapes disp_est {disp_est.shape} mask {mask.shape} disp_gt {disp_gt.shape}')
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    
    return sum(all_losses)
    
def model_loss_test(disp_ests, disp_gt, mask):
    weights = [1.0] 
    all_losses = []
    # print(f'disp_est shape = {disp_ests[0].shape} disp_gt shape = {disp_gt.shape} mask shape = {mask.shape}')
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.l1_loss(disp_est.unsqueeze(1)[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)
