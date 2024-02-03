import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from .layers_for_ss import BackprojectDepth, Project3D, disp_to_depth,transformation_from_parameters_gsr
from .layers_for_ss import apply_disparity

def gradient_x( img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx

def gradient_y( img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy

def disp_smoothness(disp,img): # , pyramid21
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img) 
    image_gradients_y = gradient_y(img) 

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1,
                    keepdim=True)) 
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1,
                    keepdim=True)) 

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return torch.abs(smoothness_x) + torch.abs(smoothness_y)

def SSIM( x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


# def apply_disparity(img, disp):
#     batch_size, _, height, width = img.size()

#     # Original coordinates of pixels
#     x_base = torch.linspace(0, 1, width).repeat(batch_size,
#                 height, 1).type_as(img)
#     y_base = torch.linspace(0, 1, height).repeat(batch_size,
#                 width, 1).transpose(1, 2).type_as(img)

#     disp = disp.float()/255.0
#     # Apply shift in X direction
#     x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
#     flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
#     # In grid_sample coordinates are assumed to be between -1 and 1
#     output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
#                            padding_mode='zeros')
# #     output = F.grid_sample(img, flow_field, mode='bilinear',
# #                            padding_mode='zeros')

#     return output

# (img,pred_disp,view_to_syn): 


def model_loss_train_ss(imgL, imgR, disp_ests,left_view_syn,right_view_syn):
    weights = [0.5, 0.5, 0.7, 1.0] 
    all_losses = []
    # print(f'model_loss_train_ss disp_ests len = ',len(disp_ests))
    ssim_w = 0.85
    for disp_est, weight in zip(disp_ests, weights):

        # print(f'disp_est shape = {disp_est.shape}')
        
        # Generate images for image consistency loss
        warped_left = left_view_syn(imgR,disp_est)
        warped_right = right_view_syn(imgL,disp_est)

        l1_left = torch.mean(torch.abs(warped_left-imgL))
        l1_right = torch.mean(torch.abs(warped_right-imgR))


        ssim_left = SSIM(warped_left,imgL)
        ssim_right = SSIM(warped_right,imgR)

        image_loss_left = ssim_w*ssim_left + (1-ssim_w)*l1_left
        image_loss_right = ssim_w*ssim_right + (1-ssim_w)*l1_right
        image_loss = sum(image_loss_left + image_loss_right)

        all_losses.append(torch.sum(image_loss)*weight)

        # Disparities smoothness loss
        disp_grad_loss = disp_smoothness(disp_est,imgL)
        disp_grad_loss = torch.sum(disp_grad_loss)*0.1
        all_losses.append(disp_grad_loss)

        # all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    # return sum(all_losses)
    return sum(all_losses),disp_grad_loss,warped_left,warped_right#torch.sum(disp_grad_loss)
    
def model_loss_test_ss(imgL,imgR,disp_ests,left_view_syn,right_view_syn):
    weights = [1.0] 
    all_losses = []
    ssim_w = 0.85
    # print(f'disp_est shape = {disp_ests[0].shape} disp_gt shape = {disp_gt.shape} mask shape = {mask.shape}')
    for disp_est, weight in zip(disp_ests, weights):
        # all_losses.append(weight * F.l1_loss(disp_est.unsqueeze(1)[mask], disp_gt[mask], size_average=True))
        warped_left = left_view_syn(imgR,disp_est)
        warped_right = right_view_syn(imgL,disp_est)
        l1_left = torch.mean(torch.abs(warped_left-imgL))
        l1_right = torch.mean(torch.abs(warped_right-imgR))


        ssim_left = SSIM(warped_left,imgL)
        ssim_right = SSIM(warped_right,imgR)

        image_loss_left = ssim_w*ssim_left + (1-ssim_w)*l1_left
        image_loss_right = ssim_w*ssim_right + (1-ssim_w)*l1_right
        image_loss = sum(image_loss_left + image_loss_right)
        # print(f'image loss shape = {image_loss.shape}')
        all_losses.append(torch.sum(image_loss))
        # Disparities smoothness loss
        # disp_grad_loss = disp_smoothness(disp_est,imgL)
        # all_losses.append(torch.sum(disp_grad_loss))

    return sum(all_losses),warped_left,warped_right

# def model_loss_train_ss_apply_disparity(imgL, imgR, disp_ests):
#     weights = [0.5, 0.5, 0.7, 1.0] 
#     all_losses = []
#     # print(f'model_loss_train_ss disp_ests len = ',len(disp_ests))
#     ssim_w = 0.85
#     for disp_est, weight in zip(disp_ests, weights):

#         # Generate images for image consistency loss
#         warped_left = apply_disparity(imgR,-disp_est,norm=imgR.shape[-1])
#         warped_right = apply_disparity(imgL,disp_est,norm=imgL.shape[-1])

#         l1_left = torch.mean(torch.abs(warped_left-imgL))
#         l1_right = torch.mean(torch.abs(warped_right-imgR))


#         ssim_left = SSIM(warped_left,imgL)
#         ssim_right = SSIM(warped_right,imgR)

#         image_loss_left = ssim_w*ssim_left + (1-ssim_w)*l1_left
#         image_loss_right = ssim_w*ssim_right + (1-ssim_w)*l1_right
#         image_loss = sum(image_loss_left + image_loss_right)

#         all_losses.append(torch.sum(image_loss)*weight)

#         # Disparities smoothness loss
#         disp_grad_loss = disp_smoothness(disp_est,imgL)
#         disp_grad_loss = torch.sum(disp_grad_loss)*0.1
#         all_losses.append(disp_grad_loss)

#         # all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
#     # return sum(all_losses)
#     return sum(all_losses),disp_grad_loss,warped_left,warped_right#torch.sum(disp_grad_loss)
    
# def model_loss_test_ss_apply_disparity(imgL,imgR,disp_ests):
#     weights = [1.0] 
#     all_losses = []
#     ssim_w = 0.85
#     # print(f'disp_est shape = {disp_ests[0].shape} disp_gt shape = {disp_gt.shape} mask shape = {mask.shape}')
#     for disp_est, weight in zip(disp_ests, weights):
#         # all_losses.append(weight * F.l1_loss(disp_est.unsqueeze(1)[mask], disp_gt[mask], size_average=True))
#         # Generate images for image consistency loss
#         warped_left = apply_disparity(imgR,-disp_est,imgR.shape[-1])
#         warped_right = apply_disparity(imgL,disp_est,imgL.shape[-1])
#         l1_left = torch.mean(torch.abs(warped_left-imgL))
#         l1_right = torch.mean(torch.abs(warped_right-imgR))


#         ssim_left = SSIM(warped_left,imgL)
#         ssim_right = SSIM(warped_right,imgR)

#         image_loss_left = ssim_w*ssim_left + (1-ssim_w)*l1_left
#         image_loss_right = ssim_w*ssim_right + (1-ssim_w)*l1_right
#         image_loss = sum(image_loss_left + image_loss_right)
#         # print(f'image loss shape = {image_loss.shape}')
#         all_losses.append(torch.sum(image_loss))
#         # Disparities smoothness loss
#         # disp_grad_loss = disp_smoothness(disp_est,imgL)
#         # all_losses.append(torch.sum(disp_grad_loss))

#     return sum(all_losses),warped_left,warped_right



def model_loss_train_ss_apply_disparity(imgL, imgR, disp_ests):
    weights = [0.5, 0.5, 0.7, 1.0] 
    all_losses = []
    # print(f'model_loss_train_ss disp_ests len = ',len(disp_ests))
    ssim_w = 0.85
    for disp_est, weight in zip(disp_ests, weights):

        # Generate images for image consistency loss
        warped_left = apply_disparity(imgR,-disp_est,norm=imgR.shape[-1])
        warped_right = apply_disparity(imgL,disp_est,norm=imgL.shape[-1])

        # l1_left = torch.mean(torch.abs(warped_left- imgL))
        # l1_right = torch.mean(torch.abs(warped_right - imgR))

        # ssim_left = SSIM(warped_left ,imgL)
        # ssim_right = SSIM(warped_right,imgR)

#        Taking only 208 pixels for loss computation
        l1_left = torch.mean(torch.abs(warped_left[:,:,:,-208:]- imgL[:,:,:,-208:]))
        l1_right = torch.mean(torch.abs(warped_right[:,:,:,:208] - imgR[:,:,:,:208]))

        ssim_left = SSIM(warped_left[:,:,:,-208:],imgL[:,:,:,-208:])
        ssim_right = SSIM(warped_right[:,:,:,:208],imgR[:,:,:,:208])

        image_loss_left = ssim_w*ssim_left + (1-ssim_w)*l1_left
        image_loss_right = ssim_w*ssim_right + (1-ssim_w)*l1_right
        image_loss = sum(image_loss_left + image_loss_right)

        all_losses.append(torch.sum(image_loss)*weight)

        # Disparities smoothness loss
        # disp_grad_loss = disp_smoothness(disp_est[:,:,:,:208],imgL[:,:,:,:208])
        disp_grad_loss = disp_smoothness(disp_est,imgL)
        # disp_grad_loss = disp_smoothness(disp_est[:,:,:,:208],imgL[:,:,:,:208])
        disp_grad_loss = torch.sum(disp_grad_loss)*0.1
        all_losses.append(disp_grad_loss)

        # all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    # return sum(all_losses)
    return sum(all_losses),disp_grad_loss,warped_left,warped_right#torch.sum(disp_grad_loss)
    
def model_loss_test_ss_apply_disparity(imgL,imgR,disp_ests):
    weights = [1.0] 
    all_losses = []
    ssim_w = 0.85
    # print(f'disp_est shape = {disp_ests[0].shape} disp_gt shape = {disp_gt.shape} mask shape = {mask.shape}')
    for disp_est, weight in zip(disp_ests, weights):
        # all_losses.append(weight * F.l1_loss(disp_est.unsqueeze(1)[mask], disp_gt[mask], size_average=True))
        # Generate images for image consistency loss
        warped_left = apply_disparity(imgR,-disp_est,imgR.shape[-1])
        warped_right = apply_disparity(imgL,disp_est,imgL.shape[-1])

        l1_left = torch.mean(torch.abs(warped_left-imgL))
        l1_right = torch.mean(torch.abs(warped_right-imgR))

        ssim_left = SSIM(warped_left,imgL)
        ssim_right = SSIM(warped_right,imgR)

        image_loss_left = ssim_w*ssim_left + (1-ssim_w)*l1_left
        image_loss_right = ssim_w*ssim_right + (1-ssim_w)*l1_right
        image_loss = sum(image_loss_left + image_loss_right)
        # print(f'image loss shape = {image_loss.shape}')
        all_losses.append(torch.sum(image_loss))
        # Disparities smoothness loss
        # disp_grad_loss = disp_smoothness(disp_est,imgL)
        # all_losses.append(torch.sum(disp_grad_loss))

    return sum(all_losses),warped_left,warped_right