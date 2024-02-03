from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def transformation_from_parameters_gsr(R_mat, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """

    R = torch.zeros((R_mat.shape[0], 4, 4),dtype=torch.float32).to(device=R_mat.device)
    # R = rot_from_axisangle(axisangle)
    R[:,:3,:3] = R_mat
    R[:,3,3]   = 1.
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        # print(f'T shape = {T.shape} R shape = {R.shape}')
        M = torch.matmul(T, R)

    return M

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        # print(f'inv_K dev = {inv_K.device} self.pix_coords dev = {self.pix_coords.device}')
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        # print(f'in backproject depth method depth shape = {depth.shape} cam_points shape = {cam_points.shape}')
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # print(f'K shape = {K.shape} T shape = {T.shape}')
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords
    
class Synthesize_view_scared(nn.Module):

    def __init__(self,batch,h,w,view_to_syn,device):
        super(Synthesize_view_scared,self).__init__()
        assert view_to_syn in ('left','right')
        self.back_project = BackprojectDepth(batch,h,w).to(device=device)
        self.project_3d = Project3D(batch,h,w).to(device=device)

        # rot_np = np.array([ 1., 1.94856493e-05, -1.52324792e-04, -1.95053162e-05, 1.,
        # -1.29114138e-04, 1.52322275e-04, 1.29117107e-04, 1. ],dtype=np.float32).reshape(3,3)
        trans_np = np.array([ -4.14339018e+00, -2.38197036e-02, -1.90685259e-03 ],dtype=np.float32)
        
        K_u = np.array([ 1.03530811e+03, 0., 5.96955017e+02, 0., 1.03508765e+03,
        5.20410034e+02, 0., 0., 1. ],dtype=np.float32).reshape(3,3)
        
        K_u[0,:] = K_u[0,:]/1280.0
        K_u[1,:] = K_u[1,:]/1024.0

        K_u4 = np.zeros((4,4),dtype=np.float32)
        K_u4[:3,:3] = K_u
        K_u4[3,3] = 1.0

        K_d = np.array([ 1.03517419e+03, 0., 6.88361877e+02, 0., 1.03497900e+03,
        5.21070801e+02, 0., 0., 1. ],dtype=np.float32).reshape(3,3)

        K_d[0,:] = K_d[0,:]/1280.0
        K_d[1,:] = K_d[1,:]/1024.0


        K_d4 = np.zeros((4,4),dtype=np.float32)
        K_d4[:3,:3] = K_d
        K_d4[3,3] = 1.0
        # Rot   = torch.from_numy(rot_np).to(device=img.device)
        # Using identity rotation matrix
        self.Rot   = torch.from_numpy(np.eye(3,dtype=np.float32)).to(device=device)

        # Following gave good result but the prediction looks like depth map not disparity map
        # Assuming up as right and down as left
        if view_to_syn == 'right':
            # right is source, left is target
            self.source_K    = torch.from_numpy(K_u4).to(device=device)
            self.target_K    = torch.from_numpy(K_d4).to(device=device)
            self.trans_vec   = torch.from_numpy(trans_np).to(device=device)

        else:
            self.source_K    = torch.from_numpy(K_d4).to(device=device)
            self.target_K    = torch.from_numpy(K_u4).to(device=device)
            self.trans_vec   = torch.from_numpy(-trans_np).to(device=device)
        
        # # Assuming up as right and down as left
        # if view_to_syn == 'left':
        #     # right is source, left is target
        #     self.source_K    = torch.from_numpy(K_u4).to(device=device)
        #     self.target_K    = torch.from_numpy(K_d4).to(device=device)
        #     self.trans_vec   = torch.from_numpy(-trans_np).to(device=device)

        # else:
        #     self.source_K    = torch.from_numpy(K_d4).to(device=device)
        #     self.target_K    = torch.from_numpy(K_u4).to(device=device)
        #     self.trans_vec   = torch.from_numpy(trans_np).to(device=device)

        self.source_K   = self.source_K.unsqueeze(0).repeat(batch,1,1)
        self.target_K   = self.target_K.unsqueeze(0).repeat(batch,1,1)
        self.Rot        = self.Rot.unsqueeze(0).repeat(batch,1,1)
        self.trans_vec  = self.trans_vec.unsqueeze(0).repeat(batch,1,1)
        
    def forward(self,img,pred_disp):

        scaled_pred_disp,pred_depth = disp_to_depth(pred_disp,10.0,255.0)

        # print(f'Synthesize_view forward pred_disp shape = {pred_disp.shape} pred_depth shape = {pred_depth.shape}')
        cam_pts = self.back_project(pred_depth,torch.inverse(self.source_K))
        Extrinsic_mat = transformation_from_parameters_gsr(self.Rot,self.trans_vec)
        pix_coords = self.project_3d(cam_pts,self.target_K,Extrinsic_mat)
        syn = F.grid_sample(img,pix_coords,padding_mode="border")
        return syn
    
class Synthesize_view_hamlyn(nn.Module):

    def __init__(self,batch,h,w,view_to_syn,device):
        super(Synthesize_view_hamlyn,self).__init__()
        assert view_to_syn in ('left','right')
        self.back_project = BackprojectDepth(batch,h,w).to(device=device)
        self.project_3d = Project3D(batch,h,w).to(device=device)

        # rot_np = np.array([ 0.999906, 0.006813, -0.011930, -0.006722, 0.999948, 0.007680,0.011981, -0.007599, 0.999899 ],dtype=np.float32).reshape(3,3)
        # self.Rot   = torch.from_numpy(rot_np).to(device=device)
        self.Rot   = torch.from_numpy(np.eye(3,dtype=np.float32)).to(device=device)

        
        K = np.array([383.1901395, 0., 155.9659519195557, 0., 383.1901395, 124.3335933685303,
        0., 0., 1. ],dtype=np.float32).reshape(3,3)
        
        K[0,:] = K[0,:]/320.0
        K[1,:] = K[1,:]/240.0

        K4 = np.zeros((4,4),dtype=np.float32)
        K4[:3,:3] = K
        K4[3,3] = 1.0

        self.source_K    = torch.from_numpy(K4).to(device=device)
        self.target_K    = torch.from_numpy(K4).to(device=device)

        trans_np = np.array([ 5.382236, 0.067659,  -0.039156 ],dtype=np.float32)
        if view_to_syn == 'right':
            # right is source, left is target
            self.trans_vec   = torch.from_numpy(-trans_np).to(device=device)

        else:
            self.trans_vec   = torch.from_numpy(trans_np).to(device=device)
        

        self.source_K   = self.source_K.unsqueeze(0).repeat(batch,1,1)
        self.target_K   = self.target_K.unsqueeze(0).repeat(batch,1,1)
        self.Rot        = self.Rot.unsqueeze(0).repeat(batch,1,1)
        self.trans_vec  = self.trans_vec.unsqueeze(0).repeat(batch,1,1)
        
    def forward(self,img,pred_disp):

        scaled_pred_disp,pred_depth = disp_to_depth(pred_disp,10.0,255.0)

        # print(f'Synthesize_view forward pred_disp shape = {pred_disp.shape} pred_depth shape = {pred_depth.shape}')
        cam_pts = self.back_project(pred_depth,torch.inverse(self.source_K))
        Extrinsic_mat = transformation_from_parameters_gsr(self.Rot,self.trans_vec)
        pix_coords = self.project_3d(cam_pts,self.target_K,Extrinsic_mat)
        syn = F.grid_sample(img,pix_coords,padding_mode="border")
        return syn

class Synthesize_view_hamlyn2(nn.Module):
    # Source https://discuss.pytorch.org/t/warping-images-using-disparity-maps-for-stereo-matching/127234
    # Source https://github.com/OniroAI/MonoDepth-PyTorch/blob/0b7d60bd1dab0e8b6a7a1bab9c0eb68ebda51c5c/loss.py#L40

    def __init__(self,batch,h,w,view_to_syn,device):
        super(Synthesize_view_hamlyn2,self).__init__()
        assert view_to_syn in ('left','right')
        self.view_to_syn = view_to_syn
        self.w = w
        
    def forward(self,img,disp):
        batch_size, _, height, width = img.size()

        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size,
                    height, 1).type_as(img)
        y_base = torch.linspace(0, 1, height).repeat(batch_size,
                    width, 1).transpose(1, 2).type_as(img)

        # print(f'disp shape = {disp.shape} x_base shape = {x_base.shape} y_base shape = {y_base.shape}')
        # Apply shift in X direction
        x_shifts = disp[:, 0, :, :] /(self.w) # Disparity is passed in NCHW format with 1 channel

        # Initial trial        
        if self.view_to_syn == 'right':
            flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

        elif self.view_to_syn == 'left':
            flow_field = torch.stack((x_base - x_shifts, y_base), dim=3)


        # In torch F grid_sample, coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',padding_mode='zeros')

        # print(f'img dtype = {img.dtype} disp dtype = {disp.dtype} output dtype = {output.dtype}')
        return output
    
def apply_disparity( img, disp,norm=1.0):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    x_shifts = x_shifts/img.shape[-1]
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros')

    return output