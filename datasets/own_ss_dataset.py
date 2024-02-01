import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, get_disp_transform #, pfm_imread
# from data_io import get_transform, read_all_lines #, pfm_imread
import cv2

class OwnSSDataset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        # self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.left_filenames, self.right_filenames = self.load_path(list_filename)

        self.training = training
        self.focal_len, self.baseline_width = 149.97001648, 4000.0
        # self.full_w, self.full_h = 263,365
        # self.full_w, self.full_h = 208,365
        self.full_w, self.full_h = 256,256

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        # disp_images = [x[2] for x in splits]
        return left_images, right_images #, disp_images

    def load_image(self, filename):
        # return Image.open(filename).convert('RGB').crop((0,0,self.full_w,self.full_h))
        return Image.open(filename).convert('RGB').resize((self.full_w,self.full_h))

    # def load_disp(self, filename):
    #     # data = Image.open(filename).convert('L').crop((0,0,self.full_w,self.full_h))
    #     data = Image.open(filename).convert('L').resize((self.full_w,self.full_h))

    #     data = np.array(data).astype(dtype=np.float32)
    #     data[data>256.0] = 256.0
    #     data[data==0]  = 256.0
    #     # data = 256.0 - data
    #     data = (self.baseline_width * (self.focal_len / self.full_w)) / data

    #     data = np.ascontiguousarray(data)

    #     return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        # disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            w, h = left_img.size

            # self.full_w, self.full_h = 263,365
            # self.full_w, self.full_h = 208,365
            # crop_w, crop_h = 256, 320
            # crop_w, crop_h = 192, 320

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            # Included by GSR
            # disp_processed = get_disp_transform()
            # disparity = disp_processed(disparity)

            return {"left": left_img,
                    "right": right_img,
                    # "disparity": disparity,
                }
        else:
            w, h = left_img.size
            # crop_w, crop_h = 960, 512
            crop_w, crop_h = self.full_w, self.full_h

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            # Included by GSR
            # disp_processed = get_disp_transform()
            # disparity = disp_processed(disparity)

            return {"left": left_img,
                    "right": right_img,
                    # "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0,
                    "left_filename": self.left_filenames[index]}
