import os
import cv2
import torch.utils.data as data
from options import args


class TestData(data.Dataset):
    def __init__(self, transform=None):
        super(TestData, self).__init__()
        self.transform = transform
        self.dir_prefix = args.dir_test

        self.img1_dir = os.listdir(self.dir_prefix + args.img_type1)
        self.img2_dir = os.listdir(self.dir_prefix + args.img_type2)

    def __getitem__(self, index):
        self.img1_dir.sort()
        self.img2_dir.sort()
        img_name = str(self.img1_dir[index])
        if args.img_type1 == 'CT/':
            img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(self.dir_prefix + args.img_type2 + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img_name, img1, img2  # img1[YCrCb]:3,256,256  img2[Gray]:1,256,256
        else:
            img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index])
            # img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(self.dir_prefix + args.img_type2 + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)  # CT/PET/SPECT 256,256,3

            img1_Y = img1[:, :, 0:1]
            img1_CrCb = img1[:, :, 1:3].transpose(2, 0, 1)

            if self.transform:
                img1_Y = self.transform(img1_Y)
                img2 = self.transform(img2)

            return img_name, img1_Y, img2, img1_CrCb  # img1[YCrCb]:3,256,256  img2[Gray]:1,256,256

    def __len__(self):
        assert len(self.img1_dir) == len(self.img2_dir)
        return len(self.img1_dir)
