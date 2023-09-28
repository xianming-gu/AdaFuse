import argparse

parser = argparse.ArgumentParser(description='Options')

parser.add_argument('--DEVICE', type=str, default='cuda:3')

parser.add_argument('--dir_test', type=str, default='HMIFDatasets/CT-MRI/test/')  # CT PET SPECT
parser.add_argument('--img_type1', type=str, default='CT/')  # CT PET SPECT
parser.add_argument('--img_type2', type=str, default='MRI/')

parser.add_argument('--model_save_path', type=str, default='Checkpoint/Model_CT_MRI.pth')  # CT PET SPECT
parser.add_argument('--img_save_dir', type=str, default='Results/CT-MRI')  # CT PET SPECT

args = parser.parse_args()
