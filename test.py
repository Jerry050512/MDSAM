import torch
import albumentations as albu
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2

import argparse
from tqdm import tqdm

from utils import *
from model import DefectSAM
from dataset.sod_dataset import getDefectDataloader

def main(args):
    # ------- model setting -------
    net = DefectSAM(args.img_size)
    net.load_state_dict(torch.load(args.model_dir))
    
    if torch.cuda.is_available():
        net.cuda()

    test(net, args)

    return

def test(net, args, epoch=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigmoid = torch.nn.Sigmoid()
    save_path = args.save_root / f"{args.model_code}-{epoch}" / f"{args.dataset}"
    save_path.mkdir(parents=True, exist_ok=True)
    args.pred_dir = save_path
    
    # ------- start testing -------
    print("------- Start Testing -------")

    net.eval()
    dataloader_test = getDefectDataloader(args, img_size=args.img_size)

    #for index, data in tqdm(enumerate(dataloader_test), total=dataloader_test.__len__()):
    for index, data in enumerate(dataloader_test):

        img = data["img"].to(device).to(torch.float32)
        ori_label = data['ori_mask'].to(device)
        mask_name = data['mask_name'][0]
        
        out= net(img)
        # out = sigmoid(out)
        out = torch.nn.functional.interpolate(out, [ori_label.shape[1],ori_label.shape[2]], mode = 'bilinear', align_corners = False)

        pred = (out.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(save_path / mask_name), pred)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=384)

    # root
    p.add_argument("--dataset_path", type=str, default="/home/sky/Dataset/2505/")
    p.add_argument("--model_dir", type=str, default="./model_save/Best.pth")
    p.add_argument("--pre_dir", type=str, default="./predicts/")
    p.add_argument("--dataset_list", type=list, default=['ESDIs','CrackSeg9k','ZJU-Leaper'])
    args = p.parse_args()

    main(args)