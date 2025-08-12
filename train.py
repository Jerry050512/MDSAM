from dataset.sod_dataset import getDefectDataloader

import torch
from tqdm import tqdm
import albumentations as albu
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2

import os
import shutil
import argparse
from torch.utils.tensorboard import SummaryWriter

from utils import *
from test import test
from eval import eval
from model import DefectSAM

from config import train_cfg, test_cfg


def main(args):
    # ------- model setting -------
    # define model
    net = DefectSAM(args.img_size)

    # # load SAM weights
    # load_info = load_sam(net, args.sam_pre, args.img_size)
    
    """print("--- Initialization of parameters ---")
    for k in load_info.missing_keys:
        print(k)"""
    
    # ------- define optimizer -------
    # adjust learning rate
    param_last = []
    param_new = []
    for k,v in net.named_parameters():
        if 'image_encoder' in k:
            if 'adapter' in k:
                param_new.append(v)
            elif 'neck' in k:
                param_last.append(v)
            else:
                v.requires_grad = False
        else:
            if any(x in k for x in ['transformer', 'iou_token', 'output_upscaling']):
                param_last.append(v)
            else:
                param_new.append(v)
    
    optimizer = torch.optim.AdamW([{"params": param_new, "lr": args.lr}, 
                       {"params" : param_last, "lr" : args.lr * 0.1}], 
                       weight_decay=args.weight_decay)
    
    # ------- load training data -------
    
    dataloader_train = getDefectDataloader(train_cfg, img_size=args.img_size)

    # ------- define loss function -------
    loss_func = LossFunc

    # ------- start training -------
    print("------- Start Training -------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    best_score = 0.0
    best_epoch = 0
    writer = SummaryWriter(args.log_dir)

    for epoch in range(1, args.epochs+1):

        print("epochs {} start".format(epoch))

        if epoch <= args.epochs_warm:
            lr_epoch = args.lr * epoch / args.epochs_warm
        else:
            lr_epoch = args.lr * (0.98 ** (epoch - args.epochs_warm))

        for index, param_group in enumerate(optimizer.param_groups):
            if index==0:
                param_group['lr'] = lr_epoch
            else:
                param_group['lr'] = lr_epoch * 0.1

        net.train()
        current_score = 0.0
        for index, data in enumerate(dataloader_train):
            img, mask = data['img'].to(device).to(torch.float32), data['mask'].to(device).unsqueeze(1)

            optimizer.zero_grad()

            pred = net(img)
            loss = loss_func(pred, mask)

            loss.backward()
            optimizer.step()

            if (index + 1) % 100 == 0:
                print("[epoch: %d/%d, batch: %d/%d] train loss: %.4f" % 
                (epoch, args.epochs, (index + 1) * args.batch_size, len(dataloader_train.dataset), loss.item()))

        # testing
        test(net, test_cfg, epoch)

        # evaluation
        # TODO: single dataset evaluation
        score = eval(test_cfg)

        current_score = score['total']
        for k in ['MAE', 'F', 'S']:
            writer.add_scalar(k, score[k], epoch)
        
        print("current score: {}".format(current_score))
        with open(f"{args.save_root}/score.log", "a+") as f:
            f.write(f"epoch: {epoch}, current score: {current_score}\n")
        rm_epoch = epoch
        #save the best result
        if current_score > best_score:
            best_score = current_score
            rm_epoch = best_epoch
            best_epoch = epoch
        pred_dir_to_remove = test_cfg.save_root / f"{test_cfg.model_code}-{rm_epoch}"
        if pred_dir_to_remove.exists():
            shutil.rmtree(pred_dir_to_remove)

        print("save epoch {} in {}".format(epoch, "{}/{}-{}.pth".format(args.ckpt_root, args.model_code, epoch)))
        args.ckpt_root.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(),"optimizer":optimizer.state_dict()}, "{}/{}-{}.pth".format(args.ckpt_root, args.model_code, epoch))
        print("best epoch: {}, best score: {}".format(best_epoch, best_score))

        with open(f"{args.save_root}/mae.log", "a+") as f:
            f.write(f"\nbest epoch: {best_epoch}, mae: {best_score}\n")
        
        best_ckpt_path = "{}/{}-{}.pth".format(args.ckpt_root, args.model_code, best_epoch)
        save_root_path = "{}/{}-{}.pth".format(args.save_root, args.model_code, best_epoch)
        shutil.move(best_ckpt_path, save_root_path)

        print("Best epoch checkpoint moved to: {}".format(save_root_path))

    print("------- Training Done -------")

    return

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # training
    p.add_argument("--epochs_warm", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-4 * .5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=384)

    # root
    p.add_argument("--dataset_path", type=str, default="/home/sky/Dataset/2505/")
    p.add_argument("--sam_pre", type=str, default="./pretrained/sam_vit_b_01ec64.pth")
    p.add_argument("--log_dir", type=str, default='./logs')
    args = p.parse_args()
    args = train_cfg.load_opt(args)
    test_cfg.load_opt(args)

    if os.path.exists(args.log_dir) == False:
        os.makedirs(args.log_dir)

    main(args)