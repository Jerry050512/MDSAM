import argparse
from model.mdsam import MDSAM
from dataset.sod_dataset import getSODDataloader
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
import cv2
import shutil
import numpy as np
from collections import OrderedDict
import time
from utils.loss import LossFunc
from utils.AvgMeter import AvgMeter

from config import train_cfg, test_cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--seed", type=int, default = 42
    )
    parser.add_argument(
        "--warmup_period", type = int, default = 5,    
    )
    parser.add_argument(
        "--batch_size", type = int, default = 8,
    )
    parser.add_argument(
        "--num_workers", type = int, default = 0
    )
    parser.add_argument(
        "--epochs", type = int, default=80
    )
    parser.add_argument(
        "--lr_rate", type = float, default = 0.0005,
    )
    parser.add_argument(
        "--img_size", type = int, default = 512
    )
    parser.add_argument(
        "--data_path", type = str, default='./datasets/DUTS', help="the postfix must to be DUTS"
    )
    parser.add_argument(
        "--sam_ckpt", type = str
    )
    parser.add_argument(
        "--save_dir", type = str, default = './output'
    )
    parser.add_argument(
        "--resume", type = str, default = "", help="If you need to train from begining, make sure 'resume' is empty str. If you want to continue training, set it to the previous checkpoint."
    )
    parser.add_argument( # 新增一个参数来指定GPU
        "--gpu_id", type=int, default=0, help="GPU ID to use for training."
    )

    args = parser.parse_args()

    return args

def trainer(net, dataloader, loss_func, optimizer):
    net.train()
    loss_avg = AvgMeter()
    mae_avg = AvgMeter()
    print("start trainning")
    
    start = time.time()

    sigmoid = torch.nn.Sigmoid()
    for data in tqdm(dataloader):
        img = data["img"].to(device).to(torch.float32)
        label = data["mask"].to(device).unsqueeze(1)
        
        optimizer.zero_grad()

        out, coarse_out = net(img)

        out = sigmoid(out)
        coarse_out = sigmoid(coarse_out)

        loss_out = loss_func(out, label)
        loss_coarse = loss_func(coarse_out, label)

        loss = loss_out + loss_coarse
        
        loss_avg.update(loss.item(), img.shape[0])

        img_mae = torch.mean(torch.abs(out - label))

        mae_avg.update(img_mae.item(),n=img.shape[0])

        loss.backward()
        optimizer.step()
    
    temp_cost=time.time()-start
    print("loss:{}, mae:{}, cost_time:{:.0f}m:{:.0f}s".format(loss_avg.avg, mae_avg.avg, temp_cost//60, temp_cost%60))
 
def valer(net, dataloader, epoch):
    net.eval()
    print("start valling")

    save_path = test_cfg.save_root / f"{test_cfg.model_code}-{epoch}" / f"{test_cfg.dataset}"
    save_path.mkdir(parents=True, exist_ok=True)

    start = time.time()

    sigmoid = torch.nn.Sigmoid()
    mae_avg = AvgMeter()
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Validating epoch {epoch}"):
                
            img = data["img"].to(device).to(torch.float32)
            ori_label = data['ori_mask'].to(device)
            mask_name = data['mask_name'][0]
            
            out, coarse_out= net(img)
            out = sigmoid(out)
            out = torch.nn.functional.interpolate(out, [ori_label.shape[1],ori_label.shape[2]], mode = 'bilinear', align_corners = False)

            # Since the float values are converted to int when saving the mask,
            # multiple decimal will be lost, which may result in minor deviations from the evaluation code.
            img_mae=torch.mean(torch.abs(out - ori_label))

            mae_avg.update(img_mae.item(),n=1)

            pred = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(save_path / mask_name), pred)

    temp_cost=time.time() - start
    print("val_mae:{}, cost_time:{:.0f}m:{:.0f}s".format(mae_avg.avg, temp_cost//60, temp_cost%60))

    return mae_avg.avg



def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed

def reshapeRel(k, rel_pos_params, img_size):
    if not ('2' in k or '5' in  k or '8' in k or '11' in k):
        return rel_pos_params
    
    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]

def load(net,ckpt, img_size):
    ckpt=torch.load(ckpt,map_location='cpu')
    from collections import OrderedDict
    dict=OrderedDict()
    for k,v in ckpt.items():
        #把pe_layer改名
        if 'pe_layer' in k:
            dict[k[15:]] = v
            continue
        if 'pos_embed' in k :
            dict[k] = reshapePos(v, img_size)
            continue
        if 'rel_pos' in k:
            dict[k] = reshapeRel(k, v, img_size)
        elif "image_encoder" in k:
            if "neck" in k:
                #Add the original final neck layer to 3, 6, and 9, initialization is the same.
                for i in range(4):
                    new_key = "{}.{}{}".format(k[:18], i, k[18:])
                    dict[new_key] = v
            else:
                dict[k]=v
        if "mask_decoder.transformer" in k:
            dict[k] = v
        if "mask_decoder.iou_token" in k:
            dict[k] = v
        if "mask_decoder.output_upscaling" in k:
            dict[k] = v
    state = net.load_state_dict(dict, strict=False)
    return state

if __name__ == "__main__":

    args = parse_args()
    args = train_cfg.load_opt(args)
    test_cfg.load_opt(args)

    print("start training, batch_size: {}, lr_rate: {}, warmup_period: {}, save to {}".format(args.batch_size, args.lr_rate, args.warmup_period, args.ckpt_root))
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}")

    #Model definition and loading SAM pre-trained weights
    net = MDSAM(args.img_size).to(device)
    if args.resume != "":
        state = load(net, args.sam_ckpt, args.img_size)
        print(state)

    trainLoader = getSODDataloader(train_cfg, img_size= args.img_size)
    valLoader = getSODDataloader(test_cfg, img_size= args.img_size)

    loss_func = LossFunc

    #freeze layers and define different lr_rate for layers
    hungry_param = []
    full_param = []
    for k,v in net.named_parameters():
        if "image_encoder" in k:
            if "adapter" in k:
                hungry_param.append(v)
            elif "neck" in k:
                full_param.append(v)
            else:
                v.requires_grad = False
        else:
            if "transformer" in k:
                full_param.append(v)
            elif "iou" in k:
                full_param.append(v)
            elif "mask_tokens" in k:
                hungry_param.append(v)
            elif "pe_layer" in k:
                full_param.append(v)
            elif "output_upscaling" in k:
                full_param.append(v)
            else:
                hungry_param.append(v)      
    
    optimizer = torch.optim.AdamW([{"params": hungry_param, "lr": args.lr_rate}, {"params" : full_param, "lr" : args.lr_rate * 0.1}], weight_decay=1e-5)
    
    best_mae = 1
    best_epoch = 0

    start_epoch = 1

    #resume from checkpoint
    if args.resume != "":
        start_epoch = int(args.resume.split("/")[-1].split(".")[0][11:]) + 1
        resume_dict = torch.load(args.resume, map_location= "cpu")
        optimizer.load_state_dict(resume_dict["optimizer"])
        net_dict = OrderedDict()

        for k,v in resume_dict['model'].items():
            if "module." in k:
                net_dict[k[7:]] = v
            else:
                net_dict[k] = v
        state = net.load_state_dict(net_dict)
        print(state)

    for i in range(start_epoch, args.epochs + 1):

        #lr_rate setting
        if i <= args.warmup_period:
            _lr = args.lr_rate * i / args.warmup_period
        else:
            _lr = args.lr_rate * (0.98 ** (i - args.warmup_period))

        t = 0
        for param_group in optimizer.param_groups:
            if t == 0:
                param_group['lr'] = _lr
            else:
                param_group['lr'] = _lr * 0.1
            t += 1
        
        print("epochs {} start".format(i))

        trainer(net, trainLoader, loss_func, optimizer)

        mae = valer(net, valLoader, i)

        print("current mae:{}".format(mae))
        with open(f"{args.save_root}/mae.log", "a+") as f:
            f.write(f"epoch:{i}, mae:{mae}\n")
        rm_epoch = i
        #save the best result
        if mae < best_mae:
            best_mae = mae
            rm_epoch = best_epoch
            best_epoch = i
        pred_dir_to_remove = test_cfg.save_root / f"{test_cfg.model_code}-{rm_epoch}"
        if pred_dir_to_remove.exists():
            shutil.rmtree(pred_dir_to_remove)

        print("save epoch {} in {}".format(i, "{}/{}-{}.pth".format(args.ckpt_root, args.model_code, i)))
        args.ckpt_root.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(),"optimizer":optimizer.state_dict()}, "{}/{}-{}.pth".format(args.ckpt_root, args.model_code, i))
        print("best epoch:{}, mae:{}".format(best_epoch,best_mae))

    with open(f"{args.save_root}/mae.log", "a+") as f:
        f.write(f"\nbest epoch:{best_epoch}, mae:{best_mae}\n")
    
    best_ckpt_path = "{}/{}-{}.pth".format(args.ckpt_root, args.model_code, best_epoch)
    save_root_path = "{}/{}-{}.pth".format(args.ckpt_root, args.model_code, best_epoch)
    shutil.move(best_ckpt_path, save_root_path)

    print("Best epoch checkpoint moved to: {}".format(save_root_path))
    print("Training Done!")

