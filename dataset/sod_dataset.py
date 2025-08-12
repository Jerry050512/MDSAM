import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import cv2
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data.distributed import DistributedSampler
from os.path import splitext, basename

class DefectDataset(Dataset):
    def __init__(self, cfg, transform):
        
        self.imgs_list=list(cfg.img_dir.glob('*'+cfg.img_ext))
        self.masks_list=list(cfg.mask_dir.glob('*'+cfg.mask_ext))
        self.imgs_list.sort()
        self.masks_list.sort()

        self.transform=transform

        self.mode = cfg.mode

        # if cfg.mode != 'train':
        #     # In order to enable distributed inference across multiple GPUs
        #     start_idx = (int)(local_rank * len(self.imgs_list) / max_rank)
        #     end_idx = (int)(local_rank * len(self.imgs_list) / max_rank + len(self.imgs_list) / max_rank)
        #     self.imgs_list = self.imgs_list[start_idx: end_idx]
        #     self.masks_list = self.masks_list[start_idx: end_idx]
    
    def __len__(self):
        return len(self.imgs_list)
    
    
    def __getitem__(self,index):
        # to prevent '\' when using glob in Windows
        img_dir= self.imgs_list[index]
        mask_dir= self.masks_list[index]

        #print(img_dir, mask_dir)

        mask_name = basename(mask_dir)

        img=cv2.imread(img_dir)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        mask=cv2.imread(mask_dir)
        mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

        # load the original resolution mask for calculating metrics when inferencing
        if not self.mode == 'train':
            ori_mask = torch.from_numpy(mask) / 255.0
            
        augmented=self.transform(image=img,mask=mask)
        img = augmented['image']
        mask = augmented['mask'] / 255.0

        if self.mode == 'train':
            return {'img':img, 'mask':mask}
        else:
            return {'img':img, 'mask':mask, 'ori_mask':ori_mask, 'mask_name':mask_name}

def get_augmentation(version=0, img_size = 512):
    if version==0:
        transforms=albu.Compose([
            albu.OneOf([
                    albu.HorizontalFlip(),
                    albu.VerticalFlip(),
                    albu.RandomRotate90()
                ], p=0.5),
            albu.OneOf([
                    albu.MotionBlur(blur_limit=5),
                    albu.MedianBlur(blur_limit=5),
                    albu.GaussianBlur(blur_limit=5),
                    albu.GaussNoise(var_limit=(5.0, 20.0)),
                ], p=0.5),
            albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        transforms=albu.Compose([albu.Resize(img_size, img_size, always_apply=True),
            albu.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    return transforms


def getDefectDataloader(cfg, img_size = 512):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    if cfg.mode == "train":
        transform = get_augmentation(0, img_size)
        dataset = DefectDataset(cfg, transform)
        # sampler = DistributedSampler(dataset)
        dataLoader = DataLoader(dataset,batch_size = batch_size, num_workers = num_workers)
    else:
        transform = get_augmentation(1, img_size)
        # max_rank represents the number of GPUs used for inference
        # local_rank represents the GPU currently in use.
        dataset = DefectDataset(cfg, transform)
        dataLoader = DataLoader(dataset, batch_size = 1, num_workers = num_workers)
    return dataLoader