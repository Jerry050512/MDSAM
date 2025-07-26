from pathlib import Path

class Config:
    def __init__(
                self,
                mode='train',
                model='MDSAM',
                dataset='DUTS',
                model_code='duts',
                img_ext='.jpg',
                mask_ext='.png',
                cuda_num=1,
                dataset_root=Path('/kaggle/temp/datasets'),
                ckpt_root = Path('/kaggle/temp/ckpt'),
                save_root = Path('/kaggle/working/output'),
                opt=None,
                **kwargs
            ):
        self.mode = mode
        self.model = model
        self.dataset = dataset
        self.model_code = model_code
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.cuda_num = cuda_num
        self.dataset_root = dataset_root
        self.ckpt_root = ckpt_root
        self.save_root = save_root
        self.prior_opt = kwargs
        self.opt = opt

        self.build()

    def build(self):
        self.dataset_dir = self.dataset_root / self.dataset
        self.img_dir = self.dataset_dir / f'Image_{self.mode}'
        self.mask_dir = self.dataset_dir / f'GT_{self.mode}'
        self.eval_img_dir = self.dataset_dir / f'Image_test'
        self.eval_mask_dir = self.dataset_dir / f'GT_test'
        self.map_dict = {
            'GPU_ID': 'cuda_num', 
            'gpu_id': 'cuda_num',
            'train_dataset': 'dataset',
            'eval_dataset': 'dataset', 
            'test_dataset': 'dataset',
        }
    
    def load_opt(self, opt):
        self.opt = opt
        return self

    def __getattr__(self, name):
        if name in self.map_dict:
            name = self.map_dict[name]
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.prior_opt:
            return self.prior_opt[name]
        elif self.opt is not None and hasattr(self.opt, name):
            return getattr(self.opt, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        

train_cfg = Config(cuda_num=0)
test_cfg = Config(mode='test', cuda_num=0)