import torch
from torch.utils.data import Dataset

class Datareader(Dataset):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def prefetch_data(self):
        raise NotImplementedError

    def __getitem__(self, index:int):
        raise NotImplementedError  

    def __len__(self):
        raise NotImplementedError

class PackedData:
    def __init__(self) -> None:
        # packed
        self.voxelgrid:torch.Tensor = None
        self.time:torch.Tensor = None
        self.fx:torch.Tensor = None
        self.v:torch.Tensor = None
        self.gt:torch.Tensor = None
        self.other:torch.Tensor = None
        self.gt_t:torch.Tensor = None
        self.occ_aps:torch.Tensor = None
        self.occ_t:torch.Tensor = None
        self.other_t:torch.Tensor = None
        self.depth_gt:torch.Tensor = None
        # runtime
        self.pred_img_depth = None
        self.pred_ev_depth = None
        self.self_warped_depth_img = None
        self.self_warped_depth_ev = None
        self.cross_warped_depth_img = None
        self.cross_warped_depth_ev = None
        self.ev_ref_frame = None
        self.img_ref_frame = None
        self.ev_depth_frame = None
        self.img_depth_frame = None
        self.pred_frame = None
        self.epoch = None

    def cuda(self):
        for key in vars(self).keys():
            if isinstance(getattr(self,key),torch.Tensor):
                setattr(self,key,getattr(self,key).cuda())
        return self
    
    def get(self):
        var_dict = vars(self)
        new_var_dict = dict()
        for key in var_dict.keys():
            if isinstance(getattr(self,key),torch.Tensor):
               new_var_dict[key] = var_dict[key]
        return new_var_dict
    
    def set(self,var_dict:dict):
        for key in var_dict.keys():
            setattr(self,key,var_dict[key])
        return self
