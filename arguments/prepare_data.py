import os
from torch.utils.data import dataloader
from argparse import ArgumentParser, Namespace



class ParamGroup:
    def __init__(self,parser:ArgumentParser,name:str,fill_none=False):
        group = parser.add_argument_group(name)
        self_var_dict = vars(self).items()
        # add self vars to Parser's args group
        for key,value in self_var_dict:
            t = type(value)
            value = value if not fill_none else None 
            if t == bool:
                group.add_argument("--" + key, default=value, action="store_true")
            elif t == list:
                group.add_argument("--" + key, nargs="*", type=t, default=value)
            else:
                group.add_argument("--" + key, type=t, default=value)

    def extract(self, args:Namespace):
        for arg in vars(args).items():
            # if vars set in args, replace them
            if arg[0] in vars(self):
                setattr(self, arg[0], arg[1])
                


class DatasetConfig(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        ## system
        self.quiet = False
        self.detect_anomaly = False
        self.seed = 42
        ## dataset
        # self.dataset = "gesai"
        # self.data_path = "./SAI-data-align"
        # self.contents = ["simple_raw_align"]
        # self.data_name = "simple_align"
        self.dataset = "FEASAI"
        self.data_path = "/afs/crc.nd.edu/user/l/lwei5/Private/FEA-SAI"
        self.contents = ["data"]
        self.data_name = "FEA_processed"
        self.time_step = 64
        self.val_ratio = 0.1
        self.h = 260
        self.w = 346
        super().__init__(parser, "Config", sentinel)

    def extract(self, args):
        super().extract(args)
        self.data_path = os.path.abspath(self.data_path)