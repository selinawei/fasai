import os
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
                group.add_argument("--" + key, nargs="+", type=t, default=value)
            else:
                group.add_argument("--" + key, type=t, default=value)

    def extract(self, args:Namespace):
        for arg in vars(args).items():
            # if vars set in args, replace them
            if arg[0] in vars(self):
                setattr(self, arg[0], arg[1])

class Config(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        ## system
        self.quiet = False
        self.detect_anomaly = False
        self.world_size = 1
        self.seed = 42
        self.ip = "localhost"
        self.port = 9888
        ## checkpoint
        # self.exp_name = "exp_test"
        # self.dataset = "gesai"
        # self.data_path = "./SAI-data-align"
        # self.contents = ["simple_align"]
        self.exp_name = "FEASAI_net_4.6_version"
        self.dataset = "FEASAI"
        self.data_path = "./data"
        self.contents = ["FEA_processed_new", "Real_Fea_data"] # "Real_Fea_data"
        ## dataset
        self.ts = 64
        self.rand_clip = True
        self.rand_flip = False
        self.occ_frame_num = 30
        # net
        self.img_size = 256
        self.framenet_n_scale = 2
        self.framenet_embed_dim = 64
        self.framenet_n_layers = 4
        self.depthnet_n_scale = 2
        self.depthnet_embed_dim = 64
        self.depthnet_n_layers = 4
        #
        
        # optim
        self.bs = 3
        self.lr = 2e-4
        self.lr_end = 1e-5
        self.max_epoch = 300
        self.fix_bn_epoch = 240
        self.per_save_model_epoch = 20
        self.loss_weights = [1.0,1.0,0.1]
        super().__init__(parser, "Config", sentinel)

    def extract(self, args):
        super().extract(args)
        self.data_path = os.path.abspath(self.data_path)

def get_combined_args(config:Config):
    cfgfile_string = "Namespace()"
    results_dir = os.path.abspath(f"./results/{config.exp_name}")
    try:
        cfgfilepath = os.path.join(results_dir, "cfg_args")
        print("Looking for config file in", results_dir)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)
    for k,v in vars(args_cfgfile).items():
        if v != None: setattr(config,k,v)
    return config