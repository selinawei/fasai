import sys
from argparse import ArgumentParser
from arguments.prepare_data import DatasetConfig
from datareader.dataset import DatasetFactory
from utils.system import safe_state,set_seed

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    config = DatasetConfig(parser)
    args = parser.parse_args(sys.argv[1:])
    config.extract(args)
    safe_state(config.quiet)
    set_seed(config.seed)
    dataset = DatasetFactory().get(config.dataset,config,None)
    dataset.preprocess(config)
    
